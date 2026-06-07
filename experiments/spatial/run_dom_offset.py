"""
Dominant-class offset spatial experiment (Pavia-U).

Target signature is computed entirely from the test box data:
    s = sig_dom_weight * mean(dominant labeled class in test box)
      + sig_mean_weight * mean(all pixels in test box)

"Dominant class" = the labeled class (cls != 0) with the most pixels in the
test box.  The resulting signature looks almost identical to a real background
pixel from that class — making this a hard, realistic detection problem where
AMF-style global detectors struggle and spatial adaptation helps.

Pipeline:
    1. Pick boxes interactively:   .venv/bin/python pick_boxes.py --config dom_offset.yaml
    2. Run experiment:             .venv/bin/python -u run_dom_offset_experiment.py --config dom_offset.yaml

Saves (results/dom_offset_<ts>/):
    config.yaml, metrics.json, rocs.json, loss_curves.json
    figures/patches.png     — crops + GT + full scene
    figures/spectra.pdf     — class mean spectra + target signature in raw space
    figures/roc_*.pdf       — ROC curves per target model
    figures/auc_bar.pdf     — AUC bar chart
    models/cfattn.pt, dsm.pt, pca.pkl
"""

import argparse, os, sys, json, pickle, time
from datetime import datetime

import numpy as np
import torch
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP)   # for cfattn_model
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive, dsm_replacement, amf_replacement,
    gmm_glrt, gmm_glrt_replacement, exact_glrt_replacement,
)
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods
from dsm_model import ScoreNet, dsm_loss
from cfattn_model import (
    CFAttnGaussianScoreNet, cfattn_dsm_loss,
    score_cfattn_additive, score_cfattn_replacement,
)

CLS_NAMES = {
    0:'unlabeled', 1:'asphalt',  2:'meadows',   3:'gravel',
    4:'trees',     5:'metal_sheets', 6:'bare_soil', 7:'bitumen',
    8:'bricks',    9:'shadows',
}
CLS_COLORS = {
    1:'#555555', 2:'#4daf4a', 3:'#a65628', 4:'#1a7a1a',
    5:'#ff7f00', 6:'#984ea3', 7:'#000000', 8:'#e41a1c', 9:'#377eb8',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auc_safe(labels, scores):
    try:    return float(roc_auc_score(labels, scores))
    except: return float('nan')

def roc_safe(labels, scores):
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr.tolist(), tpr.tolist(), auc_safe(labels, scores)
    except:
        return [0.,1.], [0.,1.], float('nan')


def save_patches_figure(data_norm, gt, cfg, fig_dir):
    H, W, _ = data_norm.shape
    bands = (60, 30, 10)
    def _rgb(crop):
        rgb = crop[..., list(bands)].astype(np.float32)
        lo = np.percentile(rgb, 2, axis=(0,1), keepdims=True)
        hi = np.percentile(rgb, 98, axis=(0,1), keepdims=True)
        return np.clip((rgb - lo)/(hi - lo + 1e-9), 0, 1)

    fig = plt.figure(figsize=(12, 7))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1,1,1.5], height_ratios=[1,1])
    n_cls = int(gt.max()) + 1

    def _plot_crop(ax_rgb, ax_gt, box, title):
        r0,r1,c0,c1 = box
        ax_rgb.imshow(_rgb(data_norm[r0:r1,c0:c1])); ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        ax_rgb.set_title(f"{title} — false color ({r1-r0}×{c1-c0})")
        gt_sub = gt[r0:r1,c0:c1]
        ax_gt.imshow(gt_sub, cmap='tab10', vmin=0, vmax=max(n_cls-1,9), interpolation='nearest')
        ax_gt.set_title(f"{title} — GT labels"); ax_gt.set_xticks([]); ax_gt.set_yticks([])
        cls_ids, cnts = np.unique(gt_sub, return_counts=True)
        note = "\n".join(f"{CLS_NAMES.get(int(c),'cls'+str(c))}={int(n)}px"
                         for c,n in zip(cls_ids, cnts) if int(c) != 0)
        ax_gt.set_xlabel(note, fontsize=7)

    _plot_crop(fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]),
               cfg['train_box'], 'TRAIN')
    _plot_crop(fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1]),
               cfg['test_box'],  'TEST')

    ax = fig.add_subplot(gs[:,2])
    ax.imshow(_rgb(data_norm))
    for box,col,lab in [(cfg['train_box'],'lime','train'),
                        (cfg['test_box'], 'red', 'test')]:
        r0,r1,c0,c1 = box
        ax.add_patch(plt.Rectangle((c0,r0),c1-c0,r1-r0, lw=2,
                     edgecolor=col, facecolor='none', label=lab))
    ax.set_title(f"Pavia-U ({H}×{W})"); ax.legend(loc='lower right')
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(fig_dir, 'patches.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}", flush=True)


def save_spectra_figure(data_norm, gt, cfg, sig_raw, dom_cls, fig_dir):
    """Raw spectra of all labeled classes + the computed target signature."""
    D_raw = data_norm.shape[-1]
    bands = np.arange(D_raw)

    def cls_in_box(box):
        r0,r1,c0,c1 = box
        return set(int(c) for c in np.unique(gt[r0:r1,c0:c1]) if c != 0)
    present = cls_in_box(cfg['train_box']) | cls_in_box(cfg['test_box'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, box_key, title in [
            (axes[0], 'train_box', 'TRAIN box'),
            (axes[1], 'test_box',  'TEST box')]:
        r0,r1,c0,c1 = cfg[box_key]
        box_flat = data_norm[r0:r1,c0:c1].reshape(-1, D_raw)
        box_gt   = gt[r0:r1,c0:c1].reshape(-1)
        bkg = box_flat[box_gt != 0]
        if len(bkg):
            bm = bkg.mean(0); bs = bkg.std(0)
            ax.fill_between(bands, bm-bs, bm+bs, alpha=0.10, color='gray', label='bkg ±1σ')
            ax.plot(bands, bm, color='gray', lw=1.2, ls='--', label='bkg mean')
        for cls_id in sorted(present):
            mask = box_gt == cls_id
            if not mask.any(): continue
            ax.plot(bands, box_flat[mask].mean(0), lw=1.5,
                    color=CLS_COLORS.get(cls_id,'black'),
                    label=f"{CLS_NAMES.get(cls_id,'cls'+str(cls_id))} (n={mask.sum()})")
        # Target signature
        ax.plot(bands, sig_raw, color='magenta', lw=2.2, ls='-',
                label=f"target sig\n(0.9·{CLS_NAMES.get(dom_cls,'?')} + 0.1·test_mean)")
        ax.set_title(f"{title} — class spectra", fontsize=10)
        ax.set_xlabel('Band index'); ax.set_ylabel('Reflectance (global_max)')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.25)
    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')
    fig.suptitle(f"Target = 0.9·{dom_name}_mean + 0.1·test_mean  "
                 f"(dominant class in test box: {dom_name})", fontsize=9)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'spectra.pdf')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}", flush=True)


def save_roc_figure(rocs, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (fpr,tpr,a) in rocs.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({a:.3f})")
    ax.plot([0,1],[0,1],'k--',lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dom_offset.yaml'))
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    t0 = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'dom_offset_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    mdl_dir = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Load + normalize, PCA                                            #
    # ------------------------------------------------------------------ #
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw = data_norm.shape
    all_flat = data_norm.reshape(-1, D_raw)
    gt_flat  = gt.reshape(-1)
    print(f"Image {H}x{W}x{D_raw}  norm={cfg['norm_mode']}", flush=True)

    D = cfg['latent_dim']
    pca = PCA(n_components=D, random_state=seed).fit(all_flat)
    evr = pca.explained_variance_ratio_.sum()
    print(f"PCA {D_raw}->{D}  explained={evr:.4f}", flush=True)
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)
    with open(os.path.join(mdl_dir, 'pca.pkl'), 'wb') as fh:
        pickle.dump(pca, fh)

    # ------------------------------------------------------------------ #
    # 2. Verify boxes (no hard target class — just report composition)    #
    # ------------------------------------------------------------------ #
    for nm, bx in [('train_box', cfg['train_box']), ('test_box', cfg['test_box'])]:
        r0,r1,c0,c1 = bx
        gt_sub = gt[r0:r1,c0:c1]
        cls_ids, cnts = np.unique(gt_sub, return_counts=True)
        comp = ", ".join(f"{CLS_NAMES.get(int(c),'cls'+str(c))}={int(n)}"
                         for c,n in zip(cls_ids,cnts))
        print(f"  {nm}: {comp}", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Compute target signature from test box data                      #
    # ------------------------------------------------------------------ #
    r0,r1,c0,c1 = cfg['test_box']
    test_raw_box = all_flat.reshape(H, W, D_raw)[r0:r1,c0:c1].reshape(-1, D_raw)
    test_gt_box  = gt[r0:r1,c0:c1].reshape(-1)

    labeled_mask = test_gt_box != 0
    if labeled_mask.any():
        cls_ids, cnts = np.unique(test_gt_box[labeled_mask], return_counts=True)
        dom_cls = int(cls_ids[cnts.argmax()])
        dom_n   = int(cnts.max())
    else:
        raise RuntimeError("Test box has no labeled pixels — cannot determine dominant class.")

    dom_mean  = test_raw_box[test_gt_box == dom_cls].mean(axis=0).astype(np.float32)
    test_mean = test_raw_box.mean(axis=0).astype(np.float32)
    w_dom  = float(cfg.get('sig_dom_weight',  0.9))
    w_mean = float(cfg.get('sig_mean_weight', 0.1))
    sig_raw = (w_dom * dom_mean + w_mean * test_mean).astype(np.float32)  # (D_raw,)
    s_pca   = pca.transform(sig_raw[None]).flatten().astype(np.float32)   # (D,)

    proj_train_pca = pca.transform(all_flat.reshape(H,W,D_raw)[
        cfg['train_box'][0]:cfg['train_box'][1],
        cfg['train_box'][2]:cfg['train_box'][3]].reshape(-1, D_raw))
    tr_std_sig = float((proj_train_pca @ s_pca / (np.linalg.norm(s_pca)+1e-12)).std())
    snr = cfg['amplitude'] * np.linalg.norm(s_pca) / (tr_std_sig + 1e-12)

    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')
    print(f"\nTarget signature: {w_dom}·{dom_name}_mean + {w_mean}·test_mean", flush=True)
    print(f"  Dominant class: cls{dom_cls} ({dom_name})  n={dom_n} px", flush=True)
    print(f"  ||s_raw|| = {np.linalg.norm(sig_raw):.4f}", flush=True)
    print(f"  SNR = {snr:.3f}\n", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Crop train/test + subsample                                      #
    # ------------------------------------------------------------------ #
    k = cfg['k']

    def crop_box(box):
        r0,r1,c0,c1 = box
        sub = torch.tensor(pca_img[r0:r1,c0:c1,:], dtype=torch.float32)
        centers, nbrs = extract_neighborhoods(sub, k)
        return centers.numpy(), nbrs.numpy()

    tr_pix, tr_nbr = crop_box(cfg['train_box'])
    te_pix, te_nbr = crop_box(cfg['test_box'])

    def subsample(pix, nbr, n):
        if len(pix) <= n: return pix, nbr
        idx = rng.choice(len(pix), n, replace=False)
        return pix[idx], nbr[idx]

    tr_pix, tr_nbr = subsample(tr_pix, tr_nbr, cfg['train_n'])
    te_pix, te_nbr = subsample(te_pix, te_nbr, cfg['test_n'])
    print(f"train={len(tr_pix)} px  test={len(te_pix)} px", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Figures: patches + spectra                                       #
    # ------------------------------------------------------------------ #
    save_patches_figure(data_norm, gt, cfg, fig_dir)
    save_spectra_figure(data_norm, gt, cfg, sig_raw, dom_cls, fig_dir)

    # ------------------------------------------------------------------ #
    # 6. Sigma                                                            #
    # ------------------------------------------------------------------ #
    sigma = compute_sigma_from_data(pca.transform(all_flat), cfg['dsm_sigma_rho'])
    baseline_loss = D / sigma**2
    print(f"sigma = {sigma:.5f}  baseline_loss = {baseline_loss:.1f}\n", flush=True)
    loss_curves = {}

    # ------------------------------------------------------------------ #
    # 7a. Train CF-Attention                                              #
    # ------------------------------------------------------------------ #
    M = k*k - 1
    print(f"[CF-Attn] D={D}  h={cfg['cfattn_h']}  K={cfg['cfattn_K']}  M={M}", flush=True)
    cfattn = CFAttnGaussianScoreNet(
        D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
        sigma=sigma, eps=cfg.get('cfattn_eps', 1e-4))

    km = KMeans(n_clusters=cfg['cfattn_K'], init='k-means++',
                n_init=5, random_state=seed, max_iter=100)
    km.fit(tr_pix)
    cfattn.comp_mu.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    print(f"  comp_mu init via k-means++ on {len(tr_pix)} px", flush=True)

    opt_cf = torch.optim.AdamW(cfattn.parameters(),
                               lr=cfg['cfattn_lr'], weight_decay=cfg['weight_decay'])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_cf, T_max=cfg['cfattn_epochs'], eta_min=cfg['cfattn_lr']/20)
    Xtr = torch.tensor(tr_pix, dtype=torch.float32)
    Ntr = torch.tensor(tr_nbr, dtype=torch.float32)
    P, bs = len(Xtr), cfg['cfattn_batch']
    cf_hist = []
    pbar = tqdm(range(1, cfg['cfattn_epochs']+1), desc='CF-Attn', dynamic_ncols=True)
    for _ in pbar:
        perm = torch.randperm(P); tot = 0.; nb = 0
        for i in range(0, P, bs):
            sel = perm[i:i+bs]
            loss, dsm_item = cfattn_dsm_loss(
                cfattn, Xtr[sel], Ntr[sel],
                lam_ent=cfg.get('lam_ent', 0.05),
                lam_div=cfg.get('lam_div', 0.05),
                lam_cov=cfg.get('lam_cov', 1e-5))
            opt_cf.zero_grad(); loss.backward(); opt_cf.step()
            tot += dsm_item; nb += 1
        sched.step()
        cf_hist.append(tot/max(nb,1))
        pbar.set_postfix(dsm=f"{cf_hist[-1]:.3f}", ratio=f"{cf_hist[-1]/baseline_loss:.3f}")
    loss_curves['CF-Attn'] = cf_hist
    cfattn.eval()
    torch.save({'state_dict': cfattn.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'cfattn.pt'))

    # ------------------------------------------------------------------ #
    # 7b. Train DSM                                                       #
    # ------------------------------------------------------------------ #
    print("\n[DSM] training ...", flush=True)
    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'])
    opt_dsm = torch.optim.Adam(dsm_net.parameters(),
                               lr=cfg['dsm_lr'], weight_decay=cfg['weight_decay'])
    X = torch.tensor(tr_pix, dtype=torch.float32)
    dsm_hist = []
    pbar = tqdm(range(1, cfg['dsm_epochs']+1), desc='DSM', dynamic_ncols=True)
    for _ in pbar:
        perm = torch.randperm(len(X)); tot = 0.; nb = 0
        for i in range(0, len(X), cfg['batch_size']):
            b = X[perm[i:i+cfg['batch_size']]]
            loss = dsm_loss(dsm_net, b, sigma)
            opt_dsm.zero_grad(); loss.backward(); opt_dsm.step()
            tot += loss.item(); nb += 1
        dsm_hist.append(tot/max(nb,1))
        pbar.set_postfix(loss=f"{dsm_hist[-1]:.3f}", ratio=f"{dsm_hist[-1]/baseline_loss:.3f}")
    loss_curves['DSM'] = dsm_hist
    dsm_net.eval()
    torch.save({'state_dict': dsm_net.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'dsm.pt'))

    # ------------------------------------------------------------------ #
    # 8. Evaluate                                                         #
    # ------------------------------------------------------------------ #
    print("\n[Eval]", flush=True)
    metrics = {}; rocs = {}

    for tm in ('additive', 'replacement'):
        planted, labels, _ = plant_targets(
            te_pix, s_pca, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        print(f"  [{tm}] planted {int(labels.sum())}/{len(labels)} targets", flush=True)

        te_nbr_f = te_nbr.astype(np.float32)
        gmm_K = cfg.get('gmm_K', 9)

        if tm == 'additive':
            sc_cf  = score_cfattn_additive(
                cfattn, planted, te_nbr_f, tr_pix, tr_nbr.astype(np.float32), s_pca)
            sc_dsm = dsm_additive(planted, tr_pix, dsm_net, s_pca)
            sc_gmm = gmm_glrt(planted, tr_pix, s_pca, K=gmm_K,
                              theta_max=cfg.get('gmm_theta_max',1.0),
                              theta_steps=cfg.get('gmm_theta_steps',50))
        else:
            sc_cf  = score_cfattn_replacement(
                cfattn, planted, te_nbr_f, tr_pix, tr_nbr.astype(np.float32), s_pca)
            sc_dsm = dsm_replacement(planted, tr_pix, dsm_net, s_pca)
            sc_gmm = gmm_glrt_replacement(planted, tr_pix, s_pca, K=gmm_K,
                                          theta_max=cfg.get('gmm_theta_max',1.0),
                                          theta_steps=cfg.get('gmm_theta_steps',50))

        det_scores = {
            'CF-Attn':  sc_cf,
            'DSM':      sc_dsm,
            'AMF':      amf(planted, tr_pix, s_pca),
            'Reg-AMF':  reg_amf(planted, tr_pix, s_pca, sigma),
            'GMM-GLRT': sc_gmm,
        }
        if tm == 'replacement':
            det_scores['AMF-rep']    = amf_replacement(planted, tr_pix, s_pca)
            det_scores['Exact-GLRT'] = exact_glrt_replacement(planted, tr_pix, s_pca)

        metrics[tm] = {k: auc_safe(labels, v) for k, v in det_scores.items()}
        rocs[tm]    = {k: roc_safe(labels, v) for k, v in det_scores.items()}
        line = "  ".join(f"{k}={v:.3f}" for k,v in metrics[tm].items())
        print(f"  {line}", flush=True)

    # ------------------------------------------------------------------ #
    # 9. Save results + figures                                           #
    # ------------------------------------------------------------------ #
    full_metrics = {
        'signature': f"{w_dom}·{dom_name} + {w_mean}·test_mean",
        'dom_cls': dom_cls, 'dom_cls_name': dom_name, 'snr': snr,
        **metrics}
    json.dump(full_metrics, open(os.path.join(run_dir,'metrics.json'),'w'), indent=2)
    json.dump(rocs,         open(os.path.join(run_dir,'rocs.json'),'w'), indent=2)
    json.dump(loss_curves,  open(os.path.join(run_dir,'loss_curves.json'),'w'))

    save_roc_figure(rocs['additive'],
                    f'Additive — {dom_name} offset (SNR={snr:.2f})',
                    os.path.join(fig_dir,'roc_additive.pdf'))
    save_roc_figure(rocs['replacement'],
                    f'Replacement — {dom_name} offset (SNR={snr:.2f})',
                    os.path.join(fig_dir,'roc_replacement.pdf'))

    det_colors = ['#d62728','#1f77b4','#aec7e8','#6baed6','#9467bd','#08306b','#ff7f0e']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tm in zip(axes, ('additive','replacement')):
        names = list(metrics[tm].keys()); vals = list(metrics[tm].values())
        bars  = ax.bar(names, vals, color=det_colors[:len(names)])
        ax.set_ylim(0.4, 1.0); ax.set_title(f'{tm.capitalize()}')
        ax.set_ylabel('AUC'); ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right')
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, min(v+0.01,0.99),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'Target: {w_dom}·{dom_name} + {w_mean}·test_mean  (SNR={snr:.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir,'auc_bar.pdf')); plt.close(fig)

    elapsed = time.time()-t0
    print(f"\nDone in {elapsed/60:.1f} min.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
