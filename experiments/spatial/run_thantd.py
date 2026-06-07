"""
run_thantd.py — THANTD baseline against the spatial score models.

Trains THANTD (Liu et al. 2025 — Triplet Hybrid Attention Network) on the
same Pavia-U train/test rectangle pair used by the CF-Attn / NeighborMLP
spatial experiments, with the same dominant-class offset target signature.
Reports additive + replacement AUC alongside AMF / DSM / NeighborMLP /
CF-Attn / CF-Attn-Jac so the spatial methods can be compared head-to-head.

THANTD is a pixel-level (non-spatial) detector — comparing the spatial
methods to it isolates how much benefit the spatial neighborhood gives.

Usage:
    .venv/bin/python -u experiments/spatial/run_thantd.py
    .venv/bin/python -u experiments/spatial/run_thantd.py --config experiments/spatial/thantd.yaml
"""

import argparse, os, sys, json, time, pickle
from datetime import datetime

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive, dsm_replacement, amf_replacement,
    gmm_glrt, gmm_glrt_replacement,
)
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods
from dsm_model import ScoreNet, dsm_loss
from cfattn_model import (
    CFAttnGaussianScoreNet, cfattn_dsm_loss,
    score_cfattn_additive, score_cfattn_replacement,
)
from neighbor_mlp_model import (
    NeighborMLPDenoiser, neighbor_mlp_dsm_loss,
    score_nmlp_additive, score_nmlp_replacement,
)
from thantd_model import (
    THANTD, build_thantd_samples, train_thantd, score_thantd,
)

CLS_NAMES = {0:'unlabeled', 1:'asphalt', 2:'meadows', 3:'gravel',
             4:'trees',     5:'metal_sheets', 6:'bare_soil', 7:'bitumen',
             8:'bricks',    9:'shadows'}


def auc_safe(lab, sc):
    try:    return float(roc_auc_score(lab, sc))
    except: return float('nan')


def roc_safe(lab, sc):
    try:
        fpr, tpr, _ = roc_curve(lab, sc)
        return fpr.tolist(), tpr.tolist(), auc_safe(lab, sc)
    except:
        return [0.,1.], [0.,1.], float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(_EXP, 'thantd.yaml'))
    p.add_argument('--no-thantd', action='store_true',
                   help='Skip THANTD training/scoring; run only the spatial baselines')
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    t0  = time.time()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'thantd_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    mdl_dir = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir,'config.yaml'),'w'), sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    seed = int(cfg['seed']); torch.manual_seed(seed)
    rng  = np.random.default_rng(seed)

    # ---- Load + PCA (PCA still used by the spatial models; THANTD uses raw) ----
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw = data_norm.shape
    all_flat = data_norm.reshape(-1, D_raw); gt_flat = gt.reshape(-1)
    print(f"Image {H}×{W}×{D_raw}  norm={cfg['norm_mode']}", flush=True)

    D = cfg['latent_dim']
    pca = PCA(n_components=D, random_state=seed).fit(all_flat)
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)
    with open(os.path.join(mdl_dir,'pca.pkl'),'wb') as fh: pickle.dump(pca, fh)

    # ---- Box compositions ----
    for nm, bx in [('train_box', cfg['train_box']), ('test_box', cfg['test_box'])]:
        r0,r1,c0,c1 = bx
        cls_ids, cnts = np.unique(gt[r0:r1,c0:c1], return_counts=True)
        comp = ", ".join(f"{CLS_NAMES.get(int(c),'cls'+str(c))}={int(n)}"
                         for c,n in zip(cls_ids,cnts))
        print(f"  {nm}: {comp}", flush=True)

    # ---- Dominant-class offset target signature ----
    r0,r1,c0,c1 = cfg['test_box']
    test_raw_box = all_flat.reshape(H,W,D_raw)[r0:r1,c0:c1].reshape(-1,D_raw)
    test_gt_box  = gt[r0:r1,c0:c1].reshape(-1)
    labeled = test_gt_box != 0
    cls_ids, cnts = np.unique(test_gt_box[labeled], return_counts=True)
    dom_cls = int(cls_ids[cnts.argmax()])
    w_dom  = float(cfg.get('sig_dom_weight',  0.9))
    w_mean = float(cfg.get('sig_mean_weight', 0.1))
    sig_raw = (w_dom * test_raw_box[test_gt_box==dom_cls].mean(0).astype(np.float32)
             + w_mean * test_raw_box.mean(0).astype(np.float32))
    s_pca = pca.transform(sig_raw[None]).flatten().astype(np.float32)
    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')
    print(f"\nTarget: {w_dom}·{dom_name} + {w_mean}·test_mean  "
          f"||s_pca||={np.linalg.norm(s_pca):.4f}\n", flush=True)

    # ---- Crop train/test pixels + spatial neighborhoods (PCA space) ----
    k = cfg['k']
    def crop_box(box):
        r0,r1,c0,c1 = box
        sub = torch.tensor(pca_img[r0:r1,c0:c1,:], dtype=torch.float32)
        centers, nbrs = extract_neighborhoods(sub, k)
        return centers.numpy(), nbrs.numpy()
    tr_pca, tr_nbr = crop_box(cfg['train_box'])
    te_pca, te_nbr = crop_box(cfg['test_box'])

    def subsample(pix, nbr, n):
        if len(pix) <= n: return pix, nbr
        idx = rng.choice(len(pix), n, replace=False)
        return pix[idx], nbr[idx]
    tr_pca, tr_nbr = subsample(tr_pca, tr_nbr, cfg['train_n'])
    te_pca, te_nbr = subsample(te_pca, te_nbr, cfg['test_n'])
    print(f"train={len(tr_pca)} px  test={len(te_pca)} px", flush=True)

    # ---- THANTD operates on the RAW (full-D) pixel; extract matching raw pixels ----
    def crop_raw(box):
        r0,r1,c0,c1 = box
        return data_norm[r0:r1,c0:c1].reshape(-1, D_raw)
    tr_raw_box = crop_raw(cfg['train_box'])
    te_raw_box = crop_raw(cfg['test_box'])
    # match the subsampled indices for the raw arrays
    # (subsample() above used cfg['train_n']/cfg['test_n'] on the FULL crop;
    #  we resample the raw box with the same seed to get the same physical pixels)
    rng2 = np.random.default_rng(seed)
    if len(tr_raw_box) > cfg['train_n']:
        idx = rng2.choice(len(tr_raw_box), cfg['train_n'], replace=False)
        tr_raw = tr_raw_box[idx]
    else:
        tr_raw = tr_raw_box
    if len(te_raw_box) > cfg['test_n']:
        idx = rng2.choice(len(te_raw_box), cfg['test_n'], replace=False)
        te_raw = te_raw_box[idx]
    else:
        te_raw = te_raw_box

    sigma = compute_sigma_from_data(pca.transform(all_flat), cfg['dsm_sigma_rho'])
    baseline = D / sigma**2
    print(f"sigma = {sigma:.5f}  baseline = {baseline:.1f}\n", flush=True)
    loss_curves = {}

    # ------------------------------------------------------------------ #
    # 1. THANTD (raw 103-D pixel)                                          #
    # ------------------------------------------------------------------ #
    thantd_model = None
    if not args.no_thantd:
        print(f"[THANTD] building samples + training ...", flush=True)
        t1 = time.time()
        # THANTD sample construction:
        # - Default (paper): CEM rough detection on tr_raw to select negatives
        # - Adaptation: if `thantd_use_secondary: true` in config, pass tr_raw
        #   directly as the background pool (target-free secondary data mode)
        bkg_for_samples = tr_raw if cfg.get('thantd_use_secondary', False) else None
        a_smp, p_smp, n_smp = build_thantd_samples(
            tr_raw, sig_raw,
            alpha=cfg.get('thantd_alpha', 0.5),
            n_samples=cfg.get('thantd_n_pairs', 1024),
            rng=rng,
            bkg_pool=bkg_for_samples)
        thantd_model = THANTD(b=D_raw, m=cfg.get('thantd_m', 7),
                              d=cfg.get('thantd_d', 64),
                              n_heads=cfg.get('thantd_heads', 4))
        train_thantd(thantd_model, a_smp, p_smp, n_smp,
                     epochs=cfg.get('thantd_epochs', 300),
                     batch_size=cfg.get('thantd_batch', 64),
                     lr=cfg.get('thantd_lr', 1e-4),
                     margin=cfg.get('thantd_margin', 0.3),
                     lam=cfg.get('thantd_lambda', 0.5),
                     device='cpu')
        torch.save({'state_dict': thantd_model.state_dict(), 'cfg': cfg},
                   os.path.join(mdl_dir, 'thantd.pt'))
        print(f"  trained in {time.time()-t1:.0f}s", flush=True)
    else:
        print(f"[THANTD] skipped (--no-thantd)", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Train CF-Attn + NeighborMLP + DSM as comparisons (spatial)        #
    # ------------------------------------------------------------------ #
    Xtr = torch.tensor(tr_pca, dtype=torch.float32)
    Ntr = torch.tensor(tr_nbr, dtype=torch.float32)
    P, bs = len(Xtr), cfg['batch_size']
    M = k*k - 1

    # CF-Attn
    print("\n[CF-Attn] training ...", flush=True)
    cfattn = CFAttnGaussianScoreNet(D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
                                     sigma=sigma, eps=cfg.get('cfattn_eps',1e-4))
    km = KMeans(n_clusters=cfg['cfattn_K'], init='k-means++',
                n_init=5, random_state=seed, max_iter=100).fit(tr_pca)
    cfattn.comp_mu.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    opt = torch.optim.AdamW(cfattn.parameters(), lr=cfg['cfattn_lr'],
                            weight_decay=cfg['weight_decay'])
    pbar = tqdm(range(cfg['cfattn_epochs']), desc='CF-Attn', dynamic_ncols=True)
    for _ in pbar:
        perm = torch.randperm(P)
        for i in range(0, P, bs):
            sel = perm[i:i+bs]
            loss, di = cfattn_dsm_loss(cfattn, Xtr[sel], Ntr[sel],
                                       lam_ent=cfg.get('lam_ent', 0.05),
                                       lam_div=cfg.get('lam_div', 0.05),
                                       lam_cov=cfg.get('lam_cov', 1e-5))
            opt.zero_grad(); loss.backward(); opt.step()
    cfattn.eval()
    torch.save({'state_dict': cfattn.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'cfattn.pt'))

    # NeighborMLP
    print("\n[NeighborMLP] training ...", flush=True)
    nmlp = NeighborMLPDenoiser(D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
                                hidden=cfg['nmlp_hidden'],
                                n_layers=cfg['nmlp_n_layers'],
                                sigma=sigma, activation=cfg['activation'])
    opt = torch.optim.AdamW(nmlp.parameters(), lr=cfg['nmlp_lr'],
                            weight_decay=cfg['weight_decay'])
    pbar = tqdm(range(cfg['nmlp_epochs']), desc='NeighborMLP', dynamic_ncols=True)
    for _ in pbar:
        perm = torch.randperm(P)
        for i in range(0, P, cfg['nmlp_batch']):
            sel = perm[i:i+cfg['nmlp_batch']]
            loss = neighbor_mlp_dsm_loss(nmlp, Xtr[sel], Ntr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
    nmlp.eval()
    torch.save({'state_dict': nmlp.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'nmlp.pt'))

    # DSM (global score net, no spatial context)
    print("\n[DSM] training ...", flush=True)
    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'])
    opt = torch.optim.Adam(dsm_net.parameters(), lr=cfg['dsm_lr'],
                           weight_decay=cfg['weight_decay'])
    pbar = tqdm(range(cfg['dsm_epochs']), desc='DSM', dynamic_ncols=True)
    for _ in pbar:
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = Xtr[perm[i:i+bs]]
            loss = dsm_loss(dsm_net, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
    dsm_net.eval()
    torch.save({'state_dict': dsm_net.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'dsm.pt'))

    # ------------------------------------------------------------------ #
    # 3. Evaluate                                                          #
    # ------------------------------------------------------------------ #
    print("\n[Eval]", flush=True)
    metrics = {}; rocs = {}
    for tm in ('additive', 'replacement'):
        planted_pca, labels, _ = plant_targets(
            te_pca, s_pca, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        planted_raw, _, _ = plant_targets(
            te_raw, sig_raw, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)

        te_nbr_f = te_nbr.astype(np.float32)
        tr_nbr_f = tr_nbr.astype(np.float32)

        if tm == 'additive':
            sc_cf   = score_cfattn_additive(cfattn,  planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca)
            sc_nmlp = score_nmlp_additive(  nmlp,    planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca)
            sc_dsm  = dsm_additive(planted_pca, tr_pca, dsm_net, s_pca)
        else:
            sc_cf   = score_cfattn_replacement(cfattn, planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca)
            sc_nmlp = score_nmlp_replacement(  nmlp,   planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca)
            sc_dsm  = dsm_replacement(planted_pca, tr_pca, dsm_net, s_pca)

        sc_amf    = amf(planted_pca, tr_pca, s_pca)
        sc_regamf = reg_amf(planted_pca, tr_pca, s_pca, sigma)

        det_scores = {
            'CF-Attn':     sc_cf,
            'NeighborMLP': sc_nmlp,
            'DSM':         sc_dsm,
            'AMF':         sc_amf,
            'Reg-AMF':     sc_regamf,
        }
        if thantd_model is not None:
            det_scores['THANTD'] = score_thantd(thantd_model, sig_raw, planted_raw)
        if tm == 'replacement':
            det_scores['AMF-rep'] = amf_replacement(planted_pca, tr_pca, s_pca)

        metrics[tm] = {kk: auc_safe(labels, v) for kk,v in det_scores.items()}
        rocs[tm]    = {kk: roc_safe(labels, v) for kk,v in det_scores.items()}
        line = "  ".join(f"{kk}={v:.3f}" for kk,v in metrics[tm].items())
        print(f"  [{tm}]  {line}", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Save metrics + figures                                            #
    # ------------------------------------------------------------------ #
    json.dump({'signature': f'{w_dom}·{dom_name} + {w_mean}·test_mean',
               'dom_cls': dom_cls, 'dom_cls_name': dom_name, **metrics},
              open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)
    json.dump(rocs, open(os.path.join(run_dir, 'rocs.json'), 'w'), indent=2)

    # ROC + AUC bar chart
    COLORS = {'THANTD':'#d62728','CF-Attn':'#1f77b4','NeighborMLP':'#2ca02c',
              'DSM':'#ff7f0e','AMF':'#9467bd','Reg-AMF':'#aec7e8','AMF-rep':'#08306b'}

    def save_roc(rocs_d, title, path):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for name, (fpr, tpr, a) in rocs_d.items():
            ax.plot(fpr, tpr, lw=1.8, color=COLORS.get(name,'k'),
                    label=f"{name} ({a:.3f})")
        ax.plot([0,1],[0,1],'k--',lw=0.8)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)

    save_roc(rocs['additive'],    f'Additive — {dom_name} offset',
             os.path.join(fig_dir,'roc_additive.pdf'))
    save_roc(rocs['replacement'], f'Replacement — {dom_name} offset',
             os.path.join(fig_dir,'roc_replacement.pdf'))

    # AUC bar
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, tm in zip(axes, ('additive', 'replacement')):
        names = list(metrics[tm].keys()); vals = list(metrics[tm].values())
        bars = ax.bar(names, vals, color=[COLORS.get(n,'k') for n in names])
        ax.set_ylim(0.4, 1.0); ax.set_ylabel('AUC')
        ax.set_title(f'{tm.capitalize()}')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, min(v+0.01,0.99),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'Target: {w_dom}·{dom_name} + {w_mean}·test_mean')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'auc_bar.pdf')); plt.close(fig)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
