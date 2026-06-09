"""
NeighborMLPDenoiser spatial experiment — comparison with CF-Attn, DSM, and
classical baselines on the same train/test blocks and target signature.

Target: dominant-class offset —
    s = sig_dom_weight * mean(dominant_cls in test box)
      + sig_mean_weight * mean(all test box pixels)

Trains three models on the same data split:
    1. NeighborMLP   — spatial, nonlinear Tweedie denoiser (new)
    2. DSM           — global, no spatial context (baseline)
    3. CF-Attn       — spatial, closed-form Gaussian score (prior work)

Plus classical detectors: AMF, Reg-AMF, GMM-GLRT, AMF-rep, Exact-GLRT.

Usage:
    .venv/bin/python -u experiments/spatial/run_neighbor_mlp.py
    .venv/bin/python -u experiments/spatial/run_neighbor_mlp.py --config experiments/spatial/neighbor_mlp.yaml
"""

import argparse, os, sys, json, pickle, time
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
    gmm_glrt, gmm_glrt_replacement, exact_glrt_replacement,
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

CLS_NAMES = {
    0:'unlabeled', 1:'asphalt',  2:'meadows',   3:'gravel',
    4:'trees',     5:'metal_sheets', 6:'bare_soil', 7:'bitumen',
    8:'bricks',    9:'shadows',
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

def save_roc(rocs, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (fpr, tpr, a) in rocs.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({a:.3f})")
    ax.plot([0,1],[0,1],'k--',lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def save_bar(metrics, title, path):
    COLORS = ['#d62728','#e377c2','#1f77b4','#aec7e8','#6baed6','#9467bd','#08306b','#ff7f0e']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tm in zip(axes, ('additive', 'replacement')):
        names = list(metrics[tm].keys())
        vals  = list(metrics[tm].values())
        bars  = ax.bar(names, vals, color=COLORS[:len(names)])
        ax.set_ylim(0.4, 1.0); ax.set_ylabel('AUC')
        ax.set_title(f'{tm.capitalize()}'); ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, min(v+0.01, 0.99),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path); plt.close(fig)

def save_patches(data_norm, gt, cfg, fig_dir):
    H, W, _ = data_norm.shape
    bands = (60, 30, 10)
    def _rgb(c):
        r = c[...,list(bands)].astype(np.float32)
        lo = np.percentile(r,2,axis=(0,1),keepdims=True)
        hi = np.percentile(r,98,axis=(0,1),keepdims=True)
        return np.clip((r-lo)/(hi-lo+1e-9),0,1)
    fig = plt.figure(figsize=(12, 6))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1,1,1.5], height_ratios=[1,1])
    ncls = int(gt.max())+1
    def _plot(arx, agt, box, ttl):
        r0,r1,c0,c1 = box
        arx.imshow(_rgb(data_norm[r0:r1,c0:c1])); arx.set_xticks([]); arx.set_yticks([])
        arx.set_title(f"{ttl} — RGB ({r1-r0}×{c1-c0})")
        agt.imshow(gt[r0:r1,c0:c1], cmap='tab10', vmin=0, vmax=max(ncls-1,9),
                   interpolation='nearest')
        cls_ids, cnts = np.unique(gt[r0:r1,c0:c1], return_counts=True)
        note = "\n".join(f"{CLS_NAMES.get(int(c),'cls'+str(c))}={int(n)}"
                         for c,n in zip(cls_ids,cnts) if int(c)!=0)
        agt.set_title(f"{ttl} — GT"); agt.set_xlabel(note, fontsize=7)
        agt.set_xticks([]); agt.set_yticks([])
    _plot(fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]), cfg['train_box'], 'TRAIN')
    _plot(fig.add_subplot(gs[0,1]), fig.add_subplot(gs[1,1]), cfg['test_box'],  'TEST')
    ax = fig.add_subplot(gs[:,2]); ax.imshow(_rgb(data_norm))
    for box,col,lab in [(cfg['train_box'],'lime','train'),(cfg['test_box'],'red','test')]:
        r0,r1,c0,c1=box
        ax.add_patch(plt.Rectangle((c0,r0),c1-c0,r1-r0,lw=2,edgecolor=col,facecolor='none',label=lab))
    ax.set_title(f"Pavia-U ({H}×{W})"); ax.legend(loc='lower right')
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir,'patches.png'),dpi=130)
    plt.close(fig)

def save_spectra(data_norm, gt, cfg, sig_raw, dom_cls, fig_dir):
    D_raw = data_norm.shape[-1]; bands = np.arange(D_raw)
    def cls_in(box):
        r0,r1,c0,c1=box
        return set(int(c) for c in np.unique(gt[r0:r1,c0:c1]) if c!=0)
    present = cls_in(cfg['train_box']) | cls_in(cfg['test_box'])
    CLS_COLORS = {1:'#555555',2:'#4daf4a',3:'#a65628',4:'#1a7a1a',
                  5:'#ff7f00',6:'#984ea3',7:'#000000',8:'#e41a1c',9:'#377eb8'}
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    for ax, bk, ttl in [(axes[0],'train_box','TRAIN'),(axes[1],'test_box','TEST')]:
        r0,r1,c0,c1=cfg[bk]
        bf = data_norm[r0:r1,c0:c1].reshape(-1,D_raw)
        bg = gt[r0:r1,c0:c1].reshape(-1)
        bkg = bf[bg!=0]
        if len(bkg):
            bm=bkg.mean(0); bs=bkg.std(0)
            ax.fill_between(bands,bm-bs,bm+bs,alpha=0.1,color='gray',label='bkg ±1σ')
            ax.plot(bands,bm,color='gray',lw=1.2,ls='--',label='bkg mean')
        for c in sorted(present):
            m=(bg==c);
            if not m.any(): continue
            ax.plot(bands,bf[m].mean(0),lw=1.5,color=CLS_COLORS.get(c,'k'),
                    label=f"{CLS_NAMES.get(c,'cls'+str(c))} (n={m.sum()})")
        ax.plot(bands,sig_raw,color='magenta',lw=2.2,
                label=f"target (0.9·{CLS_NAMES.get(dom_cls,'?')}+0.1·mean)")
        ax.set_title(f"{ttl} spectra"); ax.set_xlabel('Band')
        ax.set_ylabel('Reflectance'); ax.legend(fontsize=7); ax.grid(True,alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir,'spectra.pdf'),bbox_inches='tight')
    plt.close(fig)

def save_loss_curves(loss_curves, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, vals in loss_curves.items():
        ax.plot(vals, lw=1.5, label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('DSM loss')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(_EXP, 'neighbor_mlp.yaml'))
    args = p.parse_args()
    cfg  = yaml.safe_load(open(args.config))

    t0  = time.time()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'neighbor_mlp_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    mdl_dir = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir,'config.yaml'),'w'), sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    seed = int(cfg['seed']); torch.manual_seed(seed)
    rng  = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Load + PCA                                                       #
    # ------------------------------------------------------------------ #
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw = data_norm.shape
    all_flat = data_norm.reshape(-1, D_raw); gt_flat = gt.reshape(-1)
    print(f"Image {H}×{W}×{D_raw}  norm={cfg['norm_mode']}", flush=True)

    D   = cfg['latent_dim']
    pca = PCA(n_components=D, random_state=seed).fit(all_flat)
    print(f"PCA {D_raw}→{D}  explained={pca.explained_variance_ratio_.sum():.4f}",flush=True)
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)
    with open(os.path.join(mdl_dir,'pca.pkl'),'wb') as fh: pickle.dump(pca, fh)

    # ------------------------------------------------------------------ #
    # 2. Boxes: report composition                                        #
    # ------------------------------------------------------------------ #
    for nm, bx in [('train_box',cfg['train_box']),('test_box',cfg['test_box'])]:
        r0,r1,c0,c1 = bx
        cls_ids, cnts = np.unique(gt[r0:r1,c0:c1], return_counts=True)
        comp = ", ".join(f"{CLS_NAMES.get(int(c),'cls'+str(c))}={int(n)}"
                         for c,n in zip(cls_ids,cnts))
        print(f"  {nm}: {comp}", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Dominant-class offset target signature                           #
    # ------------------------------------------------------------------ #
    r0,r1,c0,c1 = cfg['test_box']
    test_raw_box = all_flat.reshape(H,W,D_raw)[r0:r1,c0:c1].reshape(-1,D_raw)
    test_gt_box  = gt[r0:r1,c0:c1].reshape(-1)
    labeled = test_gt_box != 0
    if labeled.any():
        cls_ids, cnts = np.unique(test_gt_box[labeled], return_counts=True)
        dom_cls = int(cls_ids[cnts.argmax()])
    else:
        raise RuntimeError("Test box has no labeled pixels.")
    w_dom  = float(cfg.get('sig_dom_weight',  0.9))
    w_mean = float(cfg.get('sig_mean_weight', 0.1))
    sig_raw = (w_dom  * test_raw_box[test_gt_box==dom_cls].mean(0).astype(np.float32)
             + w_mean * test_raw_box.mean(0).astype(np.float32))
    s_pca   = pca.transform(sig_raw[None]).flatten().astype(np.float32)
    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')
    print(f"\nTarget: {w_dom}·{dom_name} + {w_mean}·test_mean  "
          f"||s_pca||={np.linalg.norm(s_pca):.4f}\n", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Figures: patches + spectra (before training)                     #
    # ------------------------------------------------------------------ #
    save_patches(data_norm, gt, cfg, fig_dir)
    save_spectra(data_norm, gt, cfg, sig_raw, dom_cls, fig_dir)

    # ------------------------------------------------------------------ #
    # 5. Crop + subsample                                                 #
    # ------------------------------------------------------------------ #
    k = cfg['k']
    def crop(box):
        r0,r1,c0,c1 = box
        sub = torch.tensor(pca_img[r0:r1,c0:c1,:], dtype=torch.float32)
        return [x.numpy() for x in extract_neighborhoods(sub, k)]
    def sub(pix, nbr, n):
        if len(pix)<=n: return pix, nbr
        idx = rng.choice(len(pix), n, replace=False)
        return pix[idx], nbr[idx]

    tr_pix, tr_nbr = crop(cfg['train_box'])
    te_pix, te_nbr = crop(cfg['test_box'])
    tr_pix, tr_nbr = sub(tr_pix, tr_nbr, cfg['train_n'])
    te_pix, te_nbr = sub(te_pix, te_nbr, cfg['test_n'])
    print(f"train={len(tr_pix)} px  test={len(te_pix)} px", flush=True)

    sigma = compute_sigma_from_data(pca.transform(all_flat), cfg['dsm_sigma_rho'])
    baseline = D / sigma**2
    print(f"sigma={sigma:.5f}  baseline_loss={baseline:.1f}\n", flush=True)
    loss_curves = {}

    # ------------------------------------------------------------------ #
    # 6a. Train NeighborMLP                                               #
    # ------------------------------------------------------------------ #
    print("[NeighborMLP] training ...", flush=True)
    nmlp = NeighborMLPDenoiser(
        D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
        hidden=cfg['nmlp_hidden'], n_layers=cfg['nmlp_n_layers'],
        sigma=sigma, activation=cfg['activation'])
    opt_n = torch.optim.AdamW(nmlp.parameters(),
                              lr=cfg['nmlp_lr'], weight_decay=cfg['weight_decay'])
    sched_n = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_n, T_max=cfg['nmlp_epochs'], eta_min=cfg['nmlp_lr']/20)
    Xtr = torch.tensor(tr_pix, dtype=torch.float32)
    Ntr = torch.tensor(tr_nbr, dtype=torch.float32)
    P, bs = len(Xtr), cfg['nmlp_batch']
    nmlp_hist = []
    pbar = tqdm(range(1, cfg['nmlp_epochs']+1), desc='NeighborMLP', dynamic_ncols=True)
    for _ in pbar:
        perm = torch.randperm(P); tot=0.; nb=0
        for i in range(0,P,bs):
            sel=perm[i:i+bs]
            loss = neighbor_mlp_dsm_loss(nmlp, Xtr[sel], Ntr[sel])
            opt_n.zero_grad(); loss.backward(); opt_n.step(); tot+=loss.item(); nb+=1
        sched_n.step()
        nmlp_hist.append(tot/max(nb,1))
        pbar.set_postfix(loss=f"{nmlp_hist[-1]:.3f}", ratio=f"{nmlp_hist[-1]/baseline:.3f}")
    loss_curves['NeighborMLP'] = nmlp_hist
    nmlp.eval()
    torch.save({'state_dict':nmlp.state_dict(),'cfg':cfg},os.path.join(mdl_dir,'nmlp.pt'))

    # ------------------------------------------------------------------ #
    # 6b. Train DSM                                                       #
    # ------------------------------------------------------------------ #
    print("\n[DSM] training ...", flush=True)
    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'])
    opt_d = torch.optim.Adam(dsm_net.parameters(),
                             lr=cfg['dsm_lr'], weight_decay=cfg['weight_decay'])
    X = torch.tensor(tr_pix, dtype=torch.float32)
    dsm_hist = []
    pbar = tqdm(range(1, cfg['dsm_epochs']+1), desc='DSM', dynamic_ncols=True)
    for _ in pbar:
        perm=torch.randperm(len(X)); tot=0.; nb=0
        for i in range(0,len(X),cfg['batch_size']):
            b=X[perm[i:i+cfg['batch_size']]]
            loss=dsm_loss(dsm_net,b,sigma)
            opt_d.zero_grad(); loss.backward(); opt_d.step(); tot+=loss.item(); nb+=1
        dsm_hist.append(tot/max(nb,1))
        pbar.set_postfix(loss=f"{dsm_hist[-1]:.3f}",ratio=f"{dsm_hist[-1]/baseline:.3f}")
    loss_curves['DSM'] = dsm_hist
    dsm_net.eval()
    torch.save({'state_dict':dsm_net.state_dict(),'cfg':cfg},os.path.join(mdl_dir,'dsm.pt'))

    # ------------------------------------------------------------------ #
    # 6c. Train CF-Attn                                                   #
    # ------------------------------------------------------------------ #
    print("\n[CF-Attn] training ...", flush=True)
    M = k*k-1
    cfattn = CFAttnGaussianScoreNet(
        D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
        sigma=sigma, eps=cfg.get('cfattn_eps',1e-4))
    km = KMeans(n_clusters=cfg['cfattn_K'], init='k-means++',
                n_init=5, random_state=seed, max_iter=100)
    km.fit(tr_pix)
    cfattn.comp_mu.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    opt_c = torch.optim.AdamW(cfattn.parameters(),
                              lr=cfg['cfattn_lr'], weight_decay=cfg['weight_decay'])
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_c, T_max=cfg['cfattn_epochs'], eta_min=cfg['cfattn_lr']/20)
    cf_hist = []
    pbar = tqdm(range(1,cfg['cfattn_epochs']+1), desc='CF-Attn', dynamic_ncols=True)
    for _ in pbar:
        perm=torch.randperm(P); tot=0.; nb=0
        for i in range(0,P,bs):
            sel=perm[i:i+bs]
            loss,di = cfattn_dsm_loss(cfattn,Xtr[sel],Ntr[sel],
                                      lam_ent=cfg.get('lam_ent',0.05),
                                      lam_div=cfg.get('lam_div',0.05),
                                      lam_cov=cfg.get('lam_cov',1e-5))
            opt_c.zero_grad(); loss.backward(); opt_c.step(); tot+=di; nb+=1
        sched_c.step()
        cf_hist.append(tot/max(nb,1))
        pbar.set_postfix(dsm=f"{cf_hist[-1]:.3f}",ratio=f"{cf_hist[-1]/baseline:.3f}")
    loss_curves['CF-Attn'] = cf_hist
    cfattn.eval()
    torch.save({'state_dict':cfattn.state_dict(),'cfg':cfg},os.path.join(mdl_dir,'cfattn.pt'))

    # ------------------------------------------------------------------ #
    # 7. Evaluate                                                         #
    # ------------------------------------------------------------------ #
    print("\n[Eval]", flush=True)
    metrics = {}; rocs = {}
    gmm_K  = cfg.get('gmm_K', 9)
    th_max = cfg.get('gmm_theta_max', 0.5)
    th_stp = cfg.get('gmm_theta_steps', 50)

    for tm in ('additive', 'replacement'):
        planted, labels, _ = plant_targets(
            te_pix, s_pca, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        te_nbr_f = te_nbr.astype(np.float32)
        tr_nbr_f = tr_nbr.astype(np.float32)

        # --- Neural models ---
        if tm == 'additive':
            sc_nmlp   = score_nmlp_additive(nmlp, planted, te_nbr_f, tr_pix, tr_nbr_f, s_pca)
            sc_dsm    = dsm_additive(planted, tr_pix, dsm_net, s_pca)
            sc_cf     = score_cfattn_additive(cfattn, planted, te_nbr_f, tr_pix, tr_nbr_f, s_pca)
            sc_gmm    = gmm_glrt(planted, tr_pix, s_pca, K=gmm_K,
                                 theta_max=th_max, theta_steps=th_stp)
        else:
            sc_nmlp   = score_nmlp_replacement(nmlp, planted, te_nbr_f, tr_pix, tr_nbr_f, s_pca)
            sc_dsm    = dsm_replacement(planted, tr_pix, dsm_net, s_pca)
            sc_cf     = score_cfattn_replacement(cfattn, planted, te_nbr_f, tr_pix, tr_nbr_f, s_pca)
            sc_gmm    = gmm_glrt_replacement(planted, tr_pix, s_pca, K=gmm_K,
                                             theta_max=th_max, theta_steps=th_stp)

        det_scores = {
            'NeighborMLP': sc_nmlp,
            'CF-Attn':     sc_cf,
            'DSM':         sc_dsm,
            'AMF':         amf(planted, tr_pix, s_pca),
            'Reg-AMF':     reg_amf(planted, tr_pix, s_pca, sigma),
            'GMM-GLRT':    sc_gmm,
        }
        if tm == 'replacement':
            det_scores['AMF-rep']    = amf_replacement(planted, tr_pix, s_pca)
            det_scores['Exact-GLRT']= exact_glrt_replacement(planted, tr_pix, s_pca)

        metrics[tm] = {k: auc_safe(labels, v) for k,v in det_scores.items()}
        rocs[tm]    = {k: roc_safe(labels, v) for k,v in det_scores.items()}
        line = "  ".join(f"{k}={v:.3f}" for k,v in metrics[tm].items())
        print(f"  [{tm}]  {line}", flush=True)

    # ------------------------------------------------------------------ #
    # 8. Save                                                             #
    # ------------------------------------------------------------------ #
    full_metrics = {
        'signature': f"{w_dom}·{dom_name} + {w_mean}·test_mean",
        'dom_cls': dom_cls, 'dom_cls_name': dom_name,
        **metrics}
    json.dump(full_metrics, open(os.path.join(run_dir,'metrics.json'),'w'), indent=2)
    json.dump(rocs,         open(os.path.join(run_dir,'rocs.json'),'w'), indent=2)
    json.dump(loss_curves,  open(os.path.join(run_dir,'loss_curves.json'),'w'))

    save_roc(rocs['additive'],    f"Additive — {dom_name} offset",
             os.path.join(fig_dir,'roc_additive.pdf'))
    save_roc(rocs['replacement'], f"Replacement — {dom_name} offset",
             os.path.join(fig_dir,'roc_replacement.pdf'))
    save_bar(metrics, f"Target: {w_dom}·{dom_name} + {w_mean}·test_mean",
             os.path.join(fig_dir,'auc_bar.pdf'))
    save_loss_curves(loss_curves, os.path.join(fig_dir,'loss_curves.png'))

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
