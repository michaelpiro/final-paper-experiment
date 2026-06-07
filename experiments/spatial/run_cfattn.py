"""
Closed-Form Attention Gaussian Score (CF-Attn) experiment (Pavia-U).

Compares CF-Attn against DSM and AMF in the spatial setting:
  - Train on a background rectangle (no target class)
  - Plant weak targets in a test rectangle
  - Evaluate additive + replacement AUC

CF-Attn uses spatial neighbors for per-pixel Gaussian estimation via
attention over neighbor point-masses and K learned global Gaussian atoms.
The score is closed-form and affine in the query pixel — no MLP in the
score path, only in the attention (which is query-pixel-independent).

Usage:
    .venv/bin/python -u run_cfattn_experiment.py --config cfattn.yaml
"""

import argparse, os, sys, json, time, pickle
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
from dsm_model import ScoreNet, dsm_loss, compute_scores
from cfattn_model import (
    CFAttnGaussianScoreNet, cfattn_dsm_loss,
    score_cfattn_additive, score_cfattn_replacement,
)


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
        return [0., 1.], [0., 1.], float('nan')


def plot_roc(rocs: dict, title: str, path: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (fpr, tpr, a) in rocs.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({a:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _save_patch_figures(data: np.ndarray, gt: np.ndarray,
                        cfg: dict, fig_dir: str):
    """Save false-color RGBs and GT maps of the train/test rectangles.

    - false_color uses three bands far apart in the spectrum (HSI viz convention)
    - GT panel labels every class found in the crop and confirms the target
      class is absent (red overlay would show planted-target locations — here
      there are none, which is the point)
    """
    bands = (60, 30, 10)   # R, G, B (NIR/red/blue-ish for ROSIS-103)
    H, W, B = data.shape

    def _norm_rgb(crop):
        rgb = crop[..., list(bands)].astype(np.float32)
        lo  = np.percentile(rgb, 2.0, axis=(0, 1), keepdims=True)
        hi  = np.percentile(rgb, 98.0, axis=(0, 1), keepdims=True)
        return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)

    # Build a context image: the whole scene with the two boxes drawn on it.
    full_rgb = _norm_rgb(data)
    fig = plt.figure(figsize=(11.5, 6.5))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.4], height_ratios=[1, 1])

    # Left: train crop RGB + GT
    # Middle: test crop RGB + GT
    # Right: full scene with boxes
    tgt = cfg['target_cls']
    n_cls = int(gt.max()) + 1

    def _plot_crop(ax_rgb, ax_gt, box, title):
        r0, r1, c0, c1 = box
        rgb = _norm_rgb(data[r0:r1, c0:c1, :])
        gt_sub = gt[r0:r1, c0:c1]
        ax_rgb.imshow(rgb); ax_rgb.set_title(f"{title} — false color  ({r1-r0}×{c1-c0})")
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        im = ax_gt.imshow(gt_sub, cmap='tab10', vmin=0, vmax=max(n_cls - 1, 9))
        ax_gt.set_title(f"{title} — GT (target cls {tgt}: "
                        f"{int(np.sum(gt_sub == tgt))} px → must be 0)")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        return im

    ax_tr_rgb = fig.add_subplot(gs[0, 0]); ax_tr_gt = fig.add_subplot(gs[1, 0])
    ax_te_rgb = fig.add_subplot(gs[0, 1]); ax_te_gt = fig.add_subplot(gs[1, 1])
    _plot_crop(ax_tr_rgb, ax_tr_gt, cfg['train_box'], 'TRAIN')
    _plot_crop(ax_te_rgb, ax_te_gt, cfg['test_box'],  'TEST')

    # Right: whole scene with both boxes drawn
    ax_full = fig.add_subplot(gs[:, 2])
    ax_full.imshow(full_rgb)
    for box, color, lab in [(cfg['train_box'], 'lime', 'train'),
                            (cfg['test_box'],  'red',  'test')]:
        r0, r1, c0, c1 = box
        rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0, linewidth=2,
                             edgecolor=color, facecolor='none', label=lab)
        ax_full.add_patch(rect)
    ax_full.set_title(f"Pavia-U  ({H}×{W})  — train/test rectangles")
    ax_full.set_xticks([]); ax_full.set_yticks([])
    ax_full.legend(loc='lower right')

    fig.tight_layout()
    out = os.path.join(fig_dir, 'patches.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}", flush=True)


def plot_loss(curves: dict, path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, vals in curves.items():
        ax.plot(vals, lw=1.5, label=name)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cfattn.yaml'))
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_start = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'cfattn_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    mdl_dir = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'))
    print(f"Run dir: {run_dir}", flush=True)

    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    rng  = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Load + normalize whole image, PCA -> latent_dim                  #
    # ------------------------------------------------------------------ #
    data, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw = data.shape
    gt_flat  = gt.reshape(-1)
    all_flat = data.reshape(-1, D_raw)
    print(f"Image {H}x{W}x{D_raw}  norm={cfg['norm_mode']}", flush=True)

    D   = cfg['latent_dim']
    pca = PCA(n_components=D, random_state=seed).fit(all_flat)
    evr = pca.explained_variance_ratio_.sum()
    print(f"PCA {D_raw} -> {D}   explained var = {evr:.4f}", flush=True)
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)
    with open(os.path.join(mdl_dir, 'pca.pkl'), 'wb') as fh:
        pickle.dump(pca, fh)

    # Target signature in PCA space — the CENTERED mean of target pixels
    # (NOT unit-normalized: matches the IID experiments' convention).
    # Planting happens directly in PCA space (plant_targets adds amp*s to
    # te_pix in PCA), so the SAME `s` must be used for both additive and
    # replacement detection — using an uncentered or rescaled variant
    # introduces a direction/magnitude mismatch.
    tgt_pca = pca.transform(all_flat[gt_flat == cfg['target_cls']])
    assert len(tgt_pca) > 0, f"No pixels for target class {cfg['target_cls']}"
    s = tgt_pca.mean(axis=0).astype(np.float32)        # (D,) centered, NOT normalized
    print(f"||s|| = {np.linalg.norm(s):.4f}  (target class {cfg['target_cls']})",
          flush=True)

    # Verify boxes don't contain target — HARD CHECK.
    # We plant WEAK targets in test, so the test rectangle MUST be free of
    # the target class (otherwise we'd be planting on top of real targets).
    # The train rectangle must also be free of the target class.
    CLS_NAMES = {0:'unlabeled', 1:'asphalt', 2:'meadows', 3:'gravel',
                 4:'trees', 5:'metal_sheets', 6:'bare_soil', 7:'bitumen',
                 8:'bricks', 9:'shadows'}
    tgt_cls = cfg['target_cls']
    for nm, bx in [('train_box', cfg['train_box']), ('test_box', cfg['test_box'])]:
        r0, r1, c0, c1 = bx
        gt_sub = gt[r0:r1, c0:c1]
        n_tgt_in_box = int(np.sum(gt_sub == tgt_cls))
        cls, cnt = np.unique(gt_sub, return_counts=True)
        comp = ", ".join(f"{CLS_NAMES.get(int(c), f'cls{c}')}={int(n)}"
                        for c, n in zip(cls, cnt))
        print(f"  {nm} {bx}  ({gt_sub.size} px)  composition: {comp}", flush=True)
        assert n_tgt_in_box == 0, (
            f"{nm} contains {n_tgt_in_box} pixels of target class {tgt_cls}! "
            f"Cannot plant weak targets here — choose a different rectangle.")

    # Save false-color visualizations of the two crops + GT overlays.
    _save_patch_figures(data, gt, cfg, fig_dir)

    # ------------------------------------------------------------------ #
    # 2. Crop train/test from PCA image, extract spatial neighborhoods    #
    # ------------------------------------------------------------------ #
    k = cfg['k']   # neighborhood window size (M = k*k - 1 neighbors)

    def crop_box(box):
        r0, r1, c0, c1 = box
        sub = torch.tensor(pca_img[r0:r1, c0:c1, :], dtype=torch.float32)
        centers, nbrs = extract_neighborhoods(sub, k)     # (P,D), (P,M,D)
        return centers.numpy(), nbrs.numpy()

    tr_pix, tr_nbr = crop_box(cfg['train_box'])
    te_pix, te_nbr = crop_box(cfg['test_box'])
    print(f"train crop: {len(tr_pix)} px | test crop: {len(te_pix)} px", flush=True)

    # Subsample for speed
    def subsample(pix, nbr, n, rs=0):
        if len(pix) <= n: return pix, nbr
        idx = rng.choice(len(pix), n, replace=False)
        return pix[idx], nbr[idx]

    tr_pix, tr_nbr = subsample(tr_pix, tr_nbr, cfg['train_n'])
    te_pix, te_nbr = subsample(te_pix, te_nbr, cfg['test_n'])

    # ------------------------------------------------------------------ #
    # 3. Plant targets in test set                                        #
    # ------------------------------------------------------------------ #
    test_sets = {}
    for tm in ('additive', 'replacement'):
        planted, labels, _ = plant_targets(
            te_pix, s, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        test_sets[tm] = (planted, labels)
    n_tgt = int(test_sets['additive'][1].sum())
    print(f"Planted {n_tgt} targets in {D}-D PCA space (amp={cfg['amplitude']})\n",
          flush=True)

    # Noise level (from full image)
    sigma_full = compute_sigma_from_data(pca.transform(all_flat), cfg['dsm_sigma_rho'])
    sigma      = sigma_full
    print(f"sigma = {sigma:.5f}  (rho={cfg['dsm_sigma_rho']})\n", flush=True)
    baseline_loss = D / sigma ** 2
    loss_curves = {}

    # ------------------------------------------------------------------ #
    # 4a. Train CF-Attention model                                        #
    # ------------------------------------------------------------------ #
    M = k * k - 1   # number of spatial neighbors
    print(f"[CF-Attn] D={D}  h={cfg['cfattn_h']}  K={cfg['cfattn_K']}  M={M}", flush=True)
    cfattn = CFAttnGaussianScoreNet(
        D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
        sigma=sigma, eps=cfg.get('cfattn_eps', 1e-4))

    # Initialize comp_mu with k-means++ on training pixels
    km = KMeans(n_clusters=cfg['cfattn_K'], init='k-means++',
                n_init=5, random_state=seed, max_iter=100)
    km.fit(tr_pix)
    cfattn.comp_mu.data.copy_(
        torch.tensor(km.cluster_centers_, dtype=torch.float32))
    print(f"  comp_mu initialized via k-means++ on {len(tr_pix)} train pixels", flush=True)

    opt_cf = torch.optim.AdamW(cfattn.parameters(),
                               lr=cfg['cfattn_lr'], weight_decay=cfg['weight_decay'])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_cf, T_max=cfg['cfattn_epochs'], eta_min=cfg['cfattn_lr'] / 20)

    Xtr = torch.tensor(tr_pix, dtype=torch.float32)
    Ntr = torch.tensor(tr_nbr, dtype=torch.float32)
    P, bs = len(Xtr), cfg['cfattn_batch']

    cf_hist = []
    pbar = tqdm(range(1, cfg['cfattn_epochs'] + 1), desc='CF-Attn', dynamic_ncols=True)
    for ep in pbar:
        perm = torch.randperm(P); tot_dsm = 0.; nb = 0
        for i in range(0, P, bs):
            sel = perm[i:i + bs]
            loss, dsm_item = cfattn_dsm_loss(
                cfattn, Xtr[sel], Ntr[sel],
                lam_ent=cfg.get('lam_ent', 0.05),
                lam_div=cfg.get('lam_div', 0.05),
                lam_cov=cfg.get('lam_cov', 1e-5),
            )
            opt_cf.zero_grad(); loss.backward(); opt_cf.step()
            tot_dsm += dsm_item; nb += 1
        sched.step()
        avg = tot_dsm / max(nb, 1)
        cf_hist.append(avg)
        pbar.set_postfix(dsm=f"{avg:.3f}", ratio=f"{avg/baseline_loss:.3f}")
    loss_curves['CF-Attn'] = cf_hist
    cfattn.eval()
    torch.save({'state_dict': cfattn.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'cfattn.pt'))

    # ------------------------------------------------------------------ #
    # 4b. Train DSM (global, no spatial context)                          #
    # ------------------------------------------------------------------ #
    print("\n[DSM] training global score net ...", flush=True)
    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'])
    opt_dsm = torch.optim.Adam(dsm_net.parameters(),
                               lr=cfg['dsm_lr'], weight_decay=cfg['weight_decay'])
    X = torch.tensor(tr_pix, dtype=torch.float32)
    dsm_hist = []
    pbar = tqdm(range(1, cfg['dsm_epochs'] + 1), desc='DSM', dynamic_ncols=True)
    for ep in pbar:
        perm = torch.randperm(len(X)); tot = 0.; nb = 0
        for i in range(0, len(X), cfg['batch_size']):
            b = X[perm[i:i + cfg['batch_size']]]
            loss = dsm_loss(dsm_net, b, sigma)
            opt_dsm.zero_grad(); loss.backward(); opt_dsm.step()
            tot += loss.item(); nb += 1
        dsm_hist.append(tot / max(nb, 1))
        pbar.set_postfix(loss=f"{dsm_hist[-1]:.3f}", ratio=f"{dsm_hist[-1]/baseline_loss:.3f}")
    loss_curves['DSM'] = dsm_hist
    dsm_net.eval()
    torch.save({'state_dict': dsm_net.state_dict(), 'cfg': cfg},
               os.path.join(mdl_dir, 'dsm.pt'))

    # ------------------------------------------------------------------ #
    # 5. Evaluate all detectors                                           #
    # ------------------------------------------------------------------ #
    print("\n[Eval] scoring test pixels ...", flush=True)
    metrics = {}; rocs = {}

    for tm in ('additive', 'replacement'):
        planted, labels = test_sets[tm]
        te_nbr_f = te_nbr.astype(np.float32)

        # CF-Attention
        if tm == 'additive':
            sc_cf = score_cfattn_additive(
                cfattn, planted, te_nbr_f, tr_pix, tr_nbr.astype(np.float32), s)
        else:
            sc_cf = score_cfattn_replacement(
                cfattn, planted, te_nbr_f, tr_pix, tr_nbr.astype(np.float32), s)

        # DSM
        if tm == 'additive':
            sc_dsm = dsm_additive(planted, tr_pix, dsm_net, s)
        else:
            sc_dsm = dsm_replacement(planted, tr_pix, dsm_net, s)

        # Classical baselines (no spatial context, same train pixels)
        gmm_K = cfg.get('gmm_K', 9)
        gmm_theta_max   = cfg.get('gmm_theta_max', 1.0)
        gmm_theta_steps = cfg.get('gmm_theta_steps', 50)

        sc_amf    = amf(planted, tr_pix, s)
        sc_regamf = reg_amf(planted, tr_pix, s, sigma)

        print(f"    GMM-GLRT (K={gmm_K}) ...", flush=True)
        if tm == 'additive':
            sc_gmm = gmm_glrt(planted, tr_pix, s, K=gmm_K,
                              theta_max=gmm_theta_max, theta_steps=gmm_theta_steps)
        else:
            sc_gmm = gmm_glrt_replacement(planted, tr_pix, s, K=gmm_K,
                                          theta_max=gmm_theta_max,
                                          theta_steps=gmm_theta_steps)

        sc_amfrep     = amf_replacement(planted, tr_pix, s) if tm == 'replacement' else None
        sc_exactglrt  = exact_glrt_replacement(planted, tr_pix, s) if tm == 'replacement' else None

        det_scores = {
            'CF-Attn': sc_cf,
            'DSM':     sc_dsm,
            'AMF':     sc_amf,
            'Reg-AMF': sc_regamf,
            'GMM-GLRT': sc_gmm,
        }
        if tm == 'replacement':
            det_scores['AMF-rep']    = sc_amfrep
            det_scores['Exact-GLRT'] = sc_exactglrt

        metrics[tm] = {k: auc_safe(labels, v) for k, v in det_scores.items()}
        rocs[tm]    = {k: roc_safe(labels, v) for k, v in det_scores.items()}

        line = "  ".join(f"{k}={v:.3f}" for k, v in metrics[tm].items())
        print(f"  [{tm}]  {line}", flush=True)

    # ------------------------------------------------------------------ #
    # 6. Save results + figures                                           #
    # ------------------------------------------------------------------ #
    json.dump(metrics,      open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)
    json.dump(rocs,         open(os.path.join(run_dir, 'rocs.json'),    'w'), indent=2)
    json.dump(loss_curves,  open(os.path.join(run_dir, 'loss_curves.json'), 'w'))

    plot_roc(rocs['additive'],    'Additive model',    os.path.join(fig_dir, 'roc_additive.pdf'))
    plot_roc(rocs['replacement'], 'Replacement model', os.path.join(fig_dir, 'roc_replacement.pdf'))
    plot_loss(loss_curves, os.path.join(fig_dir, 'loss_curves.png'))

    # Summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, tm in zip(axes, ('additive', 'replacement')):
        names = list(metrics[tm].keys()); vals = list(metrics[tm].values())
        bars = ax.bar(names, vals, color=['#d62728', '#1f77b4', '#aec7e8', '#6baed6', '#08306b'][:len(names)])
        ax.set_ylim(0.4, 1.0); ax.set_title(f'{tm.capitalize()} model')
        ax.set_ylabel('AUC'); ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, min(v + 0.01, 0.99),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'auc_bar.pdf'))
    plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
