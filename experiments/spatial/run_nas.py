"""
============================================================================
SPATIAL NEIGHBOR-ADAPTED SCORE EXPERIMENT  (paper Section 6 / 8.2)
============================================================================

Pipeline (exactly as specified):
  1. Load Pavia, build false color (see show_pavia_grid.py for picking boxes).
  2. Mark a TRAIN rectangle and a TEST rectangle — both must EXCLUDE the
     target class.  (Set them in the CONFIG block below.)
  3. Normalize the WHOLE image, PCA(all pixels) -> latent_dim, then crop the
     two rectangles.  Every detector runs in this shared PCA space.
  4. Train the spatial neighbor-adapted score model on the train rectangle.
     Also train DSM and (optionally) LRao on the same pixels, and prepare all
     classical multiclass baselines (secondary = train pixels).
  5. Plant weak targets on the test rectangle.
  6. Evaluate every detector on the test set, additive AND replacement.

Saves (under results/spatial_<ts>/):
  - config.yaml, metrics.json (all AUCs, additive + replacement)
  - loss_curves.json + figures/loss_curves.png  (NAS, DSM, [LRao])
  - models/nas.pt, models/dsm.pt[, models/lrao.pt]
  - figures/roc_additive.pdf, roc_replacement.pdf
  - figures/auc_bar_additive.pdf, auc_bar_replacement.pdf

RUN:
  cd /Users/mac/Desktop/final_paper_experiment/pythonProject
  .venv/bin/python -u run_spatial_experiment.py  > /tmp/spatial.log 2>&1 &
============================================================================
"""

import os, sys, json, time
from datetime import datetime
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_target_signature, compute_sigma_from_data,
    plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, cem, dsm_additive, dsm_replacement, amf_replacement,
    gmm_glrt, gmm_glrt_replacement, dltd, smglrt, exact_glrt_replacement,
)
from final_paper_experiments.models.neighbor_adapted import (
    NeighborAdaptedScore, extract_neighborhoods, dsm_loss as nas_dsm_loss,
    adapted_score_field, LinearAutoencoder, train_linear_ae,
)
from dsm_model import ScoreNet, dsm_loss, lfi_loss_mode2, compute_scores

# ===========================================================================
# CONFIG  — EDIT THESE  (rectangles are [row0:row1, col0:col1], end-exclusive)
# ===========================================================================
CFG = dict(
    dataset      = 'real_datasets/pavia-u.mat',
    norm_mode    = 'per_band',         # whole-image normalization (then PCA below)

    target_cls   = 0,                  # <-- TARGET class (its mean = signature s)
    # TRAIN rectangle (background only, must NOT contain target_cls pixels)
    train_box    = (380, 470, 0, 200), # rows 380..470, cols 0..200
    # TEST rectangle (background only, targets get planted here)
    test_box     = (480, 570, 0, 200),

    amplitude       = 0.15,            # WEAK target amplitude (fill factor theta)
    target_fraction = 0.10,            # fraction of test pixels with a planted target
    seed            = 42,

    train_n      = 4000,               # cap on train pixels used (speed)
    test_n       = 4000,               # cap on test pixels evaluated (speed)

    # ----- Dimensionality reduction: PCA on the WHOLE image -> latent_dim.
    #       Everything downstream (all detectors) runs in this PCA space. -----
    latent_dim   = 8,

    # ----- Neighbor-adapted score model -----
    k            = 11,                  # k x k spatial neighborhood (K = k*k - 1 neighbors)
    M            = 1024,                # feature dim ("power to the last layer")
    nas_hidden   = (64, 64),
    nas_lambda   = 0.5,                # ridge regularization (init value if trainable)
    nas_learn_lambda = True,           # train lambda end-to-end
    nas_epochs   = 300,
    nas_batch    = 64,
    nas_mc       = 8,                  # MC noise draws at inference

    # ----- DSM / LRao (same arch family) -----
    dsm_hidden   = (64, 64),
    activation   = 'silu',
    dsm_epochs   = 3000,
    lr           = 5e-4,
    weight_decay = 1e-4,
    batch_size   = 256,
    dsm_sigma_rho= 0.01,
    lfi_delta    = 0.01,
    lfi_cutoff   = 1e-3,

    run_lrao     = False,               # LRao Mode-2 baseline (SLOW at D=103; set False to skip)
    lrao_epochs  = 100,

    gmm_K        = 9,                  # GMM components for GMM-GLRT/DLTD/SMGLRT (multiclass setting)
    gmm_theta_max= 1.0,
    gmm_theta_steps = 50,
)
# ===========================================================================

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

# Load rectangle coordinates from the interactive picker if available
_boxes_file = '/tmp/spatial_boxes.json'
if os.path.exists(_boxes_file):
    _b = json.load(open(_boxes_file))
    CFG['train_box'] = tuple(_b['train_box'])
    CFG['test_box']  = tuple(_b['test_box'])
    print(f"Loaded boxes from {_boxes_file}")
    print(f"  train_box = {CFG['train_box']}")
    print(f"  test_box  = {CFG['test_box']}")


def auc(lab, sc):
    try:    return float(roc_auc_score(lab, sc))
    except Exception: return float('nan')

def roc(lab, sc):
    try:
        fpr, tpr, _ = roc_curve(lab, sc); return fpr.tolist(), tpr.tolist(), auc(lab, sc)
    except Exception:
        return [0,1], [0,1], float('nan')


def crop_pixels_and_neighbors(img, box, k):
    """img: (H,W,D). Returns centers (P,D) and neighbors (P,K,D) for the crop."""
    r0, r1, c0, c1 = box
    sub = torch.tensor(img[r0:r1, c0:c1, :], dtype=torch.float32)   # (h,w,D)
    centers, neighbors = extract_neighborhoods(sub, k)             # (P,D),(P,K,D)
    return centers, neighbors


def main():
    t_start = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('final_paper_experiments', 'results', f'spatial_{ts}')
    fig_dir = os.path.join(run_dir, 'figures'); mdl_dir = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(CFG, f)
    print(f"Run dir: {run_dir}")

    rng = np.random.default_rng(CFG['seed'])
    torch.manual_seed(CFG['seed'])

    # ---- 3. Load + normalize whole image, then PCA(all pixels) -> latent_dim ----
    data, gt = load_and_normalize(CFG['dataset'], mode=CFG['norm_mode'])
    H, W, D_RAW = data.shape
    gt_flat = gt.flatten()
    all_flat = data.reshape(-1, D_RAW)
    print(f"Image {H}x{W}x{D_RAW}  (raw bands)")

    from sklearn.decomposition import PCA
    D   = CFG['latent_dim']
    pca = PCA(n_components=D).fit(all_flat)                       # fit on WHOLE image
    evr = pca.explained_variance_ratio_.sum()
    print(f"[PCA] {D_RAW} -> {D}   explained variance = {evr:.4f}")
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)

    # save the PCA (so the latent is reproducible)
    import pickle
    with open(os.path.join(mdl_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # Target signature in PCA space (mean of transformed target pixels, unit-norm)
    tgt_pca = pca.transform(all_flat[gt_flat == CFG['target_cls']])
    assert len(tgt_pca) > 0, f"no pixels for target class {CFG['target_cls']}"
    s = compute_target_signature(tgt_pca)                        # (D,)

    # sanity: rectangles must exclude the target class
    def box_has_target(box):
        r0, r1, c0, c1 = box
        return np.any(gt[r0:r1, c0:c1] == CFG['target_cls'])
    for nm, bx in [('train_box', CFG['train_box']), ('test_box', CFG['test_box'])]:
        if box_has_target(bx):
            print(f"  WARNING: {nm} {bx} CONTAINS target class {CFG['target_cls']} pixels!")

    # ---- crop train/test from the PCA image, extract spatial neighborhoods (in D dims) ----
    tr_centers, tr_neigh = crop_pixels_and_neighbors(pca_img, CFG['train_box'], CFG['k'])
    te_centers, te_neigh = crop_pixels_and_neighbors(pca_img, CFG['test_box'],  CFG['k'])
    print(f"train crop: {tr_centers.shape[0]} px | test crop: {te_centers.shape[0]} px")

    # subsample for speed (keep neighborhoods aligned)
    def subsample(centers, neigh, n):
        if centers.shape[0] <= n: return centers, neigh, np.arange(centers.shape[0])
        idx = rng.choice(centers.shape[0], n, replace=False)
        return centers[idx], neigh[idx], idx
    tr_centers, tr_neigh, _ = subsample(tr_centers, tr_neigh, CFG['train_n'])
    te_centers, te_neigh, _ = subsample(te_centers, te_neigh, CFG['test_n'])

    # ---- 5. Plant weak targets on the test centers (in PCA space; clean neighbors kept) ----
    test_clean = te_centers.numpy()
    test_sets  = {}
    for tm in ('additive', 'replacement'):
        planted, labels, _ = plant_targets(
            test_clean, s, CFG['amplitude'], CFG['target_fraction'],
            model=tm, seed=CFG['seed'])
        test_sets[tm] = (planted, labels)
    n_tgt = int(test_sets['additive'][1].sum())
    print(f"planted {n_tgt} targets in {D}-D PCA space (amplitude={CFG['amplitude']})")

    train_flat = tr_centers.numpy()                              # (Ntr, D)
    # sigma from the FULL image's PCA variance
    sigma_full = compute_sigma_from_data(pca.transform(all_flat), CFG['dsm_sigma_rho'])
    sigma_crop = compute_sigma_from_data(train_flat, CFG['dsm_sigma_rho'])
    sigma      = sigma_full
    print(f"\nlatent_dim={D}  ||s||={np.linalg.norm(s):.4f}  "
          f"sigma_full={sigma_full:.5f}  sigma_crop={sigma_crop:.5f}  USING sigma_full")
    baseline_loss = D / (sigma ** 2)
    print(f"zero-score DSM baseline loss = latent_dim/sigma^2 = {baseline_loss:.2f}")

    loss_curves = {}

    # =======================================================================
    # 4a. Train the NEIGHBOR-ADAPTED spatial model (DSM loss over neighborhoods)
    # =======================================================================
    print("\n[NAS] training neighbor-adapted spatial model ...")
    nas = NeighborAdaptedScore(D, M=CFG['M'], hidden=CFG['nas_hidden'],
                               k=CFG['k'], lam_init=CFG['nas_lambda'],
                               learn_lambda=CFG['nas_learn_lambda'])
    opt = torch.optim.Adam(nas.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    P = tr_centers.shape[0]; bs = CFG['nas_batch']
    nas_hist = []
    pbar = tqdm(range(1, CFG['nas_epochs'] + 1), desc='NAS', dynamic_ncols=True)
    for ep in pbar:
        perm = torch.randperm(P); tot = 0.0; nb = 0
        for i in range(0, P, bs):
            sel = perm[i:i+bs]
            loss = nas_dsm_loss(nas, tr_centers[sel], tr_neigh[sel], sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        nas_hist.append(tot / nb)
        pbar.set_postfix(loss=f"{nas_hist[-1]:.2f}",
                         ratio=f"{nas_hist[-1]/baseline_loss:.3f}",
                         lam=f"{float(nas.lam):.3f}")
    loss_curves['NAS'] = nas_hist
    print(f"   final lambda = {float(nas.lam):.4f}  (init {CFG['nas_lambda']}, "
          f"{'trained' if CFG['nas_learn_lambda'] else 'fixed'})")
    torch.save({'state_dict': nas.state_dict(), 'cfg': CFG}, os.path.join(mdl_dir, 'nas.pt'))

    # =======================================================================
    # 4b. Train DSM (global score net) on the same train pixels
    # =======================================================================
    print("\n[DSM] training global DSM ...")
    dsm = ScoreNet(D, list(CFG['dsm_hidden']), CFG['activation'])
    optd = torch.optim.Adam(dsm.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    Xtr = torch.tensor(train_flat, dtype=torch.float32)
    dsm_hist = []
    pbar = tqdm(range(1, CFG['dsm_epochs'] + 1), desc='DSM', dynamic_ncols=True)
    for ep in pbar:
        perm = torch.randperm(len(Xtr)); tot = 0.0; nb = 0
        for i in range(0, len(Xtr), CFG['batch_size']):
            b = Xtr[perm[i:i+CFG['batch_size']]]
            loss = dsm_loss(dsm, b, sigma)
            optd.zero_grad(); loss.backward(); optd.step()
            tot += loss.item(); nb += 1
        dsm_hist.append(tot / nb)
        pbar.set_postfix(loss=f"{dsm_hist[-1]:.2f}",
                         ratio=f"{dsm_hist[-1]/baseline_loss:.3f}")
    loss_curves['DSM'] = dsm_hist
    dsm.eval()
    torch.save({'state_dict': dsm.state_dict(), 'cfg': CFG}, os.path.join(mdl_dir, 'dsm.pt'))

    # =======================================================================
    # 4c. (optional) Train LRao Mode-2 (Sigma in-graph) — SLOW at D=103
    # =======================================================================
    def _train_lrao(detach):
        torch.manual_seed(CFG['seed'])
        model = ScoreNet(D, list(CFG['dsm_hidden']), CFG['activation'])
        optl = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        hist = []
        pbar = tqdm(range(1, CFG['lrao_epochs'] + 1),
                    desc=f'LRao(detach={detach})', dynamic_ncols=True)
        for ep in pbar:
            perm = torch.randperm(len(Xtr)); tot = 0.0; nb = 0
            for i in range(0, len(Xtr), CFG['batch_size']):
                b = Xtr[perm[i:i+CFG['batch_size']]]
                loss = lfi_loss_mode2(model, b, CFG['lfi_delta'], CFG['lfi_cutoff'],
                                      detach_sigma=detach)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite LRao loss")
                optl.zero_grad(); loss.backward(); optl.step()
                tot += loss.item(); nb += 1
            hist.append(-tot / nb)
            pbar.set_postfix(trJ=f"{hist[-1]:.3f}")
        return model, hist

    lrao = None
    if CFG['run_lrao']:
        print("\n[LRao] training LRao Mode-2 (slow; unstable at high D) ...")
        for detach in (False, True):           # try Sigma in-graph, fall back to detached
            try:
                lrao, lr_hist = _train_lrao(detach)
                loss_curves['LRao'] = lr_hist
                lrao.eval()
                torch.save({'state_dict': lrao.state_dict(), 'cfg': CFG, 'detach_sigma': detach},
                           os.path.join(mdl_dir, 'lrao.pt'))
                print(f"   LRao trained (detach_sigma={detach}).")
                break
            except Exception as e:
                print(f"   [warn] LRao (detach_sigma={detach}) failed: {e}")
                lrao = None

    json.dump(loss_curves, open(os.path.join(run_dir, 'loss_curves.json'), 'w'))

    # =======================================================================
    # 6. Evaluate every detector (additive + replacement)
    # =======================================================================
    print("\n[EVAL] scoring all detectors ...")

    # NAS score fields (train for statistics; test per target model)
    def nas_field(centers, neigh):
        return adapted_score_field(nas, centers, neigh, sigma,
                                   n_mc=CFG['nas_mc']).numpy()
    psi_tr_nas = nas_field(tr_centers, tr_neigh)               # (Ntr, D)
    z_bar = psi_tr_nas.mean(0); C = np.cov(psi_tr_nas, rowvar=False)
    Cs = float(s @ C @ s); norm_add = np.sqrt(max(Cs, 1e-12))
    r_tr = (psi_tr_nas * (train_flat - s)).sum(1); r_bar = r_tr.mean(); r_std = r_tr.std() + 1e-12

    metrics = {'additive': {}, 'replacement': {}}
    rocs    = {'additive': {}, 'replacement': {}}

    from final_paper_experiments.baselines.detectors import lrao_iid

    for tm in ('additive', 'replacement'):
        planted, labels = test_sets[tm]
        td = planted                                           # (Nte, D)

        def safe(name, fn):
            try:
                return fn()
            except Exception as e:
                print(f"   [warn] {name} failed: {e}")
                return np.full(len(labels), np.nan)

        # ---- classical / DSM / LRao baselines (multiclass set) ----
        jobs = {}
        if tm == 'additive':
            jobs['AMF']      = lambda: amf(td, train_flat, s)
            jobs['Reg-AMF']  = lambda: reg_amf(td, train_flat, s, sigma)
            jobs['CEM']      = lambda: cem(td, train_flat, s)
            jobs['GMM-GLRT'] = lambda: gmm_glrt(td, train_flat, s, K=CFG['gmm_K'])
            jobs['DLTD']     = lambda: dltd(td, train_flat, s, K=CFG['gmm_K'])
            jobs['SMGLRT']   = lambda: smglrt(td, train_flat, s, K=CFG['gmm_K'])
            jobs['DSM']      = lambda: dsm_additive(td, train_flat, dsm, s)
            if lrao is not None:
                jobs['LRao'] = lambda: lrao_iid(td, train_flat, lrao, s, CFG['lfi_delta'])
        else:
            jobs['AMF-rep']      = lambda: amf_replacement(td, train_flat, s)
            jobs['CEM']          = lambda: cem(td, train_flat, s)
            jobs['GMM-GLRT-rep'] = lambda: gmm_glrt_replacement(
                td, train_flat, s, K=CFG['gmm_K'],
                theta_max=CFG['gmm_theta_max'], theta_steps=CFG['gmm_theta_steps'])
            jobs['DLTD']         = lambda: dltd(td, train_flat, s, K=CFG['gmm_K'])
            jobs['SMGLRT']       = lambda: smglrt(td, train_flat, s, K=CFG['gmm_K'])
            jobs['Exact-GLRT']   = lambda: exact_glrt_replacement(td, train_flat, s)
            jobs['DSM-rep']      = lambda: dsm_replacement(td, train_flat, dsm, s)
            if lrao is not None:
                jobs['LRao'] = lambda: lrao_iid(td, train_flat, lrao, s, CFG['lfi_delta'])

        scores = {name: safe(name, fn) for name, fn in jobs.items()}

        # ---- OUR neighbor-adapted spatial model (NAS) ----
        def nas_score():
            psi_te = nas_field(torch.tensor(td, dtype=torch.float32), te_neigh)
            if tm == 'additive':
                return -((psi_te - z_bar) @ s) / norm_add
            r_te = (psi_te * (td - s)).sum(1)
            return (r_te - r_bar) / r_std
        scores['NAS'] = safe('NAS', nas_score)

        for det, sc in scores.items():
            metrics[tm][det] = auc(labels, sc)
            rocs[tm][det] = roc(labels, sc)
        print(f"  [{tm}]  " + "  ".join(f"{k}={metrics[tm][k]:.3f}" for k in scores))

    json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)
    json.dump(rocs, open(os.path.join(run_dir, 'rocs.json'), 'w'))

    # =======================================================================
    # Figures: loss curves, ROC, AUC bars
    # =======================================================================
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt

    # loss curves
    nplots = len(loss_curves)
    fig, ax = plt.subplots(1, nplots, figsize=(4.2 * nplots, 3.2))
    if nplots == 1: ax = [ax]
    for a, (name, h) in zip(ax, loss_curves.items()):
        a.plot(h); a.set_title(f'{name} training'); a.set_xlabel('epoch'); a.grid(alpha=.3)
        a.set_ylabel('tr(J*)' if name == 'LRao' else 'loss')
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, 'loss_curves.png'), dpi=130)
    plt.close(fig)

    # ROC + AUC bar per target model
    for tm in ('additive', 'replacement'):
        fig, a = plt.subplots(figsize=(5.2, 4.2))
        for det, (fpr, tpr, au) in rocs[tm].items():
            lw = 2.5 if det == 'NAS' else 1.3
            a.plot(fpr, tpr, lw=lw, label=f'{det} ({au:.3f})')
        a.plot([0,1],[0,1],'k--',lw=.7); a.set_xlabel('FPR'); a.set_ylabel('TPR')
        a.set_title(f'ROC — {tm}'); a.legend(fontsize=7, loc='lower right'); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f'roc_{tm}.pdf')); plt.close(fig)

        fig, a = plt.subplots(figsize=(6.5, 3.4))
        dets = list(metrics[tm].keys()); vals = [metrics[tm][d] for d in dets]
        colors = ['#d62728' if d == 'NAS' else '#888888' for d in dets]
        a.bar(range(len(dets)), vals, color=colors)
        a.set_xticks(range(len(dets))); a.set_xticklabels(dets, rotation=40, ha='right', fontsize=8)
        a.set_ylabel('AUC'); a.set_ylim(0.45, 1.0); a.axhline(0.5, color='k', lw=.5)
        a.set_title(f'AUC — {tm}  (NAS in red)'); a.grid(alpha=.3, axis='y')
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, f'auc_bar_{tm}.pdf')); plt.close(fig)

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min.  Results -> {run_dir}")
    print("metrics.json:")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
