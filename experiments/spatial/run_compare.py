"""
run_compare.py — Focused spatial detector comparison (single scenario).

Detectors
---------
  DSM            — our global per-pixel score (ScoreNet, no spatial context)
  LinearDSM      — neighbor-adapted LINEAR score (ridge regression head, paper §6.1)
  NeighborMLP    — spatial denoiser score net                    (Ours, spatial)
  AMF            — Adaptive Matched Filter (global SCM)
  AMF-local      — AMF on the per-pixel k×k window SCM (same window as spatial nets)
  GMM-Levin      — Levin product-GMM GLRT

Deep nets train on GPU (cuda) when available.

Detection is run TWICE (per scenario):
  1. in-patch  : target signature = dominant class of the test patch (as before)
  2. foreign   : target signature = a class NOT present in the patch, scaled so
                 ||s|| equals the mean per-pixel norm of the patch.
The in-patch outputs land in   <run>/ ,  the foreign outputs in  <run>/foreign/.

Per-run deliverables
--------------------
  false_color.pdf             — RGB false color of the test box
  label_map_targets.pdf       — GT class map + planted-target locations (cyan)
  signatures.pdf              — per-class mean spectra + the target signature
  detection_maps.pdf          — per-detector score maps + target locations (cyan)
  detected_pfa.pdf            — detected pixels @ Pfa=0.05 (hits green / FA red)
  false_alarms_falsecolor.pdf — false-alarm pixels @ Pfa∈{.01,.05,.1} on false color
  false_alarms_labelmap.pdf   — false-alarm pixels @ Pfa∈{.01,.05,.1} on label map
  roc.pdf                     — all-detector ROC overlay
  pfa_per_class.pdf           — grouped per-class Pfa bars (ALL classes incl. 0)
  summary_table.csv/.md, metrics.json, scores.npz

Usage
-----
  .venv/bin/python -u experiments/spatial/run_compare.py --dry-run
  python -u experiments/spatial/run_compare.py --config experiments/spatial/colab.yaml \
        --results_dir /content/drive/MyDrive/final_paper/compare_results
"""
import argparse, copy, json, os, sys, time
from datetime import datetime

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import uniform_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from final_paper_experiments.data_utils import load_and_normalize, plant_targets
from final_paper_experiments.baselines.detectors import dsm_additive
from final_paper_experiments.baselines.gmm_glrt_levin import gmm_glrt_levin_additive
from final_paper_experiments.evaluation import (
    partial_auc, dr_at_fpr, auc_safe, roc_safe, cfar_threshold, per_class_fpr,
    compute_signature, generate_random_boxes, scores_to_spatial_map,
)
from dsm_model import ScoreNet, dsm_loss
from neighbor_mlp_model import NeighborMLPDenoiser, score_nmlp_additive, neighbor_mlp_dsm_loss
from local_detectors import amf_cem_local_scm, amf_global
from final_paper_experiments.models.neighbor_adapted import (
    NeighborAdaptedScore, dsm_loss as ridge_dsm_loss, adapted_score_field,
)
from tqdm import tqdm

# Shared helpers from the main runner.
from run_colab import (
    _crop_pca_box,
    _make_whitening, _whiten_np, _whitened_sigma, _placeholder_whitening,
)

CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees', 5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks', 9: 'shadows',
}
CLS_COLORS_HEX = {
    0: '#000000', 1: '#808080', 2: '#00cc44', 3: '#d2691e',
    4: '#006400', 5: '#add8e6', 6: '#a52a2a', 7: '#9400d3',
    8: '#ff4500', 9: '#00008b',
}

# Overlay colors — chosen distinct from BOTH the label palette and inferno.
TARGET_MARK = '#00ffff'   # cyan — planted target locations
FA_MARK     = '#ff2d2d'   # red  — false alarms
HIT_MARK    = '#39ff14'   # lime — true detections

# Fixed display order + colors for the detectors.
DET_ORDER = [
    'DSM',
    'LinearDSM',
    'NeighborMLP',
    'AMF',
    'AMF-local',
    'GMM-Levin',
]
DET_COLORS = {
    'DSM':        '#ff7f0e',   # orange
    'LinearDSM':  '#17becf',   # teal
    'NeighborMLP':'#2ca02c',   # green
    'AMF':        '#9467bd',   # purple
    'AMF-local':  '#c5b0d5',   # light purple
    'GMM-Levin':  '#e377c2',   # pink
}

# Pfa levels for the false-alarm overlays.
PFA_LEVELS = [0.01, 0.05, 0.10]
FIGSIZE = (11, 7)   # every saved figure uses this exact size

DEFAULT_CFG = dict(
    dataset='data/pavia-u.mat',
    norm_mode='none',
    manual_boxes_path='experiments/spatial/manual_boxes.json',
    scenario_index=0,
    min_pixels=2000,
    random_scenario_seeds=[42, 123, 456, 789],
    sig_dom_weight=0.8, sig_mean_weight=0.2,
    amplitude=0.15, target_fraction=0.10, edge_guard=5,
    n_budget=None,               # None = full train box (no subsampling); int = side-crop
    k=5,
    local_scm_loading=1e-8,
    baseline_eig_floor=1e-12,
    # NeighborMLP — encoder: D→enc_hidden→d_lat ; denoiser: (D+(K+1)*d_lat)→score_hidden→D
    nmlp_d_lat=16, nmlp_K=8, nmlp_enc_hidden=[128, 64], nmlp_score_hidden=[128],
    nmlp_epochs=1000, nmlp_lr=3e-4, nmlp_batch=256,
    # DSM
    dsm_hidden=[64, 64], dsm_epochs=1000, dsm_lr=5e-4,
    # LinearDSM (neighbor-adapted ridge score head, paper §6.1)
    ridge_M=256, ridge_hidden=[128, 128], ridge_lam_init=0.1,
    ridge_epochs=1000, ridge_lr=3e-4, ridge_batch=256, ridge_n_mc=8,
    # shared
    activation='silu', dsm_sigma_rho=0.01,
    whiten_mode='zca', whiten_eig_floor=1e-5,
    batch_size=256, weight_decay=1e-4,
    gmm_steps=50, gmm_K=3,
    pfa_target=0.05,
    seed=42,
    results_dir='final_paper_experiments/results',
)

DRYRUN_OVERRIDES = dict(
    nmlp_epochs=8, dsm_epochs=20, ridge_epochs=8, ridge_n_mc=2,
    nmlp_K=4, ridge_M=64,
    n_budget=400,   # dry-run only: contiguous side-crop for speed (still spatial)
)


# ---------------------------------------------------------------------------
def _side_crop_box(box, budget):
    """Cut pixels from the SIDES only → a contiguous, centered sub-box of about
    `budget` pixels. Preserves spatial context (no random subsampling)."""
    r0, r1, c0, c1 = box
    H, Wd = r1 - r0, c1 - c0
    if not budget or H * Wd <= int(budget):
        return [r0, r1, c0, c1]
    scale = (float(budget) / (H * Wd)) ** 0.5
    newH = max(int(round(H * scale)), 1)
    newW = max(int(round(Wd * scale)), 1)
    dr, dc = (H - newH) // 2, (Wd - newW) // 2
    return [r0 + dr, r0 + dr + newH, c0 + dc, c0 + dc + newW]


def _false_color(data_raw, box, bands=(60, 30, 10)):
    r0, r1, c0, c1 = box
    rgb = data_raw[r0:r1, c0:c1][..., list(bands)].astype(np.float32)
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def _gt_colorimage(gt_crop):
    H, W = gt_crop.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for cid, hex_ in CLS_COLORS_HEX.items():
        img[gt_crop == cid] = to_rgb(hex_)
    return img


def _rc(flat_idx, W_b):
    """Flat box index → (rows, cols) for scatter (x=col, y=row)."""
    flat_idx = np.asarray(flat_idx, dtype=int)
    return flat_idx // W_b, flat_idx % W_b


def _bg_class_means(te_raw, te_gt):
    """Mean spectrum for every class present in the patch (incl. class 0)."""
    out = {}
    for c in sorted(np.unique(te_gt)):
        m = (te_gt == c)
        if m.sum() > 0:
            out[int(c)] = te_raw[m].mean(axis=0)
    return out


def _pick_foreign_class(gt, present):
    """A labeled class (1..9) NOT present in the patch, with most global pixels."""
    present = {int(c) for c in present}
    cand = [c for c in range(1, 10) if c not in present and int((gt == c).sum()) > 0]
    if not cand:
        return None
    return max(cand, key=lambda c: int((gt == c).sum()))


def _cfar_normalize_map(flat_scores, shape, bg, guard, eps=1e-6, cfar_lam=0.0):
    """Local-CFAR normalization of a projected-score MAP (paper Eq 70/80).

    Standardize each pixel by a BLENDED mean/std:

        mean_eff = (1 - lam) * mean_annulus(q)  +  lam * mean_global(q)
        std_eff  = (1 - lam) * std_annulus(q)   +  lam * std_global(q)
        T_i      = (q_i - mean_eff) / (std_eff + eps)

    cfar_lam=0  → pure local annulus (standard CFAR; default)
    cfar_lam=1  → pure global mean/std (same as original LMP normalization)
    cfar_lam∈(0,1) → smoothly interpolates (robust near boundaries)

    The local component is a 1-D scalar variance of the already-projected score
    (NO matrix inverse, NO rank issue). Cheap windowed stats via uniform_filter.
    """
    H, W = shape
    q = np.asarray(flat_scores, dtype=np.float64).reshape(H, W)
    nb, ng = bg * bg, (guard * guard if guard and guard > 0 else 0)
    m_bg  = uniform_filter(q,     size=bg, mode='wrap')
    s2_bg = uniform_filter(q * q, size=bg, mode='wrap')
    if ng > 0:
        m_g  = uniform_filter(q,     size=guard, mode='wrap')
        s2_g = uniform_filter(q * q, size=guard, mode='wrap')
    else:
        m_g = s2_g = 0.0
    denom = max(nb - ng, 1)
    mean_local = (nb * m_bg - ng * m_g) / denom          # annulus mean
    e2_local   = (nb * s2_bg - ng * s2_g) / denom         # annulus E[q^2]
    std_local  = np.sqrt(np.maximum(e2_local - mean_local ** 2, 0.0))

    # Global stats (over the whole map)
    mean_global = float(q.mean())
    std_global  = float(q.std()) + eps

    lam = float(cfar_lam)
    mean_eff = (1.0 - lam) * mean_local + lam * mean_global
    std_eff  = (1.0 - lam) * std_local  + lam * std_global + eps
    return ((q - mean_eff) / std_eff).reshape(-1).astype(np.float32)


def _knn_fisher_normalize(score_flat, model, pix, nbr, shape, k, eps=1e-6, cfar_lam=0.0):
    """Local-Fisher CFAR for NeighborMLP using the model's OWN top-K selected
    neighbors (not a spatial window). For each pixel, standardize its projected
    score q_i by a BLENDED mean/std:

        mean_eff = (1-lam) * mean_{kNN}(q)  +  lam * mean_global(q)
        std_eff  = (1-lam) * std_{kNN}(q)   +  lam * std_global(q)
        T_i = (q_i - mean_eff) / (std_eff + eps)

    cfar_lam=0  → pure kNN local (default)
    cfar_lam=1  → pure global mean/std

    q at neighbor positions is read straight off the score map via unfold
    (no extra forward passes). The k×k window order matches extract_neighborhoods
    so model selection indices line up correctly.
    """
    H, W = shape
    dev = next(model.parameters()).device
    q_i = torch.tensor(np.asarray(score_flat, np.float32), device=dev)        # (HW,)
    # neighbor q-values via unfold of the score map (circular-padded k×k window)
    p = k // 2
    qmap = q_i.reshape(1, 1, H, W)
    patches = F.unfold(F.pad(qmap, (p, p, p, p), mode='circular'), kernel_size=k)  # (1, k*k, HW)
    patches = patches.reshape(k * k, H * W).t()                               # (HW, k*k)
    cidx = (k * k) // 2
    keep = [m for m in range(k * k) if m != cidx]
    q_neigh = patches[:, keep]                                               # (HW, M)
    # the model's top-K selected neighbor indices
    idx = model.topk_indices(
        torch.tensor(np.asarray(pix, np.float32), device=dev),
        torch.tensor(np.asarray(nbr, np.float32), device=dev))               # (HW, K)
    q_topk = torch.gather(q_neigh, 1, idx)                                    # (HW, K)
    mu_local  = q_topk.mean(dim=1)
    std_local = q_topk.var(dim=1, unbiased=False).sqrt()

    # Global stats (over the whole score map)
    mu_global  = q_i.mean()
    std_global = q_i.std() + eps

    lam = float(cfar_lam)
    mu_eff  = (1.0 - lam) * mu_local  + lam * mu_global
    std_eff = (1.0 - lam) * std_local + lam * std_global + eps
    return ((q_i - mu_eff) / std_eff).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Best-epoch training functions (restore lowest-loss checkpoint at end of training)
# ---------------------------------------------------------------------------

def _train_dsm_best(tr_raw, cfg, device):
    """Train global DSM with best-epoch tracking (restores lowest-loss weights)."""
    D = tr_raw.shape[1]
    W = _make_whitening(tr_raw, cfg, device)
    sigma = _whitened_sigma(cfg)
    net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'], whitening=W).to(device)
    net.sigma = sigma
    opt = torch.optim.Adam(net.parameters(), lr=cfg['dsm_lr'], weight_decay=cfg['weight_decay'])
    Xtr = torch.tensor(tr_raw, dtype=torch.float32).to(device)
    P = len(Xtr)
    E = int(cfg['dsm_epochs'])
    best_loss, best_state = float('inf'), None
    pbar = tqdm(range(E), desc='DSM', dynamic_ncols=True, leave=False)
    last = float('nan')
    for ep in pbar:
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, P, cfg['batch_size']):
            b = Xtr[perm[i:i + cfg['batch_size']]]
            loss = dsm_loss(net, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        pbar.set_postfix(loss=f'{last:.4f}')
        if last < best_loss:
            best_loss = last
            best_state = copy.deepcopy(net.state_dict())
        if ep == 0 or (ep + 1) % max(E // 10, 1) == 0:
            print(f"    [DSM] epoch {ep+1}/{E}  loss={last:.4f}  best={best_loss:.4f}", flush=True)
    net.load_state_dict(best_state)
    net._final_loss = best_loss
    net.eval()
    return net


def _train_nmlp_best(tr_raw, tr_nbr, cfg, device):
    """Train NeighborMLP with best-epoch tracking."""
    D = tr_raw.shape[1]
    W = _make_whitening(tr_raw, cfg, device)
    sigma = _whitened_sigma(cfg)
    nmlp = NeighborMLPDenoiser(
        D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
        enc_hidden=cfg.get('nmlp_enc_hidden'),
        score_hidden=cfg.get('nmlp_score_hidden'),
        hidden=cfg.get('nmlp_hidden', 128),
        n_layers=cfg.get('nmlp_n_layers', 3),
        sigma=sigma, activation=cfg['activation'], whitening=W).to(device)
    opt = torch.optim.AdamW(nmlp.parameters(), lr=cfg['nmlp_lr'], weight_decay=cfg['weight_decay'])
    Xtr = torch.tensor(tr_raw, dtype=torch.float32).to(device)
    Ntr = torch.tensor(tr_nbr, dtype=torch.float32).to(device)
    P = len(Xtr)
    E = int(cfg['nmlp_epochs'])
    best_loss, best_state = float('inf'), None
    pbar = tqdm(range(E), desc='NeighborMLP', dynamic_ncols=True, leave=False)
    last = float('nan')
    for ep in pbar:
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, P, cfg['nmlp_batch']):
            sel = perm[i:i + cfg['nmlp_batch']]
            loss = neighbor_mlp_dsm_loss(nmlp, Xtr[sel], Ntr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        pbar.set_postfix(loss=f'{last:.4f}')
        if last < best_loss:
            best_loss = last
            best_state = copy.deepcopy(nmlp.state_dict())
        if ep == 0 or (ep + 1) % max(E // 10, 1) == 0:
            print(f"    [NeighborMLP] epoch {ep+1}/{E}  loss={last:.4f}  best={best_loss:.4f}",
                  flush=True)
    nmlp.load_state_dict(best_state)
    nmlp._final_loss = best_loss
    nmlp.eval()
    return nmlp


# ---------------------------------------------------------------------------
# LinearDSM (NeighborRidge) — neighbor-adapted ridge score head (paper §6.1).
# Shared encoder + global linear head W0 + per-pixel LOCAL head solved in closed
# form by ridge regression over the k×k neighbors, shrinking toward W0 when the
# neighborhood is uninformative (boundaries). Whitening is applied externally
# (the model has no whitening hook), so it operates in whitened space and the
# scores are mapped back to data space for detection.
# ---------------------------------------------------------------------------

def _train_ridge(tr_raw, tr_nbr, cfg, device, seed):
    """Train LinearDSM (NeighborAdaptedScore) with best-epoch tracking."""
    torch.manual_seed(seed)
    D = tr_raw.shape[1]
    W = _make_whitening(tr_raw, cfg, device)
    sigma = _whitened_sigma(cfg)
    Xc = torch.tensor(_whiten_np(W, tr_raw, device), dtype=torch.float32).to(device)
    Xn = torch.tensor(_whiten_np(W, tr_nbr, device), dtype=torch.float32).to(device)
    model = NeighborAdaptedScore(
        D=D, M=int(cfg.get('ridge_M', 256)),
        hidden=tuple(cfg.get('ridge_hidden', [128, 128])),
        k=int(cfg['k']), lam_init=float(cfg.get('ridge_lam_init', 0.1))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get('ridge_lr', 3e-4)),
                            weight_decay=cfg['weight_decay'])
    P  = len(Xc)
    bs = int(cfg.get('ridge_batch', 256))
    E  = int(cfg.get('ridge_epochs', 1000))
    best_loss, best_state = float('inf'), None
    pbar = tqdm(range(E), desc='LinearDSM', dynamic_ncols=True, leave=False)
    last = float('nan')
    for ep in pbar:
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, P, bs):
            sel = perm[i:i + bs]
            loss = ridge_dsm_loss(model, Xc[sel], Xn[sel], sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        pbar.set_postfix(loss=f'{last:.4f}')
        if last < best_loss:
            best_loss = last
            best_state = copy.deepcopy(model.state_dict())
        if ep == 0 or (ep + 1) % max(E // 10, 1) == 0:
            print(f"    [LinearDSM] epoch {ep+1}/{E}  loss={last:.4f}  best={best_loss:.4f}",
                  flush=True)
    model.load_state_dict(best_state)
    model._final_loss = best_loss
    model._whitening = W
    model.eval()
    return model


def score_ridge_additive(model, test_pix, test_nbr, train_pix, train_nbr, s, cfg):
    """Additive LMP using the neighbor-adapted ridge score (same convention as
    score_nmlp_additive). Scores are computed in whitened space, mapped to data
    space, then normalized by the training-score covariance."""
    W = model._whitening.cpu()
    Wn = W.W.detach().cpu().numpy()                      # (D, D)
    sigma = _whitened_sigma(cfg)
    n_mc = int(cfg.get('ridge_n_mc', 8))
    model.cpu().eval()

    def _scores(pix, nbr):
        cw = torch.tensor(_whiten_np(W, pix, 'cpu'), dtype=torch.float32)   # (B, D)
        nw = torch.tensor(_whiten_np(W, nbr, 'cpu'), dtype=torch.float32)   # (B, K, D)
        psi_w = adapted_score_field(model, cw, nw, sigma, n_mc=n_mc).numpy()  # (B, D) whitened
        return psi_w @ Wn                                                    # data-space score

    z_tr = _scores(train_pix, train_nbr)
    z_te = _scores(test_pix,  test_nbr)
    z_bar = z_tr.mean(axis=0)
    C = np.cov(z_tr, rowvar=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    norm = float(np.sqrt(max(float(s @ C @ s), 1e-12)))
    return -((z_te - z_bar) @ s) / norm


def _savefig(fig, path):
    # NO bbox_inches='tight' — that crops each figure to its content and breaks
    # the uniform size. Save the full FIGSIZE canvas at a fixed dpi so every
    # output file has identical dimensions (FIGSIZE inches; FIGSIZE×150 px PNG).
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path)}", flush=True)


# ---------------------------------------------------------------------------
def score_all(pix, nbr, models, tr_raw, tr_nbr, sig, cfg, device):
    """Score every AVAILABLE detector on (pix, nbr).  Returns {det_name: scores}.

    Deep detectors are skipped gracefully if their model is absent from `models`.
    """
    pix = pix.astype(np.float32)
    nbr = nbr.astype(np.float32)
    out = {}
    floor = float(cfg.get('baseline_eig_floor', 1e-12))
    if models.get('dsm') is not None:
        out['DSM'] = dsm_additive(pix, tr_raw, models['dsm'], sig)
    if models.get('nmlp') is not None:
        out['NeighborMLP'] = score_nmlp_additive(models['nmlp'], pix, nbr, tr_raw, tr_nbr, sig)
    if models.get('ridge') is not None:
        out['LinearDSM'] = score_ridge_additive(models['ridge'], pix, nbr, tr_raw, tr_nbr, sig, cfg)
    out['AMF'] = amf_global(pix, tr_raw, sig, eig_floor=floor)
    amf_loc, _ = amf_cem_local_scm(
        pix, nbr, sig, device=device,
        loading=float(cfg.get('local_scm_loading', 1e-8)))
    out['AMF-local'] = amf_loc
    out['GMM-Levin'] = gmm_glrt_levin_additive(pix, tr_raw, sig,
                                               p_steps=cfg.get('gmm_steps', 50))
    return out


# ---------------------------------------------------------------------------
def run_detection(sig, sig_label, out_dir, ctx):
    """Plant targets with `sig`, score all detectors, write the table + figures."""
    cfg, device, seed = ctx['cfg'], ctx['device'], ctx['seed']
    models = ctx['models']
    tr_raw, tr_nbr = ctx['tr_raw'], ctx['tr_nbr']
    te_raw, te_nbr, te_gt = ctx['te_raw'], ctx['te_nbr'], ctx['te_gt']
    data_norm, gt = ctx['data_norm'], ctx['gt']
    test_box, sidx = ctx['test_box'], ctx['sidx']

    os.makedirs(out_dir, exist_ok=True)
    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    te_idx = np.arange(len(te_raw))
    pfa_t = float(cfg.get('pfa_target', 0.05))
    all_pfa = sorted(set(PFA_LEVELS) | {pfa_t})

    print(f"\n########## DETECTION RUN: {sig_label} ##########", flush=True)
    edge_guard = int(cfg.get('edge_guard', 5))
    planted, labels, tgt_idx = plant_targets(
        te_raw, sig, cfg['amplitude'], cfg['target_fraction'],
        model='additive', seed=seed,
        spatial_shape=(H_b, W_b), edge_guard=edge_guard)
    planted = planted.astype(np.float32)
    print(f"[{sig_label}] planted {int(labels.sum())} targets  ||s||={np.linalg.norm(sig):.4g}",
          flush=True)

    print("Scoring detectors (test) ...", flush=True)
    test_scores = score_all(planted, te_nbr, models, tr_raw, tr_nbr, sig, cfg, device)
    print("Scoring detectors (train, for CFAR thresholds) ...", flush=True)
    train_scores = score_all(tr_raw, tr_nbr, models, tr_raw, tr_nbr, sig, cfg, device)

    DETS = [d for d in DET_ORDER if d in test_scores]
    print(f"Active detectors: {DETS}", flush=True)

    # CFAR thresholds per detector per Pfa level (from TRAIN scores only).
    thr = {d: {p: cfar_threshold(np.asarray(train_scores[d], float), target_fpr=p)
               for p in all_pfa} for d in DETS}

    # ---- Metrics (per-class Pfa includes ALL classes, incl. class 0) ----
    rows, pfa_per_class, roc_curves = [], {}, {}
    for det in DETS:
        sc = np.asarray(test_scores[det], dtype=np.float64)
        pcf = per_class_fpr(sc, labels, te_gt, thr[det][pfa_t])
        pfa_per_class[det] = pcf
        pfa_vals = list(pcf.values()) if pcf else [float('nan')]
        fpr, tpr, auc_v = roc_safe(labels, sc)
        roc_curves[det] = (fpr, tpr, auc_v)
        rows.append({
            'Detector': det,
            'pAUC@0.05': partial_auc(labels, sc, fpr_max=0.05),
            'AUC': auc_v,
            'Pd@Pfa=0.05': dr_at_fpr(labels, sc, fpr_list=(pfa_t,))[str(pfa_t)],
            'Pfa_avg': float(np.nanmean(pfa_vals)),
            'Pfa_max': float(np.nanmax(pfa_vals)),
        })

    # ---- Per-class Pfa columns (ALL classes incl. 0, ordered by class id) ----
    name_to_id = {v: k for k, v in CLS_NAMES.items()}
    all_cls = sorted({c for d in DETS for c in pfa_per_class[d]},
                     key=lambda nm: name_to_id.get(nm, 999))
    for r in rows:
        pcf = pfa_per_class[r['Detector']]
        for nm in all_cls:
            r[f'Pfa[{nm}]'] = float(pcf.get(nm, 0.0))

    # ---- Summary table ----
    cols = (['Detector', 'pAUC@0.05', 'AUC', 'Pd@Pfa=0.05', 'Pfa_avg', 'Pfa_max']
            + [f'Pfa[{nm}]' for nm in all_cls])
    with open(os.path.join(out_dir, 'summary_table.csv'), 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) if c == 'Detector' else f'{r[c]:.4f}'
                             for c in cols) + '\n')
    md_path = os.path.join(out_dir, 'summary_table.md')
    with open(md_path, 'w') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r['Detector'] if c == 'Detector'
                                      else f'{r[c]:.3f}' for c in cols) + ' |\n')
    print(f"\n=== Summary [{sig_label}] ===", flush=True)
    print(open(md_path).read(), flush=True)

    json.dump({'scenario_index': sidx, 'signature': sig_label, 'test_box': test_box,
               'pfa_target': pfa_t, 'rows': rows, 'pfa_per_class': pfa_per_class},
              open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=2, default=str)
    npz = {f'score_{d}': test_scores[d] for d in DETS}
    npz['labels'] = labels; npz['te_gt'] = te_gt; npz['tgt_idx'] = tgt_idx
    np.savez(os.path.join(out_dir, 'scores.npz'), **npz)

    # ---- Figures ----
    print("Saving figures ...", flush=True)
    train_box = ctx['train_box']
    r0t, r1t, c0t, c1t = train_box
    fc = _false_color(data_norm, test_box)
    fc_tr = _false_color(data_norm, train_box)
    lm = _gt_colorimage(gt[r0:r1, c0:c1])
    lm_tr = _gt_colorimage(gt[r0t:r1t, c0t:c1t])
    t_r, t_c = _rc(tgt_idx, W_b)

    # false color — TRAIN + TEST boxes
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(fc_tr); axes[0].axis('off'); axes[0].set_title('TRAIN box', fontsize=9)
    axes[1].imshow(fc);    axes[1].axis('off'); axes[1].set_title('TEST box', fontsize=9)
    fig.suptitle(f'False color — scen {sidx}', fontsize=11)
    _savefig(fig, os.path.join(out_dir, 'false_color.pdf'))

    # (1) label maps — TRAIN + TEST (targets overlaid on TEST only)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    axes[0].imshow(lm_tr); axes[0].axis('off'); axes[0].set_title('TRAIN box', fontsize=9)
    axes[1].imshow(lm);    axes[1].axis('off'); axes[1].set_title('TEST box + targets', fontsize=9)
    axes[1].scatter(t_c, t_r, s=12, facecolors='none', edgecolors=TARGET_MARK, linewidths=0.8)
    present = sorted(set(np.unique(te_gt)) | set(np.unique(gt[r0t:r1t, c0t:c1t])))
    handles = [mpatches.Patch(color=CLS_COLORS_HEX.get(int(c), '#777'),
                              label=CLS_NAMES.get(int(c), f'cls{c}')) for c in present]
    handles.append(Line2D([], [], marker='o', ls='', mfc='none', mec=TARGET_MARK, label='target'))
    axes[1].legend(handles=handles, fontsize=6, loc='upper right', framealpha=0.75, ncol=2)
    fig.suptitle(f'Label map — {sig_label}', fontsize=11)
    _savefig(fig, os.path.join(out_dir, 'label_map_targets.pdf'))

    # (7) class signatures + target signature
    means = _bg_class_means(te_raw, te_gt)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(te_raw.shape[1])
    for c, mu in means.items():
        ax.plot(x, mu, color=CLS_COLORS_HEX.get(c, '#777'), lw=1.0,
                label=CLS_NAMES.get(c, f'cls{c}'))
    ax.plot(x, sig, color='k', lw=2.4, ls='--', label=f'target [{sig_label}]')
    ax.set_xlabel('band'); ax.set_ylabel('value')
    ax.set_title(f'Class signatures + target — {sig_label}', fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _savefig(fig, os.path.join(out_dir, 'signatures.pdf'))

    # (2) detection score maps (no target overlay)
    ncol = 3
    nrow = int(np.ceil(len(DETS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=FIGSIZE)
    axes = np.atleast_1d(axes).ravel()
    for j, det in enumerate(DETS):
        smap = scores_to_spatial_map(test_scores[det], te_idx, (H_b, W_b))
        ax = axes[j]
        im = ax.imshow(smap, cmap='inferno'); ax.axis('off')
        ax.set_title(f'{det}  (AUC={roc_curves[det][2]:.3f})', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(DETS), len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'Detection maps — {sig_label}', fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, 'detection_maps.pdf'))

    # (5) detected pixels @ Pfa=0.05 — hits (green) vs false alarms (red)
    fig, axes = plt.subplots(nrow, ncol, figsize=FIGSIZE)
    axes = np.atleast_1d(axes).ravel()
    for j, det in enumerate(DETS):
        sc = np.asarray(test_scores[det], float)
        th = thr[det][pfa_t]
        ax = axes[j]; ax.imshow(fc); ax.axis('off')
        fa = np.where((sc > th) & (labels == 0))[0]
        hit = np.where((sc > th) & (labels == 1))[0]
        fr, fcl = _rc(fa, W_b); ax.scatter(fcl, fr, s=4, c=FA_MARK, marker='s', linewidths=0)
        hr, hcl = _rc(hit, W_b); ax.scatter(hcl, hr, s=9, c=HIT_MARK, marker='o', linewidths=0)
        ax.set_title(f'{det}  (#det={int((sc > th).sum())})', fontsize=8)
    for j in range(len(DETS), len(axes)):
        axes[j].axis('off')
    axes[0].legend(handles=[
        Line2D([], [], marker='o', ls='', mfc=HIT_MARK, mec=HIT_MARK, label='hit'),
        Line2D([], [], marker='s', ls='', mfc=FA_MARK, mec=FA_MARK, label='false alarm')],
        fontsize=6, loc='upper right')
    fig.suptitle(f'Detected pixels @ Pfa={pfa_t} — {sig_label}', fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, 'detected_pfa.pdf'))

    # (3,4) false-alarm pixels @ Pfa∈{.01,.05,.1} on false color AND on label map
    for bg_img, tag in [(fc, 'falsecolor'), (lm, 'labelmap')]:
        nr, nc = len(DETS), len(PFA_LEVELS)
        fig, axes = plt.subplots(nr, nc, figsize=FIGSIZE, squeeze=False)
        for i, det in enumerate(DETS):
            sc = np.asarray(test_scores[det], float)
            for jj, p in enumerate(PFA_LEVELS):
                ax = axes[i][jj]; ax.imshow(bg_img); ax.axis('off')
                fa = np.where((labels == 0) & (sc > thr[det][p]))[0]
                fr, fcl = _rc(fa, W_b)
                ax.scatter(fcl, fr, s=3, c=FA_MARK, marker='s', linewidths=0)
                if i == 0:
                    ax.set_title(f'Pfa={p:g}', fontsize=9)
                if jj == 0:
                    ax.text(-0.04, 0.5, det, transform=ax.transAxes, rotation=90,
                            va='center', ha='right', fontsize=7)
        fig.suptitle(f'False alarms on {tag} — {sig_label}', fontsize=11)
        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, f'false_alarms_{tag}.pdf'))

    # ROC overlay
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
    for det in DETS:
        fpr, tpr, auc_v = roc_curves[det]
        ax.plot(fpr, tpr, color=DET_COLORS[det], lw=1.6, label=f'{det} (AUC={auc_v:.3f})')
    ax.set_xlabel('False Alarm Rate'); ax.set_ylabel('Detection Rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.set_title(f'ROC — {sig_label}', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    _savefig(fig, os.path.join(out_dir, 'roc.pdf'))

    # (6) per-class Pfa grouped bars (ALL classes incl. class 0)
    classes = sorted({c for d in DETS for c in pfa_per_class[d]})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bw = 0.8 / max(len(DETS), 1)
    xpos = np.arange(len(classes))
    for di, det in enumerate(DETS):
        vals = [pfa_per_class[det].get(c, 0.0) for c in classes]
        ax.bar(xpos + di * bw, vals, bw, label=det, color=DET_COLORS[det])
    ax.axhline(pfa_t, color='k', ls=':', lw=1, label=f'target Pfa={pfa_t}')
    ax.set_xticks(xpos + 0.4 - bw / 2)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Per-class Pfa')
    ax.set_title(f'Per-class Pfa (all classes incl. 0) — {sig_label}', fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _savefig(fig, os.path.join(out_dir, 'pfa_per_class.pdf'))

    return rows


# ---------------------------------------------------------------------------
# Save / load the trained models so plots can be re-extracted WITHOUT retraining.
# The whitening front-end is a registered submodule of every net, so it is
# already inside each state_dict.
# ---------------------------------------------------------------------------

# Keys safe to override when re-extracting from saved models (they don't change
# the trained nets or the train/test boxes — only scoring / planting / plots).
RELOAD_OVERRIDE_KEYS = [
    'cfar_bg_window', 'cfar_guard', 'cfar_lam', 'pfa_target', 'amplitude', 'target_fraction',
    'edge_guard', 'ridge_n_mc', 'gmm_steps', 'gmm_K', 'local_scm_loading', 'baseline_eig_floor',
]


def _save_models(run_dir, models, cfg, sidx, train_box, test_box):
    torch.save({
        'cfg': cfg, 'sidx': sidx, 'train_box': train_box, 'test_box': test_box,
        'dsm':   models['dsm'].state_dict(),
        'nmlp':  models['nmlp'].state_dict(),
        'ridge': models['ridge'].state_dict(),
    }, os.path.join(run_dir, 'models.pt'))
    print(f"  saved models -> {os.path.join(run_dir, 'models.pt')}", flush=True)


def _build_models_from_ckpt(ckpt, D, cfg, device):
    """Reconstruct the 3 nets and load their weights (incl. frozen whitening)."""
    dsm = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'],
                   whitening=_placeholder_whitening(D))
    dsm.load_state_dict(ckpt['dsm']); dsm.to(device).eval()

    nm = NeighborMLPDenoiser(
        D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
        enc_hidden=cfg.get('nmlp_enc_hidden'),
        score_hidden=cfg.get('nmlp_score_hidden'),
        hidden=cfg.get('nmlp_hidden', 128),
        n_layers=cfg.get('nmlp_n_layers', 3),
        sigma=_whitened_sigma(cfg), activation=cfg['activation'],
        whitening=_placeholder_whitening(D))
    nm.load_state_dict(ckpt['nmlp']); nm.to(device).eval()

    rg = NeighborAdaptedScore(
        D=D, M=int(cfg.get('ridge_M', 256)),
        hidden=tuple(cfg.get('ridge_hidden', [128, 128])),
        k=int(cfg['k']), lam_init=float(cfg.get('ridge_lam_init', 0.1)))
    rg._whitening = _placeholder_whitening(D)
    rg.load_state_dict(ckpt['ridge']); rg.to(device).eval()

    return {'dsm': dsm, 'nmlp': nm, 'ridge': rg}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(_EXP, 'colab.yaml'))
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--scenario', type=int, default=None,
                    help='Override scenario_index from config')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--from-models', default=None,
                    help='Path to a models.pt: load trained nets and re-extract '
                         'all plots WITHOUT retraining.')
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if os.path.exists(args.config):
        with open(args.config) as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if k in DEFAULT_CFG})
    if args.dry_run:
        cfg.update(DRYRUN_OVERRIDES)
    if args.results_dir:
        cfg['results_dir'] = args.results_dir
    if args.scenario is not None:
        cfg['scenario_index'] = args.scenario

    # ---- Reload mode: use the SAVED training cfg (same scenario/boxes/nets),
    #      but let a whitelist of scoring/plot keys be overridden from --config. ----
    ckpt = None
    if args.from_models:
        ckpt = torch.load(args.from_models, map_location='cpu')
        base = dict(ckpt['cfg'])
        for kk in RELOAD_OVERRIDE_KEYS:
            if kk in cfg:
                base[kk] = cfg[kk]
        base['results_dir'] = cfg['results_dir']
        cfg = base
        print(f"Reload mode: models from {args.from_models}", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'compare_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)

    # ---- Data + scenario ----
    print("Loading Pavia-U ...", flush=True)
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D = data_norm.shape
    print(f"Image {H}×{W}×{D}", flush=True)

    manual_path = cfg.get('manual_boxes_path')
    manual = json.load(open(manual_path)) if (manual_path and os.path.exists(manual_path)) else []
    random_sc = generate_random_boxes(
        gt, n=4, min_pixels=int(cfg['min_pixels']),
        seeds=tuple(cfg['random_scenario_seeds']))
    scenarios = manual + random_sc
    sidx = int(cfg['scenario_index']) % len(scenarios)
    scenario = scenarios[sidx]
    train_box, test_box = scenario['train_box'], scenario['test_box']
    print(f"Scenario {sidx}: train_box={train_box}  test_box={test_box}", flush=True)

    k = int(cfg['k'])

    # ---- Crop train / test (raw bands + k×k neighbors). NO subsampling. ----
    tr_box_eff = _side_crop_box(train_box, cfg.get('n_budget'))
    tr_raw, tr_nbr = _crop_pca_box(data_norm, tr_box_eff, k)
    tr_raw = tr_raw.astype(np.float32)
    tr_nbr = tr_nbr.astype(np.float32)
    print(f"train={len(tr_raw)} px  (box {tr_box_eff}, full={cfg.get('n_budget') is None})",
          flush=True)

    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    te_raw, te_nbr = _crop_pca_box(data_norm, test_box, k)
    te_raw = te_raw.astype(np.float32)
    te_nbr = te_nbr.astype(np.float32)
    te_gt = gt[r0:r1, c0:c1].ravel()
    print(f"test={len(te_raw)} px  ({H_b}×{W_b})", flush=True)

    # ---- In-patch signature (dominant class of the test patch) ----
    sig_in, dom_cls, dom_name = compute_signature(
        gt[r0:r1, c0:c1], data_norm[r0:r1, c0:c1],
        w_dom=float(cfg['sig_dom_weight']), w_mean=float(cfg['sig_mean_weight']))
    sig_in = sig_in.astype(np.float32)
    print(f"in-patch signature: dominant={dom_name}  ||s||={np.linalg.norm(sig_in):.4g}",
          flush=True)

    # ---- Foreign signature: a class NOT in the patch, scaled to mean patch-pixel norm ----
    fcls = _pick_foreign_class(gt, np.unique(te_gt))
    sig_for = None
    if fcls is not None:
        mu_for = data_norm.reshape(-1, D)[gt.ravel() == fcls].mean(axis=0)
        scalar = float(np.linalg.norm(te_raw, axis=1).mean())   # mean ||pixel|| over patch
        sig_for = (mu_for / (np.linalg.norm(mu_for) + 1e-12) * scalar).astype(np.float32)
        print(f"foreign signature: class={CLS_NAMES[fcls]}  scaled ||s||={scalar:.4g}",
              flush=True)
    else:
        print("No labeled class is absent from the patch — skipping foreign run.", flush=True)

    # ---- Models: LOAD (reload mode) or TRAIN once (signature-independent) ----
    if ckpt is not None:
        print("Loading trained nets (no training) ...", flush=True)
        models = _build_models_from_ckpt(ckpt, D, cfg, device)
    else:
        print("Training deep nets (best-epoch checkpointing) ...", flush=True)
        def _fl(m):
            return getattr(m, '_final_loss', float('nan'))
        t0 = time.time()
        dsm_net = _train_dsm_best(tr_raw, cfg, device)
        print(f"  DSM done ({time.time()-t0:.0f}s)  best loss={_fl(dsm_net):.4f}", flush=True)
        t0 = time.time()
        nmlp = _train_nmlp_best(tr_raw, tr_nbr, cfg, device)
        print(f"  NeighborMLP done ({time.time()-t0:.0f}s)  best loss={_fl(nmlp):.4f}", flush=True)
        t0 = time.time()
        ridge = _train_ridge(tr_raw, tr_nbr, cfg, device, seed)
        print(f"  LinearDSM done ({time.time()-t0:.0f}s)  best loss={_fl(ridge):.4f}", flush=True)
        models = {'dsm': dsm_net, 'nmlp': nmlp, 'ridge': ridge}
        _save_models(run_dir, models, cfg, sidx, tr_box_eff, test_box)

    ctx = dict(cfg=cfg, device=device, seed=seed, models=models,
               tr_raw=tr_raw, tr_nbr=tr_nbr, te_raw=te_raw, te_nbr=te_nbr,
               te_gt=te_gt, data_norm=data_norm, gt=gt,
               test_box=test_box, train_box=tr_box_eff, sidx=sidx)

    # ---- Run detection twice ----
    run_detection(sig_in, f'inpatch-{dom_name}', run_dir, ctx)
    if sig_for is not None:
        run_detection(sig_for, f'foreign-{CLS_NAMES[fcls]}',
                      os.path.join(run_dir, 'foreign'), ctx)

    print(f"\nDone.  Results: {run_dir}", flush=True)

    if args.dry_run:
        expect = ['summary_table.csv', 'metrics.json', 'scores.npz',
                  'false_color.pdf', 'label_map_targets.pdf', 'signatures.pdf',
                  'detection_maps.pdf', 'detected_pfa.pdf', 'roc.pdf',
                  'pfa_per_class.pdf', 'false_alarms_falsecolor.pdf',
                  'false_alarms_labelmap.pdf']
        ok = all(os.path.exists(os.path.join(run_dir, e)) for e in expect)
        if sig_for is not None:
            ok = ok and os.path.exists(os.path.join(run_dir, 'foreign', 'summary_table.csv'))
        print("DRY-RUN:", "ALL OUTPUTS PRESENT ✓" if ok else "MISSING OUTPUTS ✗")
        sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# Programmatic API (for Colab notebooks — no argparse needed)
# ---------------------------------------------------------------------------

def run_from_cfg(overrides: dict, dry_run: bool = False):
    """Run one full scenario (train → detect → save) from a plain dict.

    Merges ``overrides`` on top of ``DEFAULT_CFG``.
    Returns the output ``run_dir``.
    """
    cfg = dict(DEFAULT_CFG)
    cfg.update(overrides)
    if dry_run:
        cfg.update(DRYRUN_OVERRIDES)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'compare_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)

    print("Loading data ...", flush=True)
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D = data_norm.shape
    print(f"Image {H}×{W}×{D}", flush=True)

    manual_path = cfg.get('manual_boxes_path')
    manual = json.load(open(manual_path)) if (manual_path and os.path.exists(manual_path)) else []
    random_sc = generate_random_boxes(
        gt, n=4, min_pixels=int(cfg['min_pixels']),
        seeds=tuple(cfg['random_scenario_seeds']))
    scenarios = manual + random_sc
    sidx = int(cfg['scenario_index']) % len(scenarios)
    scenario = scenarios[sidx]
    train_box, test_box = scenario['train_box'], scenario['test_box']
    print(f"Scenario {sidx}: train_box={train_box}  test_box={test_box}", flush=True)

    k = int(cfg['k'])
    tr_box_eff = _side_crop_box(train_box, cfg.get('n_budget'))
    tr_raw, tr_nbr = _crop_pca_box(data_norm, tr_box_eff, k)
    tr_raw = tr_raw.astype(np.float32)
    tr_nbr = tr_nbr.astype(np.float32)
    print(f"train={len(tr_raw)} px  (box {tr_box_eff})", flush=True)

    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    te_raw, te_nbr = _crop_pca_box(data_norm, test_box, k)
    te_raw = te_raw.astype(np.float32)
    te_nbr = te_nbr.astype(np.float32)
    te_gt = gt[r0:r1, c0:c1].ravel()
    print(f"test={len(te_raw)} px  ({H_b}×{W_b})", flush=True)

    sig_in, dom_cls, dom_name = compute_signature(
        gt[r0:r1, c0:c1], data_norm[r0:r1, c0:c1],
        w_dom=float(cfg['sig_dom_weight']), w_mean=float(cfg['sig_mean_weight']))
    sig_in = sig_in.astype(np.float32)
    print(f"in-patch signature: dominant={dom_name}  ||s||={np.linalg.norm(sig_in):.4g}",
          flush=True)

    fcls = _pick_foreign_class(gt, np.unique(te_gt))
    sig_for = None
    if fcls is not None:
        mu_for = data_norm.reshape(-1, D)[gt.ravel() == fcls].mean(axis=0)
        scalar = float(np.linalg.norm(te_raw, axis=1).mean())
        sig_for = (mu_for / (np.linalg.norm(mu_for) + 1e-12) * scalar).astype(np.float32)
        print(f"foreign signature: class={CLS_NAMES[fcls]}  scaled ||s||={scalar:.4g}", flush=True)

    print("Training deep nets (best-epoch checkpointing) ...", flush=True)
    def _fl(m): return getattr(m, '_final_loss', float('nan'))
    t0 = time.time()
    dsm_net = _train_dsm_best(tr_raw, cfg, device)
    print(f"  DSM done ({time.time()-t0:.0f}s)  best loss={_fl(dsm_net):.4f}", flush=True)
    t0 = time.time()
    nmlp = _train_nmlp_best(tr_raw, tr_nbr, cfg, device)
    print(f"  NeighborMLP done ({time.time()-t0:.0f}s)  best loss={_fl(nmlp):.4f}", flush=True)
    t0 = time.time()
    ridge = _train_ridge(tr_raw, tr_nbr, cfg, device, seed)
    print(f"  LinearDSM done ({time.time()-t0:.0f}s)  best loss={_fl(ridge):.4f}", flush=True)
    models = {'dsm': dsm_net, 'nmlp': nmlp, 'ridge': ridge}
    _save_models(run_dir, models, cfg, sidx, tr_box_eff, test_box)

    ctx = dict(cfg=cfg, device=device, seed=seed, models=models,
               tr_raw=tr_raw, tr_nbr=tr_nbr, te_raw=te_raw, te_nbr=te_nbr,
               te_gt=te_gt, data_norm=data_norm, gt=gt,
               test_box=test_box, train_box=tr_box_eff, sidx=sidx)

    run_detection(sig_in, f'inpatch-{dom_name}', run_dir, ctx)
    if sig_for is not None:
        run_detection(sig_for, f'foreign-{CLS_NAMES[fcls]}',
                      os.path.join(run_dir, 'foreign'), ctx)

    print(f"\nDone.  Results: {run_dir}", flush=True)
    return run_dir


def show_plots_from_dir(run_dir: str, sub: str = None, inline: bool = True):
    """Load and display all saved PNG figures from a result directory.

    Parameters
    ----------
    run_dir : path to the timestamped results folder (e.g.
              '/content/drive/MyDrive/spatial_results/compare_20260611_123456')
    sub     : optional sub-folder name ('foreign') to show the foreign-signature run
    inline  : if True, display using IPython (Colab); if False, just print paths.
    """
    target_dir = os.path.join(run_dir, sub) if sub else run_dir
    FIGS = [
        'false_color', 'label_map_targets', 'signatures',
        'detection_maps', 'detected_pfa', 'roc', 'pfa_per_class',
        'false_alarms_falsecolor', 'false_alarms_labelmap',
    ]
    found = []
    for name in FIGS:
        p = os.path.join(target_dir, f'{name}.png')
        if os.path.exists(p):
            found.append(p)
        else:
            print(f"  [missing] {name}.png")

    if inline:
        try:
            from IPython.display import display, Image as IPImage
            for p in found:
                print(f"── {os.path.basename(p)} ──")
                display(IPImage(p, width=900))
        except ImportError:
            print("IPython not available; printing paths only:")
            for p in found:
                print(p)
    else:
        for p in found:
            print(p)

    # Summary table
    csv_path = os.path.join(target_dir, 'summary_table.md')
    if os.path.exists(csv_path):
        print("\n=== Summary Table ===")
        print(open(csv_path).read())


if __name__ == '__main__':
    main()
