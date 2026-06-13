"""
run_compare.py — Focused spatial detector comparison (single scenario).

Detectors
---------
  DSM               — global per-pixel score (ScoreNet, no spatial context)
  NeighborMLP       — spatial denoiser score net                 (Ours, spatial)
  NeighborMLP-CFAR  — NeighborMLP with local kNN-Fisher normalization
  AMF               — Adaptive Matched Filter (global SCM)
  AMF-local         — AMF on the per-pixel k×k window SCM
  GMM-Levin         — Levin product-GMM GLRT

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
from tqdm import tqdm

# Shared helpers from the main runner.
from run_colab import (
    _crop_pca_box,
    _make_whitening, _whitened_sigma, _placeholder_whitening,
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
    'DSM-CFAR',
    'NeighborMLP',
    'NeighborMLP-CFAR',
    'AMF',
    'AMF-local',
    'GMM-Levin',
]
DET_COLORS = {
    'DSM':              '#ff7f0e',   # orange
    'DSM-CFAR':         '#d94801',   # dark orange
    'NeighborMLP':      '#2ca02c',   # green
    'NeighborMLP-CFAR': '#006d2c',   # dark green
    'AMF':              '#9467bd',   # purple
    'AMF-local':        '#c5b0d5',   # light purple
    'GMM-Levin':        '#e377c2',   # pink
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
    target_class=None,    # None = auto dominant labeled class; 1..9 = manual
    foreign_class=None,   # None = auto class absent from patch; 1..9 = manual
    active_detectors=None,  # None = all of DET_ORDER; else a subset to run/show
    run_inpatch=True,       # run the in-patch (dominant/target_class) signature
    run_foreign=True,       # run the foreign (absent/foreign_class) signature
    amplitude=0.15, target_fraction=0.10, edge_guard=5,
    n_budget=None,               # None = full train box (no subsampling); int = side-crop
    k=5,
    local_scm_loading=1e-8,
    baseline_eig_floor=1e-12,
    # AMF-local window: None → use the shared neighborhood k; int → AMF-local
    # re-extracts its OWN (amf_local_window×amf_local_window) window, independent
    # of the NeighborMLP neighborhood. Adjustable straight from the notebook.
    amf_local_window=None,
    # NeighborMLP — encoder: D→enc_hidden→d_lat ; denoiser: (D+(K+1)*d_lat)→score_hidden→D
    nmlp_d_lat=16, nmlp_K=8, nmlp_enc_hidden=[128, 64], nmlp_score_hidden=[128],
    nmlp_epochs=1000, nmlp_lr=3e-4, nmlp_batch=256,
    # NeighborMLP-CFAR local normalization
    cfar_bg_window=11, cfar_guard=3, cfar_lam=0.0,
    # local mean/Fisher set: False = ALL window neighbors (default),
    # True = restrict to the model's top-K latent-selected neighbors.
    cfar_fisher_use_topk=False,
    # SDSM-CFAR Fisher window (None → use model's k) + guard block side length.
    sdsm_cfar_window=None, sdsm_cfar_guard=1,
    # DSM-CFAR: global DSM score map standardized by a local window mean/std
    # (local mean reduce + local Fisher/variance normalization). Own window.
    dsm_cfar_window=11, dsm_cfar_guard=3,
    # DSM
    dsm_hidden=[64, 64], dsm_epochs=1000, dsm_lr=5e-4,
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
    nmlp_epochs=8, dsm_epochs=20,
    nmlp_K=4,
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


def _class_signature(cls_id, data_norm, gt, te_raw, w_dom, w_mean):
    """In-patch-style signature for an EXPLICIT class id.

    Mean spectrum of `cls_id` over the WHOLE image (cleaner than box-only),
    blended with the test-patch mean exactly as compute_signature does:
        s = w_dom * mu_class + w_mean * mu_patch
    """
    D = data_norm.shape[-1]
    mu_cls = data_norm.reshape(-1, D)[gt.ravel() == int(cls_id)].mean(axis=0)
    mu_patch = te_raw.reshape(-1, D).mean(axis=0)
    return (w_dom * mu_cls + w_mean * mu_patch).astype(np.float32)


def _resolve_inpatch_signature(cfg, gt, data_norm, te_raw, box):
    """Return (sig_in, dom_cls, dom_name). Honors cfg['target_class'] (1..9) if
    set, otherwise auto-picks the dominant labeled class of the box."""
    r0, r1, c0, c1 = box
    tcls = cfg.get('target_class')
    w_dom, w_mean = float(cfg['sig_dom_weight']), float(cfg['sig_mean_weight'])
    if tcls is not None:
        tcls = int(tcls)
        sig = _class_signature(tcls, data_norm, gt, te_raw, w_dom, w_mean)
        return sig, tcls, CLS_NAMES.get(tcls, f'cls{tcls}')
    sig, dom_cls, dom_name = compute_signature(
        gt[r0:r1, c0:c1], data_norm[r0:r1, c0:c1], w_dom=w_dom, w_mean=w_mean)
    return sig.astype(np.float32), dom_cls, dom_name


def _resolve_foreign_class(cfg, gt, te_gt):
    """Return the foreign class id. Honors cfg['foreign_class'] if set,
    otherwise auto-picks a labeled class absent from the patch."""
    fcls = cfg.get('foreign_class')
    if fcls is not None:
        return int(fcls)
    return _pick_foreign_class(gt, np.unique(te_gt))


def _cfar_normalize_map(flat_scores, shape, bg, guard, eps=1e-6, cfar_lam=0.0):
    """Local-CFAR normalization of a projected-score MAP.

    Always subtracts the local annulus mean (pure CFAR mean reduction).
    cfar_lam regularizes ONLY the Fisher (std) toward the global std:

        mean_eff = mean_annulus(q)                             # always local
        std_eff  = (1 - lam) * std_annulus(q) + lam * std_global(q)
        T_i      = (q_i - mean_eff) / (std_eff + eps)

    cfar_lam=0  → pure local annulus std (standard CFAR; default)
    cfar_lam=1  → local mean, global std
    cfar_lam∈(0,1) → local mean, blended std (robust near boundaries)

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

    # Global variance (mean is always local; shrink variance, not std)
    var_global = float(q.var()) + eps

    lam = float(cfar_lam)
    # Matches paper: sqrt(s'*[(1-lam)*Psi_i + lam*Psi]*s)
    #              = sqrt((1-lam)*var_local + lam*var_global)
    var_eff = (1.0 - lam) * (std_local ** 2) + lam * var_global
    std_eff = np.sqrt(np.maximum(var_eff, 0.0)) + eps
    return ((q - mean_local) / std_eff).reshape(-1).astype(np.float32)


def _knn_fisher_normalize(score_flat, model, pix, nbr, shape, k, eps=1e-6, cfar_lam=0.0,
                          use_topk=False, win=None, guard=1):
    """Local-Fisher CFAR for NeighborMLP.

    The local mean/Fisher are estimated over a neighbor set A_i, selected one of
    two ways (use_topk):
      use_topk=False (default) → ALL neighbors in a `win`×`win` window, minus a
                                 `guard`×`guard` center block. Window/guard are
                                 INDEPENDENT of the score model's k.
      use_topk=True            → the model's OWN top-K latent-selected neighbors
                                 (the same set the score forward pass uses); this
                                 mode is tied to the model's k×k window.

    Always subtracts the local mean. cfar_lam regularizes ONLY the Fisher
    (variance) toward the global variance:

        mean_eff = mean_{A_i}(q)                                # always local
        var_eff  = (1-lam) * var_{A_i}(q) + lam * var_global(q)
        T_i = (q_i - mean_eff) / (sqrt(var_eff) + eps)

    cfar_lam=0  → pure local Fisher (default)
    cfar_lam=1  → local mean, global Fisher

    Parameters
    ----------
    win   : window side length for the all-neighbors set. None → use k.
            (Ignored when use_topk=True, which must use the model's k.)
    guard : side length of the excluded center block (guard=1 → only the center
            pixel; guard=3 → 3×3 center block, classical CFAR guard ring).

    q at neighbor positions is read straight off the score map via unfold
    (no extra forward passes). For use_topk the k×k window order matches
    extract_neighborhoods so model selection indices line up correctly.
    """
    H, W = shape
    dev = next(model.parameters()).device
    q_i = torch.tensor(np.asarray(score_flat, np.float32), device=dev)        # (HW,)
    # window size: top-K is tied to the model's k; all-neighbors uses `win`.
    wsize = int(k) if use_topk else int(win or k)
    p = wsize // 2
    qmap = q_i.reshape(1, 1, H, W)
    patches = F.unfold(F.pad(qmap, (p, p, p, p), mode='circular'), kernel_size=wsize)
    patches = patches.reshape(wsize * wsize, H * W).t()                       # (HW, wsize^2)
    cc = wsize // 2
    if use_topk:
        # restrict A_i to the model's top-K latent-selected neighbors (center excluded)
        keep = [m for m in range(wsize * wsize) if m != (wsize * wsize) // 2]
        q_neigh = patches[:, keep]                                           # (HW, M)
        idx = model.topk_indices(
            torch.tensor(np.asarray(pix, np.float32), device=dev),
            torch.tensor(np.asarray(nbr, np.float32), device=dev))           # (HW, K)
        q_set = torch.gather(q_neigh, 1, idx)                                 # (HW, K)
    else:
        # all neighbors in the win×win window minus a guard×guard center block
        gr = max(int(guard), 1) // 2                                          # guard radius
        keep = [r * wsize + c for r in range(wsize) for c in range(wsize)
                if not (abs(r - cc) <= gr and abs(c - cc) <= gr)]
        q_set = patches[:, keep]                                             # (HW, M')
    mu_local  = q_set.mean(dim=1)
    std_local = q_set.var(dim=1, unbiased=False).sqrt()

    lam = float(cfar_lam)
    # Matches paper: sqrt(s'*[(1-lam)*Psi_i + lam*Psi]*s)
    #              = sqrt((1-lam)*var_local + lam*var_global)
    var_global = (q_i.var() + eps)
    var_eff = (1.0 - lam) * (std_local ** 2) + lam * var_global
    std_eff = torch.sqrt(torch.clamp(var_eff, min=0.0)) + eps
    return ((q_i - mu_local) / std_eff).cpu().numpy().astype(np.float32)


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
    if best_state is not None:
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
    if best_state is not None:
        nmlp.load_state_dict(best_state)
    nmlp._final_loss = best_loss
    nmlp.eval()
    return nmlp


def _savefig(fig, path):
    # NO bbox_inches='tight' — that crops each figure to its content and breaks
    # the uniform size. Save the full FIGSIZE canvas at a fixed dpi so every
    # output file has identical dimensions (FIGSIZE inches; FIGSIZE×150 px PNG).
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"  [fig] {os.path.relpath(path)}", flush=True)


def _active_set(cfg):
    """The detectors to compute/show. cfg['active_detectors']=None → all of
    DET_ORDER; otherwise the given subset. 'NeighborMLP-CFAR' implies its base
    'NeighborMLP' is scored too (the CFAR variant is derived from it)."""
    act = cfg.get('active_detectors')
    if act is None:
        return set(DET_ORDER)
    s = set(act)
    if 'NeighborMLP-CFAR' in s:
        s.add('NeighborMLP')
    if 'DSM-CFAR' in s:        # CFAR variant is derived from the global DSM map
        s.add('DSM')
    return s


# ---------------------------------------------------------------------------
def score_all(pix, nbr, models, tr_raw, tr_nbr, sig, cfg, device, nbr_amf=None):
    """Score the REQUESTED detectors on (pix, nbr).  Returns {det_name: scores}.

    A detector is skipped if it is not in cfg['active_detectors'] (None = all),
    or if its deep model is absent from `models`.

    nbr_amf : optional separate neighbor tensor for AMF-local (its own window).
              None → fall back to the shared `nbr`.
    """
    pix = pix.astype(np.float32)
    nbr = nbr.astype(np.float32)
    act = _active_set(cfg)
    out = {}
    floor = float(cfg.get('baseline_eig_floor', 1e-12))
    if 'DSM' in act and models.get('dsm') is not None:
        out['DSM'] = dsm_additive(pix, tr_raw, models['dsm'], sig)
    if 'NeighborMLP' in act and models.get('nmlp') is not None:
        out['NeighborMLP'] = score_nmlp_additive(models['nmlp'], pix, nbr, tr_raw, tr_nbr, sig)
    if 'AMF' in act:
        out['AMF'] = amf_global(pix, tr_raw, sig, eig_floor=floor)
    if 'AMF-local' in act:
        nbr_for_amf = (nbr_amf.astype(np.float32) if nbr_amf is not None else nbr)
        amf_loc, _ = amf_cem_local_scm(
            pix, nbr_for_amf, sig, device=device,
            loading=float(cfg.get('local_scm_loading', 1e-8)))
        out['AMF-local'] = amf_loc
    if 'GMM-Levin' in act:
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
    test_scores = score_all(planted, te_nbr, models, tr_raw, tr_nbr, sig, cfg, device,
                            nbr_amf=ctx.get('te_nbr_amf'))
    print("Scoring detectors (train, for CFAR thresholds) ...", flush=True)
    train_scores = score_all(tr_raw, tr_nbr, models, tr_raw, tr_nbr, sig, cfg, device,
                             nbr_amf=ctx.get('tr_nbr_amf'))

    # ---- NeighborMLP-CFAR: local kNN-Fisher normalization ----
    k_win = int(cfg['k'])
    lam   = float(cfg.get('cfar_lam', 0.0))
    # default: estimate the local mean/Fisher from ALL window neighbors;
    # set cfar_fisher_use_topk=True to restrict to the model's top-K selection.
    use_topk = bool(cfg.get('cfar_fisher_use_topk', False))
    # SDSM-CFAR Fisher window/guard, INDEPENDENT of the model's k.
    #   sdsm_cfar_window: window side length (None → use k)
    #   sdsm_cfar_guard : excluded center block side (1 → only center pixel)
    nmlp_win   = cfg.get('sdsm_cfar_window') or None
    nmlp_guard = int(cfg.get('sdsm_cfar_guard', 1))
    tr0, tr1, tc0, tc1 = ctx['train_box']
    tr_shape = (tr1 - tr0, tc1 - tc0)
    if ('NeighborMLP-CFAR' in _active_set(cfg)
            and 'NeighborMLP' in test_scores and models.get('nmlp') is not None):
        test_scores['NeighborMLP-CFAR'] = _knn_fisher_normalize(
            test_scores['NeighborMLP'], models['nmlp'], planted, te_nbr,
            (H_b, W_b), k_win, cfar_lam=lam, use_topk=use_topk,
            win=nmlp_win, guard=nmlp_guard)
        train_scores['NeighborMLP-CFAR'] = _knn_fisher_normalize(
            train_scores['NeighborMLP'], models['nmlp'], tr_raw, tr_nbr,
            tr_shape, k_win, cfar_lam=lam, use_topk=use_topk,
            win=nmlp_win, guard=nmlp_guard)

    # ---- DSM-CFAR: global DSM score map with local mean-reduce + local-Fisher
    # (variance) normalization over an (dsm_cfar_window) k×k window. ----
    if ('DSM-CFAR' in _active_set(cfg) and 'DSM' in test_scores):
        dsm_bg    = int(cfg.get('dsm_cfar_window', cfg.get('cfar_bg_window', 11)))
        dsm_guard = int(cfg.get('dsm_cfar_guard', cfg.get('cfar_guard', 3)))
        test_scores['DSM-CFAR'] = _cfar_normalize_map(
            test_scores['DSM'], (H_b, W_b), dsm_bg, dsm_guard, cfar_lam=lam)
        train_scores['DSM-CFAR'] = _cfar_normalize_map(
            train_scores['DSM'], tr_shape, dsm_bg, dsm_guard, cfar_lam=lam)

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
        # Pd at the *train* CFAR threshold (honest, deployable operating point):
        # apply the threshold set on TRAIN scores to the test targets directly,
        # unlike 'Pd@Pfa=0.05' which re-fits the operating point on the test ROC.
        tgt_mask = (labels == 1)
        pd_cfar = (float(np.mean(sc[tgt_mask] > thr[det][pfa_t]))
                   if tgt_mask.any() else float('nan'))
        rows.append({
            'Detector': det,
            'pAUC@0.05': partial_auc(labels, sc, fpr_max=0.05),
            'AUC': auc_v,
            'Pd@Pfa=0.05': dr_at_fpr(labels, sc, fpr_list=(pfa_t,))[str(pfa_t)],
            'Pd_cfar': pd_cfar,
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
    cols = (['Detector', 'pAUC@0.05', 'AUC', 'Pd@Pfa=0.05', 'Pd_cfar', 'Pfa_avg', 'Pfa_max']
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
    'cfar_bg_window', 'cfar_guard', 'cfar_lam', 'cfar_fisher_use_topk',
    'sdsm_cfar_window', 'sdsm_cfar_guard',
    'dsm_cfar_window', 'dsm_cfar_guard', 'amf_local_window',
    'pfa_target', 'amplitude', 'target_fraction', 'edge_guard',
    'target_class', 'foreign_class', 'active_detectors', 'run_inpatch', 'run_foreign',
    'gmm_steps', 'gmm_K', 'local_scm_loading', 'baseline_eig_floor',
]


def _save_models(run_dir, models, cfg, sidx, train_box, test_box):
    blob = {'cfg': cfg, 'sidx': sidx, 'train_box': train_box, 'test_box': test_box}
    if models.get('dsm') is not None:
        blob['dsm'] = models['dsm'].state_dict()
    if models.get('nmlp') is not None:
        blob['nmlp'] = models['nmlp'].state_dict()
    torch.save(blob, os.path.join(run_dir, 'models.pt'))
    print(f"  saved models -> {os.path.join(run_dir, 'models.pt')}", flush=True)


def _build_models_from_ckpt(ckpt, D, cfg, device):
    """Reconstruct whichever of DSM / NeighborMLP are present in the checkpoint."""
    models = {}
    if 'dsm' in ckpt:
        dsm = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'],
                       whitening=_placeholder_whitening(D))
        dsm.load_state_dict(ckpt['dsm']); dsm.to(device).eval()
        models['dsm'] = dsm
    if 'nmlp' in ckpt:
        nm = NeighborMLPDenoiser(
            D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
            enc_hidden=cfg.get('nmlp_enc_hidden'),
            score_hidden=cfg.get('nmlp_score_hidden'),
            hidden=cfg.get('nmlp_hidden', 128),
            n_layers=cfg.get('nmlp_n_layers', 3),
            sigma=_whitened_sigma(cfg), activation=cfg['activation'],
            whitening=_placeholder_whitening(D))
        nm.load_state_dict(ckpt['nmlp']); nm.to(device).eval()
        models['nmlp'] = nm
    return models


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

    # ---- AMF-local own window (independent of the NeighborMLP neighborhood k) ----
    amf_k = int(cfg.get('amf_local_window') or k)
    if amf_k != k:
        _, tr_nbr_amf = _crop_pca_box(data_norm, tr_box_eff, amf_k)
        _, te_nbr_amf = _crop_pca_box(data_norm, test_box, amf_k)
        tr_nbr_amf = tr_nbr_amf.astype(np.float32)
        te_nbr_amf = te_nbr_amf.astype(np.float32)
        print(f"AMF-local window={amf_k}×{amf_k} (independent of k={k})", flush=True)
    else:
        tr_nbr_amf, te_nbr_amf = tr_nbr, te_nbr

    # ---- In-patch signature (dominant class of the test patch) ----
    sig_in, dom_cls, dom_name = _resolve_inpatch_signature(
        cfg, gt, data_norm, te_raw, test_box)
    sig_in = sig_in.astype(np.float32)
    _tlbl = 'manual' if cfg.get('target_class') is not None else 'dominant'
    print(f"in-patch signature: {_tlbl}={dom_name}  ||s||={np.linalg.norm(sig_in):.4g}",
          flush=True)

    # ---- Foreign signature: a class NOT in the patch, scaled to mean patch-pixel norm ----
    fcls = _resolve_foreign_class(cfg, gt, te_gt)
    sig_for = None
    if fcls is not None:
        mu_for = data_norm.reshape(-1, D)[gt.ravel() == fcls].mean(axis=0)
        scalar = float(np.linalg.norm(te_raw, axis=1).mean())   # mean ||pixel|| over patch
        sig_for = (mu_for / (np.linalg.norm(mu_for) + 1e-12) * scalar).astype(np.float32)
        print(f"foreign signature: class={CLS_NAMES[fcls]}  scaled ||s||={scalar:.4g}",
              flush=True)
    else:
        print("No labeled class is absent from the patch — skipping foreign run.", flush=True)

    # ---- Models: LOAD (reload mode) or TRAIN only what's requested ----
    act = _active_set(cfg)
    if ckpt is not None:
        print("Loading trained nets (no training) ...", flush=True)
        models = _build_models_from_ckpt(ckpt, D, cfg, device)
    else:
        def _fl(m):
            return getattr(m, '_final_loss', float('nan'))
        models = {}
        if 'DSM' in act:
            print("Training DSM ...", flush=True)
            t0 = time.time(); models['dsm'] = _train_dsm_best(tr_raw, cfg, device)
            print(f"  DSM done ({time.time()-t0:.0f}s)  best loss={_fl(models['dsm']):.4f}", flush=True)
        if {'NeighborMLP', 'NeighborMLP-CFAR'} & act:
            print("Training NeighborMLP ...", flush=True)
            t0 = time.time(); models['nmlp'] = _train_nmlp_best(tr_raw, tr_nbr, cfg, device)
            print(f"  NeighborMLP done ({time.time()-t0:.0f}s)  best loss={_fl(models['nmlp']):.4f}", flush=True)
        _save_models(run_dir, models, cfg, sidx, tr_box_eff, test_box)

    ctx = dict(cfg=cfg, device=device, seed=seed, models=models,
               tr_raw=tr_raw, tr_nbr=tr_nbr, te_raw=te_raw, te_nbr=te_nbr,
               tr_nbr_amf=tr_nbr_amf, te_nbr_amf=te_nbr_amf,
               te_gt=te_gt, data_norm=data_norm, gt=gt,
               test_box=test_box, train_box=tr_box_eff, sidx=sidx)

    # ---- Run detection (in-patch and/or foreign) ----
    if cfg.get('run_inpatch', True):
        run_detection(sig_in, f'inpatch-{dom_name}', run_dir, ctx)
    if cfg.get('run_foreign', True) and sig_for is not None:
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

    # ---- AMF-local own window (independent of the NeighborMLP neighborhood k) ----
    amf_k = int(cfg.get('amf_local_window') or k)
    if amf_k != k:
        _, tr_nbr_amf = _crop_pca_box(data_norm, tr_box_eff, amf_k)
        _, te_nbr_amf = _crop_pca_box(data_norm, test_box, amf_k)
        tr_nbr_amf = tr_nbr_amf.astype(np.float32)
        te_nbr_amf = te_nbr_amf.astype(np.float32)
        print(f"AMF-local window={amf_k}×{amf_k} (independent of k={k})", flush=True)
    else:
        tr_nbr_amf, te_nbr_amf = tr_nbr, te_nbr

    sig_in, dom_cls, dom_name = _resolve_inpatch_signature(
        cfg, gt, data_norm, te_raw, test_box)
    sig_in = sig_in.astype(np.float32)
    _tlbl = 'manual' if cfg.get('target_class') is not None else 'dominant'
    print(f"in-patch signature: {_tlbl}={dom_name}  ||s||={np.linalg.norm(sig_in):.4g}",
          flush=True)

    fcls = _resolve_foreign_class(cfg, gt, te_gt)
    sig_for = None
    if fcls is not None:
        mu_for = data_norm.reshape(-1, D)[gt.ravel() == fcls].mean(axis=0)
        scalar = float(np.linalg.norm(te_raw, axis=1).mean())
        sig_for = (mu_for / (np.linalg.norm(mu_for) + 1e-12) * scalar).astype(np.float32)
        print(f"foreign signature: class={CLS_NAMES[fcls]}  scaled ||s||={scalar:.4g}", flush=True)

    # Only train the deep nets that the requested detector set actually needs.
    act = _active_set(cfg)
    def _fl(m): return getattr(m, '_final_loss', float('nan'))
    models = {}
    if 'DSM' in act:
        print("Training DSM ...", flush=True)
        t0 = time.time(); models['dsm'] = _train_dsm_best(tr_raw, cfg, device)
        print(f"  DSM done ({time.time()-t0:.0f}s)  best loss={_fl(models['dsm']):.4f}", flush=True)
    if {'NeighborMLP', 'NeighborMLP-CFAR'} & act:
        print("Training NeighborMLP ...", flush=True)
        t0 = time.time(); models['nmlp'] = _train_nmlp_best(tr_raw, tr_nbr, cfg, device)
        print(f"  NeighborMLP done ({time.time()-t0:.0f}s)  best loss={_fl(models['nmlp']):.4f}", flush=True)
    _save_models(run_dir, models, cfg, sidx, tr_box_eff, test_box)

    ctx = dict(cfg=cfg, device=device, seed=seed, models=models,
               tr_raw=tr_raw, tr_nbr=tr_nbr, te_raw=te_raw, te_nbr=te_nbr,
               tr_nbr_amf=tr_nbr_amf, te_nbr_amf=te_nbr_amf,
               te_gt=te_gt, data_norm=data_norm, gt=gt,
               test_box=test_box, train_box=tr_box_eff, sidx=sidx)

    if cfg.get('run_inpatch', True):
        run_detection(sig_in, f'inpatch-{dom_name}', run_dir, ctx)
    if cfg.get('run_foreign', True) and sig_for is not None:
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


# ---------------------------------------------------------------------------
# Multi-seed runner: repeat a scenario over several seeds and aggregate the
# per-detector metrics into a mean±std table + bar figures with error bars.
# Each seed reseeds BOTH target planting and net training, giving honest error
# bars over the two sources of randomness.
# ---------------------------------------------------------------------------

_AGG_METRICS = ['AUC', 'pAUC@0.05', 'Pd@Pfa=0.05', 'Pd_cfar', 'Pfa_avg', 'Pfa_max']


def _collect_sig_dirs(run_dir):
    """Return [(sig_label, dir)] for each signature run present under run_dir."""
    out = []
    if os.path.exists(os.path.join(run_dir, 'metrics.json')):
        out.append(('inpatch', run_dir))
    fdir = os.path.join(run_dir, 'foreign')
    if os.path.exists(os.path.join(fdir, 'metrics.json')):
        out.append(('foreign', fdir))
    return out


def _aggregate_rows(rows_per_seed):
    """rows_per_seed: list (over seeds) of {detector: {metric: value}}.
    Returns {detector: {metric: (mean, std, n)}}, ordered by DET_ORDER."""
    dets = [d for d in DET_ORDER if any(d in r for r in rows_per_seed)]
    agg = {}
    for d in dets:
        agg[d] = {}
        for m in _AGG_METRICS:
            vals = [r[d][m] for r in rows_per_seed if d in r and m in r[d]
                    and r[d][m] == r[d][m]]   # drop NaN
            if vals:
                agg[d][m] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return agg


def _write_agg_table(agg, out_path, title):
    """Write a mean±std markdown + csv table. Returns the markdown string."""
    dets = list(agg.keys())
    lines_md = [f'### {title}  (mean ± std over seeds)', '',
                '| Detector | ' + ' | '.join(_AGG_METRICS) + ' |',
                '|' + '|'.join(['---'] * (len(_AGG_METRICS) + 1)) + '|']
    csv = ['Detector,' + ','.join(f'{m}_mean,{m}_std' for m in _AGG_METRICS)]
    for d in dets:
        cells = []
        csv_cells = [d]
        for m in _AGG_METRICS:
            if m in agg[d]:
                mu, sd, _ = agg[d][m]
                cells.append(f'{mu:.3f} ± {sd:.3f}')
                csv_cells += [f'{mu:.4f}', f'{sd:.4f}']
            else:
                cells.append('—'); csv_cells += ['', '']
        lines_md.append('| ' + d + ' | ' + ' | '.join(cells) + ' |')
        csv.append(','.join(csv_cells))
    md = '\n'.join(lines_md) + '\n'
    open(out_path + '.md', 'w').write(md)
    open(out_path + '.csv', 'w').write('\n'.join(csv) + '\n')
    return md


def _agg_bar_fig(agg, metrics, out_path, title, target_line=None):
    """Grouped bar chart of mean±std for the given metrics, per detector."""
    dets = list(agg.keys())
    x = np.arange(len(dets))
    nb = len(metrics)
    bw = 0.8 / max(nb, 1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for j, m in enumerate(metrics):
        mus = [agg[d].get(m, (np.nan,))[0] for d in dets]
        sds = [agg[d].get(m, (np.nan, np.nan))[1] for d in dets]
        ax.bar(x + j * bw, mus, bw, yerr=sds, capsize=3, label=m,
               color=[DET_COLORS.get(d, '#888') for d in dets] if nb == 1 else None,
               alpha=0.9)
    if target_line is not None:
        ax.axhline(target_line, color='k', ls=':', lw=1, label=f'target={target_line:g}')
    ax.set_xticks(x + 0.4 - bw / 2)
    ax.set_xticklabels(dets, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('value'); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.25)
    _savefig(fig, out_path)


def run_multiseed(overrides: dict, seeds=(42, 43, 44, 45, 46)):
    """Run one scenario over `seeds`, aggregate per-detector metrics, and write
    a mean±std table + bar figures. Returns the parent run directory."""
    base = dict(DEFAULT_CFG); base.update(overrides)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    parent = os.path.join(base['results_dir'], f'multiseed_{ts}')
    os.makedirs(parent, exist_ok=True)
    print(f"\n########## MULTI-SEED  seeds={list(seeds)}  -> {parent}", flush=True)

    seed_dirs = []
    per_sig = {}     # sig_label -> list over seeds of {detector: {metric: val}}
    for si, s in enumerate(seeds):
        print(f"\n===== seed {s}  ({si+1}/{len(seeds)}) =====", flush=True)
        ov = dict(overrides)
        ov['seed'] = int(s)
        ov['results_dir'] = os.path.join(parent, 'seeds')
        rd = run_from_cfg(ov)
        seed_dirs.append(rd)
        for sig_label, d in _collect_sig_dirs(rd):
            m = json.load(open(os.path.join(d, 'metrics.json')))
            row_map = {r['Detector']: r for r in m['rows']}
            per_sig.setdefault(sig_label, []).append(row_map)

    # ---- aggregate + write per signature ----
    pfa_t = float(base.get('pfa_target', 0.05))
    summary = {}
    for sig_label, rows_seeds in per_sig.items():
        agg = _aggregate_rows(rows_seeds)
        summary[sig_label] = agg
        md = _write_agg_table(
            agg, os.path.join(parent, f'summary_{sig_label}'),
            f'{sig_label} — {len(rows_seeds)} seeds')
        print(f"\n=== AGGREGATE [{sig_label}] ===\n{md}", flush=True)
        _agg_bar_fig(agg, ['AUC', 'pAUC@0.05'],
                     os.path.join(parent, f'{sig_label}_auc_bar.pdf'),
                     f'{sig_label}: AUC & pAUC@0.05 (mean±std)')
        _agg_bar_fig(agg, ['Pd@Pfa=0.05', 'Pd_cfar'],
                     os.path.join(parent, f'{sig_label}_pd_bar.pdf'),
                     f'{sig_label}: Pd@Pfa vs Pd at train-CFAR threshold (mean±std)')
        _agg_bar_fig(agg, ['Pfa_avg', 'Pfa_max'],
                     os.path.join(parent, f'{sig_label}_pfa_bar.pdf'),
                     f'{sig_label}: realized Pfa at train-CFAR threshold (mean±std)',
                     target_line=pfa_t)

    json.dump({'seeds': list(seeds), 'seed_dirs': seed_dirs,
               'overrides': {k: v for k, v in overrides.items()}},
              open(os.path.join(parent, 'multiseed_meta.json'), 'w'),
              indent=2, default=str)
    print(f"\nDone.  Multi-seed results: {parent}", flush=True)
    print(f"Representative per-seed figures: {seed_dirs[0]}", flush=True)
    return parent


def show_multiseed(parent_dir: str, rep_seed_figs: bool = True, inline: bool = True):
    """Display the aggregate tables + bar figures, and (optionally) the
    qualitative figures from the first seed as a representative example."""
    import glob
    # aggregate tables + bar charts
    for md in sorted(glob.glob(os.path.join(parent_dir, 'summary_*.md'))):
        print('\n' + open(md).read())
    bars = sorted(glob.glob(os.path.join(parent_dir, '*_bar.png')))
    if inline:
        try:
            from IPython.display import display, Image as IPImage
            for p in bars:
                print(f'── {os.path.basename(p)} ──'); display(IPImage(p, width=850))
        except ImportError:
            for p in bars: print(p)
    else:
        for p in bars: print(p)

    if rep_seed_figs:
        meta_p = os.path.join(parent_dir, 'multiseed_meta.json')
        if os.path.exists(meta_p):
            meta = json.load(open(meta_p))
            rep = meta['seed_dirs'][0]
            print(f"\n=== Representative qualitative figures (seed {meta['seeds'][0]}) ===")
            for sig in _collect_sig_dirs(rep):
                show_plots_from_dir(rep, sub=(None if sig[0] == 'inpatch' else 'foreign'),
                                    inline=inline)


if __name__ == '__main__':
    main()
