"""
============================================================================
IID experiment core — shared pipeline for single-class and multiclass runs.
============================================================================

What this does (in one call to run_iid(cfg, mode)):

  Data (raw 103-D)
  ----------------
    1. Load + normalize.
    2. Extract background / target pools (single = one bkg class multi =
       union of all non-target classes minus exclude_classes).
    3. Target signature  s_raw = mean(tgt_pix)   (NOT unit-normalized).
    4. Shuffle bkg with one seed -> train pool of size max(n_train_list)
       + held-out test pool of size test_size.
    5. Plant targets in raw 103-D (additive + replacement) ONCE the test
       set is shared across every (latent_dim, n_train) combination.

  Reduced spaces (per latent_dim d in cfg['latent_dim_list'])
  -----------------------------------------------------------
    6a. PCA-d fit on the whole image transform train pool, test sets,
        signature (s_pca = pca.transform(s_raw[None]).flatten(), NOT
        re-normalized).  Save pca_d{d}.pkl.
    6b. Linear autoencoder D_RAW -> d -> D_RAW trained on the whole image
        (MSE).  Encode train pool, test sets, signature.  Save ae_d{d}.pt
        + per-epoch loss curve.

  Sweep (for d in latent_dim_list, for n in n_train_list)
  -------------------------------------------------------
    7. Train 4 score models: DSM-PCA, DSM-AE, LRao-PCA, LRao-AE.
       Local training loops per-epoch loss recorded weights saved.
       Score test set with each (DSM has separate additive / replacement
       statistics the LRao Mode-2 statistic is the same for both).

  Classical baselines (raw 103-D, depend only on n)
  -------------------------------------------------
    8. For each n: run AMF, Reg-AMF, CEM, GMM-GLRT, (DLTD, SMGLRT in multi),
       AMF-rep, GMM-GLRT-rep, Exact-GLRT.  Cached so we don't recompute
       inside the d loop.

  Save
  ----
    9. config.yaml, metrics.json (hierarchical AUCs), loss_curves.json
       (per-epoch losses, flat keys), scores.npz (per-pixel scores + labels,
       enough to re-render any figure offline), models/*, figures/*.

Use this module from run_iid_single.py and run_iid_multi.py.
============================================================================
"""

import os
import sys
import json
import time
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, cem, dsm_additive, dsm_replacement, amf_replacement,
    gmm_glrt, gmm_glrt_replacement, dltd, smglrt, exact_glrt_replacement,
)
from final_paper_experiments.models.neighbor_adapted import LinearAutoencoder
from dsm_model import (
    ScoreNet, dsm_loss, lfi_loss_mode2, compute_lfi_detector_scores_mode2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


def _roc(labels: np.ndarray, scores: np.ndarray):
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr, tpr, _auc(labels, scores)
    except Exception:
        n = len(labels)
        return np.linspace(0, 1, n), np.linspace(0, 1, n), float('nan')


def _safe(name: str, fn, n_out: int):
    try:
        return fn()
    except Exception as exc:
        print(f"      [warn] {name}: {exc}", flush=True)
        return np.full(n_out, np.nan)


def _ensure_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


# ---------------------------------------------------------------------------
# Local training loops (record per-epoch loss tqdm with ratio diagnostic)
# ---------------------------------------------------------------------------

def _make_loader(X: np.ndarray, batch_size: int):
    Xt = torch.tensor(X, dtype=torch.float32)
    return Xt


def train_ae(D_raw: int, latent: int, pixels: np.ndarray, cfg: dict,
             seed: int, label: str) -> Tuple[LinearAutoencoder, List[float]]:
    """Train a linear autoencoder on (N, D_raw) pixels with MSE."""
    torch.manual_seed(seed)
    ae = LinearAutoencoder(D_raw, latent, bias=cfg['ae_bias'])
    opt = torch.optim.Adam(ae.parameters(), lr=cfg['ae_lr'],
                           weight_decay=cfg['ae_wd'])
    X = torch.tensor(pixels, dtype=torch.float32)
    N, bs = len(X), cfg['ae_batch']
    hist = []
    pbar = tqdm(range(1, cfg['ae_epochs'] + 1), desc=f'AE {label}',
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i in range(0, N, bs):
            x = X[perm[i:i + bs]]
            x_hat, _ = ae(x)
            loss = ((x_hat - x) ** 2).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        hist.append(tot / max(nb, 1))
        pbar.set_postfix(loss=f"{hist[-1]:.4f}")
    ae.eval()
    return ae, hist


def train_dsm_local(d: int, train_data: np.ndarray, sigma: float, cfg: dict,
                    seed: int, label: str) -> Tuple[ScoreNet, List[float]]:
    """Local DSM training loop (returns final model + per-epoch loss)."""
    torch.manual_seed(seed)
    model = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    X = torch.tensor(train_data, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch_size'], len(X))
    baseline = d / (sigma ** 2)
    hist = []
    pbar = tqdm(range(1, cfg['dsm_epochs'] + 1), desc=f'DSM {label}',
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N)
        tot = 0.0
        nb = 0
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        hist.append(tot / max(nb, 1))
        pbar.set_postfix(loss=f"{hist[-1]:.2f}",
                         ratio=f"{hist[-1] / baseline:.3f}")
    model.eval()
    return model, hist


def train_lrao_local(d: int, train_data: np.ndarray, cfg: dict,
                     seed: int, label: str) -> Tuple[ScoreNet, List[float]]:
    """Local LRao Mode-2 training loop — fixed epochs, no early stopping.

    NaN-guarded: if the in-graph SVD blows up, abort and return what we have.
    """
    torch.manual_seed(seed)
    model = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    X     = torch.tensor(train_data, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch_size'], len(train_data))
    hist  = []

    pbar = tqdm(range(1, cfg['lrao_epochs'] + 1), desc=f'LRao {label}',
                dynamic_ncols=True, leave=False)
    for ep in pbar:
        model.train()
        perm = torch.randperm(N); tot = 0.0; nb = 0
        try:
            for i in range(0, N, bs):
                b    = X[perm[i:i + bs]]
                loss = lfi_loss_mode2(model, b, cfg['lfi_delta_theta'],
                                      cfg['lfi_sigma_cutoff'],
                                      detach_sigma=cfg['lfi_detach_sigma'])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite LRao loss")
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
        except Exception as exc:
            print(f"      [warn] LRao {label} aborted at epoch {ep}: {exc}",
                  flush=True)
            break
        hist.append(-tot / max(nb, 1))   # tr(J*)
        pbar.set_postfix(trJ=f"{hist[-1]:.2f}")

    model.eval()
    return model, hist


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def build_pools(data: np.ndarray, gt_flat: np.ndarray, cfg: dict, mode: str):
    """Return (bkg_pixels, tgt_pixels) in raw 103-D."""
    H_W_D = data.shape
    flat = data.reshape(-1, H_W_D[-1])
    if mode == 'single':
        bkg = flat[gt_flat == cfg['bkg_cls']]
    else:
        excl = list(cfg.get('exclude_classes', []))
        bkg_classes = sorted(int(c) for c in np.unique(gt_flat)
                             if c != cfg['target_cls'] and c not in excl)
        bkg = np.vstack([flat[gt_flat == c] for c in bkg_classes])
    tgt = flat[gt_flat == cfg['target_cls']]
    return bkg, tgt


# ---------------------------------------------------------------------------
# DSM / LRao scoring helpers in latent
# ---------------------------------------------------------------------------

def score_dsm_add(model, train_lat, test_lat, s_lat):
    return dsm_additive(test_lat, train_lat, model, s_lat)


def score_dsm_rep(model, train_lat, test_lat, s_lat):
    return dsm_replacement(test_lat, train_lat, model, s_lat)


def score_lrao(model, train_lat, test_lat, s_lat, cfg):
    return compute_lfi_detector_scores_mode2(
        model, train_lat, test_lat, s_lat,
        delta_theta=cfg['lfi_delta_theta'],
        sigma_cutoff=cfg['lfi_sigma_cutoff'])


# ---------------------------------------------------------------------------
# Classical baselines (cached by n)
# ---------------------------------------------------------------------------

def run_classical_for_n(train_raw, test_raw_planted, labels, s_raw, sig_raw,
                        cfg, mode):
    """Returns dict {tm -> {detector -> scores}}."""
    out = {'additive': {}, 'replacement': {}}
    for tm in ('additive', 'replacement'):
        traw = test_raw_planted[tm]
        n = len(labels[tm])
        if tm == 'additive':
            jobs = {
                'AMF': lambda: amf(traw, train_raw, s_raw),
                'Reg-AMF': lambda: reg_amf(traw, train_raw, s_raw, sig_raw),
                'CEM': lambda: cem(traw, train_raw, s_raw),
                'GMM-GLRT': lambda: gmm_glrt(traw, train_raw, s_raw,
                                             K=cfg['gmm_K']),
            }
            if mode == 'multi':
                jobs['DLTD'] = lambda: dltd(traw, train_raw, s_raw, K=cfg['gmm_K'])
                jobs['SMGLRT'] = lambda: smglrt(traw, train_raw, s_raw, K=cfg['gmm_K'])
        else:
            jobs = {
                'G-rep-LMP': lambda: amf_replacement(traw, train_raw, s_raw),
                'CEM': lambda: cem(traw, train_raw, s_raw),
                'GMM-GLRT-rep': lambda: gmm_glrt_replacement(
                    traw, train_raw, s_raw, K=cfg['gmm_K'],
                    theta_max=cfg['gmm_theta_max'],
                    theta_steps=cfg['gmm_theta_steps']),
                'Exact-GLRT': lambda: exact_glrt_replacement(
                    traw, train_raw, s_raw),
            }
            if mode == 'multi':
                jobs['DLTD'] = lambda: dltd(traw, train_raw, s_raw, K=cfg['gmm_K'])
                jobs['SMGLRT'] = lambda: smglrt(traw, train_raw, s_raw, K=cfg['gmm_K'])
        for nm, fn in jobs.items():
            out[tm][nm] = _safe(nm, fn, n)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

DETECTOR_COLORS = {
    'AMF': '#1f77b4',
    'Reg-AMF': '#6baed6',
    'G-rep-LMP': '#08306b',
    'CEM': '#aec7e8',
    'GMM-GLRT': '#9467bd',
    'GMM-GLRT-rep': '#c5b0d5',
    'DLTD': '#e6550d',
    'SMGLRT': '#8c564b',
    'Exact-GLRT': '#7f3b08',
    'DSM-PCA': '#d62728',
    'DSM-PCA-rep': '#fc8d59',
    'DSM-AE': '#9e0142',
    'DSM-AE-rep': '#f46d43',
    'LRao-PCA': '#2ca02c',
    'LRao-AE': '#006400',
}


def _det_color(det: str):
    return DETECTOR_COLORS.get(det, '#444444')


def plot_auc_vs_n(metrics: dict, d: int, tm: str, n_list: list,
                  out_pdf: str, classical_dets: list,
                  score_dets: list):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(n_list))
    for det in classical_dets:
        ys = metrics['classical'][tm].get(det)
        if ys is None:
            continue
        ax.plot(x, ys, marker='o', lw=1.4, color=_det_color(det), label=det)
    score_branch = metrics['score'].get(f'd_{d}', {}).get(tm, {})
    for det in score_dets:
        ys = score_branch.get(det)
        if ys is None:
            continue
        ax.plot(x, ys, marker='D', lw=2.2, color=_det_color(det), label=det)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in n_list], rotation=0)
    ax.set_xlabel('training samples  n')
    ax.set_ylabel('AUC')
    ax.set_title(f'AUC vs n  (latent_dim={d}, target model: {tm})')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_auc_vs_d(metrics: dict, n_idx: int, tm: str, d_list: list,
                  out_pdf: str, score_dets: list):
    """Score methods vs latent_dim at one fixed n (specifically the last n in the list)."""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for det in score_dets:
        ys = [metrics['score'][f'd_{d}'][tm].get(det, [None] * (n_idx + 1))[n_idx]
              for d in d_list]
        ax.plot(d_list, ys, marker='D', lw=2.0, color=_det_color(det), label=det)
    ax.set_xlabel('latent_dim  d')
    ax.set_ylabel('AUC')
    ax.set_title(f'AUC vs d  (n=max, target model: {tm})')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def plot_loss_panel(loss_curves: dict, d: int, n_max: int, out_png: str):
    keys = [
        (f'AE_d{d}', f'AE (d={d})', 'loss'),
        (f'DSM_PCA_d{d}_n{n_max}', f'DSM-PCA (d={d}, n={n_max})', 'loss'),
        (f'DSM_AE_d{d}_n{n_max}', f'DSM-AE  (d={d}, n={n_max})', 'loss'),
        (f'LRao_PCA_d{d}_n{n_max}', f'LRao-PCA (d={d}, n={n_max})', 'tr(J*)'),
        (f'LRao_AE_d{d}_n{n_max}', f'LRao-AE  (d={d}, n={n_max})', 'tr(J*)'),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
    for ax, (key, ttl, ylab) in zip(axes, keys):
        hist = loss_curves.get(key, [])
        ax.plot(hist) if hist else ax.text(0.5, 0.5, 'no data', ha='center')
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_rocs(roc_dict: dict, out_pdf: str, title: str):
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for det, (fpr, tpr, au) in roc_dict.items():
        ax.plot(fpr, tpr, lw=1.5, color=_det_color(det),
                label=f'{det} ({au:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.6)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_iid(cfg: dict, mode: str):
    assert mode in ('single', 'multi')
    t_start = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'iid_{mode}_{ts}')
    mdl_dir = os.path.join(run_dir, 'models')
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(mdl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'))
    print(f"Run dir: {run_dir}", flush=True)

    # ----- canonicalize sweep lists -----
    n_list = sorted(set(_ensure_list(cfg['n_train_list'])))
    d_list = sorted(set(_ensure_list(cfg['latent_dim_list'])))
    print(f"n_train_list  = {n_list}", flush=True)
    print(f"latent_dim_list = {d_list}", flush=True)

    # ----- load + normalize -----
    seed = int(cfg['seed'])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ----- load data: two normalizations if baseline_norm_mode differs -----
    # Score models (DSM, LRao, AE) use cfg['norm_mode'] — optimized for training.
    # Classical baselines use cfg['baseline_norm_mode'] — honest, unwhitened data.
    # If both are the same, data_base == data (no duplicate work).
    score_norm    = cfg['norm_mode']
    baseline_norm = cfg.get('baseline_norm_mode', score_norm)
    data, gt = load_and_normalize(cfg['dataset'], mode=score_norm)
    H, W, D_RAW = data.shape
    gt_flat  = gt.flatten()
    all_flat = data.reshape(-1, D_RAW).astype(np.float32)
    print(f"Image {H}x{W}x{D_RAW}", flush=True)
    if baseline_norm != score_norm:
        data_base, _ = load_and_normalize(cfg['dataset'], mode=baseline_norm)
        print(f"  score models  → norm='{score_norm}'", flush=True)
        print(f"  baselines     → norm='{baseline_norm}'", flush=True)
    else:
        data_base = data

    # ----- background/target pools -----
    # Score space (PCA + AE training)
    bkg_raw, tgt_raw = build_pools(data, gt_flat, cfg, mode)
    s_raw = tgt_raw.mean(axis=0)
    if cfg.get('normalize_signature', False):
        s_raw = s_raw / (np.linalg.norm(s_raw) + 1e-12)

    # Baseline space (classical detectors)
    bkg_base, tgt_base = build_pools(data_base, gt_flat, cfg, mode)
    s_raw_base = tgt_base.mean(axis=0)
    if cfg.get('normalize_signature', False):
        s_raw_base = s_raw_base / (np.linalg.norm(s_raw_base) + 1e-12)

    print(f"bkg pool: {len(bkg_raw):>6} | tgt pool: {len(tgt_raw):>6} | "
          f"||s_raw|| = {np.linalg.norm(s_raw):.4f} (score)  "
          f"||s_base|| = {np.linalg.norm(s_raw_base):.4f} (baseline)", flush=True)

    # ----- shuffle — SAME indices for both spaces so same pixels are train/test -----
    assert len(bkg_raw) == len(bkg_base), "bkg pool size mismatch between normalizations"
    idx = np.arange(len(bkg_raw))
    rng.shuffle(idx)
    bkg_shuf      = bkg_raw[idx]
    bkg_base_shuf = bkg_base[idx]
    max_n     = max(n_list)
    test_size = int(cfg['test_size'])
    assert len(bkg_shuf) >= max_n + test_size, \
        f"need {max_n + test_size} bkg pixels, have {len(bkg_shuf)}"
    train_pool_raw  = bkg_shuf[:max_n]          # score model training pool
    train_pool_base = bkg_base_shuf[:max_n]     # baseline training pool
    test_bkg_raw    = bkg_shuf[-test_size:]     # score model test set
    test_bkg_base   = bkg_base_shuf[-test_size:]  # baseline test set

    # ----- plant targets — independently in each space, SAME planted indices -----
    # The planted indices are determined by seed alone (same physical pixels get targets).
    test_planted_raw  = {}   # score model test set (with targets)
    test_planted_base = {}   # baseline test set (with targets)
    labels = {}
    for tm in ('additive', 'replacement'):
        planted_score, lab, _ = plant_targets(
            test_bkg_raw, s_raw, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        planted_base, _, _ = plant_targets(
            test_bkg_base, s_raw_base, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)          # same seed → same indices → same labels
        test_planted_raw[tm]  = planted_score
        test_planted_base[tm] = planted_base
        labels[tm] = lab
    n_pos = int(labels['additive'].sum())
    print(f"planted {n_pos} targets in test (amp={cfg['amplitude']})\n",
          flush=True)

    # ----- per-dim setup: PCA + AE -----
    loss_curves: Dict[str, list] = {}
    per_d = {}  # d -> dict of artifacts
    for d in d_list:
        print(f"=== Setting up d={d} ===", flush=True)
        # PCA
        pca = PCA(n_components=d).fit(all_flat)
        evr = pca.explained_variance_ratio_.sum()
        print(f"  PCA d={d}  explained var = {evr:.4f}", flush=True)
        pickle.dump(pca, open(os.path.join(mdl_dir, f'pca_d{d}.pkl'), 'wb'))

        train_pool_pca = pca.transform(train_pool_raw).astype(np.float32)
        test_pca = {tm: pca.transform(test_planted_raw[tm]).astype(np.float32)
                    for tm in test_planted_raw}
        # Two signature vectors for PCA space:
        #   additive  y = w + θs  →  z = z_w + θ·(s @ V.T)   (no centering!)
        #   replacement y=(1-θ)w+θs → z=(1-θ)z_w + θ·(s-μ)@V.T  (with centering)
        s_pca_add = (pca.components_ @ s_raw).astype(np.float32)
        s_pca_rep = pca.transform(s_raw[None]).flatten().astype(np.float32)

        # AE
        ae, ae_hist = train_ae(D_RAW, d, all_flat, cfg, seed=seed,
                               label=f'd={d}')
        torch.save({'state_dict': ae.state_dict(),
                    'D': D_RAW, 'latent': d, 'bias': cfg['ae_bias']},
                   os.path.join(mdl_dir, f'ae_d{d}.pt'))
        loss_curves[f'AE_d{d}'] = ae_hist
        with torch.no_grad():
            train_pool_aelat = ae.encode(
                torch.tensor(train_pool_raw, dtype=torch.float32)).numpy().astype(np.float32)
            test_aelat = {tm: ae.encode(
                torch.tensor(test_planted_raw[tm], dtype=torch.float32)
            ).numpy().astype(np.float32) for tm in test_planted_raw}
            # AE signature:
            #   additive  enc(w + θs) = enc(w) + θ·(s @ W.T)   → no bias term
            #   replacement enc((1-θ)w+θs) = (1-θ)enc(w) + θ·enc(s)  → enc(s) correct
            if cfg.get('ae_bias', True):
                # W has shape (latent, D_raw) W @ s_raw = enc(s) - bias
                W = ae.enc.weight.detach().cpu().numpy()  # (latent, D_raw)
                s_aelat_add = (W @ s_raw).astype(np.float32)
            else:
                s_aelat_add = ae.encode(
                    torch.tensor(s_raw[None], dtype=torch.float32)
                ).numpy().flatten().astype(np.float32)
            s_aelat_rep = ae.encode(
                torch.tensor(s_raw[None], dtype=torch.float32)
            ).numpy().flatten().astype(np.float32)

        per_d[d] = dict(pca=pca, train_pool_pca=train_pool_pca,
                        test_pca=test_pca,
                        s_pca_add=s_pca_add, s_pca_rep=s_pca_rep,
                        ae=ae, train_pool_aelat=train_pool_aelat,
                        test_aelat=test_aelat,
                        s_aelat_add=s_aelat_add, s_aelat_rep=s_aelat_rep)

    # ----- structured metrics + scores stores -----
    metrics = {
        'n_train_list': n_list,
        'latent_dim_list': d_list,
        'classical': {'additive': {}, 'replacement': {}},
        'score': {f'd_{d}': {'additive': {}, 'replacement': {}} for d in d_list},
    }
    scores = {
        'labels_additive': labels['additive'].astype(np.int8),
        'labels_replacement': labels['replacement'].astype(np.int8),
    }

    # ----- classical baselines once per n (cache) -----
    print("\n=== CLASSICAL baselines (raw 103-D) ===", flush=True)
    classical_dets_add = [
        'AMF',
        'Reg-AMF',
        # 'CEM',
        'GMM-GLRT'
    ]
    classical_dets_rep = [
        'G-rep-LMP',
        # 'CEM',
        'GMM-GLRT-rep',
        'Exact-GLRT'
    ]
    if mode == 'multi':
        pass
        # classical_dets_add += ['DLTD', 'SMGLRT']
        # classical_dets_rep += ['DLTD', 'SMGLRT']
    for det in classical_dets_add:
        metrics['classical']['additive'][det] = []
    for det in classical_dets_rep:
        metrics['classical']['replacement'][det] = []

    for n in n_list:
        t0 = time.time()
        train_raw_n  = train_pool_base[:n]                  # baseline normalization
        sig_raw = compute_sigma_from_data(train_raw_n, cfg['dsm_sigma_rho'])
        cl = run_classical_for_n(train_raw_n, test_planted_base, labels,
                                 s_raw_base, sig_raw, cfg, mode)
        for tm in ('additive', 'replacement'):
            for det, sc in cl[tm].items():
                metrics['classical'][tm].setdefault(det, []).append(
                    _auc(labels[tm], sc))
                scores[f'classical/{det}_n{n}_{tm}'] = sc.astype(np.float32)
        line_add = " ".join(f"{d}={metrics['classical']['additive'][d][-1]:.3f}"
                            for d in classical_dets_add)
        line_rep = " ".join(f"{d}={metrics['classical']['replacement'][d][-1]:.3f}"
                            for d in classical_dets_rep)
        print(f"  n={n:>4}  ({time.time() - t0:.0f}s)", flush=True)
        print(f"     add: {line_add}", flush=True)
        print(f"     rep: {line_rep}", flush=True)

    # ----- sweep (d, n) for score methods -----
    print("\n=== SCORE METHODS (PCA-d + AE-d) ===", flush=True)
    score_dets_add = ['DSM-PCA', 'DSM-AE', 'LRao-PCA', 'LRao-AE']
    score_dets_rep = ['DSM-PCA-rep', 'DSM-AE-rep', 'LRao-PCA', 'LRao-AE']

    for d in d_list:
        for det in score_dets_add:
            metrics['score'][f'd_{d}']['additive'].setdefault(det, [])
        for det in score_dets_rep:
            metrics['score'][f'd_{d}']['replacement'].setdefault(det, [])

        for n in n_list:
            t0 = time.time()
            train_pca_n = per_d[d]['train_pool_pca'][:n]
            train_aelat_n = per_d[d]['train_pool_aelat'][:n]
            sigma_pca = compute_sigma_from_data(train_pca_n, cfg['dsm_sigma_rho'])
            sigma_ae = compute_sigma_from_data(train_aelat_n, cfg['dsm_sigma_rho'])

            # ---- DSM-PCA ----
            dsm_pca, hist = train_dsm_local(
                d, train_pca_n, sigma_pca, cfg, seed, f'PCA d={d} n={n}')
            loss_curves[f'DSM_PCA_d{d}_n{n}'] = hist
            torch.save({'state_dict': dsm_pca.state_dict(),
                        'd': d, 'n': n, 'sigma': sigma_pca,
                        'hidden_dims': cfg['hidden_dims'],
                        'activation': cfg['activation']},
                       os.path.join(mdl_dir, f'dsm_pca_d{d}_n{n}.pt'))
            sc_dsm_pca_add = score_dsm_add(
                dsm_pca, train_pca_n, per_d[d]['test_pca']['additive'], per_d[d]['s_pca_add'])
            sc_dsm_pca_rep = score_dsm_rep(
                dsm_pca, train_pca_n, per_d[d]['test_pca']['replacement'], per_d[d]['s_pca_rep'])

            # ---- DSM-AE ----
            dsm_ae, hist = train_dsm_local(
                d, train_aelat_n, sigma_ae, cfg, seed, f'AE  d={d} n={n}')
            loss_curves[f'DSM_AE_d{d}_n{n}'] = hist
            torch.save({'state_dict': dsm_ae.state_dict(),
                        'd': d, 'n': n, 'sigma': sigma_ae,
                        'hidden_dims': cfg['hidden_dims'],
                        'activation': cfg['activation']},
                       os.path.join(mdl_dir, f'dsm_ae_d{d}_n{n}.pt'))
            sc_dsm_ae_add = score_dsm_add(
                dsm_ae, train_aelat_n, per_d[d]['test_aelat']['additive'], per_d[d]['s_aelat_add'])
            sc_dsm_ae_rep = score_dsm_rep(
                dsm_ae, train_aelat_n, per_d[d]['test_aelat']['replacement'], per_d[d]['s_aelat_rep'])

            # ---- LRao-PCA ----
            lrao_pca, hist = train_lrao_local(
                d, train_pca_n, cfg, seed, f'PCA d={d} n={n}')
            loss_curves[f'LRao_PCA_d{d}_n{n}'] = hist
            torch.save({'state_dict': lrao_pca.state_dict(),
                        'd': d, 'n': n,
                        'hidden_dims': cfg['hidden_dims'],
                        'activation': cfg['activation']},
                       os.path.join(mdl_dir, f'lrao_pca_d{d}_n{n}.pt'))
            sc_lrao_pca_add = _safe(
                f'LRao-PCA d={d} n={n} add',
                lambda: score_lrao(lrao_pca, train_pca_n,
                                   per_d[d]['test_pca']['additive'],
                                   per_d[d]['s_pca_add'], cfg),
                len(labels['additive']))
            sc_lrao_pca_rep = _safe(
                f'LRao-PCA d={d} n={n} rep',
                lambda: score_lrao(lrao_pca, train_pca_n,
                                   per_d[d]['test_pca']['replacement'],
                                   per_d[d]['s_pca_rep'], cfg),
                len(labels['replacement']))

            # ---- LRao-AE ----
            lrao_ae, hist = train_lrao_local(
                d, train_aelat_n, cfg, seed, f'AE  d={d} n={n}')
            loss_curves[f'LRao_AE_d{d}_n{n}'] = hist
            torch.save({'state_dict': lrao_ae.state_dict(),
                        'd': d, 'n': n,
                        'hidden_dims': cfg['hidden_dims'],
                        'activation': cfg['activation']},
                       os.path.join(mdl_dir, f'lrao_ae_d{d}_n{n}.pt'))
            sc_lrao_ae_add = _safe(
                f'LRao-AE d={d} n={n} add',
                lambda: score_lrao(lrao_ae, train_aelat_n,
                                   per_d[d]['test_aelat']['additive'],
                                   per_d[d]['s_aelat_add'], cfg),
                len(labels['additive']))
            sc_lrao_ae_rep = _safe(
                f'LRao-AE d={d} n={n} rep',
                lambda: score_lrao(lrao_ae, train_aelat_n,
                                   per_d[d]['test_aelat']['replacement'],
                                   per_d[d]['s_aelat_rep'], cfg),
                len(labels['replacement']))

            # ---- collect AUCs + scores ----
            sc = {
                'additive': {
                    'DSM-PCA': sc_dsm_pca_add,
                    'DSM-AE': sc_dsm_ae_add,
                    'LRao-PCA': sc_lrao_pca_add,
                    'LRao-AE': sc_lrao_ae_add,
                },
                'replacement': {
                    'DSM-PCA-rep': sc_dsm_pca_rep,
                    'DSM-AE-rep': sc_dsm_ae_rep,
                    'LRao-PCA': sc_lrao_pca_rep,
                    'LRao-AE': sc_lrao_ae_rep,
                },
            }
            for tm in ('additive', 'replacement'):
                for det, arr in sc[tm].items():
                    metrics['score'][f'd_{d}'][tm][det].append(
                        _auc(labels[tm], arr))
                    scores[f'score/d{d}_n{n}_{det}_{tm}'] = arr.astype(np.float32)

            elapsed = time.time() - t0
            line_add = " ".join(
                f"{k}={metrics['score'][f'd_{d}']['additive'][k][-1]:.3f}"
                for k in score_dets_add)
            line_rep = " ".join(
                f"{k}={metrics['score'][f'd_{d}']['replacement'][k][-1]:.3f}"
                for k in score_dets_rep)
            print(f"  d={d} n={n:>4}  ({elapsed:.0f}s)", flush=True)
            print(f"     add: {line_add}", flush=True)
            print(f"     rep: {line_rep}", flush=True)
            # incremental save so we never lose all progress
            json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'),
                      indent=2, default=str)
            json.dump(loss_curves,
                      open(os.path.join(run_dir, 'loss_curves.json'), 'w'),
                      default=str)

    # ---- final saves ----
    np.savez_compressed(os.path.join(run_dir, 'scores.npz'), **scores)

    # ---- figures ----
    n_max = max(n_list)
    n_idx_max = n_list.index(n_max)
    for d in d_list:
        plot_auc_vs_n(metrics, d, 'additive', n_list,
                      os.path.join(fig_dir, f'auc_vs_n_additive_d{d}.pdf'),
                      classical_dets_add, score_dets_add)
        plot_auc_vs_n(metrics, d, 'replacement', n_list,
                      os.path.join(fig_dir, f'auc_vs_n_replacement_d{d}.pdf'),
                      classical_dets_rep, score_dets_rep)
        plot_loss_panel(loss_curves, d, n_max,
                        os.path.join(fig_dir, f'loss_curves_d{d}_n{n_max}.png'))
        # ROC at (d, n_max)
        for tm, score_dets in (('additive', score_dets_add),
                               ('replacement', score_dets_rep)):
            cl_dets = classical_dets_add if tm == 'additive' else classical_dets_rep
            roc_dict = {}
            for det in cl_dets:
                roc_dict[det] = _roc(labels[tm], scores[f'classical/{det}_n{n_max}_{tm}'])
            for det in score_dets:
                key = f'score/d{d}_n{n_max}_{det}_{tm}'
                if key in scores:
                    roc_dict[det] = _roc(labels[tm], scores[key])
            plot_rocs(roc_dict,
                      os.path.join(fig_dir, f'roc_d{d}_n{n_max}_{tm}.pdf'),
                      title=f'ROC  d={d}, n={n_max}, {tm}')

    plot_auc_vs_d(metrics, n_idx_max, 'additive', d_list,
                  os.path.join(fig_dir, f'auc_vs_d_n{n_max}_additive.pdf'),
                  score_dets_add)
    plot_auc_vs_d(metrics, n_idx_max, 'replacement', d_list,
                  os.path.join(fig_dir, f'auc_vs_d_n{n_max}_replacement.pdf'),
                  score_dets_rep)

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\nDONE in {elapsed_min:.1f} min. Results -> {run_dir}", flush=True)
    return run_dir, metrics
