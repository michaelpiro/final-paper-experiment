"""
data.py — data loading, target planting, ZCA whitening, spatial neighbourhoods,
random scene boxes, and target-signature construction.

The hyperspectral cube is consumed RAW (no normalization): every learned score
model carries a frozen ZCA whitening first layer fitted on the training
background, so per-band scaling is unnecessary and PCA is avoided (it could
discard the unknown target direction).

CLS_NAMES — Pavia-University class id -> name
load_hsi  — read the .mat cube + ground-truth map (raw float64)
compute_sigma_from_data — DSM noise level sigma^2 = rho * tr(Sigma)/d
split_background — seeded shuffle + train/val/test split of a pixel pool
plant_targets — additive (or replacement) target planting into test pixels
Whitening — frozen ZCA whitening module (W = V Lambda^{-1/2} V^T)
extract_neighborhoods — k x k spatial windows (centre + neighbours), circular pad
compute_signature / generate_random_boxes / scores_to_spatial_map — spatial helpers
_crop_pca_box / _crop_raw_box / make_whitening / whitened_sigma — spatial crop helpers
"""

import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks',    9: 'shadows',
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_hsi(path: str, gt_path: str = None):
    """Load a .mat hyperspectral dataset. Returns the RAW cube (no normalization).

    The repackaged datasets in repro/data store the cube under 'data' and the
    labels under 'map', and that fast path is unchanged. Datasets distributed in
    their original form (Salinas, Indian Pines, ...) instead name the arrays
    after the scene and ship the labels in a SEPARATE `*_gt.mat`, so if 'data' /
    'map' are absent we fall back to picking the only 3-D array as the cube and
    the only 2-D integer array as the labels, looking in `gt_path` (or a sibling
    `<stem>_gt.mat`) when the cube file has no labels of its own.

    Returns
    -------
    data : (H, W, B) float64 — raw sensor values
    gt   : (H, W)    int     — ground-truth class labels (0 = unlabeled)
    """
    if not os.path.exists(path):          # tolerate a missing 'repro/' prefix
        alt = os.path.join('repro', path)
        if os.path.exists(alt):
            path = alt
    mat = scipy.io.loadmat(path)
    if 'data' in mat and 'map' in mat:                       # repackaged (fast path)
        return mat['data'].astype(np.float64), mat['map'].astype(int)

    def _arrays(m, ndim):
        return [v for k, v in m.items()
                if not k.startswith('__') and getattr(v, 'ndim', 0) == ndim]

    cubes = _arrays(mat, 3)
    if len(cubes) != 1:
        raise ValueError(f"{path}: expected exactly one 3-D cube, found {len(cubes)}")
    data = cubes[0].astype(np.float64)

    labels = _arrays(mat, 2)
    if not labels:                                           # labels live elsewhere
        if gt_path is None:
            stem = os.path.splitext(path)[0]
            for cand in (f'{stem}_gt.mat', f'{stem.replace("_corrected", "")}_gt.mat'):
                if os.path.exists(cand):
                    gt_path = cand
                    break
        if gt_path is None or not os.path.exists(gt_path):
            raise ValueError(f"{path}: no label array here and no *_gt.mat alongside; "
                             f"pass gt_path explicitly")
        labels = _arrays(scipy.io.loadmat(gt_path), 2)
        if not labels:
            raise ValueError(f"{gt_path}: no 2-D label array found")
    gt = max(labels, key=lambda a: a.size)                   # the full-scene map
    if gt.shape != data.shape[:2]:
        raise ValueError(f"{path}: label map {gt.shape} does not match cube "
                         f"{data.shape[:2]}")
    return data, gt.astype(int)


# ---------------------------------------------------------------------------
# DSM noise level + train/test split
# ---------------------------------------------------------------------------

def compute_sigma_from_data(train_data: np.ndarray, rho: float = 0.01) -> float:
    """Data-driven DSM noise level: sigma^2 = rho * (1/d) * tr(Sigma_hat)."""
    s2 = np.mean(np.var(train_data, axis=0))
    return float(np.sqrt(rho * s2))


def split_background(bkg: np.ndarray, n_train: int, n_test: int,
                     n_val: int = 0, seed: int = 42):
    """Shuffle and split a background pixel pool into train / val / test."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(bkg))
    rng.shuffle(idx)
    bkg = bkg[idx]
    assert n_train + n_val + n_test <= len(bkg), \
        f"Not enough background pixels: need {n_train+n_val+n_test}, have {len(bkg)}"
    train = bkg[:n_train]
    val   = bkg[n_train:n_train + n_val] if n_val > 0 else np.empty((0, bkg.shape[1]))
    test  = bkg[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


# ---------------------------------------------------------------------------
# Target planting
# ---------------------------------------------------------------------------

def plant_targets(test_bkg: np.ndarray, s: np.ndarray, amplitude: float,
                  tgt_fraction: float, model: str = 'additive', seed: int = 0,
                  spatial_shape: tuple = None, edge_guard: int = 0):
    """Plant target signatures into a random subset of test pixels.

    model='additive'    -> y = w + amplitude * s
    model='replacement' -> y = amplitude * s + (1 - amplitude) * w
    edge_guard>0 (with spatial_shape) excludes pixels within `edge_guard` of the
    box border from the candidate pool, so every planted target has a full
    neighbourhood. Returns (test_data, labels, tgt_idx).
    """
    n_test = len(test_bkg)
    if edge_guard > 0 and spatial_shape is not None:
        H, W = spatial_shape
        g = int(edge_guard)
        rows = np.arange(n_test) // W
        cols = np.arange(n_test) % W
        interior = np.where((rows >= g) & (rows < H - g) &
                            (cols >= g) & (cols < W - g))[0]
        if len(interior) == 0:
            interior = np.arange(n_test)
    else:
        interior = np.arange(n_test)

    n_tgt   = max(1, int(round(len(interior) * tgt_fraction)))
    labels  = np.zeros(n_test, dtype=int)
    tgt_idx = np.random.default_rng(seed).choice(interior, size=n_tgt, replace=False)
    labels[tgt_idx] = 1

    test_data = test_bkg.copy()
    if model == 'additive':
        test_data[tgt_idx] += amplitude * s
    elif model == 'replacement':
        test_data[tgt_idx] = amplitude * s + (1.0 - amplitude) * test_bkg[tgt_idx]
    else:
        raise ValueError(f"Unknown target model: {model!r}")
    return test_data, labels, tgt_idx


# ---------------------------------------------------------------------------
# Frozen ZCA whitening
# ---------------------------------------------------------------------------

class Whitening(nn.Module):
    """Frozen ZCA whitening front-end: x -> (x - mu) @ W^T, with
    W = V Lambda^{-1/2} V^T computed from the background covariance so
    cov(output) = I. Symmetric (stays closest to the original axes); replaces
    PCA. The module is frozen (buffers, no grad) and also whitens a (B, M, D)
    neighbour tensor (broadcast over the last axis)."""

    def __init__(self, mu, W):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("W",  torch.as_tensor(W,  dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) @ self.W.t()

    def transform_direction(self, s) -> np.ndarray:
        """Whiten a DIRECTION (additive signature; no mean subtraction): s -> W s."""
        Wn = self.W.detach().cpu().numpy()
        return (np.asarray(s, dtype=np.float32) @ Wn.T).astype(np.float32)

    @classmethod
    def from_data(cls, X: np.ndarray, eig_floor: float = 0.0, eps: float = 1e2,
                  mode: str = 'zca'):
        """Fit a frozen linear whitener from background pixels X.

        eig_floor : relative eigenvalue floor (x lambda_max). 0 -> auto 1e-5.
        eps       : absolute minimum floor (raw sensor values are large, so the
                    default is intentionally O(1e2)).
        mode      : linear front-end (all return a square D->D Whitening, so they
                    are drop-in for one another and carry a signature direction
                    through transform_direction):
                      'zca'       W = V Λ^{-1/2} Vᵀ  (symmetric; DEFAULT — closest
                                  to the original axes; cov(output)=I).
                      'pca'       W = Λ^{-1/2} Vᵀ    (rotate to the eigenbasis).
                      'cholesky'  W = L^{-1}, Σ = L Lᵀ.
                      'normalize' W = diag(1/σ_b)    (per-channel standardization
                                  only — centers + unit-variances each band but
                                  keeps cross-band correlations; diagonal W, so no
                                  eigendecomposition/LAPACK is used).
        """
        X = np.asarray(X, dtype=np.float64)
        n, D = X.shape
        mu = X.mean(0)
        Xc = X - mu

        if mode == 'normalize':
            std = Xc.std(0)
            floor = max(float(np.max(std)) * (eig_floor if eig_floor > 0 else 1e-5),
                        np.sqrt(eps))
            std = np.clip(std, floor, None)
            W = np.diag(1.0 / std)                  # diagonal, no decorrelation
            return cls(mu.astype(np.float32), W.astype(np.float32))

        Sigma = (Xc.T @ Xc) / max(n - 1, 1)
        Sigma = (Sigma + Sigma.T) / 2
        if mode == 'cholesky':
            rel = eig_floor if eig_floor > 0 else 1e-5
            floor = max(float(np.trace(Sigma) / D) * rel, eps)
            L = np.linalg.cholesky(Sigma + floor * np.eye(D))
            W = np.linalg.inv(L)                    # cov(Wx) = I
            return cls(mu.astype(np.float32), W.astype(np.float32))

        evals, evecs = np.linalg.eigh(Sigma)        # ascending
        rel = eig_floor if eig_floor > 0 else 1e-5
        floor = max(float(evals[-1]) * rel, eps)
        evals = np.clip(evals, floor, None)
        inv_sqrt = np.diag(1.0 / np.sqrt(evals))
        if mode == 'pca':
            W = inv_sqrt @ evecs.T                  # rotate to eigenbasis
        else:                                       # 'zca' (default)
            W = evecs @ inv_sqrt @ evecs.T          # ZCA (symmetric)
        return cls(mu.astype(np.float32), W.astype(np.float32))


def make_whitening(tr_raw, cfg, device):
    """Frozen ZCA whitener fit on the RAW training background (-> device)."""
    W = Whitening.from_data(np.asarray(tr_raw, dtype=np.float32),
                            eig_floor=float(cfg.get('whiten_eig_floor', 1e-5)))
    return W.to(device)


def whitened_sigma(cfg):
    """In whitened space cov ~ I, so sigma^2 = rho * 1 => sigma = sqrt(rho)."""
    return float(np.sqrt(cfg['dsm_sigma_rho']))


def placeholder_whitening(D):
    """Identity whitener of the right shape so load_state_dict can fill its buffers."""
    return Whitening(np.zeros(D, dtype=np.float32), np.eye(D, dtype=np.float32))


# ---------------------------------------------------------------------------
# Spatial neighbourhood extraction + crop helpers
# ---------------------------------------------------------------------------

def extract_neighborhoods(img: torch.Tensor, k: int):
    """k x k spatial windows of a (H, W, D) feature image (circular-padded).

    Returns centers (H*W, D) and neighbors (H*W, k*k-1, D) (window minus centre).
    """
    H, W, D = img.shape
    p = k // 2
    x = img.permute(2, 0, 1).unsqueeze(0)            # (1, D, H, W)
    x = F.pad(x, (p, p, p, p), mode='circular')
    patches = F.unfold(x, kernel_size=k, padding=0)  # (1, D*k*k, H*W)
    patches = patches.reshape(D, k * k, H * W).permute(2, 1, 0)  # (HW, k*k, D)
    center_idx = (k * k) // 2
    centers = patches[:, center_idx, :].contiguous()
    mask = torch.ones(k * k, dtype=torch.bool)
    mask[center_idx] = False
    neighbors = patches[:, mask, :].contiguous()
    return centers, neighbors


def _crop_pca_box(img, box, k):
    """Crop a box and return (centers (N,D), neighbors (N,k*k-1,D)) numpy arrays."""
    r0, r1, c0, c1 = box
    sub = torch.tensor(img[r0:r1, c0:c1, :], dtype=torch.float32)
    centers, nbrs = extract_neighborhoods(sub, k)
    return centers.numpy(), nbrs.numpy()


def _crop_raw_box(data, box):
    """Flatten a box to (N, D) raw pixels."""
    r0, r1, c0, c1 = box
    return data[r0:r1, c0:c1].reshape(-1, data.shape[-1])


# ---------------------------------------------------------------------------
# Target-signature construction (training side only)
# ---------------------------------------------------------------------------

def compute_signature(gt_patch: np.ndarray, raw_patch: np.ndarray,
                      w_dom: float = 0.8, w_mean: float = 0.2,
                      external_cls_pixels: np.ndarray = None):
    """In-patch signature: s = w_dom * mu_dominant_class + w_mean * mu_patch.
    Returns (s_raw, dom_cls_id, dom_cls_name)."""
    gt_flat  = gt_patch.ravel()
    raw_flat = raw_patch.reshape(-1, raw_patch.shape[-1])
    labeled  = gt_flat != 0
    if labeled.sum() == 0:
        labeled = np.ones(len(gt_flat), dtype=bool)
    cls_ids, cnts = np.unique(gt_flat[labeled], return_counts=True)
    dom_cls  = int(cls_ids[cnts.argmax()])
    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')
    if external_cls_pixels is not None:
        mu_dom = external_cls_pixels.mean(axis=0).astype(np.float32)
    else:
        dom_mask = (gt_flat == dom_cls)
        if dom_mask.sum() == 0:
            dom_mask = np.ones(len(gt_flat), dtype=bool)
        mu_dom = raw_flat[dom_mask].mean(axis=0).astype(np.float32)
    mu_patch = raw_flat.mean(axis=0).astype(np.float32)
    s_raw = (w_dom * mu_dom + w_mean * mu_patch).astype(np.float32)
    return s_raw, dom_cls, dom_name


def generate_random_boxes(gt: np.ndarray, n: int = 4, min_pixels: int = 2000,
                          seeds=(42, 123, 456, 789)) -> list:
    """Generate n random (train_box, test_box) pairs on disjoint image halves.
    Each pair: {'train_box': [r0,r1,c0,c1], 'test_box': [...], + class stats}."""
    H, W = gt.shape
    pairs = []
    for i, seed in enumerate(seeds[:n]):
        rng = np.random.default_rng(seed)
        if i % 2 == 0:
            train_region = (0, H // 2, 0, W)
            test_region  = (H // 2, H, 0, W)
        else:
            train_region = (0, H, 0, W // 2)
            test_region  = (0, H, W // 2, W)

        def _random_box(region, rng, min_pix):
            r0r, r1r, c0r, c1r = region
            for _ in range(500):
                target = int(min_pix * (1.5 + rng.uniform(0, 1)))
                side   = int(np.sqrt(target))
                h_box  = max(side, 40) + int(rng.integers(0, max(side // 2, 20)))
                w_box  = max(side, 40) + int(rng.integers(0, max(side // 2, 20)))
                h_box  = min(h_box, r1r - r0r)
                w_box  = min(w_box, c1r - c0r)
                if h_box < 10 or w_box < 10:
                    continue
                r0 = int(rng.integers(r0r, r1r - h_box + 1))
                c0 = int(rng.integers(c0r, c1r - w_box + 1))
                r1, c1 = r0 + h_box, c0 + w_box
                if (r1 - r0) * (c1 - c0) >= min_pix:
                    return [r0, r1, c0, c1]
            pad = 10
            return [r0r + pad, r1r - pad, c0r + pad, c1r - pad]

        tr_box = _random_box(train_region, rng, min_pixels)
        te_box = _random_box(test_region,  rng, min_pixels)

        def _stats(box):
            r0, r1, c0, c1 = box
            patch = gt[r0:r1, c0:c1].ravel()
            cls_ids, cnts = np.unique(patch, return_counts=True)
            dom_cls = int(cls_ids[cnts.argmax()])
            stats = {CLS_NAMES.get(int(c), f'cls{c}'): int(n_)
                     for c, n_ in zip(cls_ids, cnts)}
            return stats, int(cnts.sum()), dom_cls

        tr_stats, tr_total, tr_dom = _stats(tr_box)
        te_stats, te_total, te_dom = _stats(te_box)
        pairs.append({
            'train_box': tr_box, 'train_stats': tr_stats, 'train_total': tr_total,
            'train_dominant': CLS_NAMES.get(tr_dom, f'cls{tr_dom}'),
            'test_box': te_box, 'test_stats': te_stats, 'test_total': te_total,
            'test_dominant': CLS_NAMES.get(te_dom, f'cls{te_dom}'),
        })
    return pairs


def scores_to_spatial_map(scores: np.ndarray, te_idx: np.ndarray,
                          box_shape: tuple, fill: float = float('nan')) -> np.ndarray:
    """Scatter test-pixel scores back into a 2D grid matching the test box."""
    smap = np.full(box_shape[0] * box_shape[1], fill, dtype=np.float32)
    smap[te_idx] = scores.astype(np.float32)
    return smap.reshape(box_shape)
