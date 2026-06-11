"""
data_utils.py — Data loading, normalization, PCA, sigma selection, and target planting.

Pipeline (always this order):
  1. load_and_normalize(path, mode)      → (H,W,B) image normalized to [0,1]
  2. pca_reduce(all_flat, ...)            → PCA fit on ALL image pixels, then transform
  3. compute_target_signature(tgt_pca)   → unit-norm mean in PCA space
  4. plant_targets(test_bkg_pca, s, ...) → plant in PCA space (after norm+PCA)

All functions used by both single_class and multiclass experiments.
"""

import os
import pickle
import numpy as np
import scipy.io
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Loading & normalization
# ---------------------------------------------------------------------------

def load_and_normalize(path: str, mode: str = 'none'):
    """
    Load a .mat hyperspectral dataset and normalize to [0, 1].

    Parameters
    ----------
    path : str
        Path to .mat file containing 'data' (H×W×B) and 'map' (H×W).
    mode : str
        'global'        — subtract global min, divide by global range.
        'per_band'      — per-band min/max normalization (each band independently).
        'per_band_max'  — divide each band by its maximum only (zero stays zero).
        'none'          — no normalization; return raw sensor values as float64.

    Returns
    -------
    data : np.ndarray  (H, W, B)  float64
    gt   : np.ndarray  (H, W)     int
    """
    mat  = scipy.io.loadmat(path)
    data = mat['data'].astype(np.float64)
    gt   = mat['map'].astype(int)

    if mode == 'global':
        lo, hi = data.min(), data.max()
        data = (data - lo) / (hi - lo + 1e-12)
    elif mode == 'per_band':
        lo = data.min(axis=(0, 1), keepdims=True)   # (1, 1, B)
        hi = data.max(axis=(0, 1), keepdims=True)
        data = (data - lo) / (hi - lo + 1e-12)
    elif mode == 'per_band_max':
        hi = data.max(axis=(0, 1), keepdims=True)   # (1, 1, B)
        data = data / (hi + 1e-12)                  # zero stays zero
    elif mode == 'global_max':
        hi = data.max()                             # scalar
        data = data / (hi + 1e-12)                  # zero stays zero
    elif mode == 'none':
        pass   # return raw values unchanged
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}. "
                         f"Use 'global', 'per_band', 'per_band_max', or 'none'.")

    return data, gt


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_reduce(all_flat: np.ndarray,
               train: np.ndarray,
               test: np.ndarray,
               tgt_pixels: np.ndarray,
               n_components: int):
    """
    Fit PCA on all image pixels, transform train / test / target pixels.

    Parameters
    ----------
    all_flat   : (N_total, B) — all image pixels (fit domain)
    train      : (n_train, B)
    test       : (n_test,  B)
    tgt_pixels : (n_tgt,   B)
    n_components : int

    Returns
    -------
    pca        : fitted sklearn PCA object
    train_pca  : (n_train, n_components)
    test_pca   : (n_test,  n_components)
    tgt_pca    : (n_tgt,   n_components)
    """
    pca = PCA(n_components=n_components)
    pca.fit(all_flat)
    return pca, pca.transform(train), pca.transform(test), pca.transform(tgt_pixels)


# ---------------------------------------------------------------------------
# Target signature
# ---------------------------------------------------------------------------

def compute_target_signature(tgt_pca: np.ndarray) -> np.ndarray:
    """
    Target signature = mean of target pixels in PCA space, unit-normalized.

    Parameters
    ----------
    tgt_pca : (n_tgt, d)

    Returns
    -------
    s : (d,)  unit-norm vector
    """
    s_raw = tgt_pca.mean(axis=0)
    return s_raw / (np.linalg.norm(s_raw) + 1e-12)


# ---------------------------------------------------------------------------
# Sigma selection from data
# ---------------------------------------------------------------------------

def compute_sigma_from_data(train_data: np.ndarray, rho: float = 0.01) -> float:
    """
    Data-driven DSM noise level: σ² = ρ · (1/d) · tr(Σ̂).

    Follows the KDE bandwidth analogy (Silverman 1986, Scott 1992):
    σ is set as sqrt(ρ) × empirical RMS scale of the data.

    Parameters
    ----------
    train_data : (n, d)
    rho        : noise-to-signal ratio (default 0.01 → σ = 10% of RMS scale)

    Returns
    -------
    sigma : float
    """
    s2 = np.mean(np.var(train_data, axis=0))   # (1/d) tr(Σ̂)
    return float(np.sqrt(rho * s2))


def compute_sigma_diag_from_data(train_data: np.ndarray,
                                 rho: float = 0.01) -> np.ndarray:
    """Data-driven *diagonal* DSM noise level: σ_b = sqrt(ρ · Var(x_b)).

    The per-band generalization of compute_sigma_from_data: instead of using
    one pooled scale (the average band variance), each band gets noise
    proportional to its OWN std.  The scalar rule is exactly this with every
    band's variance replaced by the mean.

    Parameters
    ----------
    train_data : (n, d)
    rho        : noise-to-signal ratio (default 0.01 → σ_b = 10% of band std)

    Returns
    -------
    sigma : (d,) float32 array of per-band noise stds.
    """
    var = np.var(train_data, axis=0)            # (d,)
    return np.sqrt(rho * var).astype(np.float32)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_background(bkg_pca: np.ndarray,
                     n_train: int,
                     n_test: int,
                     n_val: int = 0,
                     seed: int = 42):
    """
    Shuffle and split background pixels into train / val / test.

    Parameters
    ----------
    bkg_pca : (N, d) — all background pixels in PCA space
    n_train, n_test, n_val : sizes (n_val=0 means no validation split)
    seed : RNG seed for shuffle

    Returns
    -------
    train, val, test — np.ndarray slices  (val is empty array if n_val=0)
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(bkg_pca))
    rng.shuffle(idx)
    bkg_pca = bkg_pca[idx]

    assert n_train + n_val + n_test <= len(bkg_pca), \
        f"Not enough background pixels: need {n_train+n_val+n_test}, have {len(bkg_pca)}"

    train = bkg_pca[:n_train]
    val   = bkg_pca[n_train: n_train + n_val] if n_val > 0 else np.empty((0, bkg_pca.shape[1]))
    test  = bkg_pca[n_train + n_val: n_train + n_val + n_test]
    return train, val, test


# ---------------------------------------------------------------------------
# Target planting
# ---------------------------------------------------------------------------

def plant_targets(test_bkg: np.ndarray,
                  s: np.ndarray,
                  amplitude: float,
                  tgt_fraction: float,
                  model: str = 'additive',
                  seed: int = 0,
                  spatial_shape: tuple = None,
                  edge_guard: int = 0):
    """
    Plant target signatures into a random subset of test pixels.

    Parameters
    ----------
    test_bkg       : (n_test, d) — clean background test pixels (row-major flat)
    s              : (d,) — unit-norm target signature
    amplitude      : target amplitude θ
    tgt_fraction   : fraction of test pixels to contaminate
    model          : 'additive'    → y = w + θ·s
                     'replacement' → y = θ·s + (1-θ)·w
    seed           : RNG seed
    spatial_shape  : (H, W) of the test box — required when edge_guard > 0
    edge_guard     : exclude pixels within this many pixels of any box edge
                     from the candidate pool. Use ≥ k//2 to avoid boundary
                     pixels where local-SCM detectors get reflect-padded
                     neighbors (which inflate false alarms). Default 0 = off.

    Returns
    -------
    test_data : (n_test, d) — test pixels with planted targets
    labels    : (n_test,)   — binary (1 = target pixel)
    tgt_idx   : indices of contaminated pixels (into the flat n_test array)
    """
    n_test = len(test_bkg)

    # Build the candidate pool (all pixels, or interior-only when edge_guard>0).
    if edge_guard > 0 and spatial_shape is not None:
        H, W = spatial_shape
        g = int(edge_guard)
        rows = np.arange(n_test) // W
        cols = np.arange(n_test) %  W
        interior = np.where(
            (rows >= g) & (rows < H - g) &
            (cols >= g) & (cols < W - g)
        )[0]
        if len(interior) == 0:
            interior = np.arange(n_test)   # fallback: guard too large
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
        raise ValueError(f"Unknown target model: {model!r}. Use 'additive' or 'replacement'.")

    return test_data, labels, tgt_idx


# ---------------------------------------------------------------------------
# Pre-processing save / load  (used by pretrain + fast experiment)
# ---------------------------------------------------------------------------

def save_preprocessing(save_dir: str,
                        pca: PCA,
                        norm_mode: str,
                        vmin: np.ndarray,
                        ranges: np.ndarray,
                        gt_flat: np.ndarray,
                        class_pixels_pca: dict):
    """
    Save everything needed to reconstruct the normalized-PCA representation.

    Parameters
    ----------
    save_dir          : directory to write into
    pca               : fitted sklearn PCA object
    norm_mode         : 'global' or 'per_band'
    vmin, ranges      : normalization parameters (per-band arrays)
    gt_flat           : (N,) ground-truth labels for all pixels
    class_pixels_pca  : {class_id: (n_pixels, pca_dim) array}
    """
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'norm_params.npz'),
             norm_mode=np.array([norm_mode]),
             vmin=vmin, ranges=ranges)
    with open(os.path.join(save_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    np.save(os.path.join(save_dir, 'gt_flat.npy'), gt_flat)
    for cls_id, pixels in class_pixels_pca.items():
        np.save(os.path.join(save_dir, f'cls{cls_id}_pca.npy'), pixels)


def load_preprocessing(save_dir: str):
    """
    Load pre-saved normalization + PCA artifacts.

    Returns
    -------
    pca, norm_mode, vmin, ranges, gt_flat, class_pixels_pca
    """
    params = np.load(os.path.join(save_dir, 'norm_params.npz'), allow_pickle=True)
    norm_mode = str(params['norm_mode'][0])
    vmin      = params['vmin']
    ranges    = params['ranges']
    with open(os.path.join(save_dir, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)
    gt_flat = np.load(os.path.join(save_dir, 'gt_flat.npy'))

    # Load all available per-class PCA arrays
    class_pixels_pca = {}
    for fname in os.listdir(save_dir):
        if fname.startswith('cls') and fname.endswith('_pca.npy'):
            cls_id = int(fname[3:-8])
            class_pixels_pca[cls_id] = np.load(os.path.join(save_dir, fname))

    return pca, norm_mode, vmin, ranges, gt_flat, class_pixels_pca


def normalize_with_params(data: np.ndarray,
                           vmin: np.ndarray,
                           ranges: np.ndarray) -> np.ndarray:
    """Apply pre-computed normalization parameters to raw data."""
    return (data - vmin) / ranges
