"""
data_utils.py — Data loading, normalization, PCA, sigma selection, and target planting.

All functions used by both single_class and multiclass experiments.
"""

import numpy as np
import scipy.io
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Loading & normalization
# ---------------------------------------------------------------------------

def load_and_normalize(path: str, mode: str = 'global'):
    """
    Load a .mat hyperspectral dataset and normalize to [0, 1].

    Parameters
    ----------
    path : str
        Path to .mat file containing 'data' (H×W×B) and 'map' (H×W).
    mode : str
        'global'   — subtract global min, divide by global range.
        'per_band' — per-band min/max normalization (each band independently).

    Returns
    -------
    data : np.ndarray  (H, W, B)  float64, values in [0, 1]
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
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}. Use 'global' or 'per_band'.")

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
                  seed: int = 0):
    """
    Plant target signatures into a random subset of test pixels.

    Parameters
    ----------
    test_bkg     : (n_test, d) — clean background test pixels
    s            : (d,) — unit-norm target signature
    amplitude    : target amplitude θ
    tgt_fraction : fraction of test pixels to contaminate
    model        : 'additive'    → y = w + θ·s
                   'replacement' → y = θ·s + (1-θ)·w
    seed         : RNG seed

    Returns
    -------
    test_data : (n_test, d) — test pixels with planted targets
    labels    : (n_test,)   — binary (1 = target pixel)
    tgt_idx   : indices of contaminated pixels
    """
    n_test  = len(test_bkg)
    n_tgt   = max(1, int(round(n_test * tgt_fraction)))
    labels  = np.zeros(n_test, dtype=int)
    tgt_idx = np.random.default_rng(seed).choice(n_test, size=n_tgt, replace=False)
    labels[tgt_idx] = 1

    test_data = test_bkg.copy()
    if model == 'additive':
        test_data[tgt_idx] += amplitude * s
    elif model == 'replacement':
        test_data[tgt_idx] = amplitude * s + (1.0 - amplitude) * test_bkg[tgt_idx]
    else:
        raise ValueError(f"Unknown target model: {model!r}. Use 'additive' or 'replacement'.")

    return test_data, labels, tgt_idx
