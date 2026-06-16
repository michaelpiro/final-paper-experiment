"""
detectors.py — detection statistics. Each returns a (n_test,) score array
(higher = more likely target).

Classical
  amf                     — AMF (global adaptive matched filter)
  amf_local               — AMF on a per-pixel local k x k sample covariance
  gmm_glrt_levin_additive — Gaussian-mixture GLRT (Levin 2019), fill factor by grid search

Score-based
  dsm_additive            — DART / L-DART additive-LMP statistic (uses a trained ScoreNet)

(DARTS / DARTS-CFAR scoring lives in models.py / spatial.py; the learned-Rao
LRao / L-LRao statistic lives in models.compute_lfi_detector_scores_mode2.)
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from .models import compute_scores


# ---------------------------------------------------------------------------
# AMF (global)
# ---------------------------------------------------------------------------

def amf(test_data: np.ndarray, train_data: np.ndarray, s: np.ndarray,
        eig_floor: float = 1e-18) -> np.ndarray:
    """Adaptive Matched Filter: T(y) = s^T Sigma^{-1} (y - mu) / sqrt(s^T Sigma^{-1} s).

    eig_floor : relative eigenvalue floor (x lambda_max) on the background
                covariance before inversion. Default 1e-18 = numerical floor only
                (pure AMF). Pass a larger value to regularize at small n.
    """
    mu    = train_data.mean(axis=0)
    Sigma = np.cov(train_data, rowvar=False)
    Sigma = (Sigma + Sigma.T) / 2
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv = np.clip(eigv, eigv.max() * float(eig_floor), None)
    Si   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    Si_s = Si @ s
    norm = np.sqrt(float(s @ Si_s) + 1e-18)
    return (test_data - mu) @ Si_s / norm


# ---------------------------------------------------------------------------
# AMF-local (per-pixel local sample covariance)
# ---------------------------------------------------------------------------

@torch.no_grad()
def amf_local(test_pix: np.ndarray, test_nbr: np.ndarray, s: np.ndarray,
              device: str = 'cpu', loading: float = 1e-16,
              chunk: int = 1024) -> np.ndarray:
    """AMF on a per-pixel local SCM drawn from each pixel's k x k window.

    For test pixel y_i with neighbours X_i (K = k*k - 1 pixels in D bands):
        mu_i    = mean(X_i)
        Sigma_i = cov(X_i) + loading * mean(diag(Sigma_i)) * I   (diagonal loading)
        T_i     = s^T Sigma_i^{-1} (y_i - mu_i) / sqrt(s^T Sigma_i^{-1} s)

    Diagonal loading keeps the rank-deficient local SCM invertible (with k=5 the
    window holds only 24 samples while D ~ 103). loading=0 -> numerical floor only.
    Batched/chunked on `device`.
    """
    N, K, D = test_nbr.shape
    dev  = torch.device(device)
    y    = torch.as_tensor(test_pix, dtype=torch.float32, device=dev)
    nbr  = torch.as_tensor(test_nbr, dtype=torch.float32, device=dev)
    s_t  = torch.as_tensor(s,        dtype=torch.float32, device=dev)
    eyeD = torch.eye(D, device=dev)
    out  = torch.empty(N, dtype=torch.float32)

    for i0 in range(0, N, chunk):
        yb  = y[i0:i0 + chunk]                 # (B, D)
        nb  = nbr[i0:i0 + chunk]               # (B, K, D)
        B   = yb.shape[0]
        s_b = s_t.expand(B, D).unsqueeze(-1)   # (B, D, 1)
        mu  = nb.mean(dim=1)                   # (B, D)
        cen = nb - mu.unsqueeze(1)             # (B, K, D)
        Sigma = cen.transpose(1, 2) @ cen / max(K - 1, 1)
        load_s = loading * Sigma.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-8)
        Sigma = Sigma + load_s.view(B, 1, 1) * eyeD
        Sinv_s = torch.linalg.solve(Sigma, s_b).squeeze(-1)
        num = ((yb - mu) * Sinv_s).sum(-1)
        den = (s_t * Sinv_s).sum(-1).clamp_min(1e-12).sqrt()
        out[i0:i0 + chunk] = (num / den).cpu()
    return out.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# DART / L-DART additive-LMP (score-based)
# ---------------------------------------------------------------------------

def dsm_additive(test_data: np.ndarray, train_data: np.ndarray,
                 model, s: np.ndarray) -> np.ndarray:
    """DART / L-DART additive-LMP statistic with a trained ScoreNet:
        T(y) = -s^T (psi(y) - psi_bar) / sqrt(s^T C_psi s)."""
    model.eval()
    z_train = compute_scores(model, train_data)
    z_bar   = z_train.mean(axis=0)
    C_psi   = np.cov(z_train, rowvar=False)
    if C_psi.ndim == 0:
        C_psi = np.array([[float(C_psi)]])
    z_test  = compute_scores(model, test_data)
    norm    = np.sqrt(max(float(s @ C_psi @ s), 1e-12))
    return -((z_test - z_bar) @ s) / norm


# ---------------------------------------------------------------------------
# GMM-Levin: cluster-based product-of-GMMs GLRT (Levin 2019)
# ---------------------------------------------------------------------------

def _eigen_subsets(eigvals, cond_tol=1e3, max_dim=5):
    """Partition descending eigenvalue indices into consecutive, well-conditioned,
    low-dimensional subsets (each subset feeds one marginal GMM)."""
    subsets, cur = [], [0]
    for i in range(1, len(eigvals)):
        cond = eigvals[cur[0]] / max(eigvals[i], 1e-12)
        if cond <= cond_tol and len(cur) < max_dim:
            cur.append(i)
        else:
            subsets.append(cur); cur = [i]
    subsets.append(cur)
    return subsets


class ProductGMM:
    """Background density f0(z) = sum_i log f_{S_i}(z_{S_i})."""

    def __init__(self, subsets, gmms):
        self.subsets = subsets
        self.gmms    = gmms

    def logpdf(self, Z):
        out = np.zeros(len(Z), dtype=np.float64)
        for sub, gm in zip(self.subsets, self.gmms):
            if gm is not None:
                out += gm.score_samples(Z[:, sub])
        return out


def fit_product_gmm(Z, cond_tol=1e3, max_dim=5, k_max=5, reg_covar=1e-6, seed=0):
    """Fit a per-subset GMM (EM, model order by AIC) on PCA scores Z (n, D)."""
    var     = Z.var(axis=0)
    subsets = _eigen_subsets(var, cond_tol, max_dim)
    gmms    = []
    for sub in subsets:
        Zs = Z[:, sub]
        best_gmm, best_aic = None, np.inf
        for k in range(1, k_max + 1):
            if len(Zs) < 2 * k:
                break
            try:
                gm = GaussianMixture(n_components=k, covariance_type='full',
                                     reg_covar=reg_covar, random_state=seed,
                                     max_iter=200, n_init=1)
                gm.fit(Zs)
                aic = gm.aic(Zs)
            except Exception:
                continue
            if aic < best_aic:
                best_aic, best_gmm = aic, gm
        gmms.append(best_gmm)
    return ProductGMM(subsets, gmms)


class GMMGLRTLevin:
    """Levin GMM-GLRT: fit a product-of-GMMs background once, score the additive
    GLRT with the fill factor p estimated per pixel by a 1-D grid search."""

    def __init__(self, cond_tol=1e3, max_dim=5, k_max=5,
                 rcond=1e-8, reg_covar=1e-6, seed=0):
        self.cond_tol  = cond_tol
        self.max_dim   = max_dim
        self.k_max     = k_max
        self.rcond     = rcond
        self.reg_covar = reg_covar
        self.seed      = seed

    def fit(self, train):
        self.mu = train.mean(axis=0)
        Xc      = train - self.mu
        cov     = np.cov(Xc, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        keep = evals > self.rcond * max(evals[0], 1e-12)
        self.Psi   = evecs[:, keep]
        self.evals = evals[keep]
        Ztr = Xc @ self.Psi
        self.pgmm = fit_product_gmm(Ztr, self.cond_tol, self.max_dim,
                                    self.k_max, self.reg_covar, self.seed)
        return self

    def score(self, test, t, p_steps=50, p_max=1.0, oracle_p=None):
        """Additive GLRT: T(x) = max_p [log f0(x - p t)] - log f0(x)."""
        z     = (test - self.mu) @ self.Psi
        logf0 = self.pgmm.logpdf(z)
        psi_t = t @ self.Psi
        p_grid = (np.array([float(oracle_p)]) if oracle_p is not None
                  else np.linspace(0.0, p_max, p_steps))
        best = np.full(len(test), -np.inf)
        for p in p_grid:
            best = np.maximum(best, self.pgmm.logpdf(z - p * psi_t) - logf0)
        return best


def gmm_glrt_levin_additive(test, train, t, p_steps=50, p_max=1.0, **kw):
    """GMM-GLRT (Levin 2019), additive model. Fill factor estimated by grid search."""
    return GMMGLRTLevin(**kw).fit(train).score(test, t, p_steps, p_max)
