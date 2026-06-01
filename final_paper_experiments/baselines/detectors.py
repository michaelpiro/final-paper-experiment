"""
baselines/detectors.py — Unified detector API for all methods.

All detectors accept numpy arrays and return (n_test,) score arrays
where higher values indicate more likely target.

Detectors implemented here:
  AMF / Reg-AMF / Oracle        — classical (imported from existing code)
  DSM additive                  — score-based LMP (imported)
  DSM replacement               — NEW: replacement model LMP
  AMF replacement               — NEW: Gaussian replacement model
  LRao-IID                      — imported from dsm_model
  LRao-MLP                      — wrapper for adapted CNN-LRao
  GMM-GLRT                      — imported from gmm_iid_experiment
  DLTD                          — NEW: Distribution-Level Target Detection (Ma et al. 2026)
  SMGLRT                        — NEW: Segmented-Mixing GLRT (Ma et al. 2025)
"""

import sys
import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — import from the parent pythonProject directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gaussian_iid_experiment import (
    detector_oracle,
    detector_amf,
    detector_reg_amf,
    detector_dsm,
)
from gmm_iid_experiment import detector_gmm_glrt
from dsm_model import compute_scores, compute_lfi_detector_scores


# ===========================================================================
# Re-exports (existing detectors, pass-through)
# ===========================================================================

def amf(test_data, train_data, s):
    """Adaptive Matched Filter."""
    return detector_amf(test_data, train_data, s)


def reg_amf(test_data, train_data, s, sigma):
    """Diagonal-loaded AMF (Theorem 1, Reg-AMF)."""
    return detector_reg_amf(test_data, train_data, s, sigma)


def oracle(test_data, mu, Sigma, s, amplitude):
    """Oracle LRT with known Gaussian parameters."""
    return detector_oracle(test_data, mu, Sigma, s, amplitude)


def dsm_additive(test_data, train_data, model, s):
    """DSM-LMP for the additive target model."""
    return detector_dsm(test_data, train_data, model, s)


def gmm_glrt(test_data, train_data, s, K=3, theta_min=0.0,
             theta_max=2.0, theta_steps=50):
    """GMM-GLRT with amplitude grid search."""
    return detector_gmm_glrt(test_data, train_data, s, K, theta_min,
                              theta_max, theta_steps)


def lrao_iid(test_data, train_data, model, s, delta_theta=0.01):
    """LRao detector for i.i.d. data using trained ScoreNet + LFI statistic."""
    return compute_lfi_detector_scores(model, train_data, test_data, s, delta_theta)


def lrao_mlp(test_data, train_data, mlp_model, s, delta_theta=0.01):
    """LRao detector using the MLP-adapted Trafo model (see lrao_mlp.py)."""
    return compute_lfi_detector_scores(mlp_model, train_data, test_data, s, delta_theta)


# ===========================================================================
# NEW: Replacement model detectors
# ===========================================================================

def dsm_replacement(test_data: np.ndarray, train_data: np.ndarray,
                    model, s: np.ndarray) -> np.ndarray:
    """
    DSM-LMP statistic for the REPLACEMENT target model.

        T_rep(y) = (ψ(y)^T(y-s) - r̄) / sqrt(Î_rep)

    where r̄  = mean{ψ(wᵢ)^T(wᵢ-s)}  over training samples
          Î_rep = sample variance of {ψ(wᵢ)^T(wᵢ-s)}

    From the replacement model score identity:
        u_rep(y; 0) = ψ(y)^T(y-s) + d
    The +d term is constant under H0 so it cancels after centering.
    """
    model.eval()
    psi_train = compute_scores(model, train_data)          # (n, d)
    psi_test  = compute_scores(model, test_data)           # (n_test, d)

    # Per-sample scalar r_i = ψ(wᵢ)^T(wᵢ - s)
    r_train = (psi_train * (train_data - s)).sum(axis=1)   # (n,)
    r_bar   = r_train.mean()
    r_std   = r_train.std() + 1e-12

    r_test  = (psi_test * (test_data - s)).sum(axis=1)     # (n_test,)
    return (r_test - r_bar) / r_std


def amf_replacement(test_data: np.ndarray, train_data: np.ndarray,
                    s: np.ndarray) -> np.ndarray:
    """
    Gaussian-case LMP for the REPLACEMENT target model (closed form).

        T_rep(y) = [d - (y-μ̂)^T Σ̂^{-1}(y-s)]
                   / sqrt(2d + (μ̂-s)^T Σ̂^{-1}(μ̂-s))

    Derived in main2.tex Section 2.3.
    """
    d      = train_data.shape[1]
    mu_hat = train_data.mean(axis=0)
    S      = np.cov(train_data, rowvar=False)
    S      = (S + S.T) / 2
    eigv, eigvec = np.linalg.eigh(S)
    eigv   = np.clip(eigv, eigv.max() * 1e-12, None)
    S_inv  = eigvec @ np.diag(1.0 / eigv) @ eigvec.T

    denom = np.sqrt(2 * d + (mu_hat - s) @ S_inv @ (mu_hat - s) + 1e-12)

    # (n_test, d) → scalar per pixel
    diff_test = test_data - s                              # (n_test, d)
    quad      = (test_data - mu_hat) @ S_inv              # (n_test, d)
    scores    = (d - (quad * diff_test).sum(axis=1)) / denom
    return scores


# ===========================================================================
# NEW: DLTD — Distribution-Level Target Detection (Ma et al., 2026)
# ===========================================================================

def _fit_gmm_shared_cov(data: np.ndarray, K: int,
                         max_iter: int = 100, tol: float = 1e-6,
                         seed: int = 0):
    """
    EM for GMM with K components and SHARED covariance matrix.

    E-step: γ_ik = π_k N(xᵢ; μ_k, Σ) / Σ_l π_l N(xᵢ; μ_l, Σ)
    M-step: μ_k = Σᵢ γᵢₖ xᵢ / Σᵢ γᵢₖ
            Σ   = (1/n) Σᵢ Σₖ γᵢₖ (xᵢ-μₖ)(xᵢ-μₖ)^T   (shared, pooled)
            π_k = (1/n) Σᵢ γᵢₖ

    Returns
    -------
    means  : (K, d)
    Sigma  : (d, d)  shared covariance
    weights: (K,)
    """
    n, d  = data.shape
    rng   = np.random.default_rng(seed)

    # K-means initialisation for means
    idx   = rng.choice(n, K, replace=False)
    means = data[idx].copy().astype(float)

    # Init weights and covariance
    weights = np.ones(K) / K
    Sigma   = np.cov(data, rowvar=False).astype(float)
    Sigma   = (Sigma + Sigma.T) / 2 + 1e-6 * np.eye(d)

    log_lik_prev = -np.inf

    for _ in range(max_iter):
        # ---- E-step ----
        # Compute log N(xᵢ; μ_k, Σ) for all i, k
        S_inv  = np.linalg.inv(Sigma)
        log_det = np.linalg.slogdet(Sigma)[1]
        log_gamma = np.zeros((n, K))
        for k in range(K):
            diff         = data - means[k]          # (n, d)
            maha         = (diff @ S_inv * diff).sum(axis=1)  # (n,)
            log_gamma[:, k] = (np.log(weights[k] + 1e-300)
                                - 0.5 * (d * np.log(2 * np.pi) + log_det + maha))

        log_lik = np.logaddexp.reduce(log_gamma, axis=1).mean()

        # Normalise to get posteriors
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)                   # (n, K)

        # ---- M-step ----
        nk = gamma.sum(axis=0) + 1e-10              # (K,)
        weights = nk / nk.sum()
        means   = (gamma.T @ data) / nk[:, None]    # (K, d)

        # Shared covariance (pooled)
        Sigma = np.zeros((d, d))
        for k in range(K):
            diff   = data - means[k]                # (n, d)
            Sigma += (gamma[:, k:k+1] * diff).T @ diff
        Sigma /= n
        Sigma  = (Sigma + Sigma.T) / 2 + 1e-6 * np.eye(d)

        if abs(log_lik - log_lik_prev) < tol:
            break
        log_lik_prev = log_lik

    return means, Sigma, weights


def dltd(test_data: np.ndarray, train_data: np.ndarray,
         s: np.ndarray, K: int = 3) -> np.ndarray:
    """
    Distribution-Level Target Detection (DLTD).
    Ma et al., "Distribution-Level Hyperspectral Target Detection
    Under Mixture of Gaussian", IEEE GRSL 2026.

    Algorithm:
      1. Fit K-component GMM with shared covariance Σ on training data.
      2. For each pixel xᵢ and component k, define:
            u_ik = Σ^{-1/2}(μ_k - xᵢ)
            v_i  = Σ^{-1/2}(xᵢ - s)
      3. Detection score (Eq. 6–8 in paper):
            d_i = w_i · g_i
            w_i = Σ_k π_k exp{-u_ik^T(u_ik + v_i)}
            g_i = exp{-½ v_i^T v_i} / Σ_l π_l exp{-½ u_il^T u_il}

    Parameters
    ----------
    test_data  : (n_test, d)
    train_data : (n_train, d)
    s          : (d,) target signature (unit-norm)
    K          : number of Gaussian components

    Returns
    -------
    scores : (n_test,)  — higher = more likely target
    """
    means, Sigma, weights = _fit_gmm_shared_cov(train_data, K)

    # Σ^{-1/2} via eigendecomposition
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv         = np.clip(eigv, 1e-12, None)
    Sigma_invsqrt = eigvec @ np.diag(1.0 / np.sqrt(eigv)) @ eigvec.T  # (d, d)

    n_test = len(test_data)
    scores = np.zeros(n_test)

    # Precompute v_i for all test pixels: (n_test, d)
    V = (test_data - s) @ Sigma_invsqrt.T       # v_i = Σ^{-1/2}(xᵢ - s)

    # Precompute u_ik for all k: U[k] is (n_test, d)
    U = []
    for k in range(K):
        # u_ik = Σ^{-1/2}(μ_k - xᵢ)  for all i
        U.append((means[k] - test_data) @ Sigma_invsqrt.T)   # (n_test, d)

    # Denominator: Σ_l π_l exp{-½ ||u_il||²}
    denom_log = np.array([
        np.log(weights[l] + 1e-300) - 0.5 * (U[l] ** 2).sum(axis=1)
        for l in range(K)
    ])   # (K, n_test)
    log_denom = np.logaddexp.reduce(denom_log, axis=0)   # (n_test,)

    # Numerator w_i · g_i  (compute in log space for stability)
    # log(d_i) = log(w_i) + log(g_i)
    # log(g_i) = -½ v_i^T v_i - log_denom
    log_g = -0.5 * (V ** 2).sum(axis=1) - log_denom      # (n_test,)

    # w_i = Σ_k π_k exp{-u_ik^T(u_ik + v_i)}
    w_terms = np.array([
        np.log(weights[k] + 1e-300) - (U[k] * (U[k] + V)).sum(axis=1)
        for k in range(K)
    ])   # (K, n_test)
    log_w = np.logaddexp.reduce(w_terms, axis=0)          # (n_test,)

    scores = log_w + log_g    # log(d_i)
    return scores


# ===========================================================================
# NEW: SMGLRT — Segmented-Mixing GLRT (Ma et al., 2025)
# ===========================================================================

def smglrt(test_data: np.ndarray, train_data: np.ndarray,
           s: np.ndarray, K: int = 3,
           n_segments: int = None) -> np.ndarray:
    """
    Segmented-Mixing-based Generalized Likelihood Ratio Test (SMGLRT).
    Ma et al., "Generalized Likelihood Ratio Test for Hyperspectral
    Subpixel Target Detection Based on Segmented Mixing Model",
    IEEE JSTARS 2025.

    The SMM assigns per-segment mixing coefficients:
        H0: x = b
        H1: x = BlockDiag(α¹I_{p1},...,α^m I_{pm}) t
              + BlockDiag(β¹I_{p1},...,β^m I_{pm}) b

    Background b obeys a GMM with K components and shared covariance Σ.

    For each test pixel yᵢ and component k, the per-segment MLE of α^j is:
        α̂^j_k = (t^j - μ_k^j)^T Σ_jj^{-1} (yᵢ^j - μ_k^j)
                 / ||t^j - μ_k^j||²_{Σ_jj^{-1}}
    clamped to [0, 1], then β^j = 1 - α̂^j.

    The GLRT statistic under GMM background:
        T(y) = log Σ_k π_k N(y; α̂_k ⊙ t + (1-α̂_k) ⊙ μ_k, Σ)
             - log Σ_k π_k N(y; μ_k, Σ)

    Note: for post-PCA data with d ≤ 5, segments with p_j = 1 make
    the test per-dimension independent. When n_segments = 1, this
    reduces to the standard CMM-GLRT with scalar α.

    Parameters
    ----------
    test_data   : (n_test, d)
    train_data  : (n_train, d)
    s           : (d,) target signature
    K           : number of GMM components
    n_segments  : number of spectral segments (default: d // 2 or 1)

    Returns
    -------
    scores : (n_test,)
    """
    d = test_data.shape[1]
    if n_segments is None:
        n_segments = max(1, d // 2)
    n_segments = min(n_segments, d)

    # Build segment index ranges
    base  = d // n_segments
    extra = d % n_segments
    segs  = []
    start = 0
    for j in range(n_segments):
        end = start + base + (1 if j < extra else 0)
        segs.append(slice(start, end))
        start = end

    # Fit GMM with shared covariance
    means, Sigma, weights = _fit_gmm_shared_cov(train_data, K)

    # Precompute Σ inverse and log-det
    Sigma  = (Sigma + Sigma.T) / 2
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv   = np.clip(eigv, 1e-12, None)
    S_inv  = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    log_det = np.sum(np.log(eigv))

    n_test = len(test_data)
    log_lik_H0 = np.zeros((n_test, K))
    log_lik_H1 = np.zeros((n_test, K))

    for k in range(K):
        # H0 log-likelihood
        diff0        = test_data - means[k]             # (n_test, d)
        maha0        = (diff0 @ S_inv * diff0).sum(1)   # (n_test,)
        log_lik_H0[:, k] = (np.log(weights[k] + 1e-300)
                             - 0.5 * (d * np.log(2 * np.pi) + log_det + maha0))

        # Per-segment MLE of α
        alpha_k = np.zeros((n_test, n_segments))
        for j, seg in enumerate(segs):
            t_j  = s[seg]                               # (p_j,)
            mu_j = means[k][seg]                        # (p_j,)
            y_j  = test_data[:, seg]                    # (n_test, p_j)

            Sj_inv = S_inv[seg, :][:, seg]              # (p_j, p_j)
            dt     = t_j - mu_j                         # (p_j,)
            Sdt    = Sj_inv @ dt                        # (p_j,)
            denom_j = dt @ Sdt + 1e-12

            # α̂^j_k = (t_j - μ_j)^T Σ_jj^{-1} (y_j - μ_j) / ||t_j - μ_j||²_{Σ^{-1}}
            num_j = (y_j - mu_j) @ Sdt                 # (n_test,)
            alpha_k[:, j] = np.clip(num_j / denom_j, 0.0, 1.0)

        # H1 mean: per-segment α̂^j_k · t^j + (1-α̂^j_k) · μ_k^j
        mu_H1 = np.zeros_like(test_data)               # (n_test, d)
        for j, seg in enumerate(segs):
            aj = alpha_k[:, j:j+1]                     # (n_test, 1)
            mu_H1[:, seg] = (aj * s[seg]
                             + (1 - aj) * means[k][seg])

        diff1        = test_data - mu_H1                # (n_test, d)
        maha1        = (diff1 @ S_inv * diff1).sum(1)   # (n_test,)
        log_lik_H1[:, k] = (np.log(weights[k] + 1e-300)
                             - 0.5 * (d * np.log(2 * np.pi) + log_det + maha1))

    # GLRT: log Σ_k exp(log_lik_H1_k) - log Σ_k exp(log_lik_H0_k)
    scores = (np.logaddexp.reduce(log_lik_H1.T, axis=0)
              - np.logaddexp.reduce(log_lik_H0.T, axis=0))
    return scores


# ===========================================================================
# Convenience: run all detectors and return dict of scores
# ===========================================================================

ADDITIVE_DETECTORS = ['AMF', 'DSM-add', 'LRao-IID', 'DLTD', 'SMGLRT']
REPLACEMENT_DETECTORS = ['AMF-rep', 'DSM-rep', 'DLTD', 'SMGLRT']


def run_all(test_data, train_data, s, dsm_model=None, lrao_model=None,
            reg_sigma=None, K=3, target_model='additive',
            include_gmm_glrt=False):
    """
    Run all applicable detectors and return {label: scores} dict.

    Parameters
    ----------
    target_model : 'additive' or 'replacement'
    """
    results = {}

    if target_model == 'additive':
        results['AMF'] = amf(test_data, train_data, s)
        if reg_sigma is not None:
            results[f'Reg-AMF σ={reg_sigma}'] = reg_amf(test_data, train_data, s, reg_sigma)
        if dsm_model is not None:
            results['DSM'] = dsm_additive(test_data, train_data, dsm_model, s)
        if lrao_model is not None:
            results['LRao-IID'] = lrao_iid(test_data, train_data, lrao_model, s)
        if include_gmm_glrt:
            results['GMM-GLRT'] = gmm_glrt(test_data, train_data, s, K=K)
        results['DLTD']   = dltd(test_data, train_data, s, K=K)
        results['SMGLRT'] = smglrt(test_data, train_data, s, K=K)

    elif target_model == 'replacement':
        results['AMF-rep'] = amf_replacement(test_data, train_data, s)
        if dsm_model is not None:
            results['DSM-rep'] = dsm_replacement(test_data, train_data, dsm_model, s)
        if lrao_model is not None:
            results['LRao-IID-rep'] = lrao_iid(test_data, train_data, lrao_model, s)
        results['DLTD']   = dltd(test_data, train_data, s, K=K)
        results['SMGLRT'] = smglrt(test_data, train_data, s, K=K)

    return results
