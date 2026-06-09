"""
baselines/detectors.py — Unified detector API.

All detectors accept numpy arrays and return (n_test,) score arrays
where higher values indicate more likely target.

Detectors implemented:
  AMF                  — Adaptive Matched Filter
  Reg-AMF              — Diagonal-loaded AMF
  CEM                  — Constrained Energy Minimization
  DSM additive         — Score-based LMP, additive model
  DSM replacement      — Score-based LMP, replacement model
  AMF replacement      — Gaussian closed-form, replacement model
  GMM-GLRT (additive)  — Grid-search GLRT with sklearn GMM
  GMM-GLRT (replace)   — Grid-search GLRT with Jacobian correction
  Exact-GLRT           — One-step GLRT for replacement (Vincent & Besson 2019)
  DLTD                 — Distribution-Level Target Detection (Ma et al. 2026)
  SMGLRT               — Segmented-Mixing GLRT (Ma et al. 2025)
"""

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dsm_model import compute_scores


# ===========================================================================
# Classical additive detectors
# ===========================================================================

def amf(test_data: np.ndarray, train_data: np.ndarray,
        s: np.ndarray) -> np.ndarray:
    """Adaptive Matched Filter: T(y) = sᵀΣ̂⁻¹(y−μ̂) / sqrt(sᵀΣ̂⁻¹s)."""
    mu    = train_data.mean(axis=0)
    Sigma = np.cov(train_data, rowvar=False)
    Sigma = (Sigma + Sigma.T) / 2
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv = np.clip(eigv, eigv.max() * 1e-12, None)
    Si   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    Si_s = Si @ s
    norm = np.sqrt(float(s @ Si_s) + 1e-12)
    return (test_data - mu) @ Si_s / norm


def reg_amf(test_data: np.ndarray, train_data: np.ndarray,
            s: np.ndarray, sigma: float) -> np.ndarray:
    """Diagonal-loaded AMF: T(y) = sᵀ(Σ̂+σ²I)⁻¹(y−μ̂) / sqrt(sᵀ(Σ̂+σ²I)⁻¹Σ̂(Σ̂+σ²I)⁻¹s)."""
    mu    = train_data.mean(axis=0)
    Sigma = np.cov(train_data, rowvar=False)
    Sigma = (Sigma + Sigma.T) / 2
    Sr    = Sigma + float(sigma) ** 2 * np.eye(len(s))
    Si    = np.linalg.inv(Sr)
    Si_s  = Si @ s
    denom = np.sqrt(float(Si_s @ Sigma @ Si_s) + 1e-12)
    return (test_data - mu) @ Si_s / denom


def cem(test_data: np.ndarray, train_data: np.ndarray,
        s: np.ndarray) -> np.ndarray:
    """
    Constrained Energy Minimization.
    Filter w = R⁻¹s / (sᵀR⁻¹s), R = XᵀX/n (autocorrelation, no mean subtraction).
    """
    n, d = train_data.shape
    R    = (train_data.T @ train_data) / n
    R    = (R + R.T) / 2 + 1e-8 * np.eye(d)
    eigv, eigvec = np.linalg.eigh(R)
    eigv = np.clip(eigv, eigv.max() * 1e-12, None)
    Ri   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    Ri_s = Ri @ s
    w    = Ri_s / (float(s @ Ri_s) + 1e-12)
    return test_data @ w


def dsm_additive(test_data: np.ndarray, train_data: np.ndarray,
                 model, s: np.ndarray) -> np.ndarray:
    """
    DSM-LMP for the additive target model.
    T(y) = −sᵀ(ψ̂(y) − z̄) / sqrt(sᵀĈ_ψs)
    """
    model.eval()
    z_train = compute_scores(model, train_data)        # (n, d)
    z_bar   = z_train.mean(axis=0)
    C_psi   = np.cov(z_train, rowvar=False)
    if C_psi.ndim == 0:
        C_psi = np.array([[float(C_psi)]])
    z_test  = compute_scores(model, test_data)         # (n_test, d)
    norm    = np.sqrt(max(float(s @ C_psi @ s), 1e-12))
    # return -((z_test - z_bar) @ s) / norm
    return -((z_test - z_bar) @ s) / norm


def gmm_glrt(test_data: np.ndarray, train_data: np.ndarray,
             s: np.ndarray, K: int = 3, theta_min: float = 0.0,
             theta_max: float = 2.0, theta_steps: int = 50) -> np.ndarray:
    """
    GMM-GLRT for the additive model — GRID-SEARCH variant.
    Fits a K-component GMM on train_data, then:
        T(y) = max_θ [log p(y − θs)] − log p(y)   (θ over [theta_min, theta_max])
    """
    gm = GaussianMixture(n_components=K, covariance_type='full',
                         n_init=5, random_state=0)
    gm.fit(train_data)
    log_p0   = gm.score_samples(test_data)            # (n_test,)
    thetas   = np.linspace(theta_min, theta_max, theta_steps)
    log_grid = np.column_stack([
        gm.score_samples(test_data - th * s) for th in thetas
    ])                                                  # (n_test, steps)
    return log_grid.max(axis=1) - log_p0


def gmm_glrt_oracle(test_data: np.ndarray, train_data: np.ndarray,
                    s: np.ndarray, theta: float, K: int = 3) -> np.ndarray:
    """
    GMM-GLRT for the additive model — ORACLE (known amplitude θ).
    Same GMM background as gmm_glrt(), but skips the grid search and
    plugs in the TRUE θ directly. Clairvoyant upper bound for grid GMM-GLRT.
        T(y) = log p_GMM(y − θ·s) − log p_GMM(y)

    ⚠️  UNUSED by the honest pipeline. This is a SEPARATE, WEAKER clairvoyant
        built on the single K-component GMM density of gmm_glrt() — it is NOT
        the Levin product-GMM oracle. The honest pipeline's oracle curve is
        GMMGLRTLevin.score(..., oracle_p=θ) in gmm_glrt_levin.py, which uses the
        same (stronger) product-GMM density as the honest 'GMM-Levin' curve and
        is therefore a valid upper bound. Do not use this function as the paper
        oracle — it can score BELOW the honest Levin curve.
    """
    gm = GaussianMixture(n_components=K, covariance_type='full',
                         n_init=5, random_state=0)
    gm.fit(train_data)
    log_p0 = gm.score_samples(test_data)
    log_p1 = gm.score_samples(test_data - theta * s)
    return log_p1 - log_p0


# ===========================================================================
# Replacement model detectors
# ===========================================================================

def dsm_replacement(test_data: np.ndarray, train_data: np.ndarray,
                    model, s: np.ndarray) -> np.ndarray:
    """
    DSM-LMP for the replacement target model y = (1−θ)w + θs.

    Paper formula:
        u_rep(y) = (ψ(y) − ψ̄)ᵀ(y−s) + d
        I_rep    = E[(ψ(w)−ψ̄)ᵀ(w−s))²] − d²
        T(y)     = u_rep(y) / sqrt(I_rep)

    The only adaptation: center ψ by its empirical mean ψ̄ = E_train[ψ(x)].
    For a well-trained model the Stein identity gives E[r] ≈ −d, so
    I_rep = E[r²] − d² = Var[r].
    """
    model.eval()
    psi_train = compute_scores(model, train_data)
    psi_test  = compute_scores(model, test_data)
    psi_bar   = psi_train.mean(axis=0)          # (D,) per-dim score mean
    d         = train_data.shape[1]

    r_train = ((psi_train - psi_bar) * (train_data - s)).sum(axis=1)
    I_rep   = max(float((r_train**2).mean()) - d**2, 1e-12)
    r_test  = ((psi_test  - psi_bar) * (test_data  - s)).sum(axis=1)
    return (r_test + d) / np.sqrt(I_rep)


def amf_replacement(test_data: np.ndarray, train_data: np.ndarray,
                    s: np.ndarray) -> np.ndarray:
    """
    Gaussian closed-form LMP for the replacement model.
        T(y) = [d − (y−μ̂)ᵀΣ̂⁻¹(y−s)] / sqrt(2d + (μ̂−s)ᵀΣ̂⁻¹(μ̂−s))
    """
    d      = train_data.shape[1]
    mu     = train_data.mean(axis=0)
    Sigma  = np.cov(train_data, rowvar=False)
    Sigma  = (Sigma + Sigma.T) / 2
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv   = np.clip(eigv, eigv.max() * 1e-12, None)
    Si     = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    denom  = np.sqrt(2 * d + float((mu - s) @ Si @ (mu - s)) + 1e-12)
    scores = (d - ((test_data - mu) @ Si * (test_data - s)).sum(axis=1)) / denom
    return scores


def gmm_glrt_replacement(test_data: np.ndarray, train_data: np.ndarray,
                          s: np.ndarray, K: int = 3,
                          theta_min: float = 1e-4, theta_max: float = 0.5,
                          theta_steps: int = 50) -> np.ndarray:
    """
    GMM-GLRT for the replacement model y = (1−θ)w + θs.
    Includes Jacobian correction −d·log(1−θ) which is mandatory.
        T(y) = max_θ [log p_GMM(ŷ) − d·log(1−θ)] − log p_GMM(y)
        where ŷ = (y − θs)/(1−θ)
    """
    d  = test_data.shape[1]
    means, Sigma, weights = _fit_gmm_shared_cov(train_data, K)
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv = np.clip(eigv, 1e-12, None)
    Si   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    log_det = np.sum(np.log(eigv))

    def _log_gmm(X):
        n = len(X)
        lc = np.zeros((n, K))
        for k in range(K):
            diff = X - means[k]
            maha = (diff @ Si * diff).sum(1)
            lc[:, k] = (np.log(weights[k] + 1e-300)
                        - 0.5 * (d * np.log(2 * np.pi) + log_det + maha))
        return np.logaddexp.reduce(lc.T, axis=0)

    log_p0   = _log_gmm(test_data)
    theta_hi = min(theta_max, 0.99)
    thetas   = np.linspace(theta_min, theta_hi, theta_steps)
    best     = np.full(len(test_data), -np.inf)
    for th in thetas:
        y_hat = (test_data - th * s) / (1.0 - th)
        llr   = _log_gmm(y_hat) - d * np.log(1.0 - th) - log_p0
        best  = np.maximum(best, llr)
    return best


def gmm_glrt_replacement_oracle(test_data: np.ndarray, train_data: np.ndarray,
                                 s: np.ndarray, theta: float,
                                 K: int = 3) -> np.ndarray:
    """
    GMM-GLRT for the replacement model — ORACLE (known amplitude θ).
        T(y) = log p_GMM((y−θs)/(1−θ)) − d·log(1−θ) − log p_GMM(y)

    ⚠️  UNUSED by the honest pipeline, and the paper now uses the additive model
        only. Like gmm_glrt_oracle(), this is a SEPARATE, WEAKER single-GMM
        clairvoyant — NOT the Levin product-GMM oracle. Do not use as the paper
        oracle; see GMMGLRTLevin.score(..., oracle_p=θ) in gmm_glrt_levin.py.
    """
    d  = test_data.shape[1]
    means, Sigma, weights = _fit_gmm_shared_cov(train_data, K)
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv = np.clip(eigv, 1e-12, None)
    Si   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    log_det = np.sum(np.log(eigv))

    def _log_gmm(X):
        n = len(X)
        lc = np.zeros((n, K))
        for k in range(K):
            diff = X - means[k]
            maha = (diff @ Si * diff).sum(1)
            lc[:, k] = (np.log(weights[k] + 1e-300)
                        - 0.5 * (d * np.log(2 * np.pi) + log_det + maha))
        return np.logaddexp.reduce(lc.T, axis=0)

    theta = float(min(max(theta, 1e-6), 0.99))
    y_hat = (test_data - theta * s) / (1.0 - theta)
    return _log_gmm(y_hat) - d * np.log(1.0 - theta) - _log_gmm(test_data)


def exact_glrt_replacement(test_data: np.ndarray, train_data: np.ndarray,
                            s: np.ndarray, alpha_max: float = 0.98,
                            n_grid: int = 400) -> np.ndarray:
    """
    Exact one-step GLRT for the replacement model
    (Vincent & Besson, CAMSAP 2019 — the 'Kelly counterpart' for replacement).

    log T = max_{α} [−N log(1−α) − (K+1)/2·log(1 + c·q1(α))]
            + (K+1)/2·log(1 + c·q0)
    """
    n, d  = train_data.shape
    K, N  = n, d
    c     = K / (K + 1.0)
    zbar  = train_data.mean(axis=0)
    Zc    = train_data - zbar
    S     = (Zc.T @ Zc + (Zc.T @ Zc).T) / 2.0
    eigv, eigvec = np.linalg.eigh(S)
    eigv  = np.clip(eigv, eigv.max() * 1e-12, None)
    Si    = eigvec @ np.diag(1.0 / eigv) @ eigvec.T

    yc    = test_data - zbar
    q0    = np.einsum('md,dk,mk->m', yc, Si, yc)
    h0    = 0.5 * (K + 1) * np.log1p(c * q0)
    smz   = s - zbar
    best  = np.zeros(len(test_data))
    for a in np.linspace(0.0, alpha_max, n_grid)[1:]:
        u    = (yc - a * smz) / (1.0 - a)
        q1   = np.einsum('md,dk,mk->m', u, Si, u)
        obj  = -N * np.log(1.0 - a) - 0.5 * (K + 1) * np.log1p(c * q1) + h0
        best = np.maximum(best, obj)
    return best


# ===========================================================================
# DLTD and SMGLRT (shared-covariance GMM helpers)
# ===========================================================================

def _fit_gmm_shared_cov(data: np.ndarray, K: int,
                         max_iter: int = 100, tol: float = 1e-6,
                         seed: int = 0):
    """
    K-component GMM with SHARED (tied) covariance, fit via sklearn.
    Uses covariance_type='tied' — identical to the custom EM above but
    backed by LAPACK/BLAS: 10-20x faster, and scale-aware regularisation
    (reg_covar scales with mean band variance, not an absolute 1e-6).
    """
    from sklearn.mixture import GaussianMixture
    d = data.shape[1]
    # reg_covar: sklearn's absolute floor — make it relative to data scale
    # so it doesn't vanish on raw HSI bands (variance ~ 1e4–1e6).
    rel_reg = 1e-3 * float(np.mean(np.var(data, axis=0)))
    gm = GaussianMixture(
        n_components=K,
        covariance_type='tied',
        max_iter=max_iter,
        tol=tol,
        reg_covar=max(rel_reg, 1e-6),
        random_state=seed,
        n_init=1,
    ).fit(data)
    means   = gm.means_                       # (K, d)
    Sigma   = gm.covariances_.copy()          # (d, d)  — single shared matrix
    Sigma   = (Sigma + Sigma.T) / 2           # enforce symmetry
    weights = gm.weights_                     # (K,)
    return means, Sigma, weights


def _dltd_score(test_data: np.ndarray, means: np.ndarray,
                Sigma: np.ndarray, weights: np.ndarray,
                s: np.ndarray) -> np.ndarray:
    """DLTD scoring given a pre-fitted shared-cov GMM."""
    K = len(means)
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv  = np.clip(eigv, 1e-12, None)
    Sis   = eigvec @ np.diag(1.0 / np.sqrt(eigv)) @ eigvec.T
    V     = (test_data - s) @ Sis.T
    U     = [(means[k] - test_data) @ Sis.T for k in range(K)]
    denom_log = np.array([np.log(weights[l] + 1e-300) - 0.5 * (U[l]**2).sum(1)
                          for l in range(K)])
    log_denom = np.logaddexp.reduce(denom_log, axis=0)
    log_g     = -0.5 * (V**2).sum(1) - log_denom
    w_terms   = np.array([np.log(weights[k] + 1e-300) - (U[k] * (U[k] + V)).sum(1)
                          for k in range(K)])
    return np.logaddexp.reduce(w_terms, axis=0) + log_g


def _smglrt_score(test_data: np.ndarray, means: np.ndarray,
                  Sigma: np.ndarray, weights: np.ndarray,
                  s: np.ndarray, n_segments: int = None) -> np.ndarray:
    """SMGLRT scoring given a pre-fitted shared-cov GMM."""
    K, d = len(means), test_data.shape[1]
    if n_segments is None:
        n_segments = max(1, d // 2)
    n_segments = min(n_segments, d)
    base, extra = d // n_segments, d % n_segments
    segs, start = [], 0
    for j in range(n_segments):
        end = start + base + (1 if j < extra else 0)
        segs.append(slice(start, end)); start = end

    Sigma = (Sigma + Sigma.T) / 2
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv    = np.clip(eigv, 1e-12, None)
    Si      = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    log_det = np.sum(np.log(eigv))

    n_test = len(test_data)
    lH0 = np.zeros((n_test, K))
    lH1 = np.zeros((n_test, K))
    for k in range(K):
        diff0      = test_data - means[k]
        lH0[:, k]  = (np.log(weights[k] + 1e-300)
                      - 0.5 * (d * np.log(2 * np.pi) + log_det
                                + (diff0 @ Si * diff0).sum(1)))
        alpha_k = np.zeros((n_test, n_segments))
        for j, seg in enumerate(segs):
            dt           = s[seg] - means[k][seg]
            Sdt          = Si[seg, :][:, seg] @ dt
            num_j        = (test_data[:, seg] - means[k][seg]) @ Sdt
            alpha_k[:, j] = np.clip(num_j / (float(dt @ Sdt) + 1e-12), 0.0, 1.0)
        mu1 = np.zeros_like(test_data)
        for j, seg in enumerate(segs):
            mu1[:, seg] = alpha_k[:, j:j+1] * s[seg] + (1 - alpha_k[:, j:j+1]) * means[k][seg]
        diff1      = test_data - mu1
        lH1[:, k]  = (np.log(weights[k] + 1e-300)
                      - 0.5 * (d * np.log(2 * np.pi) + log_det
                                + (diff1 @ Si * diff1).sum(1)))
    return np.logaddexp.reduce(lH1.T, axis=0) - np.logaddexp.reduce(lH0.T, axis=0)


def dltd(test_data: np.ndarray, train_data: np.ndarray,
         s: np.ndarray, K: int = 3) -> np.ndarray:
    """
    Distribution-Level Target Detection (Ma et al., IEEE GRSL 2026).
    Requires K ≥ 3 (K=1 gives a degenerate constant score).
    """
    means, Sigma, weights = _fit_gmm_shared_cov(train_data, K)
    return _dltd_score(test_data, means, Sigma, weights, s)


def smglrt(test_data: np.ndarray, train_data: np.ndarray,
           s: np.ndarray, K: int = 3,
           n_segments: int = None) -> np.ndarray:
    """
    Segmented-Mixing GLRT (Ma et al., IEEE JSTARS 2025).
    Requires K ≥ 3.
    """
    means, Sigma, weights = _fit_gmm_shared_cov(train_data, K)
    return _smglrt_score(test_data, means, Sigma, weights, s, n_segments)
