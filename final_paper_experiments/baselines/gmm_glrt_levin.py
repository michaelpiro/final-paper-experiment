"""
gmm_glrt_levin.py — Cluster-based GMM-GLRT detector (ML estimation of fill factor).

Implements the detector of:
    I. Levin, T. Hershkovitz, S. Rotman, "Hyperspectral target detection
    using cluster-based probability models implemented in a generalized
    likelihood ratio test", Proc. SPIE 11155, 2019.

Core idea
---------
The background density is estimated as a PRODUCT of low-dimensional GMMs
over consecutive-eigenvalue subsets of the PCA-rotated data (a probabilistic
graphical model with independent cliques), beating the curse of dimensionality
that a single full-D GMM suffers:

    z = Psi^T (x - mu)               # PCA scores (rotation, decorrelated)
    f0(z) = prod_i  f_{S_i}(z_{S_i}) # product of per-subset GMMs (EM + AIC)

Target models (PCA space, with Psi orthogonal):
    additive     H1: z = z_b + p * Psi^T t
                 f1(z) = f0(z - p * Psi^T t)
    replacement  H1: z = p*Psi^T(t-mu) + (1-p) z_b
                 f1(z) = (1-p)^{-r} f0( (z - p*Psi^T(t-mu)) / (1-p) )
                 where r = retained PCA rank (see note on Jacobian below).

GLRT statistic:  T(x) = max_p [log f1(z; p)] - log f0(z).

The fill factor p is UNKNOWN and estimated per pixel via 1-D grid search
(ML estimation), exactly as prescribed in Sec. 4.2 of the paper.  No oracle
knowledge of the true amplitude is used.

Note on replacement Jacobian
-----------------------------
The paper writes the full-D Jacobian as (1-p)^{-N} where N is the number of
spectral bands.  In the PCA-reduced space with r retained components the
Jacobian becomes (1-p)^{-r}, because Psi is orthonormal:
    |d z_b / d z|  =  (1-p)^r   →   log |J^{-1}| = -r log(1-p).
Here r ≤ N; null dimensions are dropped via rcond filtering and contribute no
log-Jacobian term (their density is treated as a delta function at the
mean — a standard PCA approximation).

Note on centering asymmetry (additive vs. replacement)
------------------------------------------------------
Additive model (z = z_b + p * Psi^T t):
    The target direction in PCA space is Psi^T t (no centering of t), because
    the target is added on top of the background.  Eq.(16): f1(z) = f0(z - p*psi_t).

Replacement model (x = p*t + (1-p)*b, z = Psi^T(x-mu)):
    z = p*Psi^T(t-mu) + (1-p)*Psi^T(b-mu) = p*psi_tc + (1-p)*z_b.
    The target direction in PCA space is Psi^T(t-mu) (centering IS applied),
    because t is treated as a full pixel (mean-centered like background pixels).
"""

import numpy as np
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# Eigenvalue-subset partition (consecutive PCA components grouped by scale)
# ---------------------------------------------------------------------------

def _eigen_subsets(eigvals, cond_tol=1e3, max_dim=5):
    """Partition descending eigenvalue indices into consecutive subsets.

    A subset is grown while (lambda_first / lambda_next <= cond_tol) AND
    (size < max_dim).  This keeps each subset's marginal covariance well
    conditioned and low-dimensional, as required for stable GMM fitting.
    """
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
    """Background density f0(z) = sum_i log f_{S_i}(z_{S_i})  (in log domain)."""

    def __init__(self, subsets, gmms):
        self.subsets = subsets
        self.gmms    = gmms

    def logpdf(self, Z):
        out = np.zeros(len(Z), dtype=np.float64)
        for sub, gm in zip(self.subsets, self.gmms):
            if gm is not None:
                out += gm.score_samples(Z[:, sub])
        return out


def fit_product_gmm(Z, cond_tol=1e3, max_dim=5, k_max=5,
                    reg_covar=1e-6, seed=0):
    """Fit a per-subset GMM (EM, model order by AIC) on PCA scores Z (n, D)."""
    var     = Z.var(axis=0)                       # = eigenvalues (PCA scores)
    subsets = _eigen_subsets(var, cond_tol, max_dim)
    gmms    = []
    for sub in subsets:
        Zs = Z[:, sub]
        # Drop non-finite rows: sklearn's EM drops into a LAPACK Cholesky that
        # SEGFAULTS on non-finite input on macOS (Accelerate) instead of raising.
        Zs = Zs[np.all(np.isfinite(Zs), axis=1)]
        if len(Zs) < 2:
            gmms.append(None)
            continue
        # reg_covar is an ABSOLUTE diagonal floor in sklearn; its 1e-6 default
        # assumes ~unit-scale data. These are raw PCA scores with variance up
        # to ~1e6, so 1e-6 fails to keep component covariances positive-
        # definite and a collapsed EM component -> singular covariance ->
        # Cholesky crash. Floor it relative to the subset's data scale (the
        # scale-correct analogue of the sklearn default).
        scale = float(np.mean(np.var(Zs, axis=0))) + 1e-12
        rc    = max(reg_covar, 1e-6 * scale)
        best_gmm, best_aic = None, np.inf
        for k in range(1, k_max + 1):
            if len(Zs) < 2 * k:           # not enough data for k comps
                break
            try:
                gm = GaussianMixture(n_components=k, covariance_type='full',
                                     reg_covar=rc, random_state=seed,
                                     max_iter=200, n_init=1)
                gm.fit(Zs)
                aic = gm.aic(Zs)
            except Exception:
                continue
            if aic < best_aic:
                best_aic, best_gmm = aic, gm
        gmms.append(best_gmm)
    return ProductGMM(subsets, gmms)


# ---------------------------------------------------------------------------
# Oracle GMM-GLRT detector
# ---------------------------------------------------------------------------

class GMMGLRTLevin:
    """Levin GMM-GLRT: fit once on background, score via ML estimation of fill factor p.

    Fit on full-D background training pixels.  The background GMM is
    amplitude-independent, so fit ONCE and score under any target model.
    The fill factor p is estimated per pixel by 1-D grid search (no oracle).
    """

    def __init__(self, cond_tol=1e3, max_dim=5, k_max=5,
                 rcond=1e-8, reg_covar=1e-6, seed=0):
        self.cond_tol  = cond_tol
        self.max_dim   = max_dim
        self.k_max     = k_max
        self.rcond     = rcond          # numerical-rank cutoff (drop null dims)
        self.reg_covar = reg_covar
        self.seed      = seed

    def fit(self, train):
        """train: (n, D) background pixels in the detection space."""
        self.mu = train.mean(axis=0)
        Xc      = train - self.mu
        cov     = np.cov(Xc, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]

        # drop numerical-null directions (eigenvalue < rcond * lambda_max)
        keep = evals > self.rcond * max(evals[0], 1e-12)
        self.Psi    = evecs[:, keep]                 # (D, r)
        self.evals  = evals[keep]
        Ztr = Xc @ self.Psi                          # (n, r) PCA scores
        self.pgmm = fit_product_gmm(
            Ztr, self.cond_tol, self.max_dim, self.k_max,
            self.reg_covar, self.seed)
        return self

    def score(self, test, t, model='additive',
              p_steps=50, p_max=1.0, oracle_p=None):
        """
        GLRT statistic for the cluster-based (product-GMM) density.

        Paper Sec. II / eq. (15)-(16):
            Additive:    T(x) = max_p [log f(x − p·t)] − log f(x)
            Replacement: T(x) = max_p [log f((x−p·t)/(1−p)) − N·log(1−p)] − log f(x)

        Fill-factor handling
        --------------------
        oracle_p is None (default) — HONEST detector:
            p is the unknown fill factor, estimated per pixel by 1-D grid
            search (ML estimation), exactly as in Sec. 4.2 of the paper.
            No knowledge of the true amplitude is used. This is the curve
            that must be reported as "GMM-Levin".

        oracle_p = <float> — ORACLE (clairvoyant) reference:
            the grid is replaced by the single TRUE fill factor. The density
            model is IDENTICAL to the honest detector, so this is a genuine
            upper bound on the honest GLRT (it can only do better, never
            worse). Must be labelled as an oracle in any figure/table.
        """
        z     = (test - self.mu) @ self.Psi           # (n, r) PCA scores
        logf0 = self.pgmm.logpdf(z)

        if model == 'additive':
            psi_t = t @ self.Psi                       # direction rule: no centering
            if oracle_p is not None:
                p_grid = np.array([float(oracle_p)])
            else:
                p_grid = np.linspace(0.0, p_max, p_steps)
            best  = np.full(len(test), -np.inf)
            for p in p_grid:
                logf1 = self.pgmm.logpdf(z - p * psi_t)
                best  = np.maximum(best, logf1 - logf0)
            return best
        else:                                          # replacement
            # r = retained PCA rank (NOT full band count D); see module docstring
            # for why the Jacobian exponent is r, not N.
            N     = self.Psi.shape[1]                  # = r (retained rank)
            psi_tc = (t - self.mu) @ self.Psi         # point rule: centering applied (see module docstring)
            if oracle_p is not None:
                p_grid = np.array([min(float(oracle_p), 0.99)])
            else:
                p_grid = np.linspace(1e-4, min(p_max, 0.99), p_steps)
            best  = np.full(len(test), -np.inf)
            for p in p_grid:
                z_b   = (z - p * psi_tc) / (1.0 - p)
                logf1 = -N * np.log(1.0 - p) + self.pgmm.logpdf(z_b)
                best  = np.maximum(best, logf1 - logf0)
            return best


# ---------------------------------------------------------------------------
# Functional API (drop-in for detectors.py)
# ---------------------------------------------------------------------------

def gmm_glrt_levin_additive(test, train, t, p_steps=50, p_max=1.0, **kw):
    """GMM-GLRT (Levin 2019), additive model. Fill factor estimated by grid search."""
    return GMMGLRTLevin(**kw).fit(train).score(test, t, 'additive', p_steps, p_max)


def gmm_glrt_levin_replacement(test, train, t, p_steps=50, p_max=1.0, **kw):
    """GMM-GLRT (Levin 2019), replacement model. Fill factor estimated by grid search."""
    return GMMGLRTLevin(**kw).fit(train).score(test, t, 'replacement', p_steps, p_max)
