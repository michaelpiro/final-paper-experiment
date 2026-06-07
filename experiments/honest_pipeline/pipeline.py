"""
pipeline.py — Honest score-based detection pipeline.

Implements the derivation:

  Step 0  Fixed invertible affine normalization N(x) = A(x - c), calibrated
          on BACKGROUND ONLY.  Default A = diag(1/sigma_b), c = mu_b
          (per-band standardization — robust, band-symmetric, invertible).
          NOT per-band min-max; NEVER clipped.

  Step 1  Carry the signature through N with the model-correct rule:
            additive     t -> A t          (direction; c cancels)
            replacement  t -> A(t - c)      (point)

  Step 2  PCA on the normalized background: z = V_d^T (N(x) - m).
          Carry the signature again:
            s_add = V_d^T (A t)             (no centering)
            s_rep = V_d^T (A(t - c) - m)    (centered)

  Step 3  Train score model psi_hat ~ grad log p_z on background PCA features.

  Step 4  Standardized (CFAR) LMP statistics in z-space.

Validity diagnostic (condition 2 of the derivation):
    rho_d = retained deflection fraction
          = [ sum_{i<=d} (v_i^T s_n)^2 / lambda_i ]
          / [ sum_{i<=D} (v_i^T s_n)^2 / lambda_i ]
    rho_d -> 1  <=>  the matched-filter direction Sigma_N^{-1} s lies in the
    top-d PCA subspace  <=>  PCA projection is detection-lossless.
"""

import numpy as np
import torch

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dsm_model import ScoreNet, dsm_loss, compute_scores


# ---------------------------------------------------------------------------
# Normalization maps (Step 0) — all invertible, calibrated on background only
# ---------------------------------------------------------------------------

def _fit_normalizer(bkg_raw: np.ndarray, mode: str):
    """Return (A_diag, c) for the affine map N(x) = A_diag * (x - c)."""
    if mode == 'per_band_std':
        c = bkg_raw.mean(axis=0)
        A = 1.0 / (bkg_raw.std(axis=0) + 1e-8)          # per-band 1/sigma
    elif mode == 'global_max':
        c = np.zeros(bkg_raw.shape[1], dtype=bkg_raw.dtype)
        A = np.full(bkg_raw.shape[1], 1.0 / (bkg_raw.max() + 1e-12),
                    dtype=bkg_raw.dtype)
    elif mode == 'per_band_minmax':
        c = bkg_raw.min(axis=0)
        A = 1.0 / (bkg_raw.max(axis=0) - bkg_raw.min(axis=0) + 1e-12)
    elif mode == 'pca_std':
        # No raw-space normalization; PC scores are standardised after PCA.
        c = np.zeros(bkg_raw.shape[1], dtype=bkg_raw.dtype)
        A = np.ones(bkg_raw.shape[1], dtype=bkg_raw.dtype)
    else:
        raise ValueError(f"unknown norm mode {mode!r}")
    return A.astype(np.float32), c.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HonestDetectionPipeline:
    def __init__(self, latent_dim: int = 5, norm: str = 'per_band_std'):
        self.d    = latent_dim
        self.norm = norm

    # ---- Step 0 + 2: fit normalization + PCA on background ----
    def fit(self, bkg_raw: np.ndarray):
        # Step 0: normalization from background only
        self.A, self.c = _fit_normalizer(bkg_raw, self.norm)
        bkg_n = (bkg_raw - self.c) * self.A                  # N(bkg)

        # Step 2: PCA on normalized background
        self.m = bkg_n.mean(axis=0).astype(np.float32)       # PCA centering
        Xc     = bkg_n - self.m
        # full eigendecomposition of the normalized-background covariance
        Sigma_N = np.cov(bkg_n, rowvar=False)
        evals, evecs = np.linalg.eigh(Sigma_N)               # ascending
        order = np.argsort(evals)[::-1]                       # descending
        self.eigvals = evals[order].astype(np.float64)        # (D,)
        self.eigvecs = evecs[:, order].astype(np.float32)     # (D, D) cols = v_i
        self.V = self.eigvecs[:, :self.d]                     # (D, d) top-d

        # Post-PCA standardisation (pca_std mode only):
        # divide each PC score by its training std so all dims have unit variance.
        if self.norm == 'pca_std':
            Z = ((bkg_n - self.m) @ self.V).astype(np.float32)
            self.post_pca_scale = (1.0 / (Z.std(axis=0) + 1e-8)).astype(np.float32)
        else:
            self.post_pca_scale = None
        return self

    # ---- Normalize + project ----
    def normalize(self, x_raw):
        return (x_raw - self.c) * self.A

    def project(self, x_raw):
        """Raw -> normalized -> PCA-d scores (-> per-PC std scale if pca_std)."""
        x_n = self.normalize(x_raw)
        z   = ((x_n - self.m) @ self.V).astype(np.float32)
        if self.post_pca_scale is not None:
            z = z * self.post_pca_scale
        return z

    # ---- Step 1+2: signature transforms (model-correct rules) ----
    def signature_additive(self, t_raw):
        """Direction rule: s_add = V^T (A t).  No centering."""
        t_n = (t_raw * self.A).astype(np.float32)            # A t  (c cancels)
        s   = (self.V.T @ t_n).astype(np.float32)
        if self.post_pca_scale is not None:
            s = s * self.post_pca_scale
        return s

    def signature_replacement(self, t_raw):
        """Point rule: s_rep = V^T (A(t - c) - m)."""
        t_n = ((t_raw - self.c) * self.A).astype(np.float32) # A(t - c)
        s   = (self.V.T @ (t_n - self.m)).astype(np.float32)
        if self.post_pca_scale is not None:
            s = s * self.post_pca_scale
        return s

    # ---- Validity diagnostic: retained deflection fraction ----
    def deflection_curve(self, t_raw):
        """
        Returns the cumulative retained-deflection fraction rho_k for
        k = 1..D, using the additive signature direction in normalized space.

        rho_k = [sum_{i<=k} (v_i^T s_n)^2 / lambda_i] / [sum_i (v_i^T s_n)^2 / lambda_i]
        """
        s_n = (t_raw * self.A).astype(np.float64)            # A t  (full D)
        proj = self.eigvecs.T.astype(np.float64) @ s_n        # (v_i^T s_n) for all i
        contrib = proj ** 2 / np.maximum(self.eigvals, 1e-12) # per-direction deflection
        cum = np.cumsum(contrib)
        total = cum[-1] + 1e-300
        return cum / total                                    # rho_k, k=1..D

    def rho_d(self, t_raw):
        return float(self.deflection_curve(t_raw)[self.d - 1])

    # ---- Step 3: train score model on background PCA features ----
    def train_dsm(self, bkg_pca, sigma, hidden=(64, 64), activation='silu',
                  epochs=2000, lr=1e-3, weight_decay=1e-4, batch_size=256,
                  seed=42):
        torch.manual_seed(seed)
        model = ScoreNet(self.d, list(hidden), activation)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        X = torch.tensor(bkg_pca, dtype=torch.float32)
        N, bs = len(X), min(batch_size, len(bkg_pca))
        for _ in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, bs):
                b = X[perm[i:i + bs]]
                loss = dsm_loss(model, b, sigma)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        self.dsm = model
        return model

    # ---- Step 4: standardized (CFAR) LMP statistics ----
    def score_dsm_additive(self, train_pca, test_pca, s_add):
        z_tr = compute_scores(self.dsm, train_pca)
        z_te = compute_scores(self.dsm, test_pca)
        z_bar = z_tr.mean(0)
        C_psi = np.cov(z_tr, rowvar=False)
        if C_psi.ndim == 0: C_psi = np.array([[float(C_psi)]])
        norm = float(np.sqrt(max(float(s_add @ C_psi @ s_add), 1e-12)))
        return -((z_te - z_bar) @ s_add) / norm

    def score_dsm_replacement(self, train_pca, test_pca, s_rep):
        psi_tr = compute_scores(self.dsm, train_pca)
        psi_te = compute_scores(self.dsm, test_pca)
        psi_bar = psi_tr.mean(0)
        r_tr = ((psi_tr - psi_bar) * (train_pca - s_rep)).sum(1)
        r_bar, r_std = r_tr.mean(), r_tr.std() + 1e-12
        r_te = ((psi_te - psi_bar) * (test_pca - s_rep)).sum(1)
        return (r_te - r_bar) / r_std


# ---------------------------------------------------------------------------
# Gaussian AMF (matched filter) — ideal reference detector
# ---------------------------------------------------------------------------

def amf_score(train, test, s):
    """Adaptive matched filter: s^T Sigma^{-1}(y - mu) / sqrt(s^T Sigma^{-1} s).

    Works in any space (full normalized D-dim, or PCA-d). `s` must be the
    additive signature expressed in that same space.
    """
    mu = train.mean(0)
    Sigma = np.cov(train, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])
    Si_s = np.linalg.solve(Sigma + 1e-8 * np.eye(len(Sigma)), s)
    denom = float(np.sqrt(max(s @ Si_s, 1e-12)))
    return ((test - mu) @ Si_s) / denom


def amf_replacement_score(train, test, s):
    """Gaussian replacement LMP: [d - (y-mu)^T Si (y-s)] / sqrt(2d + (mu-s)^T Si (mu-s))."""
    mu = train.mean(0)
    d  = train.shape[1]
    Sigma = np.cov(train, rowvar=False)
    if Sigma.ndim == 0: Sigma = np.array([[float(Sigma)]])
    Si = np.linalg.inv(Sigma + 1e-8 * np.eye(d))
    quad = ((test - mu) @ Si * (test - s)).sum(1)
    denom = float(np.sqrt(max(2 * d + (mu - s) @ Si @ (mu - s), 1e-12)))
    return (d - quad) / denom
