"""
pipeline.py — Honest score-based detection pipeline.

Implements the derivation:

  Step 1  PCA on raw background pixels: z = V_d^T (x - m_raw).
          Eigenvectors V are fitted on RAW (un-normalised) background.

  Step 2  Fixed invertible affine normalisation N(z) = A(z - c), calibrated
          on the d-dimensional PCA SCORES of the background.
          Default A = diag(1/sigma_z), c = mu_z  (per-PC standardisation).

  Step 3  Carry the signature through both transforms with the model-correct rule:
            additive     s_add = A (V^T t)                (direction; c cancels)
            replacement  s_rep = A (V^T(t - m_raw) - c)   (point rule)

  Step 4  Train score model psi_hat ~ grad log p on background features.

  Step 5  Standardised (CFAR) LMP statistics.

Validity diagnostic:
    rho_d = retained deflection fraction
          = [ sum_{i<=d} (v_i^T t)^2 / lambda_i ]
          / [ sum_{i<=D} (v_i^T t)^2 / lambda_i ]
    Computed in RAW PCA space (eigvecs of raw background covariance).
    rho_d -> 1  <=>  PCA projection is detection-lossless.
"""

import numpy as np
import torch

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.decomposition import PCA as SklearnPCA
from dsm_model import ScoreNet, dsm_loss, compute_scores


# ---------------------------------------------------------------------------
# Normalisation maps (Step 2) — fitted on d-dim PCA scores, not raw pixels
# ---------------------------------------------------------------------------

def _fit_normalizer(bkg: np.ndarray, mode: str):
    """Return (A_diag, c) for the affine map N(z) = A_diag * (z - c).

    Called on d-dimensional PCA scores (new order: PCA first, then normalise).
    pca_std / pca_elm are aliases that remain valid — they mean the same thing
    as per_band_std / elm when applied to PCA scores.
    """
    if mode in ('per_band_std', 'pca_std'):
        c = bkg.mean(axis=0)
        A = 1.0 / (bkg.std(axis=0) + 1e-8)
    elif mode == 'global_max':
        c = np.zeros(bkg.shape[1], dtype=bkg.dtype)
        A = np.full(bkg.shape[1], 1.0 / (bkg.max() + 1e-12), dtype=bkg.dtype)
    elif mode == 'per_band_minmax':
        c = bkg.min(axis=0)
        A = 1.0 / (bkg.max(axis=0) - bkg.min(axis=0) + 1e-12)
    elif mode in ('elm', 'pca_elm'):
        # Robust two-point scaling: dark = 1st pct, bright = 99th pct.
        c = np.percentile(bkg, 1, axis=0)
        A = 1.0 / (np.percentile(bkg, 99, axis=0) - c + 1e-12)
    elif mode == 'none':
        c = np.zeros(bkg.shape[1], dtype=bkg.dtype)
        A = np.ones(bkg.shape[1], dtype=bkg.dtype)
    else:
        raise ValueError(f"unknown norm mode {mode!r}")
    return A.astype(np.float32), c.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HonestDetectionPipeline:
    def __init__(self, latent_dim: int = 5, norm: str = 'per_band_std',
                 skip_pca: bool = False):
        self.d        = latent_dim
        self.norm     = norm
        self.skip_pca = skip_pca

    # ---- Step 1 + 2: PCA on raw pixels, then normalise PCA scores ----
    def fit(self, bkg_raw: np.ndarray):
        n_samples, D = bkg_raw.shape

        if self.skip_pca:
            # No PCA — V = I_D, m_raw = 0.
            # Normalisation is fitted directly on raw pixels.
            self.d       = D
            self.m       = np.zeros(D, dtype=np.float32)   # no PCA centering
            self.V       = np.eye(D, dtype=np.float32)
            self.eigvals = np.ones(D, dtype=np.float64)
            self.eigvecs = np.eye(D, dtype=np.float32)
            self.A, self.c = _fit_normalizer(bkg_raw, self.norm)
            return self

        # Step 1: PCA on RAW background (sklearn SVD — numerically stable).
        # Fit all available components so rho_d has the full eigenspectrum;
        # only the top-d are used for projection.
        n_comp_full = min(n_samples - 1, D)
        n_comp      = min(self.d, n_comp_full)   # cap at available rank

        pca = SklearnPCA(n_components=n_comp_full, svd_solver='full')
        pca.fit(bkg_raw)

        self.m       = pca.mean_.astype(np.float32)               # (D,) raw-space mean
        self.eigvals = pca.explained_variance_.astype(np.float64)  # (n_comp_full,)
        self.eigvecs = pca.components_.T.astype(np.float32)        # (D, n_comp_full)
        self.V       = self.eigvecs[:, :n_comp]                    # (D, d)
        self.d       = n_comp   # update in case capped (e.g. n=50, d=64 → d=49)

        # Step 2: fit normaliser on PCA scores of background.
        # A and c are d-dimensional (PCA score space).
        bkg_pca      = ((bkg_raw - self.m) @ self.V).astype(np.float32)
        self.A, self.c = _fit_normalizer(bkg_pca, self.norm)

        return self

    # ---- Project: raw → PCA → normalise ----
    def project(self, x_raw: np.ndarray) -> np.ndarray:
        """Raw pixels → normalised PCA scores (d-dimensional)."""
        z = ((x_raw - self.m) @ self.V).astype(np.float32)   # PCA
        return (z - self.c) * self.A                           # normalise

    # ---- Step 3: signature transforms (model-correct rules) ----
    def signature_additive(self, t_raw: np.ndarray) -> np.ndarray:
        """Direction rule: s_add = A * (V^T t).  c cancels for additive model."""
        z = (self.V.T @ t_raw).astype(np.float32)   # project direction
        return z * self.A                             # scale; c cancels

    def signature_replacement(self, t_raw: np.ndarray) -> np.ndarray:
        """Point rule: s_rep = A * (V^T(t - m_raw) - c)."""
        z = (self.V.T @ (t_raw - self.m)).astype(np.float32)   # project point
        return (z - self.c) * self.A                            # normalise

    # ---- Validity diagnostic: retained deflection fraction ----
    def deflection_curve(self, t_raw: np.ndarray) -> np.ndarray:
        """
        Cumulative retained-deflection fraction rho_k for k = 1..n_comp_full.

        Computed in raw PCA space (eigvecs of raw background covariance):
            rho_k = [sum_{i<=k} (v_i^T t)^2 / lambda_i]
                  / [sum_i      (v_i^T t)^2 / lambda_i]
        """
        s_n   = t_raw.astype(np.float64)                        # raw signature
        proj  = self.eigvecs.T.astype(np.float64) @ s_n          # (n_comp_full,)
        contr = proj ** 2 / np.maximum(self.eigvals, 1e-12)
        cum   = np.cumsum(contr)
        return cum / (cum[-1] + 1e-300)

    def rho_d(self, t_raw: np.ndarray) -> float:
        if self.skip_pca:
            return 1.0   # no projection → no loss by definition
        return float(self.deflection_curve(t_raw)[self.d - 1])

    # ---- Step 4: train score model on normalised PCA features ----
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

    # ---- Step 5: standardised (CFAR) LMP statistics ----
    def score_dsm_additive(self, train_pca, test_pca, s_add):
        z_tr = compute_scores(self.dsm, train_pca)
        z_te = compute_scores(self.dsm, test_pca)
        z_bar = z_tr.mean(0)
        C_psi = np.cov(z_tr, rowvar=False)
        if C_psi.ndim == 0: C_psi = np.array([[float(C_psi)]])
        norm = float(np.sqrt(max(float(s_add @ C_psi @ s_add), 1e-12)))
        return -((z_te - z_bar) @ s_add) / norm

    def score_dsm_replacement(self, train_pca, test_pca, s_rep):
        # Paper: u_rep(y) = (ψ(y)−ψ̄)ᵀ(y−s) + d,  I_rep = E[r²] − d²
        # Statistic: u_rep / sqrt(I_rep)
        # Adaptation: center ψ by its empirical mean ψ̄ = E_train[ψ(x)].
        psi_tr  = compute_scores(self.dsm, train_pca)
        psi_te  = compute_scores(self.dsm, test_pca)
        psi_bar = psi_tr.mean(axis=0)
        d = train_pca.shape[1]
        r_tr = ((psi_tr - psi_bar) * (train_pca - s_rep)).sum(axis=1)
        I_rep = max(float((r_tr**2).mean()) - d**2, 1e-12)
        r_te  = ((psi_te - psi_bar) * (test_pca  - s_rep)).sum(axis=1) + d
        return (r_te + d) / np.sqrt(I_rep)


# ---------------------------------------------------------------------------
# Gaussian AMF (matched filter) — ideal reference detector
# ---------------------------------------------------------------------------

def amf_score(train, test, s):
    """Adaptive matched filter: s^T Sigma^{-1}(y - mu) / sqrt(s^T Sigma^{-1} s).

    Works in any space (full normalised D-dim, or PCA-d). `s` must be the
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
