"""
local_detectors.py — Local sample-covariance (SCM) classical detectors.

AMF-local and CEM-local use a PER-PIXEL local background estimate drawn from
the SAME k×k spatial window that the spatial score nets (CF-Attn / NeighborMLP)
consume — i.e. the (k*k - 1) neighbors returned by
`models.neighbor_adapted.extract_neighborhoods`.

For each test pixel y_i with local neighbor set X_i (K = k*k-1 pixels in D bands):

  AMF-local:
      mu_i    = mean(X_i)
      Sigma_i = cov(X_i) + beta * mean(diag(Sigma_i)) * I        (diagonal loading)
      T_i     = sᵀ Sigma_i⁻¹ (y_i - mu_i) / sqrt(sᵀ Sigma_i⁻¹ s)

  CEM-local (autocorrelation, NO mean subtraction — matches global CEM):
      R_i     = X_iᵀ X_i / K + beta * mean(diag(R_i)) * I
      w_i     = R_i⁻¹ s / (sᵀ R_i⁻¹ s)
      T_i     = y_iᵀ w_i

Diagonal loading is REQUIRED: with k=5 the window holds only 24 samples while
D≈103, so the raw local SCM is rank-deficient (rank ≤ K < D) and singular.

Everything is batched + chunked on the given torch device (GPU-friendly).
"""
import numpy as np
import torch


@torch.no_grad()
def amf_cem_local_scm(test_pix: np.ndarray,
                      test_nbr: np.ndarray,
                      s: np.ndarray,
                      device: str = 'cpu',
                      loading: float = 1e-8,
                      chunk: int = 1024):
    """
    Compute AMF-local and CEM-local scores from per-pixel k×k-window SCMs.

    Parameters
    ----------
    test_pix : (N, D)        test pixels (planted)
    test_nbr : (N, K, D)     their k×k window neighbors (K = k*k - 1)
    s        : (D,)          target signature (raw band space)
    device   : torch device string
    loading  : diagonal-loading factor (× mean diagonal) for invertibility
    chunk    : pixels processed per batch (memory control)

    Returns
    -------
    amf_local : (N,) np.float32
    cem_local : (N,) np.float32
    """
    N, K, D = test_nbr.shape
    dev   = torch.device(device)
    y     = torch.as_tensor(test_pix, dtype=torch.float32, device=dev)
    nbr   = torch.as_tensor(test_nbr, dtype=torch.float32, device=dev)
    s_t   = torch.as_tensor(s,        dtype=torch.float32, device=dev)
    eyeD  = torch.eye(D, device=dev)

    amf_out = torch.empty(N, dtype=torch.float32)
    cem_out = torch.empty(N, dtype=torch.float32)

    for i0 in range(0, N, chunk):
        yb = y[i0:i0 + chunk]                       # (B, D)
        nb = nbr[i0:i0 + chunk]                      # (B, K, D)
        B  = yb.shape[0]
        s_b = s_t.expand(B, D).unsqueeze(-1)         # (B, D, 1)

        # ---- AMF-local: mean-centered local covariance ----
        mu  = nb.mean(dim=1)                          # (B, D)
        cen = nb - mu.unsqueeze(1)                    # (B, K, D)
        Sigma = cen.transpose(1, 2) @ cen / max(K - 1, 1)   # (B, D, D)
        load_s = loading * Sigma.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-8)
        Sigma = Sigma + load_s.view(B, 1, 1) * eyeD
        Sinv_s = torch.linalg.solve(Sigma, s_b).squeeze(-1)  # (B, D)
        num   = ((yb - mu) * Sinv_s).sum(-1)                 # (B,)
        den   = (s_t * Sinv_s).sum(-1).clamp_min(1e-12).sqrt()
        amf_out[i0:i0 + chunk] = (num / den).cpu()

        # ---- CEM-local: autocorrelation (no mean subtraction) ----
        R = nb.transpose(1, 2) @ nb / max(K, 1)              # (B, D, D)
        load_r = loading * R.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-8)
        R = R + load_r.view(B, 1, 1) * eyeD
        Rinv_s = torch.linalg.solve(R, s_b).squeeze(-1)      # (B, D)
        w = Rinv_s / (s_t * Rinv_s).sum(-1, keepdim=True).clamp_min(1e-12)
        cem_out[i0:i0 + chunk] = (yb * w).sum(-1).cpu()

    return amf_out.numpy().astype(np.float32), cem_out.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Global classical baselines with an explicit (minimal) eigenvalue floor.
# These mirror baselines.detectors.amf / .cem but expose `eig_floor` so the
# comparison can run the baselines at near-zero regularization (e.g. 1e-12)
# WITHOUT editing the shared detectors.py. Floor is RELATIVE (× λ_max).
# ---------------------------------------------------------------------------

def amf_global(test_data: np.ndarray, train_data: np.ndarray,
               s: np.ndarray, eig_floor: float = 1e-12) -> np.ndarray:
    """AMF with a relative eigenvalue floor on the background covariance."""
    mu    = train_data.mean(axis=0)
    Sigma = np.cov(train_data, rowvar=False)
    Sigma = (Sigma + Sigma.T) / 2
    eigv, eigvec = np.linalg.eigh(Sigma)
    eigv = np.clip(eigv, eigv.max() * float(eig_floor), None)
    Si   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    Si_s = Si @ s
    norm = np.sqrt(float(s @ Si_s) + 1e-18)
    return (test_data - mu) @ Si_s / norm


def cem_global(test_data: np.ndarray, train_data: np.ndarray,
               s: np.ndarray, eig_floor: float = 1e-12) -> np.ndarray:
    """CEM (autocorrelation, no mean subtraction) with a relative eigenvalue floor."""
    n, d = train_data.shape
    R    = (train_data.T @ train_data) / n
    R    = (R + R.T) / 2
    eigv, eigvec = np.linalg.eigh(R)
    eigv = np.clip(eigv, eigv.max() * float(eig_floor), None)
    Ri   = eigvec @ np.diag(1.0 / eigv) @ eigvec.T
    Ri_s = Ri @ s
    w    = Ri_s / (float(s @ Ri_s) + 1e-12)
    return test_data @ w
