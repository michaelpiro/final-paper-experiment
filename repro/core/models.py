"""
models.py — score networks and training/scoring objectives.

Two learned score models, both trained by denoising score matching (DSM):

  * ScoreNet                — global score psi(x); linear (L-DART) or one-hidden-
                              layer MLP (DART). Also used, with the LFI objective,
                              for the learned-Rao baselines (L-LRao / LRao).
  * NeighborMLPDenoiser     — spatially-adapted score psi_i(x; neighbours) used by
                              DARTS / DARTS-CFAR.

Objectives / scorers
  dsm_loss                          — DSM training loss for ScoreNet
  lfi_loss_mode2                    — signal-agnostic LFI loss (learned-Rao training)
  compute_lfi_detector_scores_mode2 — learned-Rao (LRao/L-LRao) detection statistic
  compute_scores                    — evaluate psi(x) on a numpy array
  neighbor_mlp_dsm_loss             — DSM loss for NeighborMLPDenoiser
  score_nmlp_additive               — DARTS additive-LMP detection statistic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Robust SVD (fallback chain for ill-conditioned score covariances)
# ---------------------------------------------------------------------------

def _robust_svd_np(A: np.ndarray):
    """SVD with a fallback chain. Tries the original matrix first (so normal runs
    are unaffected); only adds a ridge if LAPACK gesdd fails to converge. A matrix
    with NaN/Inf returns a trivial decomposition (-> pseudo-inverse 0 -> scores 0)."""
    if not np.all(np.isfinite(A)):
        n = A.shape[0]
        return np.eye(n), np.zeros(n), np.eye(n)
    try:
        return np.linalg.svd(A)
    except np.linalg.LinAlgError:
        pass
    try:
        from scipy.linalg import svd as scipy_svd
        return scipy_svd(A, full_matrices=True, lapack_driver='gesvd')
    except Exception:
        pass
    try:
        from scipy.linalg import svd as scipy_svd
        return scipy_svd(A + 1e-6 * np.eye(A.shape[0]),
                         full_matrices=True, lapack_driver='gesvd')
    except Exception:
        pass
    from scipy.linalg import svd as scipy_svd
    return scipy_svd(A + 1e-3 * np.eye(A.shape[0]),
                     full_matrices=True, lapack_driver='gesvd')


# ---------------------------------------------------------------------------
# Global score network (DART / L-DART; also the learned-Rao backbone)
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """Score network psi(x): R^d -> R^d trained by denoising score matching.

    Optional frozen `whitening` front-end (the first layer). When present the net
    operates in WHITENED space: forward(x) = W^T net(whiten(x)); the DSM loss
    whitens first then adds noise in whitened space; detection uses the RAW
    signature directly.

    hidden_dims=[]    -> linear/affine score  (L-DART, L-LRao)
    hidden_dims=[h]   -> one-hidden-layer MLP (DART, LRao)
    """

    def __init__(self, input_dim: int, hidden_dims: list = None,
                 activation: str = "silu", whitening=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        act_map = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}
        act_cls = act_map[activation]
        dims = [input_dim] + list(hidden_dims) + [input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)
        self.whitening = whitening

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        return self.whitening(x) if self.whitening is not None else x

    def to_data_space(self, score_w: torch.Tensor) -> torch.Tensor:
        """Un-whiten a whitened-space score: grad_x log p(x) = W^T grad_w log p(w)."""
        return score_w @ self.whitening.W if self.whitening is not None else score_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_data_space(self.net(self.whiten(x)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def dsm_loss(model: ScoreNet, batch: torch.Tensor, sigma,
             weighted: bool = False) -> torch.Tensor:
    """DSM objective: E[||psi(w~) - (w - w~)/sigma^2||^2], w~ = w + sigma*eps.

    sigma may be a scalar (isotropic) or a (d,) per-band vector. If the net has a
    frozen whitening front-end, the noise is added in WHITENED space and the inner
    net is scored directly.
    """
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(sigma, dtype=batch.dtype, device=batch.device)
    w = model.whiten(batch) if hasattr(model, "whiten") else batch
    inner = model.net if hasattr(model, "net") else model
    eps = torch.randn_like(w) * sigma
    w_tilde = w + eps
    target = -eps / (sigma ** 2)
    se = (inner(w_tilde) - target) ** 2
    if weighted:
        se = se * (sigma ** 2)
    return se.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Learned-Rao (LFI mode-2): signal-agnostic training, project signal at test time
# ---------------------------------------------------------------------------

def lfi_loss_mode2(model: ScoreNet, batch: torch.Tensor,
                   delta_theta: float = 0.01,
                   detach_sigma: bool = False) -> torch.Tensor:
    """Signal-agnostic LFI loss: maximize tr(J*) = tr(G^T Sigma^{-1} G), where
    G = E[d psi / d x] is the full Jacobian of the mean score and Sigma the score
    covariance (full SVD pseudo-inverse — no eigenvalue truncation).
    `detach_sigma=True` stops the gradient through Sigma (stabilises training)."""
    n, d = batch.shape
    if detach_sigma:
        ctx = torch.no_grad()
    else:
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:
        psi_0    = model(batch)
        mu_psi   = psi_0.mean(dim=0)
        centered = psi_0 - mu_psi
        Sigma    = (centered.T @ centered) / max(n - 1, 1)
        U, S, Vh = torch.linalg.svd(Sigma)
        S_inv    = torch.where(S > 0, 1.0 / S, torch.zeros_like(S))
        Sigma_inv = Vh.T @ torch.diag(S_inv) @ U.T

    from torch.func import jacrev, vmap
    def _model_1d(x1d):
        return model(x1d.unsqueeze(0)).squeeze(0)
    def _single_jac(x):
        return jacrev(_model_1d)(x)
    J_all = vmap(_single_jac)(batch)        # (n, d_out, d)
    G     = J_all.mean(dim=0)               # (d_out, d)
    J_star = G.T @ Sigma_inv @ G            # (d, d)
    return -J_star.trace()


@torch.no_grad()
def compute_lfi_detector_scores_mode2(model: ScoreNet, train_data: np.ndarray,
                                       test_data: np.ndarray, s: np.ndarray,
                                       delta_theta: float = 0.01) -> np.ndarray:
    """Learned-Rao (LRao / L-LRao) one-sided LLMP statistic. The signal s enters
    only here (not during training):
        g_s = G s,   J_s = g_s^T Sigma^{-1} g_s,
        T(y) = g_s^T Sigma^{-1} (psi(y) - mu) / sqrt(J_s).
    """
    model.eval()
    device = next(model.parameters()).device
    d    = train_data.shape[1]
    X_tr = torch.tensor(train_data, dtype=torch.float32).to(device)
    X_te = torch.tensor(test_data,  dtype=torch.float32).to(device)
    I_d  = torch.eye(d, device=device)

    psi_tr = model(X_tr).cpu().numpy()
    d_out  = psi_tr.shape[1]
    if not np.all(np.isfinite(psi_tr)):
        return np.zeros(len(test_data), dtype=np.float32)
    mu       = psi_tr.mean(axis=0)
    centered = psi_tr - mu
    n        = len(train_data)
    Sigma    = centered.T @ centered / max(n - 1, 1)
    U, S, Vh = _robust_svd_np(Sigma)
    S_inv    = np.where(S > 0, 1.0 / S, 0.0)
    Sigma_inv = Vh.T @ np.diag(S_inv) @ U.T

    G = np.zeros((d_out, d))
    for j in range(d):
        psi_plus  = model(X_tr + delta_theta * I_d[j]).cpu().numpy()
        psi_minus = model(X_tr - delta_theta * I_d[j]).cpu().numpy()
        G[:, j]   = ((psi_plus - psi_minus) / (2.0 * delta_theta)).mean(axis=0)

    g_s   = G @ s
    J_s   = float(g_s @ Sigma_inv @ g_s)
    denom = np.sqrt(max(J_s, 1e-12))
    psi_te = model(X_te).cpu().numpy()
    return (psi_te - mu) @ (Sigma_inv @ g_s) / denom


@torch.no_grad()
def compute_scores(model: ScoreNet, data: np.ndarray) -> np.ndarray:
    """Evaluate the learned score psi(w) on a numpy array. Returns (n, d)."""
    model.eval()
    device = next(model.parameters()).device
    X = torch.tensor(data, dtype=torch.float32).to(device)
    return model(X).cpu().numpy()


# ===========================================================================
# Spatially-adapted score network (DARTS)
# ===========================================================================

class NeighborMLPDenoiser(nn.Module):
    """Spatially-aware denoising score estimator (DARTS).

    1. A shared MLP encoder phi embeds the query pixel and its M spatial
       neighbours into a d_lat latent space.
    2. The K latent-nearest neighbours are selected (hard top-K, no gradient
       through the indices).
    3. The denoiser MLP f maps [y_i | z_i | z_j1..z_jK] to a clean-pixel
       estimate x_hat.
    4. Score via Tweedie's identity: psi(y_i) = (x_hat - y_i) / sigma^2.
    """

    def __init__(self, D: int, d_lat: int = 16, K: int = 8,
                 enc_hidden=None, score_hidden=None,
                 hidden: int = 128, n_layers: int = 3,
                 sigma: float = 0.1, activation: str = 'silu',
                 whitening=None):
        super().__init__()
        self.D     = D
        self.d_lat = d_lat
        self.K     = K
        self.sigma = sigma
        self.whitening = whitening

        if enc_hidden is None:
            enc_hidden = [hidden] * max(n_layers - 1, 1)
        if score_hidden is None:
            score_hidden = [hidden] * max(n_layers - 1, 1)
        act_cls = {'silu': nn.SiLU, 'relu': nn.ReLU}[activation]

        def _mlp(in_dim, hidden_list, out_dim):
            dims = [in_dim] + list(hidden_list)
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), act_cls()]
            layers.append(nn.Linear(dims[-1], out_dim))
            return nn.Sequential(*layers)

        # shared encoder: D -> enc_hidden -> d_lat
        self.phi = _mlp(D, enc_hidden, d_lat)
        # denoiser: [y_i | z_i | z_j1..z_jK] -> score_hidden -> D
        self.f = _mlp(D + d_lat * (1 + K), score_hidden, D)

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        return self.whitening(x) if self.whitening is not None else x

    def to_data_space(self, score_w: torch.Tensor) -> torch.Tensor:
        return score_w @ self.whitening.W if self.whitening is not None else score_w

    def forward(self, y: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """Whiten raw y + neighbours, run the Tweedie score in whitened space,
        then map back to DATA space (detection uses the raw signature)."""
        score_w = self._forward_inner(self.whiten(y), self.whiten(neighbors))
        return self.to_data_space(score_w)

    def _forward_inner(self, y: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        B, M, D = neighbors.shape
        z_i = self.phi(y)                                                       # (B, d_lat)
        z_j = self.phi(neighbors.reshape(B * M, D)).reshape(B, M, self.d_lat)    # (B, M, d_lat)

        # top-K latent-nearest neighbours (no gradient through the indices)
        with torch.no_grad():
            dists    = ((z_j - z_i.unsqueeze(1)) ** 2).sum(-1)                   # (B, M)
            K_eff    = min(self.K, M)
            topk_idx = dists.topk(K_eff, dim=1, largest=False).indices           # (B, K)
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.d_lat)
        z_topk = z_j.gather(1, topk_idx_exp)                                     # (B, K_eff, d_lat)
        if K_eff < self.K:                                                      # zero-pad small windows
            pad = z_topk.new_zeros(B, self.K - K_eff, self.d_lat)
            z_topk = torch.cat([z_topk, pad], dim=1)

        u = torch.cat([y, z_i, z_topk.reshape(B, self.K * self.d_lat)], dim=-1)
        x_hat = self.f(u)
        return (x_hat - y) / (self.sigma ** 2)                                  # Tweedie score

    @torch.no_grad()
    def topk_indices(self, y: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """Indices of the K latent-nearest neighbours per pixel (same selection as
        _forward_inner). Inputs are RAW (whitened internally). Used by DARTS-CFAR."""
        y = self.whiten(y)
        neighbors = self.whiten(neighbors)
        B, M, D = neighbors.shape
        z_i = self.phi(y)
        z_j = self.phi(neighbors.reshape(B * M, D)).reshape(B, M, self.d_lat)
        dists = ((z_j - z_i.unsqueeze(1)) ** 2).sum(-1)
        K_eff = min(self.K, M)
        return dists.topk(K_eff, dim=1, largest=False).indices


def neighbor_mlp_dsm_loss(model: NeighborMLPDenoiser, x: torch.Tensor,
                          neighbors: torch.Tensor) -> torch.Tensor:
    """DSM loss for DARTS: noise ONLY the query pixel; predict the denoising
    direction from the clean neighbour context. Noise is added in whitened space."""
    sigma = model.sigma
    x_w   = model.whiten(x)
    nbr_w = model.whiten(neighbors)
    eps   = torch.randn_like(x_w) * sigma
    y     = x_w + eps
    target = -eps / (sigma ** 2)
    score  = model._forward_inner(y, nbr_w)
    return ((score - target) ** 2).sum(-1).mean()


def _batch_scores(model, pix, nbr, batch_size=512):
    """Evaluate a NeighborMLPDenoiser on (pix, nbr) in batches -> (N, D) numpy."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(pix), batch_size):
            p = torch.tensor(pix[i:i + batch_size], dtype=torch.float32).to(device)
            n = torch.tensor(nbr[i:i + batch_size], dtype=torch.float32).to(device)
            out.append(model(p, n).cpu().numpy())
    return np.concatenate(out, axis=0)


def score_nmlp_additive(model: NeighborMLPDenoiser,
                        test_pix: np.ndarray, test_nbr: np.ndarray,
                        train_pix: np.ndarray, train_nbr: np.ndarray,
                        s: np.ndarray) -> np.ndarray:
    """DARTS additive-LMP statistic (same normalisation convention as DART):
        T(y) = -( (psi(y) - psi_bar)^T s ) / sqrt( s^T C_psi s )."""
    z_train = _batch_scores(model, train_pix, train_nbr)
    z_test  = _batch_scores(model, test_pix,  test_nbr)
    z_bar   = z_train.mean(axis=0)
    C_psi   = np.cov(z_train, rowvar=False)
    if C_psi.ndim == 0:
        C_psi = np.array([[float(C_psi)]])
    norm = float(np.sqrt(max(float(s @ C_psi @ s), 1e-12)))
    return -((z_test - z_bar) @ s) / norm
