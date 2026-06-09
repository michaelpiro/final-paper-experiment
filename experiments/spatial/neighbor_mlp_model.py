"""
neighbor_mlp_model.py — NeighborMLPDenoiser

A spatially-aware score estimator that is faster and simpler than CF-Attn:

  1. Encode the query pixel AND all M spatial neighbors with a SHARED MLP phi.
  2. Select the K latent-space nearest neighbors (hard top-K, no gradient
     through the selection indices).
  3. Concatenate [y_i | z_i | z_j1 | ... | z_jK] and pass through a denoiser
     MLP f to get the reconstructed clean pixel x_hat.
  4. Score via the Tweedie identity: s(y_i) = (x_hat - y_i) / sigma^2.

Key properties:
  - Score is NONLINEAR in y (unlike CF-Attn's closed-form affine score).
  - Top-K selection is a soft form of attention — no heavy matrix operations.
  - Trains with standard DSM loss: E||s(y) - (x - y)/sigma^2||^2.
  - Detection: T_i = -s(y_i) · target  (additive LMP, same as DSM).
"""

import numpy as np
import torch
import torch.nn as nn


class NeighborMLPDenoiser(nn.Module):
    """Spatially-aware denoising score estimator via top-K neighbor selection."""

    def __init__(self, D: int, d_lat: int = 32, K: int = 8,
                 hidden: int = 128, n_layers: int = 3,
                 sigma: float = 0.1, activation: str = 'silu',
                 whitening=None):
        """
        Parameters
        ----------
        D         : spectral / latent dimension
        d_lat     : shared encoder output dimension
        K         : number of latent-nearest neighbors to use (K <= M)
        hidden    : width of both phi and f MLPs
        n_layers  : total layers in each MLP (including input/output)
        sigma     : DSM noise level (also used in the Tweedie score formula)
        activation: 'silu' or 'relu'
        """
        super().__init__()
        self.D     = D
        self.d_lat = d_lat
        self.K     = K
        self.sigma = sigma
        self.whitening = whitening

        act_cls = {'silu': nn.SiLU, 'relu': nn.ReLU}[activation]

        def _mlp(in_dim, out_dim, hidden, n_layers):
            layers = [nn.Linear(in_dim, hidden), act_cls()]
            for _ in range(max(n_layers - 2, 0)):
                layers += [nn.Linear(hidden, hidden), act_cls()]
            layers.append(nn.Linear(hidden, out_dim))
            return nn.Sequential(*layers)

        # Shared encoder: D -> d_lat
        self.phi = _mlp(D, d_lat, hidden, n_layers)

        # Denoiser: [y_i | z_i | z_j1...z_jK] -> D
        self.f = _mlp(D + d_lat * (1 + K), D, hidden, n_layers)

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        return self.whitening(x) if self.whitening is not None else x

    def to_data_space(self, score_w: torch.Tensor) -> torch.Tensor:
        """Un-whiten a whitened-space score into a DATA-SPACE score (score_w @ W)."""
        return score_w @ self.whitening.W if self.whitening is not None else score_w

    def forward(self, y: torch.Tensor,
                neighbors: torch.Tensor) -> torch.Tensor:
        """Public forward: whiten raw y + neighbors, run the Tweedie score in
        whitened space, then map back to DATA space (detection uses raw signature)."""
        score_w = self._forward_inner(self.whiten(y), self.whiten(neighbors))
        return self.to_data_space(score_w)

    def _forward_inner(self, y: torch.Tensor,
                       neighbors: torch.Tensor) -> torch.Tensor:
        """
        Score on ALREADY-WHITENED inputs.

        Parameters
        ----------
        y         : (B, D)    noisy observation (corrupted for training, raw for eval)
        neighbors : (B, M, D) clean spatial neighbor spectra

        Returns
        -------
        score : (B, D)    estimated score ∇ log p(y)  =  (x_hat - y) / sigma^2
        """
        B, M, D = neighbors.shape

        # --- 1. Encode ---
        z_i = self.phi(y)                                              # (B, d_lat)
        z_j = self.phi(neighbors.reshape(B * M, D)).reshape(B, M, self.d_lat)  # (B, M, d_lat)

        # --- 2. Top-K selection by L2 in latent space ---
        # No gradient through the selection indices.
        with torch.no_grad():
            dists   = ((z_j - z_i.unsqueeze(1)) ** 2).sum(-1)        # (B, M)
            K_eff   = min(self.K, M)
            topk_idx = dists.topk(K_eff, dim=1, largest=False).indices  # (B, K)

        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.d_lat)
        z_topk = z_j.gather(1, topk_idx_exp)                          # (B, K, d_lat)

        # --- 3. Concatenate ---
        u = torch.cat([y, z_i, z_topk.reshape(B, K_eff * self.d_lat)], dim=-1)

        # --- 4. Denoiser ---
        x_hat = self.f(u)                                              # (B, D)

        # --- 5. Tweedie score ---
        return (x_hat - y) / (self.sigma ** 2)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def neighbor_mlp_dsm_loss(model: NeighborMLPDenoiser,
                          x: torch.Tensor,
                          neighbors: torch.Tensor) -> torch.Tensor:
    """Standard DSM loss for NeighborMLPDenoiser.

    Corrupts x with Gaussian noise, asks model to predict denoising direction.
    """
    sigma = model.sigma
    # Whiten first, then add DSM noise IN WHITENED SPACE.
    x_w   = model.whiten(x)
    nbr_w = model.whiten(neighbors)
    eps   = torch.randn_like(x_w) * sigma
    y     = x_w + eps
    target = -eps / (sigma ** 2)               # (B, D) — the true score direction
    score  = model._forward_inner(y, nbr_w)    # (B, D)
    return ((score - target) ** 2).sum(-1).mean()


# ---------------------------------------------------------------------------
# Scoring at inference (mirrors dsm_additive / dsm_replacement convention)
# ---------------------------------------------------------------------------

def _batch_scores(model, pix, nbr, batch_size=512):
    """Evaluate model on (pix, nbr) in batches. Returns (N, D) numpy."""
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
    """
    Additive LMP statistic:
        T(y) = -( (psi(y) - psi_bar)^T s ) / sqrt( s^T C_psi s )

    Mirrors dsm_additive exactly — same normalization convention.
    """
    z_train = _batch_scores(model, train_pix, train_nbr)
    z_test  = _batch_scores(model, test_pix,  test_nbr)
    z_bar   = z_train.mean(axis=0)
    C_psi   = np.cov(z_train, rowvar=False)
    if C_psi.ndim == 0:
        C_psi = np.array([[float(C_psi)]])
    norm = float(np.sqrt(max(float(s @ C_psi @ s), 1e-12)))
    return -((z_test - z_bar) @ s) / norm


def score_nmlp_replacement(model: NeighborMLPDenoiser,
                           test_pix: np.ndarray, test_nbr: np.ndarray,
                           train_pix: np.ndarray, train_nbr: np.ndarray,
                           s: np.ndarray) -> np.ndarray:
    """
    Replacement LMP statistic (centered score, same convention as dsm_replacement):
        T(y) = ( (psi(y) - psi_bar)^T (y - s) - r_bar ) / std(r_train)
    """
    psi_train = _batch_scores(model, train_pix, train_nbr)
    psi_test  = _batch_scores(model, test_pix,  test_nbr)
    psi_bar   = psi_train.mean(axis=0)
    r_train   = ((psi_train - psi_bar) * (train_pix - s)).sum(axis=1)
    r_bar, r_std = r_train.mean(), r_train.std() + 1e-12
    r_test    = ((psi_test - psi_bar) * (test_pix - s)).sum(axis=1)
    return (r_test - r_bar) / r_std
