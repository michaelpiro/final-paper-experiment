"""
Neighbor-Adapted Score Matching (paper Section 6.2).

A shared nonlinear encoder  h_theta : R^D -> R^M  lifts each pixel to a
high-dimensional feature.  For every pixel i, a *local* linear score head
W_i in R^{D x M} is solved IN CLOSED FORM by ridge regression over the
pixel's k x k spatial neighborhood, regularized toward a globally-shared
prior head W0.  The score at the center pixel is  s_hat_i = W_i h(x_i).

Ridge (per pixel i), with neighbor features H_i (M x K) and neighbor score
targets S_i (D x K):

    W_i = (S_i H_i^T + lambda W0) (H_i H_i^T + lambda I_M)^{-1}

Because the neighborhood is small (K = k*k - 1) but M is large ("power to the
last layer"), we never form the M x M inverse.  Woodbury reduces it to a
K x K solve:

    (H_i H_i^T + lambda I_M)^{-1} phi
        = (1/lambda) [ phi - H_i (lambda I_K + H_i^T H_i)^{-1} H_i^T phi ]

Targets are denoising-score-matching directions (no local Gaussian assumed):

    x_tilde = x + sigma * eps ,   s_target = (x - x_tilde) / sigma^2

so the head fits a local DSM score operator and the Gaussian case (Sec 6.1)
is recovered exactly when the local cloud is Gaussian.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial neighborhood extraction
# ---------------------------------------------------------------------------

def extract_neighborhoods(img: torch.Tensor, k: int):
    """
    img : (H, W, D) tensor of per-pixel features (e.g. PCA-reduced bands).

    Returns
    -------
    centers   : (H*W, D)        the center pixel of each window
    neighbors : (H*W, k*k-1, D) the k x k window minus the center
    Reflect-padded at the image border.
    """
    H, W, D = img.shape
    p = k // 2
    x = img.permute(2, 0, 1).unsqueeze(0)           # (1, D, H, W)
    x = F.pad(x, (p, p, p, p), mode='reflect')
    patches = F.unfold(x, kernel_size=k, padding=0)  # (1, D*k*k, H*W)
    patches = patches.reshape(D, k * k, H * W).permute(2, 1, 0)  # (HW, k*k, D)

    center_idx = (k * k) // 2
    centers = patches[:, center_idx, :].contiguous()            # (HW, D)
    mask = torch.ones(k * k, dtype=torch.bool)
    mask[center_idx] = False
    neighbors = patches[:, mask, :].contiguous()                # (HW, k*k-1, D)
    return centers, neighbors


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(y)))


class NeighborAdaptedScore(nn.Module):
    def __init__(self, D: int, M: int = 256, hidden=(128, 128), k: int = 7,
                 lam_init: float = 0.1, learn_lambda: bool = True):
        super().__init__()
        self.D, self.M, self.k = D, M, k

        # Encoder h_theta : R^D -> R^M  (SiLU, linear final layer => raw features)
        dims = [D] + list(hidden) + [M]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        layers = layers[:-1]                       # drop final activation
        self.encoder = nn.Sequential(*layers)

        # Global prior head W0 : D x M
        self.W0 = nn.Parameter(torch.empty(D, M))
        nn.init.normal_(self.W0, std=1.0 / np.sqrt(M))

        # Ridge regularization lambda (kept positive via softplus)
        raw = torch.tensor(_inv_softplus(lam_init), dtype=torch.float32)
        if learn_lambda:
            self._raw_lambda = nn.Parameter(raw)
        else:
            self.register_buffer('_raw_lambda', raw)

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self._raw_lambda)

    def score(self, center: torch.Tensor, neigh: torch.Tensor,
              neigh_target: torch.Tensor) -> torch.Tensor:
        """
        center        : (B, D)      (already noised) center pixel
        neigh         : (B, K, D)   (already noised) neighbors
        neigh_target  : (B, K, D)   DSM score targets for the neighbors
        returns s_hat : (B, D)      adapted score at the center
        """
        B, K, D = neigh.shape
        M, lam = self.M, self.lam

        phi = self.encoder(center)                                  # (B, M)
        Hc  = self.encoder(neigh.reshape(B * K, D)).reshape(B, K, M)  # (B, K, M) = H_i^T
        S   = neigh_target.transpose(1, 2)                          # (B, D, K) = S_i

        eyeK = torch.eye(K, device=center.device, dtype=center.dtype).unsqueeze(0)
        A = torch.bmm(Hc, Hc.transpose(1, 2)) + lam * eyeK          # (B, K, K)
        b = torch.bmm(Hc, phi.unsqueeze(-1))                        # (B, K, 1) = H_i^T phi
        c = torch.linalg.solve(A, b)                                # (B, K, 1)
        Hic = torch.bmm(Hc.transpose(1, 2), c).squeeze(-1)          # (B, M) = H_i c
        g = (phi - Hic) / lam                                       # (B, M) = (HH^T+lam I)^-1 phi

        Htg = torch.bmm(Hc, g.unsqueeze(-1))                        # (B, K, 1) = H_i^T g
        term1 = torch.bmm(S, Htg).squeeze(-1)                       # (B, D) = S_i H_i^T g
        term2 = lam * (g @ self.W0.t())                             # (B, D) = lam W0 g
        return term1 + term2                                        # (B, D)


# ---------------------------------------------------------------------------
# DSM loss helper
# ---------------------------------------------------------------------------

def dsm_loss(model: NeighborAdaptedScore,
             center_clean: torch.Tensor, neigh_clean: torch.Tensor,
             sigma: float) -> torch.Tensor:
    """
    Denoising score-matching loss at the center pixels.
    center_clean : (B, D)
    neigh_clean  : (B, K, D)
    """
    eps_c = torch.randn_like(center_clean)
    eps_n = torch.randn_like(neigh_clean)
    center_t = center_clean + sigma * eps_c
    neigh_t  = neigh_clean + sigma * eps_n

    target_c = (center_clean - center_t) / (sigma ** 2)     # (B, D)
    target_n = (neigh_clean  - neigh_t) / (sigma ** 2)      # (B, K, D)

    s_hat = model.score(center_t, neigh_t, target_n)        # (B, D)
    return ((s_hat - target_c) ** 2).sum(dim=1).mean()


@torch.no_grad()
def adapted_score_field(model, centers, neighbors, sigma,
                        n_mc: int = 8, batch: int = 512):
    """
    Deterministic local score estimate  s_hat(x)  at each (CLEAN) center pixel.

    The local ridge head is built from the neighbors' DENOISING targets, which
    are stochastic; we MC-average the resulting score over n_mc noise draws of
    the neighbors (the center is always evaluated clean — this is where we want
    the score, including any planted target).

    centers   : (B, D) tensor  — query pixels, evaluated clean
    neighbors : (B, K, D) tensor — clean background neighbors
    returns   : (B, D) tensor  — averaged score field
    """
    model.eval()
    B, D = centers.shape
    out = torch.zeros(B, D)
    for i in range(0, B, batch):
        c  = centers[i:i + batch]
        nb = neighbors[i:i + batch]
        acc = torch.zeros_like(c)
        for _ in range(n_mc):
            eps  = torch.randn_like(nb) * sigma
            nb_t = nb + eps
            tgt  = -eps / (sigma ** 2)                  # (clean - noised)/sigma^2
            acc += model.score(c, nb_t, tgt)
        out[i:i + batch] = acc / n_mc
    return out


# ---------------------------------------------------------------------------
# Linear autoencoder  (PCA-like, but learned end-to-end + differentiable)
# ---------------------------------------------------------------------------

class LinearAutoencoder(nn.Module):
    """
    Plain linear AE  D ──enc──> latent ──dec──> D.

    With MSE loss the encoder's row-space converges to the PCA subspace
    (Bourlard & Kamp 1988); the linearity is essential here because it
    commutes with the additive / replacement target models:
        enc(w + theta * s)         = enc(w) + theta * enc(s)
        enc((1-theta)*w + theta*s) = (1-theta)*enc(w) + theta*enc(s).
    """
    def __init__(self, D: int, latent: int, bias: bool = True):
        super().__init__()
        self.D, self.latent = D, latent
        self.enc = nn.Linear(D, latent, bias=bias)
        self.dec = nn.Linear(latent, D, bias=bias)

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_linear_ae(model: LinearAutoencoder, pixels: torch.Tensor,
                    epochs: int = 500, batch: int = 512,
                    lr: float = 1e-3, weight_decay: float = 1e-5,
                    pbar=None) -> list:
    """Train the linear AE by plain MSE. Returns the per-epoch loss list."""
    model.train()
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    N    = pixels.shape[0]
    hist = []
    iterator = range(1, epochs + 1) if pbar is None else pbar(range(1, epochs + 1))
    for _ in iterator:
        perm = torch.randperm(N); tot = 0.0; nb = 0
        for i in range(0, N, batch):
            x = pixels[perm[i:i + batch]]
            x_hat, _ = model(x)
            loss = ((x_hat - x) ** 2).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        hist.append(tot / nb)
        if pbar is not None and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(loss=f"{hist[-1]:.4f}")
    model.eval()
    return hist
