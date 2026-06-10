"""
cfattn_model.py — Closed-Form Attention Gaussian Score Network.

Architecture (following METHOD.md):

  Per-pixel score estimator:
    1.  Embed M spatial neighbors via shared MLP phi: D -> h -> h
    2.  Build a neighborhood-context query from (mean_pool, var_pool) of embeddings
    3.  Attend over neighbors + K learned Gaussian atoms  (attention is y-free)
    4.  Moment-match the weighted mixture to a single Gaussian (mu_i, Sigma_i)
    5.  Closed-form score: s_i = inv(Sigma_i + sigma^2*I) @ (mu_i - y_i)

CRITICAL: attention weights do NOT depend on y_i.
This keeps the score AFFINE in y_i — required for the Rao score test.

Detection statistic:
    T_i = s_i · target_signature

Larger T_i -> target more likely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CFAttnGaussianScoreNet(nn.Module):
    """Closed-Form Attention Gaussian Score Network."""

    def __init__(self, D: int, h: int = 64, K: int = 4,
                 sigma: float = 0.1, eps: float = 1e-4, whitening=None):
        """
        Parameters
        ----------
        D     : spectral / latent dimension (= D_raw when a whitening layer is used)
        h     : hidden dimension for all MLPs and attention
        K     : number of learned Gaussian atoms (global components)
        sigma : DSM noise level (also used in the closed-form inverse)
        eps   : numerical stability term added to the diagonal of Sigma_i
        whitening : optional frozen Whitening module (first layer). When present,
                    y AND neighbors are whitened before the closed-form Gaussian;
                    the net then operates in whitened space (comp_mu atoms live in
                    whitened space) and detection uses the whitened signature.
        """
        super().__init__()
        self.D = D
        self.h = h
        self.K = K
        self.sigma = sigma
        self.eps = eps
        self.whitening = whitening

        # Neighbor embedding: D -> h -> h (ReLU)
        self.phi = nn.Sequential(
            nn.Linear(D, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )

        # Query from neighborhood context: [mean | var] -> h
        self.query_net = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(),
            nn.Linear(h, h),
        )

        # Key projection for neighbor embeddings
        self.key_net = nn.Linear(h, h)

        # Context-dependent temperature: h -> scalar in (0.05, 1.05)
        self.temp_head = nn.Linear(h, 1)

        # Learned component means (initialized externally via k-means++)
        self.comp_mu  = nn.Parameter(torch.zeros(K, D))

        # Cholesky factors for component covariances: K x D x D lower-triangular
        # Initialize near 0.5 * I (near-isotropic, moderate spread)
        L_init = torch.eye(D).unsqueeze(0).expand(K, -1, -1).clone() * 0.5
        self.comp_L_raw = nn.Parameter(L_init + 0.01 * torch.randn(K, D, D))

        # Learned component keys: K x h
        self.comp_key = nn.Parameter(torch.randn(K, h) * h ** -0.5)

    # ------------------------------------------------------------------
    def _comp_cov(self) -> torch.Tensor:
        """PSD covariance matrices from Cholesky factors. Returns [K, D, D]."""
        # Lower triangular part
        L = torch.tril(self.comp_L_raw)                          # [K, D, D]
        # Positive diagonal via softplus
        d = torch.arange(self.D, device=L.device)
        eye = torch.eye(self.D, device=L.device).unsqueeze(0)    # [1, D, D]
        diag_pos = F.softplus(self.comp_L_raw[:, d, d]) + 1e-4   # [K, D]
        L = L * (1 - eye) + torch.diag_embed(diag_pos)           # [K, D, D]
        return L @ L.transpose(-1, -2)                            # [K, D, D]

    # ------------------------------------------------------------------
    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        return self.whitening(x) if self.whitening is not None else x

    def to_data_space(self, score_w: torch.Tensor) -> torch.Tensor:
        """Un-whiten a whitened-space score into a DATA-SPACE score (score_w @ W)."""
        return score_w @ self.whitening.W if self.whitening is not None else score_w

    def forward(self, y: torch.Tensor,
                neighbors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Public forward: whiten raw y + neighbors, compute the closed-form score
        in whitened space, then map back to DATA space (detection uses raw sig)."""
        s_w, w = self._forward_inner(self.whiten(y), self.whiten(neighbors))
        return self.to_data_space(s_w), w

    def _forward_inner(self, y: torch.Tensor,
                       neighbors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form score on ALREADY-WHITENED inputs.

        Parameters
        ----------
        y         : [B, D]    query pixels (may be noised for training)
        neighbors : [B, M, D] spatial neighbor spectra (always clean)

        Returns
        -------
        s_i : [B, D]    score vectors (affine in y)
        w   : [B, M+K]  attention weights (returned for regularization)
        """
        B, M, D = neighbors.shape
        device   = y.device

        # --- Step 1: embed neighbors ---
        z_nbr = self.phi(neighbors.reshape(B * M, D)).reshape(B, M, self.h)  # [B,M,h]

        # --- Step 2: neighborhood-context query (y-free) ---
        mean_pool = z_nbr.mean(dim=1)                                 # [B, h]
        var_pool  = z_nbr.var(dim=1, unbiased=False).clamp(min=0)     # [B, h]
        q = self.query_net(torch.cat([mean_pool, var_pool], dim=-1))  # [B, h]

        # --- Step 3: keys ---
        nbr_keys  = self.key_net(z_nbr)                               # [B, M, h]
        comp_keys = self.comp_key.unsqueeze(0).expand(B, -1, -1)     # [B, K, h]
        all_keys  = torch.cat([nbr_keys, comp_keys], dim=1)          # [B, M+K, h]

        # --- Step 4: attention (NO y dependency) ---
        logits = (q.unsqueeze(1) * all_keys).sum(-1) * (self.h ** -0.5)  # [B, M+K]
        temp   = torch.sigmoid(self.temp_head(q)) + 0.05                  # [B, 1]
        w      = torch.softmax(logits / temp, dim=-1)                     # [B, M+K]

        # --- Step 5: candidate means and covariances ---
        comp_mu  = self.comp_mu.unsqueeze(0).expand(B, -1, -1)    # [B, K, D]
        cand_mu  = torch.cat([neighbors, comp_mu], dim=1)         # [B, M+K, D]
        comp_cov = self._comp_cov().unsqueeze(0).expand(B, -1, -1, -1)  # [B,K,D,D]

        # --- Step 6: moment-matched Gaussian ---
        w3 = w.unsqueeze(-1)                                       # [B, M+K, 1]
        mu_i  = (w3 * cand_mu).sum(1)                             # [B, D]

        diff  = cand_mu - mu_i.unsqueeze(1)                       # [B, M+K, D]
        outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)           # [B, M+K, D, D]
        w4    = w.unsqueeze(-1).unsqueeze(-1)                     # [B, M+K, 1, 1]

        # Neighbors: point masses (zero intrinsic covariance)
        Sigma_nbr  = (w4[:, :M] * outer[:, :M]).sum(1)           # [B, D, D]
        # Components: intrinsic covariance + between-atom spread
        Sigma_comp = (w4[:, M:] * (comp_cov + outer[:, M:])).sum(1)  # [B, D, D]
        Sigma_i    = Sigma_nbr + Sigma_comp                           # [B, D, D]

        # --- Step 7: closed-form score (solve, not invert, for stability) ---
        reg     = (self.sigma ** 2 + self.eps) * torch.eye(D, device=device)
        A_Sigma = Sigma_i + reg.unsqueeze(0)                      # [B, D, D]
        rhs     = (mu_i - y).unsqueeze(-1)                        # [B, D, 1]
        s_i     = torch.linalg.solve(A_Sigma, rhs).squeeze(-1)   # [B, D]

        return s_i, w


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def cfattn_dsm_loss(model: CFAttnGaussianScoreNet,
                    x: torch.Tensor,
                    neighbors: torch.Tensor,
                    lam_ent: float = 0.05,
                    lam_div: float = 0.05,
                    lam_cov: float = 1e-5) -> tuple[torch.Tensor, float]:
    """
    DSM loss + regularization for the CF-Attention model.

    Parameters
    ----------
    x         : [B, D]    clean background pixels
    neighbors : [B, M, D] their spatial neighbors (always clean)

    Returns
    -------
    total_loss : scalar tensor (differentiable)
    dsm_item   : float (raw DSM loss, for logging)
    """
    sigma = model.sigma
    # Whiten first, then add DSM noise IN WHITENED SPACE (consistent with the
    # whitened-space score the closed-form head produces).
    x_w   = model.whiten(x)
    nbr_w = model.whiten(neighbors)
    eps_   = torch.randn_like(x_w) * sigma
    y_tilde = x_w + eps_
    target  = -eps_ / (sigma ** 2)                                # [B, D]

    s_i, w = model._forward_inner(y_tilde, nbr_w)
    loss_dsm = ((s_i - target) ** 2).sum(-1).mean()

    # Entropy penalty: high entropy (flat attention) is penalized
    H_w = -(w * (w + 1e-8).log()).sum(-1).mean()

    # Component diversity: prevent dead atoms
    M = neighbors.shape[1]
    w_comp      = w[:, M:]                           # [B, K]
    mean_w_comp = w_comp.mean(0) + 1e-8              # [K]
    H_comp      = -(mean_w_comp * mean_w_comp.log()).sum()

    # Covariance regularization: prevent unbounded Cholesky factors
    loss_cov = (torch.tril(model.comp_L_raw) ** 2).sum()

    total = loss_dsm + lam_ent * H_w - lam_div * H_comp + lam_cov * loss_cov
    return total, loss_dsm.item()


# ---------------------------------------------------------------------------
# Scoring at inference
# ---------------------------------------------------------------------------

def score_cfattn_additive(model: CFAttnGaussianScoreNet,
                           test_pix: np.ndarray,
                           test_nbr: np.ndarray,
                           train_pix: np.ndarray,
                           train_nbr: np.ndarray,
                           s: np.ndarray,
                           batch_size: int = 512) -> np.ndarray:
    """
    Additive model score: T_i = s_i · s_sig, normalized by training stats.

    Normalization mirrors dsm_additive: subtract training mean, divide by
    sqrt(s^T C_psi s), where C_psi = cov of score vectors on training set.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        def _scores(pix, nbr):
            out = []
            for i in range(0, len(pix), batch_size):
                p = torch.tensor(pix[i:i+batch_size], dtype=torch.float32).to(device)
                n = torch.tensor(nbr[i:i+batch_size], dtype=torch.float32).to(device)
                si, _ = model(p, n)
                out.append(si.cpu().numpy())
            return np.concatenate(out, axis=0)

        z_train = _scores(train_pix, train_nbr)    # (N_tr, D)
        z_test  = _scores(test_pix,  test_nbr)     # (N_te, D)

    z_bar  = z_train.mean(axis=0)
    C_psi  = np.cov(z_train, rowvar=False)
    if C_psi.ndim == 0:
        C_psi = np.array([[float(C_psi)]])
    norm   = float(np.sqrt(max(float(s @ C_psi @ s), 1e-12)))
    return -((z_test - z_bar) @ s) / norm


def score_cfattn_replacement(model: CFAttnGaussianScoreNet,
                              test_pix: np.ndarray,
                              test_nbr: np.ndarray,
                              train_pix: np.ndarray,
                              train_nbr: np.ndarray,
                              s: np.ndarray,
                              batch_size: int = 512) -> np.ndarray:
    """
    Replacement model score: T_i = psi(y)^T (y - s), normalized.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        def _scores(pix, nbr):
            out = []
            for i in range(0, len(pix), batch_size):
                p = torch.tensor(pix[i:i+batch_size], dtype=torch.float32).to(device)
                n = torch.tensor(nbr[i:i+batch_size], dtype=torch.float32).to(device)
                si, _ = model(p, n)
                out.append(si.cpu().numpy())
            return np.concatenate(out, axis=0)

        psi_train = _scores(train_pix, train_nbr)
        psi_test  = _scores(test_pix,  test_nbr)

    psi_bar = psi_train.mean(axis=0)                       # (D,) per-dim score mean
    r_train = ((psi_train - psi_bar) * (train_pix - s)).sum(axis=1)
    r_bar, r_std = r_train.mean(), r_train.std() + 1e-12
    r_test  = ((psi_test  - psi_bar) * (test_pix  - s)).sum(axis=1)
    return (r_test - r_bar) / r_std


# ---------------------------------------------------------------------------
# Local Fisher (CFAR) normalization — T̂_head  (Eq. 65 in paper)
# ---------------------------------------------------------------------------

def _cfattn_cfar_forward(model: CFAttnGaussianScoreNet,
                          y: np.ndarray,
                          nbr: np.ndarray,
                          s: np.ndarray,
                          batch_size: int = 512) -> np.ndarray:
    """
    Compute T̂_head_i = s^T Â_i (μ̂_i − y_i) / sqrt(s^T Â_i s + ε)
    where Â_i = (Σ̂_i + (σ²+ε)I)^{-1}.

    No global training statistics needed — normalization is local (per-pixel).
    This is the CFAR variant that should give a flat FPR across all background
    classes under the local Gaussian assumption.
    """
    model.eval()
    device = next(model.parameters()).device
    # The caller passes the RAW (data-space) signature. The local-Fisher head Â_i
    # is built in WHITENED space, so we whiten the signature here (W·s). The
    # resulting statistic is whitening-invariant — identical to the data-space form.
    s_use = model.whitening.transform_direction(s) if model.whitening is not None else s
    s_t = torch.tensor(np.asarray(s_use, dtype=np.float32)).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(y), batch_size):
            y_b   = torch.tensor(y[i:i+batch_size],   dtype=torch.float32).to(device)
            nbr_b = torch.tensor(nbr[i:i+batch_size], dtype=torch.float32).to(device)
            # whiten raw inputs (local Fisher computed in whitened space)
            y_b   = model.whiten(y_b)
            nbr_b = model.whiten(nbr_b)
            B, D  = y_b.shape

            # Re-run the forward pass to get mu_i and A_Sigma
            # (same computation as model.forward, but we need the matrices)
            M_      = nbr_b.shape[1]
            z_nbr   = model.phi(nbr_b.reshape(B * M_, D)).reshape(B, M_, model.h)
            mean_p  = z_nbr.mean(1)
            var_p   = z_nbr.var(1, unbiased=False).clamp(min=0)
            q       = model.query_net(torch.cat([mean_p, var_p], -1))
            nbr_keys  = model.key_net(z_nbr)
            comp_keys = model.comp_key.unsqueeze(0).expand(B, -1, -1)
            all_keys  = torch.cat([nbr_keys, comp_keys], dim=1)
            logits  = (q.unsqueeze(1) * all_keys).sum(-1) * (model.h ** -0.5)
            temp    = torch.sigmoid(model.temp_head(q)) + 0.05
            w       = torch.softmax(logits / temp, dim=-1)

            comp_mu  = model.comp_mu.unsqueeze(0).expand(B, -1, -1)
            cand_mu  = torch.cat([nbr_b, comp_mu], dim=1)
            comp_cov = model._comp_cov().unsqueeze(0).expand(B, -1, -1, -1)

            w3 = w.unsqueeze(-1)
            mu_i = (w3 * cand_mu).sum(1)

            diff  = cand_mu - mu_i.unsqueeze(1)
            outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)
            w4    = w.unsqueeze(-1).unsqueeze(-1)
            Sigma_nbr  = (w4[:, :M_] * outer[:, :M_]).sum(1)
            Sigma_comp = (w4[:, M_:] * (comp_cov + outer[:, M_:])).sum(1)
            Sigma_i    = Sigma_nbr + Sigma_comp

            reg     = (model.sigma ** 2 + model.eps) * torch.eye(D, device=y_b.device)
            A_Sigma = Sigma_i + reg.unsqueeze(0)   # [B, D, D]

            # T̂_head = s^T Â_i (μ_i − y_i) / sqrt(s^T Â_i s + ε)
            # numerator:   A_i_s = A_Sigma^{-1} s,   dot with (mu_i - y_i)
            # denominator: A_i_s · s
            s_exp   = s_t.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
            A_i_s   = torch.linalg.solve(A_Sigma, s_exp).squeeze(-1)   # [B, D]

            rhs     = (y_b - mu_i)                                      # [B, D]  (+sign: higher = target)
            numer   = (A_i_s * rhs).sum(-1)                             # [B]
            denom   = torch.sqrt((A_i_s * s_t.unsqueeze(0)).sum(-1)
                                 .clamp(min=model.eps))                  # [B]
            t_head  = numer / denom                                      # [B]
            out.append(t_head.cpu().numpy())

    return np.concatenate(out, axis=0)


def score_cfattn_additive_cfar(model: CFAttnGaussianScoreNet,
                                test_pix: np.ndarray,
                                test_nbr: np.ndarray,
                                s: np.ndarray,
                                batch_size: int = 512) -> np.ndarray:
    """
    Additive model CFAR score — local Fisher normalization T̂_head (Eq. 65).

    T̂_head_i = s^T Â_i (μ̂_i − y_i) / sqrt(s^T Â_i s + ε)

    No global training statistics needed — normalization is local (per-pixel).
    Theoretically CFAR under the local Gaussian model for the test pixels.
    """
    return _cfattn_cfar_forward(model, test_pix, test_nbr, s, batch_size)


def score_cfattn_replacement_cfar(model: CFAttnGaussianScoreNet,
                                   test_pix: np.ndarray,
                                   test_nbr: np.ndarray,
                                   s: np.ndarray,
                                   batch_size: int = 512) -> np.ndarray:
    """
    Replacement model CFAR score — local Fisher normalization T̂_head (Eq. 65).

    For the replacement model y = (1-θ)w + θs, the score inner product gives
    the same T̂_head; the replacement-specific term (y-s) is captured in
    the numerator s^T Â_i (μ̂_i - y_i) when y contains the replacement signal.
    Use this with the replacement-planted test pixels.
    """
    return _cfattn_cfar_forward(model, test_pix, test_nbr, s, batch_size)
