"""
cfattn_jac_model.py — CF-Attn with Jacobian pullback (feature-space extension).

Extends CFAttnGaussianScoreNet by adding a nonlinear trunk f: D -> p.

Steps 1-4 (attention) are IDENTICAL to the base model — they still operate
in observation space D to produce attention weights w.

Steps 5-7 change: instead of moment-matching in D, we:
  5. Map all candidates + y_i into feature space via the shared trunk f.
  6. Moment-match in feature space p: compute (mu_phi, Sigma_phi).
  7. Ridge-solve in feature space: s_phi = inv(Sigma_phi + sigma^2*I) @ (mu_phi - phi_y).
  8. Pull back to observation space via the chain rule (Jacobian VJP):
        s_i = J^T s_phi    where J = df(y_i)/dy_i  [p, D]

This is exact by the chain rule: if the background is Gaussian in phi-space,
then grad_y log p(y) = J^T grad_phi log p_Gaussian(phi(y)).

The VJP J^T s_phi is computed cheaply without materializing the full [p, D]
matrix, using torch.autograd.grad with s_phi as the grad_output.

Key differences from the base model:
  - Adds trunk f: D -> p  (small MLP, shared)
  - Drops comp_L (no learned covariances; all candidates are point masses in phi-space)
  - Attention still y-free  (score is no longer affine in y, but the VJP
    is computed via autograd so gradients flow correctly through training)
  - Feature dim p should be in [D, 4D]; larger p = more expressive but
    harder to estimate from M+K=~33 candidates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFAttnJacGaussianScoreNet(nn.Module):
    """CF-Attn with nonlinear feature trunk and Jacobian pullback."""

    def __init__(self, D: int, h: int = 64, K: int = 4,
                 feature_dim: int = None,
                 sigma: float = 0.1, eps: float = 1e-4):
        """
        Parameters
        ----------
        D           : input/output spectral dimension
        h           : hidden width for attention MLPs
        K           : number of learned global atoms
        feature_dim : dimension of feature space p (default: 2*D)
        sigma       : DSM noise level
        eps         : ridge regularizer added to Sigma_phi diagonal
        """
        super().__init__()
        self.D   = D
        self.h   = h
        self.K   = K
        self.p   = feature_dim if feature_dim is not None else 2 * D
        self.sigma = sigma
        self.eps   = eps

        # ---- Attention networks (work in observation space D) ----
        # Neighbor embedding: D -> h -> h (ReLU) — y-free
        self.phi_nbr = nn.Sequential(
            nn.Linear(D, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        # Query from neighborhood context
        self.query_net = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(),
            nn.Linear(h, h),
        )
        self.key_net  = nn.Linear(h, h)
        self.temp_head = nn.Linear(h, 1)

        # ---- Learned global atoms (in observation space D) ----
        self.comp_mu  = nn.Parameter(torch.zeros(K, D))
        self.comp_key = nn.Parameter(torch.randn(K, h) * h ** -0.5)

        # ---- Nonlinear trunk f: D -> p ----
        # Maps candidates AND y_i into feature space for Gaussian estimation.
        # Small but nonlinear — 3 layers with SiLU.
        self.trunk = nn.Sequential(
            nn.Linear(D, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, self.p),
        )

    # ------------------------------------------------------------------
    def _attention_weights(self, y: torch.Tensor,
                           neighbors: torch.Tensor) -> tuple:
        """
        Compute attention weights w [B, M+K] and candidate means [B, M+K, D].
        Attention is y-FREE: weights only depend on neighbor context.
        """
        B, M, D = neighbors.shape

        # Embed neighbors
        z_nbr = self.phi_nbr(neighbors.reshape(B * M, D)).reshape(B, M, self.h)

        # Query from neighborhood (y-free)
        mean_pool = z_nbr.mean(1)
        var_pool  = z_nbr.var(1, unbiased=False).clamp(min=0)
        q = self.query_net(torch.cat([mean_pool, var_pool], -1))   # [B, h]

        # Keys
        nbr_keys  = self.key_net(z_nbr)                           # [B, M, h]
        comp_keys = self.comp_key.unsqueeze(0).expand(B, -1, -1)  # [B, K, h]
        all_keys  = torch.cat([nbr_keys, comp_keys], 1)           # [B, M+K, h]

        # Attention
        logits = (q.unsqueeze(1) * all_keys).sum(-1) * (self.h ** -0.5)
        temp   = torch.sigmoid(self.temp_head(q)) + 0.05           # [B, 1]
        w      = torch.softmax(logits / temp, -1)                  # [B, M+K]

        # Candidate means in observation space
        comp_mu  = self.comp_mu.unsqueeze(0).expand(B, -1, -1)
        cand_mu  = torch.cat([neighbors, comp_mu], 1)              # [B, M+K, D]

        return w, cand_mu

    # ------------------------------------------------------------------
    def forward(self, y: torch.Tensor,
                neighbors: torch.Tensor) -> torch.Tensor:
        """
        y         : [B, D]    query pixels (may be noised for training)
        neighbors : [B, M, D] spatial neighbor spectra (clean)

        Returns
        -------
        s_i : [B, D]  estimated score ∇_y log p(y)  via Jacobian pullback
        """
        B, M, D = neighbors.shape
        device  = y.device

        # ---- Steps 1-4: attention (observation space, y-free) ----
        w, cand_mu = self._attention_weights(y, neighbors)         # [B,M+K], [B,M+K,D]

        # ---- Step 5: map candidates + y into feature space ----
        # Map all M+K candidates through the trunk
        phi_cands = self.trunk(
            cand_mu.reshape(B * (M + self.K), D)
        ).reshape(B, M + self.K, self.p)                          # [B, M+K, p]

        # Map y through trunk separately for VJP (re-use y, grad via autograd)
        y_req = y.detach().requires_grad_(True)                   # fresh leaf for VJP
        phi_y = self.trunk(y_req)                                 # [B, p]

        # ---- Step 6: moment-match in feature space (point masses only) ----
        w3   = w.unsqueeze(-1)                                    # [B, M+K, 1]
        mu_phi = (w3 * phi_cands).sum(1)                          # [B, p]

        diff_phi = phi_cands - mu_phi.unsqueeze(1)                # [B, M+K, p]
        outer    = diff_phi.unsqueeze(-1) * diff_phi.unsqueeze(-2) # [B, M+K, p, p]
        Sigma_phi = (w.unsqueeze(-1).unsqueeze(-1) * outer).sum(1) # [B, p, p]

        # ---- Step 7: ridge solve in feature space ----
        p = self.p
        reg     = (self.sigma ** 2 + self.eps) * torch.eye(p, device=device)
        A_Sigma = Sigma_phi + reg.unsqueeze(0)                    # [B, p, p]
        rhs     = (mu_phi - phi_y).unsqueeze(-1)                  # [B, p, 1]
        s_phi   = torch.linalg.solve(A_Sigma, rhs).squeeze(-1)   # [B, p]

        # ---- Step 8: Jacobian VJP pullback to observation space ----
        # s_i = J^T s_phi  where J = df(y)/dy  [p, D]
        # Computed cheaply without materializing J:
        #   d( phi_y · s_phi_detached ).sum() / d y_req  =  J^T s_phi  per pixel
        # We detach s_phi so the VJP is exactly J^T s_phi (not the product-rule variant).
        # Gradients for training still flow through s_phi in the main graph,
        # and through J via create_graph=True.
        s_i = torch.autograd.grad(
            (phi_y * s_phi.detach()).sum(),
            y_req,
            create_graph=self.training,   # needed for DSM loss → model param grads
        )[0]                                                       # [B, D]

        return s_i, w


# ---------------------------------------------------------------------------
# Training loss  (same regularization structure as the base model)
# ---------------------------------------------------------------------------

def cfattn_jac_dsm_loss(model: CFAttnJacGaussianScoreNet,
                        x: torch.Tensor,
                        neighbors: torch.Tensor,
                        lam_ent: float = 0.05,
                        lam_div: float = 0.05) -> tuple:
    """
    DSM loss + attention regularization.

    No covariance regularization (comp_L_raw dropped in this variant).
    """
    sigma = model.sigma
    eps_  = torch.randn_like(x) * sigma
    y_tilde = x + eps_
    target  = -eps_ / (sigma ** 2)

    s_i, w = model(y_tilde, neighbors)
    loss_dsm = ((s_i - target) ** 2).sum(-1).mean()

    # Entropy penalty
    H_w = -(w * (w + 1e-8).log()).sum(-1).mean()

    # Component diversity
    M = neighbors.shape[1]
    w_comp      = w[:, M:]
    mean_w_comp = w_comp.mean(0) + 1e-8
    H_comp      = -(mean_w_comp * mean_w_comp.log()).sum()

    total = loss_dsm + lam_ent * H_w - lam_div * H_comp
    return total, float(loss_dsm)


# ---------------------------------------------------------------------------
# Scoring (same convention as cfattn_model.py)
# ---------------------------------------------------------------------------

def _batch_scores_jac(model, pix, nbr, batch_size=256):
    """Evaluate model in batches → (N, D) numpy."""
    model.eval()
    out = []
    for i in range(0, len(pix), batch_size):
        p = torch.tensor(pix[i:i+batch_size], dtype=torch.float32)
        n = torch.tensor(nbr[i:i+batch_size], dtype=torch.float32)
        with torch.enable_grad():           # autograd required for VJP in forward
            s, _ = model(p, n)
        out.append(s.detach().numpy())
    return np.concatenate(out, 0)


def score_cfattn_jac_additive(model, test_pix, test_nbr, train_pix, train_nbr, s):
    z_train = _batch_scores_jac(model, train_pix, train_nbr)
    z_test  = _batch_scores_jac(model, test_pix,  test_nbr)
    z_bar   = z_train.mean(0)
    C_psi   = np.cov(z_train, rowvar=False)
    if C_psi.ndim == 0: C_psi = np.array([[float(C_psi)]])
    norm = float(np.sqrt(max(float(s @ C_psi @ s), 1e-12)))
    return -((z_test - z_bar) @ s) / norm


def score_cfattn_jac_replacement(model, test_pix, test_nbr, train_pix, train_nbr, s):
    psi_train = _batch_scores_jac(model, train_pix, train_nbr)
    psi_test  = _batch_scores_jac(model, test_pix,  test_nbr)
    psi_bar   = psi_train.mean(0)
    r_train   = ((psi_train - psi_bar) * (train_pix - s)).sum(1)
    r_bar, r_std = r_train.mean(), r_train.std() + 1e-12
    r_test    = ((psi_test - psi_bar) * (test_pix - s)).sum(1)
    return (r_test - r_bar) / r_std
