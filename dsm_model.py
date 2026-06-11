import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def select_sigma_parzen(W: np.ndarray, sigma_lo: float = 0.02,
                        sigma_hi: float = 5.0, seed: int = 0,
                        criterion: str = "loglik") -> float:
    """Choose the DSM noise level sigma from the samples, before training.

    DSM at noise sigma estimates the score of q_sigma = p_w * N(0, sigma^2 I)
    -- exactly the Gaussian-kernel Parzen (KDE) density with bandwidth sigma.
    So picking sigma is a KDE-bandwidth problem, solved from the samples alone
    by leave-one-out cross-validation (no network training):

      criterion="loglik"     : maximize the LOO Parzen log-likelihood.
      criterion="scorematch" : minimize the LOO implicit-score-matching loss
                               (the objective DSM itself optimizes).

    NOTE: pass the data in the SAME space the DSM noise is added in. Here the
    score net whitens internally and DSM noise is isotropic in whitened space,
    so this must be called on the WHITENED training data; the returned sigma is
    then the whitened-space sigma used by dsm_loss.

    n is small in this problem, so we just build the full (n, n) pairwise
    squared-distance matrix once and reuse it across the 1-D golden-section
    search over log-sigma.
    """
    from scipy.special import logsumexp
    mu, std = W.mean(0), W.std(0) + 1e-8
    U = ((W - mu) / std).astype(np.float64)
    n, d = U.shape
    if n < 3:
        return 1.0

    # full pairwise squared distances (n, n); self-distance excluded (LOO)
    sqn = np.einsum("ij,ij->i", U, U)
    D = np.maximum(sqn[:, None] + sqn[None, :] - 2.0 * (U @ U.T), 0.0)
    diag = np.arange(n)
    Dself0 = D.copy()                       # self-distance 0 for weighted sums
    D[diag, diag] = np.inf                  # exclude self in the LOO kernel

    def negobj(log_sigma):
        s2 = np.exp(2.0 * log_sigma)
        logk = -D / (2.0 * s2)              # (n, n), -inf on the diagonal
        lse = logsumexp(logk, axis=1)       # sum over j != i
        if criterion == "loglik":
            ll = lse - np.log(n - 1) - 0.5 * d * np.log(2 * np.pi * s2)
            return -ll.mean()
        # score-matching: LOO implicit-score-matching loss of the KDE score
        r = np.exp(logk - lse[:, None])     # responsibilities (self = 0)
        m = r @ U - U                       # (n, d) = E_r[u_j] - u_i  = s2 * psi
        m2 = np.einsum("ij,ij->i", m, m)
        rD = np.einsum("ij,ij->i", r, Dself0)
        psi2 = m2 / s2 ** 2
        div = (-d + (rD - m2) / s2) / s2
        return (0.5 * psi2 + div).mean()

    gr = (np.sqrt(5) - 1) / 2
    a, b = np.log(sigma_lo), np.log(sigma_hi)
    c, dd = b - gr * (b - a), a + gr * (b - a)
    fc, fd = negobj(c), negobj(dd)
    for _ in range(40):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a)
            fc = negobj(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a)
            fd = negobj(dd)
        if (b - a) < 1e-24:
            break
    return float(np.exp((a + b) / 2))


def select_sigma_ledoitwolf(W: np.ndarray, seed: int = 0) -> float:
    """Pick the DSM noise level sigma from a Ledoit-Wolf shrinkage covariance.

    By Proposition 1, Gaussian DSM at noise sigma is the diagonally-loaded
    covariance (Sigma_hat + sigma^2 I). Ledoit-Wolf gives the optimal linear
    shrinkage toward a scaled identity:

        Sigma_LW = (1 - rho) Sigma_hat + rho * mu * I,   mu = tr(Sigma_hat)/d.

    The isotropic ("diagonal-loading") variance that the shrinkage injects is

        sigma^2 = rho * mu,

    which we use as the DSM noise level. (We deliberately do NOT use the
    scale-invariant ratio rho*mu/(1-rho): on whitened data Sigma_hat is already
    ~= mu*I = the shrinkage target, so Ledoit-Wolf drives rho -> 1 and the ratio
    diverges; rho*mu stays bounded by mu.) rho is chosen analytically by
    Ledoit-Wolf -- no search, no training. Pass WHITENED data so sigma is in
    the space the DSM noise lives in. Unlike the Parzen-LOO selector this is a
    purely second-order (Gaussian) criterion: cheap and stable, but blind to
    non-Gaussian structure.
    """
    from sklearn.covariance import LedoitWolf
    X = np.asarray(W, dtype=np.float64)
    n, d = X.shape
    if n < 3:
        return 1.0
    lw = LedoitWolf().fit(X)
    rho = float(np.clip(lw.shrinkage_, 0.0, 1.0))
    Xc = X - X.mean(0)
    mu_scale = float(np.trace(Xc.T @ Xc) / max(n - 1, 1) / d)   # tr(Sigma)/d
    sigma2 = rho * mu_scale
    return float(np.sqrt(max(sigma2, 1e-12)))


def _robust_svd_np(A: np.ndarray):
    """
    SVD with fallback chain for ill-conditioned matrices.

    Primary attempt uses the original matrix unchanged (no ridge) so that
    normal runs are completely unaffected.  The fallback chain only activates
    when np.linalg.svd (LAPACK gesdd) fails to converge.

      1. np.linalg.svd(A)              — unchanged, fast (gesdd)
      2. scipy gesvd(A)                — slower but more robust
      3. scipy gesvd(A + 1e-6 I)       — light ridge + robust driver
      4. scipy gesvd(A + 1e-3 I)       — heavier ridge as last resort

    A matrix with NaN/Inf is handled before any SVD attempt.
    """
    if not np.all(np.isfinite(A)):
        # NaN/Inf in covariance (exploding gradients, untrained model).
        # Return trivial decomposition → pseudo-inverse = 0 → scores = 0 → AUC≈0.5.
        n = A.shape[0]
        return np.eye(n), np.zeros(n), np.eye(n)
    try:
        return np.linalg.svd(A)                         # 1. unchanged primary
    except np.linalg.LinAlgError:
        pass
    try:
        from scipy.linalg import svd as scipy_svd
        return scipy_svd(A, full_matrices=True, lapack_driver='gesvd')   # 2.
    except Exception:
        pass
    try:
        from scipy.linalg import svd as scipy_svd
        return scipy_svd(A + 1e-6 * np.eye(A.shape[0]),                 # 3.
                         full_matrices=True, lapack_driver='gesvd')
    except Exception:
        pass
    from scipy.linalg import svd as scipy_svd
    return scipy_svd(A + 1e-3 * np.eye(A.shape[0]),                     # 4.
                     full_matrices=True, lapack_driver='gesvd')


class MixtureOfLinears(nn.Module):
    """
    Lightweight score model: mixture of K linear experts with a small gating network.

        gate(x)     : x (d) → Linear(d, gate_hidden) → SiLU → Linear(gate_hidden, K) → Softmax
        expert_k(x) : x (d) → Linear(d, d)
        output      : Σ_k gate_k(x) · expert_k(x)

    Parameter count (d=20, K=2, gate_hidden=5):
        gate    : 20×5 + 5 + 5×2 + 2  ≈  117
        experts : 2 × (20×20 + 20)    ≈  840
        total   ≈  957   vs ~4300 for MLP [64,64]
    """

    def __init__(self, input_dim: int, K: int = 2, gate_hidden: int = 5):
        super().__init__()
        self.K = K
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, K),
        )
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(K)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = torch.softmax(self.gate(x), dim=-1)          # (batch, K)
        out = sum(z[:, k : k+1] * self.experts[k](x) for k in range(self.K))
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 latent_activation: str = None):
        """
        latent_activation: optional activation applied after the encoder bottleneck.
            None  → linear (PCA-style)
            'relu' → ReLU on latent codes
        """
        super().__init__()
        enc_layers, dec_layers = [], []

        act_map = {"relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}

        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        enc_layers.append(nn.Linear(dims[-1], latent_dim))
        if latent_activation is not None:
            enc_layers.append(act_map[latent_activation]())
        self.encoder = nn.Sequential(*enc_layers)

        dims_rev = [latent_dim] + list(reversed(hidden_dims))
        for i in range(len(dims_rev) - 1):
            dec_layers += [nn.Linear(dims_rev[i], dims_rev[i + 1]), nn.SiLU()]
        dec_layers.append(nn.Linear(dims_rev[-1], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Whitening(nn.Module):
    """Frozen whitening front-end: x -> (x - mu) @ W.T, where W is computed from
    a background covariance so cov(output) = I (full-dimensional; replaces PCA).

    mode (configurable — keep it a one-line swap):
      'zca'       W = V Λ^{-1/2} Vᵀ   (symmetric; stays closest to original axes)
      'pca'       W = Λ^{-1/2} Vᵀ     (rotate to eigenbasis)
      'cholesky'  W = L^{-1}, Σ = L Lᵀ
      'normalize' W = diag(1/σ_b)     (per-channel standardization only — NO
                  decorrelation: each band is centered and scaled to unit
                  variance but cross-band correlations are kept. Diagonal W,
                  so no eigendecomposition/LAPACK is used.)

    The module is FROZEN (buffers, no grad). It also whitens the (B, M, D)
    neighbor tensor (broadcast over the last axis).
    """

    def __init__(self, mu, W):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("W",  torch.as_tensor(W,  dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) @ self.W.t()

    def transform_direction(self, s) -> np.ndarray:
        """Whiten a DIRECTION (additive signature; no mean subtraction): s -> W·s."""
        Wn = self.W.detach().cpu().numpy()
        return (np.asarray(s, dtype=np.float32) @ Wn.T).astype(np.float32)

    @classmethod
    def from_data(cls, X: np.ndarray, mode: str = "zca",
                  eig_floor: float = 0.0, eps: float = 1e2):
        """Fit a frozen whitener from background pixels X.

        eig_floor : Eigenvalue floor mode.
            0.0  (default) — Spectral-gap adaptive floor.
                 Scans the BOTTOM HALF of the positive eigenvalue spectrum
                 for the first large multiplicative jump (≥ 100×).  If one
                 is found, the floor is set at the top of the bottom cluster
                 (i.e. the eigenvalue just below the gap).  If no large gap
                 exists the floor is ``eps`` — smooth spectra keep all
                 directions.

                 Examples:
                   [1e7,…,1e0, 1e-3, 1e-6,1e-6,1e-6]
                       → gap 1e-6→1e-3 is 1000×  → floor = 1e-6
                   null-space (n < D):
                       near-zero eigenvalues excluded from scan,
                       automatically clipped to eps = 1e-6
                   smooth spectrum (n >> D):
                       no gap ≥ 100× in the bottom half → floor = eps

            > 0  — Fixed relative override: floor = eig_floor × λ_max.
                 Useful for a predictable manual floor (e.g.
                 lrao_whiten_eig_floor = 0.01 to keep C_Ψ invertible).
        eps : absolute minimum floor — safety guard against 1/0.
        """
        X = np.asarray(X, dtype=np.float64)
        n, D = X.shape
        mu = X.mean(0)
        Xc = X - mu
        Sigma = (Xc.T @ Xc) / max(n - 1, 1)
        Sigma = (Sigma + Sigma.T) / 2
        if mode in ("normalize", "standardize", "channel"):
            # Per-channel standardization: diagonal W = diag(1/sigma_b).
            # No decorrelation, no eigendecomposition. Floor the variance so
            # a (near-)constant band does not produce a non-finite scale.
            var   = np.diag(Sigma).copy()
            var   = np.maximum(var, max(float(var.max()), 1.0) * 1e-12)
            W     = np.diag(1.0 / np.sqrt(var))
            return cls(mu.astype(np.float32), W.astype(np.float32))
        if mode == "cholesky":
            W = np.linalg.inv(np.linalg.cholesky(
                Sigma + eps * np.eye(D)))
        else:
            evals, evecs = np.linalg.eigh(Sigma)   # ascending order
            if eig_floor > 0:
                # Manual: fixed relative floor × λ_max
                floor = max(float(evals.max()) * eig_floor, eps)
            else:

                # # Spectral-gap adaptive floor.
                # #
                # # Look for a large multiplicative jump in the BOTTOM HALF of
                # # the positive eigenvalue sequence (ascending).  A gap ≥ 100×
                # # indicates a natural noise floor (null-space boundary or a
                # # genuine low-eigenvalue cluster).  Searching only the bottom
                # # half avoids being misled by large gaps in the signal part of
                # # the spectrum.  The floor is set to the eigenvalue just below
                # # the gap (top of the bottom cluster).
                # #
                # # Null-space directions (eigenvalue ≤ 0) are excluded from the
                # # scan; np.clip raises them to eps automatically.
                # _GAP = 100.0                          # 2 log10 decades
                # pos  = evals[evals > 0]               # ascending positive evals
                # floor = eps
                floor = max(float(evals[-1]*1e-5),eps)
                # if len(pos) >= 2:
                #     # third   = max(len(pos) // 8, 1)
                #     bottom = pos[2:]            # bottom-half + 1 element
                #     if len(bottom) >= 2:
                #         ratios = bottom[1:] / bottom[:-1]
                #         i_gap  = int(np.argmax(ratios))
                #         if ratios[i_gap] >= _GAP:
                #
                #             floor = max(float(bottom[i_gap]), eps)
            evals = np.clip(evals, floor, None)
            inv_sqrt = np.diag(1.0 / np.sqrt(evals))
            W = (inv_sqrt @ evecs.T) if mode == "pca" else (evecs @ inv_sqrt @ evecs.T)
        return cls(mu.astype(np.float32), W.astype(np.float32))


class _ResidualScoreNet(nn.Module):
    """Pre-norm residual MLP score net with a learnable affine skip.

    Runs in WHITENED space (d -> d). Two design choices make it train far
    better than a plain MLP at high d / low n:

      * Pre-norm residual blocks (LayerNorm -> Linear -> act -> Linear, added
        to the input) give stable gradients and let depth help instead of hurt.
      * A learnable global affine skip  a ⊙ w + b  with the residual branch
        zero-initialized and a init = -1, b = 0. In whitened space the
        population score is ≈ -w (the score of N(0, I)), so the net STARTS at
        the exact Gaussian/AMF score and only has to learn the non-Gaussian
        correction — a strong, well-motivated inductive prior.
    """

    def __init__(self, d: int, width: int, n_blocks: int, act_cls):
        super().__init__()
        self.inp = nn.Linear(d, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width),
                          nn.Linear(width, width), act_cls(),
                          nn.Linear(width, width))
            for _ in range(n_blocks)
        )
        self.out = nn.Linear(width, d)
        self.a = nn.Parameter(-torch.ones(d))      # affine skip: start at -w
        self.b = nn.Parameter(torch.zeros(d))
        nn.init.zeros_(self.out.weight)             # residual branch starts at 0
        nn.init.zeros_(self.out.bias)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        h = self.inp(w)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h) + (self.a * w + self.b)


class _MixtureScoreNet(nn.Module):
    """Mixture-of-affine-experts score net (the natural GMM-score model).

    Operates in WHITENED space (d -> d). The score of a Gaussian mixture is a
    responsibility-weighted sum of affine (per-component Gaussian) scores:

        psi(w) = sum_k g_k(w) * (a_k ⊙ w + b_k),   g(w) = softmax(gate(w)),

    which is exactly this network: K diagonal-affine experts and a small MLP
    gate. This matches the multimodal structure of HSI clutter (the reason
    GMM-GLRT is the baseline) and is far better suited than a generic MLP:

      * It is lightweight (K*(2d) + a tiny gate), so it works with FEW samples.
      * No feature normalization (LayerNorm/BatchNorm) — a score must carry
        magnitude, which normalization destroys (the resmlp's weakness here).
      * After VICReg the latent is ~white, so initializing every expert slope
        a_k = -1 (and b_k = 0) starts the net at the white Gaussian score -w;
        the gate then learns the soft cluster assignment and b_k the per-mode
        offsets — a strong, problem-matched prior.
    """

    def __init__(self, d: int, n_experts: int, gate_hidden: int, act_cls):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d, gate_hidden), act_cls(),
            nn.Linear(gate_hidden, n_experts),
        )
        # diagonal-affine experts; slope starts at -1 (whitened Gaussian score)
        self.a = nn.Parameter(-torch.ones(n_experts, d)
                              + 0.01 * torch.randn(n_experts, d))
        self.b = nn.Parameter(torch.zeros(n_experts, d))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        g = torch.softmax(self.gate(w), dim=-1)            # (B, K)
        experts = w.unsqueeze(1) * self.a + self.b         # (B, K, d)
        return (g.unsqueeze(-1) * experts).sum(1)          # (B, d)


class ScoreNet(nn.Module):
    """Score network ψ_η: R^d → R^d trained via denoising score matching.

    Optional frozen `whitening` front-end (the first layer). When present the net
    operates in WHITENED space: forward(x) = net(whiten(x)); the DSM loss whitens
    first then adds noise in whitened space; detection uses the whitened signature.

    arch :
      'mlp' (default)     — plain feed-forward MLP. Best empirically after the
                            VICReg front-end (the latent is a generic learned
                            representation; A/B: mlp > mixture > resmlp).
      'mixture'           — mixture of affine experts (GMM-score model; see
                            _MixtureScoreNet). n_experts from `dsm_n_experts`,
                            gate width from max(hidden_dims).
      'resmlp'            — pre-norm residual MLP + affine skip (LayerNorm can
                            hurt: it strips the score magnitude).
    """

    def __init__(self, input_dim: int, hidden_dims: list = None,
                 activation: str = "silu", whitening: "Whitening" = None,
                 arch: str = "mlp", n_experts: int = 16):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []

        act_map = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        act_cls = act_map[activation]
        self.arch = arch
        gate_hidden = int(max(hidden_dims)) if len(hidden_dims) else 64

        if arch in ("mixture", "moe"):
            self.net = _MixtureScoreNet(input_dim, n_experts, gate_hidden,
                                        act_cls)
        elif arch in ("resmlp", "residual") and len(hidden_dims) > 0:
            width = int(max(hidden_dims))
            n_blocks = len(hidden_dims)
            self.net = _ResidualScoreNet(input_dim, width, n_blocks, act_cls)
        else:                                       # plain MLP (back-compat)
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
        """Un-whiten a whitened-space score into a DATA-SPACE score:
        ∇_x log p(x) = Wᵀ ∇_w log p(w)  ==  score_w @ W  (per-row)."""
        return score_w @ self.whitening.W if self.whitening is not None else score_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the DATA-SPACE score. The net runs in whitened space (for
        conditioning), then the output is mapped back to data space via Wᵀ so the
        detection statistic uses the RAW signature directly."""
        return self.to_data_space(self.net(self.whiten(x)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())








def dsm_loss(model: ScoreNet, batch: torch.Tensor, sigma,
             weighted: bool = False, noise=None) -> torch.Tensor:
    """DSM objective: E[||ψ_η(w̃) - (w - w̃)/Σ_n||²] where w̃ = w + ε, ε ~ N(0,Σ_n).

    Parameters
    ----------
    sigma : float OR (d,) array/tensor.
        Scalar  → isotropic noise Σ_n = σ²I  (the original DSM).
        Vector  → diagonal noise Σ_n = diag(σ_1²,…,σ_d²); each band b is
                  corrupted with its own std σ_b and denoised with target
                  -ε_b/σ_b².  Broadcasts over the batch.
    weighted : if True, weight each band's squared error by σ_b² (Vincent
        preconditioning).  This rebalances fitting effort across bands so a
        tiny-σ band does not numerically dominate the loss.  For SCALAR σ it
        is just a global constant (does not change the learned score); for a
        DIAGONAL σ it is the principled, well-conditioned form.
    noise : optional FIXED unit-N(0,1) tensor (same shape as the whitened
        batch). If given it is used instead of drawing fresh noise -- this lets
        a caller (e.g. validation) compute a deterministic loss WITHOUT touching
        the global RNG, so it cannot perturb the training noise stream (this
        matters on MPS, whose RNG is not covered by get/set_rng_state).
    """
    if not torch.is_tensor(sigma):
        sigma = torch.as_tensor(sigma, dtype=batch.dtype, device=batch.device)
    # If the net has a frozen whitening front-end, operate in WHITENED space:
    # whiten first, then add the DSM noise, and score the inner net directly.
    w = model.whiten(batch) if hasattr(model, "whiten") else batch
    inner = model.net if hasattr(model, "net") else model
    z = torch.randn_like(w) if noise is None else noise.to(w.dtype)
    eps = z * sigma                                 # (B,d), per-band std
    w_tilde = w + eps
    target = -eps / (sigma ** 2)                    # (w - w̃)/Σ_n = -ε/σ_b²
    se = (inner(w_tilde) - target) ** 2             # (B,d)
    if weighted:
        se = se * (sigma ** 2)                      # precondition per band
    return se.sum(dim=-1).mean()


def ssm_loss(model: ScoreNet, batch: torch.Tensor, n_projections: int = 1,
             variance_reduction: bool = True, noise: float = 0.0,
             fixed=None) -> torch.Tensor:
    """Sliced Score Matching objective (Song et al., UAI 2019).

    Estimates the score of p_w DIRECTLY (no noise level sigma), via random
    projections of the score Jacobian:

        J = E_x E_v [ v^T (d psi/dx) v + 1/2 (v^T psi(x))^2 ]      (plain)
        J = E_x E_v [ v^T (d psi/dx) v ] + 1/2 ||psi(x)||^2        (SSM-VR)

    Like dsm_loss, it operates in the WHITENED space (the net's internal
    space): the frozen whitening is applied, then the inner net's score and
    its Jacobian-vector products are taken w.r.t. the whitened input. This
    is the sigma -> 0 counterpart of DSM and is the natural comparison.

    ``noise`` > 0 evaluates the objective at slightly perturbed points
    w + noise * N(0, I). Pure SSM (noise = 0) is ill-posed for a flexible
    network at finite n -- the empirical trace term is unbounded below (the
    net can make ||psi|| ~ 0 at each training point with a very steep
    negative slope), so the loss runs off to -inf. A small noise (or the
    val-based early stopping in the trainer) regularizes it.

    ``fixed`` : optional dict {'innoise': (B,d) or None, 'proj': (P,B,d)} of
        FIXED unit-N(0,1) tensors, so a caller (validation) can compute a
        deterministic loss WITHOUT touching the global RNG (avoids leaking into
        the training noise stream; matters on MPS).
    """
    w0 = model.whiten(batch) if hasattr(model, "whiten") else batch
    inner = model.net if hasattr(model, "net") else model
    w = w0.detach()
    if noise and noise > 0:
        innoise = None if fixed is None else fixed.get('innoise')
        z = torch.randn_like(w) if innoise is None else innoise.to(w.dtype)
        w = w + noise * z
    w = w.requires_grad_(True)
    s = inner(w)                                        # (B, d) whitened score
    proj_fixed = None if fixed is None else fixed.get('proj')
    jac = 0.0
    proj = 0.0
    for j in range(n_projections):
        v = (torch.randn_like(w) if proj_fixed is None
             else proj_fixed[j].to(w.dtype))
        gv = torch.autograd.grad((s * v).sum(), w, create_graph=True)[0]
        jac = jac + (gv * v).sum(-1)                    # v^T (ds/dw) v
        if not variance_reduction:
            proj = proj + 0.5 * ((s * v).sum(-1)) ** 2
    jac = jac / n_projections
    norm = (0.5 * (s ** 2).sum(-1) if variance_reduction
            else proj / n_projections)
    return (jac + norm).mean()


def train_dsm(model: ScoreNet, data: np.ndarray, sigma: float,
              lr: float = 1e-3, batch_size: int = 32,
              epochs: int = 500, device: str = "cpu",
              print_every: int = 100,
              weight_decay: float = 0.0,
              checkpointer=None) -> ScoreNet:
    """Train ScoreNet on background samples using DSM loss.

    Parameters
    ----------
    checkpointer : optional Checkpointer instance (from final_paper_experiments/checkpointing.py).
                   If provided, saves periodic and best-loss checkpoints during training.
    """
    model = model.to(device)
    model.train()

    X = torch.tensor(data, dtype=torch.float32).to(device)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(data)), shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = dsm_loss(model, batch, sigma)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"    epoch {epoch:>{len(str(epochs))}}/{epochs}  loss={avg:.5f}")

        if checkpointer is not None:
            checkpointer.save(epoch, model.cpu(), optimizer, {'loss': avg})
            checkpointer.save_best_loss(model.cpu(), avg, epoch, optimizer)
            model = model.to(device)

    model.eval()
    if checkpointer is not None:
        checkpointer.save_final(model.cpu(), epochs, optimizer)
    return model.cpu()


def lfi_loss(model: ScoreNet, batch: torch.Tensor, s: torch.Tensor,
             delta_theta: float = 0.01) -> torch.Tensor:
    """
    LFI training loss from Zschetzsche et al. 2026 (LRao paper).
    Minimizing C = -Ĵ  where Ĵ = ĝᵀ Σ̂_Ψ⁻¹ ĝ  (estimated Linear Fisher Information).

    ĝ  = E[(Ψ(w+sΔθ) - Ψ(w-sΔθ)) / (2Δθ)]   central-difference Jacobian of mean
    Σ̂_Ψ = Cov(Ψ(w))                             output covariance (regularized)
    """
    psi_0     = model(batch)                              # (n, d)
    psi_plus  = model(batch + delta_theta * s)            # (n, d)
    psi_minus = model(batch - delta_theta * s)            # (n, d)

    g        = ((psi_plus - psi_minus) / (2.0 * delta_theta)).mean(dim=0)  # (d,)

    mu_psi   = psi_0.mean(dim=0)
    centered = psi_0 - mu_psi                             # (n, d)
    n        = batch.shape[0]
    Sigma    = (centered.T @ centered) / max(n - 1, 1)   # (d, d)
    Sigma    = Sigma + 1e-4 * torch.eye(Sigma.shape[0], device=batch.device)

    # Guard the LAPACK call: non-finite input makes torch.linalg.inv raise on
    # Linux but SEGFAULT on macOS (Accelerate). Raise a catchable error so the
    # training loop aborts gracefully on every platform.
    if not torch.isfinite(Sigma).all():
        raise RuntimeError("non-finite Sigma in LFI loss (LRao training diverged)")
    Sigma_inv = torch.linalg.inv(Sigma)
    J         = g @ Sigma_inv @ g                         # scalar LFI
    return -J


def train_lfi(model: ScoreNet, data: np.ndarray, s: np.ndarray,
              delta_theta: float = 0.01,
              lr: float = 1e-3, batch_size: int = 32,
              epochs: int = 500, device: str = "cpu",
              print_every: int = 100,
              weight_decay: float = 0.0,
              checkpointer=None) -> ScoreNet:
    """
    Train network by maximizing the LFI (LRao paper objective).
    Uses full batch when n < batch_size for stable covariance estimate.

    Parameters
    ----------
    checkpointer : optional Checkpointer instance.
                   Saves periodic and best-loss checkpoints during training.
    """
    model   = model.to(device)
    model.train()
    s_t     = torch.tensor(s, dtype=torch.float32, device=device)
    X       = torch.tensor(data, dtype=torch.float32, device=device)
    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=min(batch_size, len(data)),
                         shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = lfi_loss(model, batch, s_t, delta_theta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"    epoch {epoch:>{len(str(epochs))}}/{epochs}  LFI={-avg:.5f}")

        if checkpointer is not None:
            checkpointer.save(epoch, model.cpu(), optimizer, {'lfi': -avg})
            checkpointer.save_best_loss(model.cpu(), avg, epoch, optimizer)
            model = model.to(device)

    model.eval()
    if checkpointer is not None:
        checkpointer.save_final(model.cpu(), epochs, optimizer)
    return model.cpu()


def train_autoencoder(model: nn.Module, data: np.ndarray,
                      lr: float = 1e-3, batch_size: int = 32,
                      epochs: int = 500, device: str = "cpu",
                      print_every: int = 100,
                      weight_decay: float = 0.0) -> nn.Module:
    """Train Autoencoder on background samples using MSE reconstruction loss."""
    model = model.to(device)
    model.train()

    # Prepare data matching your original format
    X = torch.tensor(data, dtype=torch.float32).to(device)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(data)), shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Standard reconstruction loss for Autoencoders
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()

            # Forward pass: get reconstructed output
            reconstructed = model(batch)

            # Calculate loss between original input and reconstruction
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            avg = epoch_loss / len(loader)
            print(f"    epoch {epoch:>{len(str(epochs))}}/{epochs}  loss={avg:.5f}")

    model.eval()
    return model.cpu()
@torch.no_grad()
def compute_lfi_detector_scores(model: ScoreNet, train_data: np.ndarray,
                                 test_data: np.ndarray, s: np.ndarray,
                                 delta_theta: float = 0.01) -> np.ndarray:
    """
    LLMP detector statistic (LRao paper, one-sided for θ > 0):
        T(y) = ĝᵀ Σ̂_Ψ⁻¹ (Ψ(y) - μ̂_Ψ) / √Ĵ

    All statistics estimated from train_data.
    """
    model.eval()
    s_t    = torch.tensor(s,          dtype=torch.float32)
    X_tr   = torch.tensor(train_data, dtype=torch.float32)
    X_te   = torch.tensor(test_data,  dtype=torch.float32)

    psi_tr    = model(X_tr).numpy()                          # (n, d)
    mu_psi    = psi_tr.mean(axis=0)                          # (d,)
    centered  = psi_tr - mu_psi
    Sigma     = centered.T @ centered / max(len(train_data) - 1, 1)
    Sigma    += 1e-4 * np.eye(Sigma.shape[0])
    # Guard the LAPACK call: a diverged model gives non-finite Sigma, and
    # np.linalg.inv raises on Linux but SEGFAULTS on macOS (Accelerate).
    if not np.all(np.isfinite(Sigma)):
        return np.zeros(len(test_data), dtype=np.float32)    # scores ~ 0
    Sigma_inv = np.linalg.inv(Sigma)

    # Central-difference Jacobian direction from train data
    psi_plus  = model(X_tr + delta_theta * s_t).numpy()
    psi_minus = model(X_tr - delta_theta * s_t).numpy()
    g         = ((psi_plus - psi_minus) / (2.0 * delta_theta)).mean(axis=0)  # (d,)

    J     = float(g @ Sigma_inv @ g)
    denom = np.sqrt(max(J, 1e-12))

    psi_te  = model(X_te).numpy()                            # (n_test, d)
    scores  = (psi_te - mu_psi) @ (Sigma_inv @ g) / denom
    return scores


# ---------------------------------------------------------------------------
# Mode-2 LFI: signal-agnostic training (Zschetzsche et al. B4-B5)
# ---------------------------------------------------------------------------

def lfi_loss_mode2(model: ScoreNet, batch: torch.Tensor,
                   delta_theta: float = 0.01,
                   sigma_cutoff: float = 1e-3,
                   detach_sigma: bool = False) -> torch.Tensor:
    """
    Signal-agnostic LFI loss: maximize tr(J*) = tr(G^T Sigma^{-1} G).

    G[:,j] = E[(Psi(w+e_j*dt) - Psi(w-e_j*dt)) / (2*dt)]  for each basis dir e_j.

    By the chain rule (Zschetzsche SI B4-B5), for any signal H:
        J = H^T J* H   (projected at inference, no retraining needed).

    At optimum Psi*(x) = score function ∇_x log p_w(x).
    """
    n, d = batch.shape
    device = batch.device

    # Output covariance. By default detached from gradient (our choice — see notes).
    # If detach_sigma=False, matches the original LRao code (CNN_LRao_functions.py
    # lfi_diag_autocorr) where gradient flows through Sigma as well.
    if detach_sigma:
        ctx = torch.no_grad()
    else:
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:
        psi_0    = model(batch)                      # (n, d_out)
        d_out    = psi_0.shape[1]
        mu_psi   = psi_0.mean(dim=0)
        centered = psi_0 - mu_psi
        Sigma    = (centered.T @ centered) / max(n - 1, 1)
        Sigma    = 0.5 * (Sigma + Sigma.T)           # symmetrize (numerical)
        if not torch.isfinite(Sigma).all():
            raise RuntimeError("non-finite Sigma in LFI (LRao training diverged)")
        # Regularized inverse of the (symmetric PSD) score covariance.
        # We use eigh (NOT svd): for a symmetric matrix eigh is more accurate
        # and, crucially, its BACKWARD is far better conditioned -- the svd
        # backward has 1/(s_i^2 - s_j^2) terms that blow up for close/degenerate
        # singular values, and that blow-up is BLAS/torch-version dependent
        # (so LRao diverged on some machines but not others). A relative
        # eigenvalue FLOOR (instead of a hard truncation to 0) keeps Sigma_inv
        # bounded, which also removes the unbounded-objective divergence.
        evals, evecs = torch.linalg.eigh(Sigma)      # ascending, symmetric
        lam_max = evals[-1].clamp_min(1e-12)
        floor = torch.clamp(evals, min=sigma_cutoff * lam_max)   # relative floor
        inv = 1.0 / floor
        Sigma_inv = (evecs * inv) @ evecs.T

    # Full Jacobian G = E[∂Ψ(x)/∂x] via vmap+jacrev — one vectorised backward pass
    # instead of 2d sequential forward passes. G shape: (d_out, d).
    from torch.func import jacrev, vmap
    def _model_1d(x1d):                              # x1d: (d,) → (d_out,)
        return model(x1d.unsqueeze(0)).squeeze(0)
    def _single_jac(x):                              # x: (d,) → (d_out, d)
        return jacrev(_model_1d)(x)
    J_all = vmap(_single_jac)(batch)                 # (n, d_out, d)
    G     = J_all.mean(dim=0)                        # (d_out, d)

    # cost = -tr(J*) = -tr(G^T Sigma^{-1} G)
    J_star = G.T @ Sigma_inv @ G                     # (d, d)
    return -J_star.trace()


def train_lfi_mode2(model: ScoreNet, data: np.ndarray,
                    delta_theta: float = 0.01,
                    lr: float = 1e-3, batch_size: int = 256,
                    epochs: int = 5000, device: str = "cpu",
                    print_every: int = 500,
                    weight_decay: float = 1e-4,
                    sigma_cutoff: float = 1e-3,
                    detach_sigma: bool = False,
                    checkpointer=None) -> ScoreNet:
    """
    Train by maximizing tr(J*): NO target signature s needed.
    Same hyperparameter interface as train_dsm.
    """
    model   = model.to(device)
    model.train()
    X       = torch.tensor(data, dtype=torch.float32, device=device)
    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=min(batch_size, len(data)),
                         shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = lfi_loss_mode2(model, batch, delta_theta, sigma_cutoff,
                                  detach_sigma=detach_sigma)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"    epoch {epoch:>{len(str(epochs))}}/{epochs}  tr(J*)={-avg:.5f}")

        if checkpointer is not None:
            checkpointer.save(epoch, model.cpu(), optimizer, {'lfi_m2': -avg})
            checkpointer.save_best_loss(model.cpu(), avg, epoch, optimizer)
            model = model.to(device)

    model.eval()
    if checkpointer is not None:
        checkpointer.save_final(model.cpu(), epochs, optimizer)
    return model.cpu()


@torch.no_grad()
def compute_lfi_detector_scores_mode2(model: ScoreNet, train_data: np.ndarray,
                                       test_data: np.ndarray, s: np.ndarray,
                                       delta_theta: float = 0.01,
                                       sigma_cutoff: float = 1e-3) -> np.ndarray:
    """
    Mode-2 LLMP detector. Signal s enters only here (not during training).

    Chain rule (B5): g_s = G @ s,  J_s = g_s^T Sigma^{-1} g_s
    T(y) = g_s^T Sigma^{-1} (Psi(y) - mu) / sqrt(J_s)
    """
    model.eval()
    device = next(model.parameters()).device
    d    = train_data.shape[1]
    X_tr = torch.tensor(train_data, dtype=torch.float32).to(device)
    X_te = torch.tensor(test_data,  dtype=torch.float32).to(device)
    I_d  = torch.eye(d, device=device)

    psi_tr = model(X_tr).cpu().numpy()               # (n, d_out)
    d_out  = psi_tr.shape[1]
    if not np.all(np.isfinite(psi_tr)):
        # Model outputs contain NaN/Inf (exploding gradients, untrained model).
        # Return zeros — AUC will be 0.5, flagging the run as degenerate.
        return np.zeros(len(test_data), dtype=np.float32)
    mu     = psi_tr.mean(axis=0)
    centered = psi_tr - mu
    n = len(train_data)
    Sigma     = centered.T @ centered / max(n - 1, 1)
    # Truncated pseudo-inverse (matches lfi_loss_mode2 and original LRao code).
    # _robust_svd_np tries the original matrix first; only adds ridge if gesdd fails.
    U, S, Vh  = _robust_svd_np(Sigma)
    cutoff    = sigma_cutoff * S[0]
    S_inv     = np.where(S > cutoff, 1.0 / S, 0.0)
    Sigma_inv = Vh.T @ np.diag(S_inv) @ U.T

    # Full Jacobian G: (d_out, d)
    G = np.zeros((d_out, d))
    for j in range(d):
        psi_plus  = model(X_tr + delta_theta * I_d[j]).cpu().numpy()
        psi_minus = model(X_tr - delta_theta * I_d[j]).cpu().numpy()
        G[:, j]   = ((psi_plus - psi_minus) / (2.0 * delta_theta)).mean(axis=0)

    # Project onto signal direction
    g_s   = G @ s                                    # (d_out,)
    J_s   = float(g_s @ Sigma_inv @ g_s)
    denom = np.sqrt(max(J_s, 1e-12))

    psi_te = model(X_te).cpu().numpy()
    return (psi_te - mu) @ (Sigma_inv @ g_s) / denom


# ---------------------------------------------------------------------------

def _model_lfi_stats(model: ScoreNet, train_data: np.ndarray, s: np.ndarray,
                     delta_theta: float = 0.01):
    """
    Compute LFI-related statistics for a trained model on training data.
    Returns (mu, C_inv, g, J, norm_score_fn) where:
      mu       : mean of Ψ(w) on train          (d,)
      C_inv    : inverse covariance of Ψ(w)      (d, d)
      g        : central-diff Jacobian direction  (d,)
      J        : estimated LFI = g^T C_inv g     (scalar)
      denom    : sqrt(J)                          (scalar)
    """
    s_t       = torch.tensor(s, dtype=torch.float32)
    X         = torch.tensor(train_data, dtype=torch.float32)
    with torch.no_grad():
        psi        = model(X).numpy()                                     # (n, d)
        psi_plus   = model(X + delta_theta * s_t).numpy()
        psi_minus  = model(X - delta_theta * s_t).numpy()

    mu    = psi.mean(axis=0)
    C     = np.cov(psi, rowvar=False) + 1e-4 * np.eye(psi.shape[1])
    # Guard the LAPACK call: a diverged model gives non-finite Psi -> non-finite
    # C, and np.linalg.inv raises on Linux but SEGFAULTS on macOS (Accelerate).
    if not np.all(np.isfinite(C)):
        d = C.shape[0]
        return mu, np.zeros((d, d)), np.zeros(d), 0.0, 1.0   # -> scores ~ 0
    C_inv = np.linalg.inv(C)
    g     = ((psi_plus - psi_minus) / (2.0 * delta_theta)).mean(axis=0)
    J     = float(g @ C_inv @ g)
    denom = np.sqrt(max(J, 1e-12))
    return mu, C_inv, g, J, denom


def select_sigma_by_lfi(models_dict: dict, train_data: np.ndarray,
                         s: np.ndarray, delta_theta: float = 0.01) -> tuple:
    """
    Select the sigma whose trained DSM model achieves the highest LFI on training data.
    No retraining — just evaluates each already-trained model.

    Returns: (best_sigma, {sigma: lfi_value})
    """
    lfi_vals = {}
    for sigma, model in models_dict.items():
        _, _, _, J, _ = _model_lfi_stats(model, train_data, s, delta_theta)
        lfi_vals[sigma] = J
        print(f"    σ={sigma:<6}  LFI={J:.5f}")
    best_sigma = max(lfi_vals, key=lfi_vals.get)
    print(f"  → best σ={best_sigma}  (LFI={lfi_vals[best_sigma]:.5f})")
    return best_sigma, lfi_vals


def detector_dsm_best_sigma(test_data: np.ndarray, train_data: np.ndarray,
                             models_dict: dict, s: np.ndarray,
                             delta_theta: float = 0.01) -> np.ndarray:
    """
    Run LFI-based sigma selection on training data, then apply the best model's
    LLMP statistic to test data.
    """
    best_sigma, _ = select_sigma_by_lfi(models_dict, train_data, s, delta_theta)
    return detector_dsm(test_data, train_data, models_dict[best_sigma], s)


def detector_dsm_combined(test_data: np.ndarray, train_data: np.ndarray,
                           models_dict: dict, s: np.ndarray,
                           delta_theta: float = 0.01) -> np.ndarray:
    """
    Optimal linear combination of scalar LLMP scores from multiple DSM models.

    For each model σ_k, the scalar score T_k(y) = g_kᵀ Σ̂_k⁻¹(Ψ_k(y)-μ_k) / √J_k.

    Under H0: T_k ~ N(0,1). Under H1: T_k ~ N(√J_k · θ, 1) (approximately).
    The K scores are correlated. Optimal weights using joint LMP on the score vector:

        α* = R⁻¹ √J / √(√Jᵀ R⁻¹ √J)

    where R = Cov([T_1,...,T_K]) estimated from training data.
    Combined score: T_comb(y) = α*ᵀ [T_1(y),...,T_K(y)]
    """
    sigmas = list(models_dict.keys())
    K      = len(sigmas)

    # Compute per-model stats and training scalar scores
    stats        = {}
    train_scores = np.zeros((len(train_data), K))
    J_vals       = np.zeros(K)

    for i, sigma in enumerate(sigmas):
        mu, C_inv, g, J, denom = _model_lfi_stats(
            models_dict[sigma], train_data, s, delta_theta)
        stats[sigma]       = (mu, C_inv, g, J, denom)
        J_vals[i]          = J
        psi_tr             = compute_scores(models_dict[sigma], train_data)
        train_scores[:, i] = (psi_tr - mu) @ (C_inv @ g) / denom

    # Estimate K×K correlation matrix from training scores
    if K == 1:
        return train_scores[:, 0]   # trivial — just return the single score on test

    R     = np.cov(train_scores, rowvar=False) + 1e-6 * np.eye(K)
    R_inv = np.linalg.inv(R)
    nu    = np.sqrt(np.maximum(J_vals, 0))           # non-centrality proxy
    w     = R_inv @ nu
    w    /= np.sqrt(nu @ R_inv @ nu + 1e-12)         # normalize

    # Apply combined weights to test data
    test_scores = np.zeros((len(test_data), K))
    for i, sigma in enumerate(sigmas):
        mu, C_inv, g, J, denom = stats[sigma]
        psi_te              = compute_scores(models_dict[sigma], test_data)
        test_scores[:, i]   = (psi_te - mu) @ (C_inv @ g) / denom

    return test_scores @ w


def select_sigma_loo(data: np.ndarray, sigma_grid: list) -> tuple:
    """
    Select DSM noise level σ for the LINEAR score model via LOO implicit score matching (ISM).

    For each held-out w_i, the linear score model is re-fitted analytically on the
    remaining n-1 samples. The ISM loss evaluated at the held-out point is:

        ISM(ψ̂_{-i}; w_i) = ½||ψ̂_{-i}(w_i)||² + ∇·ψ̂_{-i}(w_i)

    For ψ̂_{-i}(x) = -(Σ̂_{-i}+σ²I)⁻¹(x - μ̂_{-i}), in the eigenspace (eigenvalues λ_k):

        ½||ψ̂_{-i}(w_i)||²  = ½ Σ_k  δ̃_k² / (λ_k + σ²)²
        ∇·ψ̂_{-i}(w_i)      =  -Σ_k  1 / (λ_k + σ²)

    The sum has a proper finite minimum: d/dσ² of each term k is zero at δ̃_k² = λ_k + σ²,
    creating a balance between data-fit and regularization.

    Only valid for the linear (hidden_dims=[]) score model.
    Runs n × |sigma_grid| eigendecompositions — fast for n≤200, d≤20.

    Returns:
        best_sigma  : float
        loo_losses  : dict {sigma: avg_loo_ism_loss}
    """
    n, d = data.shape
    loo_losses = {}

    # Pre-compute LOO eigendecompositions once (reused across sigma values)
    loo_cache = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_i = data[mask]
        mu_i    = train_i.mean(axis=0)
        S_i     = np.cov(train_i, rowvar=False)
        S_i     = (S_i + S_i.T) / 2
        eigvals_i, eigvecs_i = np.linalg.eigh(S_i)
        eigvals_i = np.clip(eigvals_i, 0.0, None)
        delta_tilde = eigvecs_i.T @ (data[i] - mu_i)   # in eigenspace
        loo_cache.append((eigvals_i, delta_tilde))

    for sigma in sigma_grid:
        sig2  = float(sigma) ** 2
        total = 0.0
        for eigvals_i, delta_tilde in loo_cache:
            inv_diag = 1.0 / (eigvals_i + sig2)          # (d,)
            # ½||ψ̂_{-i}(w_i)||²
            ism_fit  = 0.5 * float(np.sum((delta_tilde * inv_diag) ** 2))
            # ∇·ψ̂_{-i}(w_i) = tr(A_i) = -Σ_k 1/(λ_k+σ²)
            ism_div  = -float(np.sum(inv_diag))
            total   += ism_fit + ism_div
        loo_losses[sigma] = total / n

    best_sigma = min(loo_losses, key=loo_losses.get)
    return best_sigma, loo_losses


@torch.no_grad()
def compute_scores(model: ScoreNet, data: np.ndarray) -> np.ndarray:
    """Evaluate learned score ψ̂(w) on a numpy array. Returns (n, d) numpy array."""
    model.eval()
    device = next(model.parameters()).device
    X = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()
