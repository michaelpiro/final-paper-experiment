import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


class ScoreNet(nn.Module):
    """Score network ψ_η: R^d → R^d trained via denoising score matching."""

    def __init__(self, input_dim: int, hidden_dims: list = None, activation: str = "silu"):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())








def dsm_loss(model: ScoreNet, batch: torch.Tensor, sigma: float) -> torch.Tensor:
    """DSM objective: E[||ψ_η(w̃) - (w - w̃)/σ²||²] where w̃ = w + ε, ε ~ N(0,σ²I)."""
    eps = torch.randn_like(batch) * sigma
    w_tilde = batch + eps
    target = -eps / (sigma ** 2)   # (w - w̃)/σ² = -ε/σ²
    score_pred = model(w_tilde)
    return ((score_pred - target) ** 2).sum(dim=-1).mean()


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
    X = torch.tensor(data, dtype=torch.float32)
    return model(X).numpy()
