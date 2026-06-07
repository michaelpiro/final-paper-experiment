"""
thantd_model.py — THANTD: Triplet Hybrid Attention Network for Target Detection.

Implements exactly:
    Liu, Wang, Cheng, Xing, Xu, "THANTD: Triplet Hybrid Attention Network for
    Hyperspectral Target Detection", IEEE JSTARS 18:8831-8844, 2025.

The ONLY adaptation from the paper: `build_thantd_samples` accepts an optional
`bkg_pool` argument. When provided, negative samples are drawn directly from
this secondary (target-free) data instead of using CEM rough detection on the
full image. This supports the "secondary data only" scenario. All other aspects
match the paper exactly.

Paper algorithm summary:
  1. SAMPLE CONSTRUCTION (Sec. II.A):
       - Rough detection via CEM on S_prior → rank all pixels
       - Top α% lowest-scoring pixels = S_negative (pure background)
       - S_positive = μ * S_negative + (1-μ) * S_prior,  μ ~ U[0, 0.1]  (R²TM, eq. 2)
       - S_prior as anchor
       [Adaptation: if bkg_pool given, use it directly as S_negative source]

  2. SPECTRA EMBEDDER (Sec. II.B):
       - s ∈ R^{b}  →  grouped X ∈ R^{c×m} via overlapping sliding window (c=b bands)
         x_i = [s_{i-⌊m/2⌋}, ..., s_i, ..., s_{i+⌊m/2⌋}] ∈ R^m
       - Linear projection X ∈ R^{c×m} → Z ∈ R^{c×d}
       - Prepend learnable class token, add learnable position embedding
       - Final Z ∈ R^{(c+1)×d}

  3. HYBRID ATTENTION BLOCK — HAB (Sec. II.C, Fig. 3):
       X_N = LN(X)
       X_M = α · CAM(X_N) + MSA(X_N) + X          [eq. 5, α learnable]
       output = X_M + MLP(LN(X_M))

  4. CHANNEL ATTENTION MODULE — CAM (Fig. 4, eq. 3–4):
       s = sigmoid(W_U · GELU(W_D · z))            global avg pool then up/down conv
       x̃_c = s_c · x_c                             channel-wise rescaling

  5. TRIPLET HYBRID NETWORK (Sec. II.C):
       Three parallel HABs with shared weights — one per branch (anchor/pos/neg).
       Equivalent to: one HAB applied independently to each branch's embedding.

  6. ETB-LOSS (Sec. II.D, eq. 6–11):
       Cosine distances (eq. 9):
         d_cos+   = 1 - cos(a, p)
         d_cos-   = 1 - cos(a, n)
         d_cos^-' = 1 - cos(p, n)
       Dual triplet (eq. 10):
         L_dual = mean[max(0, d_cos+ - d_cos- + m) + max(0, d_cos+ - d_cos^-' + m)]
       BCE on similarity scores s+, s- (eq. 11):
         L_BCE = -(1/2b) Σ [log(s+) + log(1-s-)]
       L_ETB = λ * L_dual + (1-λ) * L_BCE

  7. INFERENCE (Fig. 1, testing stage):
       score(pixel) = cosine_similarity(HAB(embed(S_prior)), HAB(embed(pixel)))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Section II.A — CEM rough detector
# ---------------------------------------------------------------------------

def cem_score(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    CEM detection score (paper cites [13]).
    w = R^{-1}t / (t^T R^{-1}t),   T(x) = x^T w.
    Smaller score → more likely background.
    """
    R = (X.T @ X) / max(len(X), 1) + 1e-6 * np.eye(X.shape[1])
    Rinv_t = np.linalg.solve(R, t)
    w = Rinv_t / (t @ Rinv_t + 1e-12)
    return X @ w


def build_thantd_samples(
        X: np.ndarray,
        t: np.ndarray,
        alpha: float = 0.5,
        n_samples: int = 1024,
        rng: np.random.Generator = None,
        bkg_pool: np.ndarray = None,
) -> tuple:
    """
    Construct (S_prior, S_positive, S_negative) triplets exactly as in Sec. II.A.

    Parameters
    ----------
    X         : (N, b) hyperspectral image pixels (used for CEM if bkg_pool=None).
    t         : (b,) prior target spectrum S_prior.
    alpha     : percentage coefficient — top α fraction of low-CEM scores = background.
    n_samples : number of triplets (= h×w×α in the paper notation).
    rng       : numpy random generator.
    bkg_pool  : ADAPTATION — if provided, use this (N', b) array as the background
                pool directly instead of running CEM on X. This supports training
                from secondary (target-free) data.

    Returns
    -------
    S_prior    : (n_samples, b) replicated target spectrum  [anchor]
    S_positive : (n_samples, b) R²TM-mixed target-like samples  [positive]
    S_negative : (n_samples, b) drawn from background pool  [negative]
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # --- Negative sample pool (background) ---
    if bkg_pool is not None:
        # Adaptation: secondary data provided directly
        neg_pool = bkg_pool.astype(np.float32)
    else:
        # Paper: rank all pixels by CEM, take top α% lowest scorers as background
        scores   = cem_score(X, t)
        n_neg    = max(int(alpha * len(X)), n_samples)
        idx_sorted = np.argsort(scores)            # ascending: lowest CEM first
        neg_pool = X[idx_sorted[:n_neg]].astype(np.float32)

    # --- Draw S_negative ---
    idx_neg    = rng.integers(0, len(neg_pool), size=n_samples)
    S_negative = neg_pool[idx_neg]                 # (n_samples, b)

    # --- Generate S_positive via R²TM (paper eq. 2) ---
    # S_positive = μ * S_negative + (1 - μ) * S_prior,  μ ~ U[0, 0.1]
    mu = rng.uniform(0.0, 0.1, size=(n_samples, 1)).astype(np.float32)
    S_positive = mu * S_negative + (1.0 - mu) * t.astype(np.float32)[None, :]

    # --- S_prior as anchor ---
    S_prior = np.tile(t.astype(np.float32)[None, :], (n_samples, 1))

    return S_prior, S_positive, S_negative


# ---------------------------------------------------------------------------
# Section II.B — Spectra Embedder
# ---------------------------------------------------------------------------

class SpectraEmbedder(nn.Module):
    """
    Maps a raw spectrum s ∈ R^b to token sequence Z ∈ R^{(c+1)×d} (paper Sec. II.B).

    Step 1: overlapping sliding window g(·) groups c=b adjacent bands into
            X = g(s) ∈ R^{c×m}, where x_i = [s_{i-⌊m/2⌋}, ..., s_i, ..., s_{i+⌊m/2⌋}].
    Step 2: linear projection X → Z ∈ R^{c×d}.
    Step 3: prepend learnable class token e_cls.
    Step 4: add learnable position embeddings.
    Output: Z ∈ R^{(c+1)×d}.
    """

    def __init__(self, b: int, m: int = 7, d: int = 64):
        """
        b : number of spectral bands (= c in the paper).
        m : group size (number of adjacent bands per token).
        d : embedding dimension.
        """
        super().__init__()
        if m % 2 == 0:
            m += 1                        # ensure odd so center band is well defined
        self.b = b                        # input bands
        self.m = m                        # group size
        self.d = d
        self.c = b                        # c = b (one token per band position)

        # Step 2: linear projection from m → d
        self.proj = nn.Linear(m, d)

        # Step 3: learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        # Step 4: learnable position embedding for c+1 tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.c + 1, d))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, b) → Z: (B, c+1, d)."""
        B = s.size(0)
        p = self.m // 2

        # Reflect-pad spectrum at boundaries so every band i has m neighbours
        sp = F.pad(s.unsqueeze(1), (p, p), mode='reflect').squeeze(1)  # (B, b+2p)

        # Build c overlapping windows: idx[i, j] = band i's j-th neighbour index
        i_idx = torch.arange(self.c, device=s.device)                  # (c,)
        j_idx = torch.arange(self.m, device=s.device)                  # (m,)
        idx   = (i_idx[:, None] + j_idx[None, :])                      # (c, m)
        X     = sp[:, idx]                                              # (B, c, m)

        # Step 2: project
        Z = self.proj(X)                                                # (B, c, d)

        # Step 3: prepend class token
        cls = self.cls_token.expand(B, -1, -1)                         # (B, 1, d)
        Z   = torch.cat([cls, Z], dim=1)                                # (B, c+1, d)

        # Step 4: add position embedding
        Z = Z + self.pos_embed                                          # (B, c+1, d)
        return Z


# ---------------------------------------------------------------------------
# Section II.C — Channel Attention Module (CAM)
# ---------------------------------------------------------------------------

class ChannelAttentionModule(nn.Module):
    """
    CAM (Fig. 4, eq. 3–4):
        s = sigmoid( W_U · GELU( W_D · global_avg_pool(z) ) )
        x̃_c = s_c · x_c

    Implemented as:
      1. Global average pool over token dim T → (B, d)
      2. Linear down (d → d//ratio)  + GELU
      3. Linear up   (d//ratio → d) + sigmoid
      4. Scale each channel of X by the gate s
    """

    def __init__(self, d: int, ratio: int = 4):
        super().__init__()
        h = max(d // ratio, 1)
        # W_D and W_U implemented as 1-D convolutions over the channel dim
        # (kernel_size=1 is equivalent to a per-channel linear layer)
        self.W_D = nn.Conv1d(d, h, kernel_size=1, bias=True)
        self.W_U = nn.Conv1d(h, d, kernel_size=1, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, T, d) → (B, T, d) channel-recalibrated."""
        # Transpose to (B, d, T) for Conv1d
        Xt = X.transpose(1, 2)                          # (B, d, T)

        # Global average pool over T → (B, d, 1)
        z  = Xt.mean(dim=-1, keepdim=True)              # (B, d, 1)

        # eq. 3: s = f(W_U(δ(W_D z)))
        s  = torch.sigmoid(self.W_U(F.gelu(self.W_D(z))))  # (B, d, 1)

        # eq. 4: x̃_c = s_c · x_c
        return (Xt * s).transpose(1, 2)                 # (B, T, d)


# ---------------------------------------------------------------------------
# Section II.C — Hybrid Attention Block (HAB)
# ---------------------------------------------------------------------------

class HybridAttentionBlock(nn.Module):
    """
    HAB (Fig. 3, eq. 5):
        X_N = LayerNorm(X)
        X_M = α · CAM(X_N) + MSA(X_N) + X        [eq. 5, α is learnable]
        output = X_M + MLP(LayerNorm(X_M))
    """

    def __init__(self, d: int = 64, n_heads: int = 4,
                 mlp_ratio: int = 2, cam_ratio: int = 4,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.msa = nn.MultiheadAttention(d, n_heads, dropout=attn_dropout,
                                          batch_first=True)
        self.cam = ChannelAttentionModule(d, ratio=cam_ratio)

        # learnable α that balances CAM vs MSA contributions (paper eq. 5)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_ratio),
            nn.GELU(),
            nn.Linear(d * mlp_ratio, d),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, T, d) → (B, T, d)."""
        X_N = self.ln1(X)

        # Parallel MSA and CAM (eq. 5)
        msa_out, _ = self.msa(X_N, X_N, X_N, need_weights=False)
        cam_out    = self.cam(X_N)
        X_M = self.alpha * cam_out + msa_out + X         # residual to original X

        # MLP sub-block with residual
        output = X_M + self.mlp(self.ln2(X_M))
        return output


# ---------------------------------------------------------------------------
# Full THANTD model
# ---------------------------------------------------------------------------

class THANTD(nn.Module):
    """
    Spectra Embedder + one Hybrid Attention Block (shared across all branches).

    Three parallel weight-sharing HABs (paper Sec. II.C) ≡ one HAB instance
    applied independently to each of the three branches (anchor, pos, neg).
    """

    def __init__(self, b: int, m: int = 7, d: int = 64,
                 n_heads: int = 4, mlp_ratio: int = 2, cam_ratio: int = 4):
        """
        b       : number of spectral bands.
        m       : adjacent-band group size for the Spectra Embedder.
        d       : embedding dimension.
        n_heads : MSA heads.
        mlp_ratio: MLP hidden dim ratio.
        cam_ratio: CAM channel compression ratio.
        """
        super().__init__()
        self.embedder = SpectraEmbedder(b, m=m, d=d)
        self.hab      = HybridAttentionBlock(d, n_heads=n_heads,
                                              mlp_ratio=mlp_ratio,
                                              cam_ratio=cam_ratio)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of spectra and return the class-token representation.
        s : (B, b)  →  class token: (B, d)
        """
        Z   = self.embedder(s)    # (B, c+1, d)
        Z   = self.hab(Z)         # (B, c+1, d)
        return Z[:, 0, :]         # class token  (B, d)


# ---------------------------------------------------------------------------
# Section II.D — ETB-Loss
# ---------------------------------------------------------------------------

def etb_loss(
        emb_a: torch.Tensor,      # anchor  embeddings  (B, d)
        emb_p: torch.Tensor,      # positive embeddings (B, d)
        emb_n: torch.Tensor,      # negative embeddings (B, d)
        margin: float = 0.3,
        lam: float = 0.5,
) -> torch.Tensor:
    """
    ETB-Loss = λ · L_dual + (1-λ) · L_BCE   (paper eq. 6–11).

    Cosine distances (eq. 9):
        d_cos+   = 1 - cos(a, p)
        d_cos-   = 1 - cos(a, n)
        d_cos^-' = 1 - cos(p, n)

    Dual triplet (eq. 10):
        L_dual = mean[ max(0, d_cos+ - d_cos- + margin)
                     + max(0, d_cos+ - d_cos^-' + margin) ]

    BCE (eq. 11), s_i+ = cos(a,p), s_i- = cos(a,n), mapped to [0,1]:
        L_BCE = -(1/2b) Σ [ y_i+ log(s_i+) + (1-y_i-) log(1-s_i-) ]
              = -0.5 * mean[ log(s+) + log(1-s-) ]
    """
    def _cos(u, v):
        return F.cosine_similarity(u, v, dim=-1)

    # Cosine distances (eq. 9)
    d_pos = 1.0 - _cos(emb_a, emb_p)     # d_cos+
    d_neg = 1.0 - _cos(emb_a, emb_n)     # d_cos-
    d_pn  = 1.0 - _cos(emb_p, emb_n)     # d_cos^-'

    # Dual triplet loss (eq. 10)
    L_dual = (F.relu(d_pos - d_neg + margin) +
              F.relu(d_pos - d_pn  + margin)).mean()

    # BCE loss (eq. 11)
    # Similarity scores: map cosine ∈ [-1, 1] → [0, 1] via (1 + cos)/2
    s_pos = ((1.0 + _cos(emb_a, emb_p)) / 2.0).clamp(1e-7, 1.0 - 1e-7)
    s_neg = ((1.0 + _cos(emb_a, emb_n)) / 2.0).clamp(1e-7, 1.0 - 1e-7)
    L_BCE = -(torch.log(s_pos).mean() + torch.log(1.0 - s_neg).mean()) / 2.0

    return lam * L_dual + (1.0 - lam) * L_BCE


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_thantd(
        model: THANTD,
        S_prior:    np.ndarray,    # (N, b) anchors
        S_positive: np.ndarray,    # (N, b) positives
        S_negative: np.ndarray,    # (N, b) negatives
        epochs:     int   = 300,
        batch_size: int   = 64,
        lr:         float = 1e-4,
        margin:     float = 0.3,
        lam:        float = 0.5,
        device:     str   = 'cpu',
) -> THANTD:
    """Train THANTD with ETB-Loss on the triplet samples (paper Sec. II.D)."""
    A = torch.tensor(S_prior,    dtype=torch.float32, device=device)
    P = torch.tensor(S_positive, dtype=torch.float32, device=device)
    N = torch.tensor(S_negative, dtype=torch.float32, device=device)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_total = len(A)
    for _ in range(epochs):
        perm = torch.randperm(n_total, device=device)
        for i in range(0, n_total, batch_size):
            sel = perm[i:i + batch_size]
            ea = model.encode(A[sel])
            ep = model.encode(P[sel])
            en = model.encode(N[sel])
            loss = etb_loss(ea, ep, en, margin=margin, lam=lam)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference: detection map (paper Fig. 1 — testing stage)
# ---------------------------------------------------------------------------

def score_thantd(
        model:      THANTD,
        t:          np.ndarray,    # (b,) prior spectrum S_prior
        X:          np.ndarray,    # (N, b) pixels to score
        batch_size: int = 512,
        device:     str = 'cpu',
) -> np.ndarray:
    """
    Compute cosine similarity between HAB(embed(S_prior)) and HAB(embed(x))
    for each pixel x.  Higher = more target-like (paper Fig. 1 test stage).
    """
    model.eval().to(device)
    with torch.no_grad():
        t_tensor = torch.tensor(t[None, :], dtype=torch.float32, device=device)
        t_emb    = model.encode(t_tensor)              # (1, d)
        scores   = []
        for i in range(0, len(X), batch_size):
            xb  = torch.tensor(X[i:i + batch_size], dtype=torch.float32,
                               device=device)
            xe  = model.encode(xb)
            scores.append(F.cosine_similarity(xe, t_emb, dim=-1).cpu().numpy())
    return np.concatenate(scores, axis=0)
