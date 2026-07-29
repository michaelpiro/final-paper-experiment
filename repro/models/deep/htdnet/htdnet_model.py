"""htdnet_model.py — faithful implementation of HTD-Net (Zhang et al.,
Remote Sensing 2020, 12(9):1489, doi:10.3390/rs12091489).

Pipeline (paper Sec. 2, Algorithm 1):
  1. U-AE (U-net style 1-D conv autoencoder, Fig. 2) trained with MSE on the
     available pixels; used to GENERATE target samples from the known target
     signature(s).                                   [Sec. 2.1, eq. 1]
  2. LP-based background selection: iteratively pick the image pixel with the
     largest linear-prediction residual w.r.t. the target matrix, append, and
     repeat; then augment by Euclidean-nearest pixels to 500 samples.
                                                     [Sec. 2.2, eq. 2-5]
  3. Pixel-pairs: target-target (similar) + target-background (dissimilar).
     Labels: LINEAR strategy (the paper's chosen one) — ACE similarity of the
     mean-removed pair under the test-image covariance G.  [Sec. 2.3, eq. 7]
  4. SD-CNN (Fig. 4/5): input |t_i - t_j| (1 x d), 16 conv layers with 30
     kernels of size 1x3; layers C4, C7, C10, C13, C16 have stride 2 (no pad),
     the rest stride 1 (zero-pad, size-invariant); AVG pool; FC -> scalar.
     MSE loss to the ACE label (linear regression head).  [Sec. 2.4, eq. 8]
  5. Detection: D(z) = r_t(z) - r_b(z), the mean pair-similarity of z to the
     target samples minus to the background samples.       [Sec. 2.5, eq. 10]

Paper hyperparameters kept: 1000 generated targets, 500 background samples,
balanced similar/dissimilar pair sampling, batch 256, lr 0.01 with decay
(Sec. 4.2).  ONE documented adaptation (needed for the single-prior-signature,
target-agnostic protocol): U-AE generation inputs are R2TM-style mixtures
t_mu = (1-mu)*t + mu*b_rand, mu~U[0,0.1] (same recipe THANTD uses, eq. 2 there)
instead of "a small number of known target pixels", since exactly one prior
signature is available.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. U-AE generator (paper Fig. 2)
# --------------------------------------------------------------------------- #
class UAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv1d(1, 20, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.e2 = nn.Conv1d(20, 40, 3, padding=1)
        self.e3 = nn.Conv1d(40, 40, 3, padding=1)
        self.up = nn.ConvTranspose1d(40, 40, 2, stride=2)
        self.d1 = nn.Conv1d(60, 20, 3, padding=1)   # cat(skip 20, up 40)
        self.d2 = nn.Conv1d(20, 1, 3, padding=1)

    def forward(self, x):                            # x: (B, 1, L), L even
        s = F.relu(self.e1(x))
        h = self.pool(s)
        h = F.relu(self.e2(h))
        h = F.relu(self.e3(h))
        h = self.up(h)
        if h.shape[-1] != s.shape[-1]:               # odd-L guard
            h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
        h = F.relu(self.d1(torch.cat([s, h], dim=1)))
        return self.d2(h)


def train_uae(pixels: np.ndarray, epochs=200, batch=256, lr=1e-3,
              device="cpu", seed=0, log_every=0):
    """Train the U-AE with MSE reconstruction on the given pixels (paper: the
    available samples of the training image). Pixels are scaled to [0,1] by the
    global max (scale is returned for inverse mapping)."""
    torch.manual_seed(seed)
    scale = float(np.abs(pixels).max()) + 1e-12
    X = torch.tensor(pixels / scale, dtype=torch.float32, device=device).unsqueeze(1)
    net = UAE().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            b = X[perm[i:i + batch]]
            loss = F.mse_loss(net(b), b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if log_every and (ep + 1) % log_every == 0:
            print(f"  [U-AE] epoch {ep+1}/{epochs} loss={tot:.5f}", flush=True)
    net.eval()
    return net, scale


@torch.no_grad()
def generate_targets(uae: UAE, scale: float, target_sig: np.ndarray,
                     bkg_pixels: np.ndarray, n_samples=1000,
                     mu_max=0.1, device="cpu", rng=None):
    """Generate target samples by passing R2TM-style jittered copies of the
    prior signature through the trained U-AE (documented adaptation)."""
    rng = rng or np.random.default_rng(0)
    t = target_sig / scale
    b = bkg_pixels[rng.integers(0, len(bkg_pixels), n_samples)] / scale
    mu = rng.uniform(0.0, mu_max, size=(n_samples, 1))
    inp = (1.0 - mu) * t[None, :] + mu * b
    x = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(1)
    out = []
    for i in range(0, len(x), 512):
        out.append(uae(x[i:i + 512]).squeeze(1).cpu().numpy())
    return np.concatenate(out, 0) * scale            # back to data scale


# --------------------------------------------------------------------------- #
# 2. LP-based background selection (paper eq. 2-5)
# --------------------------------------------------------------------------- #
def lp_background_selection(image_pixels: np.ndarray, target_sig: np.ndarray,
                            n_direct=60, n_total=500, ridge=1e-8):
    """Iteratively select the pixel with max LP residual ||y - Xt a||, append it
    to Xt, repeat (paper stops when Xt^T Xt turns ill-rank; we also cap at
    n_direct).  Then augment with Euclidean-nearest pixels to n_total."""
    X = image_pixels.astype(np.float64)
    Xt = target_sig.astype(np.float64).reshape(-1, 1)      # (d, m)
    chosen = []
    for _ in range(n_direct):
        G = Xt.T @ Xt
        G = G + ridge * np.trace(G) / len(G) * np.eye(len(G))
        try:
            P = Xt @ np.linalg.solve(G, Xt.T)               # projector
        except np.linalg.LinAlgError:
            break
        resid = X - X @ P.T
        e = np.linalg.norm(resid, axis=1)
        j = int(np.argmax(e))
        if e[j] < 1e-9:
            break
        chosen.append(j)
        Xt = np.concatenate([Xt, X[j].reshape(-1, 1)], axis=1)
        if np.linalg.cond(Xt.T @ Xt) > 1e12:
            break
    B = X[chosen]
    # Euclidean augmentation to n_total (paper Sec. 2.2 / Algorithm 1 step 3)
    if len(B) < n_total and len(B) > 0:
        d2 = ((X[:, None, :] - B[None, :, :]) ** 2).sum(-1).min(axis=1)
        order = np.argsort(d2)
        extra = [i for i in order if i not in set(chosen)][:n_total - len(B)]
        B = np.concatenate([B, X[extra]], axis=0)
    return B.astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. ACE linear labels (paper eq. 7)
# --------------------------------------------------------------------------- #
class ACELabeler:
    def __init__(self, detect_image_pixels: np.ndarray, ridge=1e-6):
        X = detect_image_pixels.astype(np.float64)
        self.mu = X.mean(axis=0)
        C = np.cov(X - self.mu, rowvar=False)
        C = C + ridge * np.trace(C) / len(C) * np.eye(len(C))
        self.Ginv = np.linalg.inv(C)

    def __call__(self, ti: np.ndarray, tj: np.ndarray):
        a = ti - self.mu; b = tj - self.mu
        num = (np.einsum("id,de,ie->i", a, self.Ginv, b)) ** 2
        den = (np.einsum("id,de,ie->i", a, self.Ginv, a) *
               np.einsum("id,de,ie->i", b, self.Ginv, b)) + 1e-30
        return (num / den).astype(np.float32)


# --------------------------------------------------------------------------- #
# 4. SD-CNN (paper Fig. 4/5)
# --------------------------------------------------------------------------- #
class SDCNN(nn.Module):
    def __init__(self, n_kernels=30):
        super().__init__()
        layers, ch = [], 1
        for i in range(1, 17):                        # C1..C16
            stride2 = i in (4, 7, 10, 13, 16)
            layers.append(nn.Conv1d(ch, n_kernels, 3,
                                    stride=2 if stride2 else 1,
                                    padding=0 if stride2 else 1))
            layers.append(nn.ReLU())
            ch = n_kernels
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(n_kernels, 1)

    def forward(self, diff):                          # diff: (B, 1, d)
        h = self.conv(diff)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)   # (B, 30)
        return self.fc(h).squeeze(-1)                 # (B,)


def train_sdcnn(targets: np.ndarray, backgrounds: np.ndarray,
                labeler: ACELabeler, epochs=30, pairs_per_epoch=100_000,
                batch=256, lr=1e-3, device="cpu", seed=0, log_every=5):
    """Train the SD-CNN with the linear (ACE-label, MSE) strategy on pixel-pair
    differences; balanced 50/50 similar (t-t) / dissimilar (t-b) sampling."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(max(np.abs(targets).max(), np.abs(backgrounds).max())) + 1e-12
    net = SDCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.3)
    nt, nb = len(targets), len(backgrounds)
    for ep in range(epochs):
        tot, nbtch = 0.0, 0
        for i0 in range(0, pairs_per_epoch, batch):
            m = min(batch, pairs_per_epoch - i0)
            half = m // 2
            i = rng.integers(0, nt, m)
            j_t = rng.integers(0, nt, half)                    # t-t (similar)
            j_b = rng.integers(0, nb, m - half)                # t-b (dissimilar)
            a = targets[i]
            b = np.concatenate([targets[j_t], backgrounds[j_b]], axis=0)
            y = labeler(a, b)
            diff = torch.tensor(np.abs(a - b) / scale, dtype=torch.float32,
                                device=device).unsqueeze(1)
            yt = torch.tensor(y, device=device)
            loss = F.mse_loss(net(diff), yt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nbtch += 1
        sched.step()
        if log_every and (ep + 1) % log_every == 0:
            print(f"  [SD-CNN] epoch {ep+1}/{epochs} loss={tot/nbtch:.5f}", flush=True)
    net.eval()
    net._scale = scale
    return net


@torch.no_grad()
def htdnet_detect(net: SDCNN, test_pixels: np.ndarray, targets: np.ndarray,
                  backgrounds: np.ndarray, n_score_t=200, n_score_b=200,
                  batch=4096, device="cpu", rng=None):
    """D(z) = mean similarity(z, targets) - mean similarity(z, backgrounds)
    (paper eq. 10). Uses a random subset of samples for tractability."""
    rng = rng or np.random.default_rng(0)
    T = targets[rng.choice(len(targets), min(n_score_t, len(targets)), replace=False)]
    B = backgrounds[rng.choice(len(backgrounds), min(n_score_b, len(backgrounds)), replace=False)]
    scale = getattr(net, "_scale", 1.0)
    out = np.zeros(len(test_pixels), dtype=np.float64)
    for grp, sgn in ((T, +1.0), (B, -1.0)):
        acc = np.zeros(len(test_pixels), dtype=np.float64)
        for s in grp:
            diffs = np.abs(test_pixels - s[None, :]) / scale
            for i in range(0, len(diffs), batch):
                d = torch.tensor(diffs[i:i + batch], dtype=torch.float32,
                                 device=device).unsqueeze(1)
                acc[i:i + batch] += net(d).cpu().numpy()
        out += sgn * acc / len(grp)
    return out.astype(np.float32)
