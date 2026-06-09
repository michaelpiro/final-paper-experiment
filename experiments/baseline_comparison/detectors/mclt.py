"""
MCLT (deep template) — momentum-contrastive transformer baseline.

Faithful self-contained re-implementation of the core of:
  Wang et al., "An Unsupervised Momentum Contrastive Learning based Transformer
  Network for Hyperspectral Target Detection", JSTARS 2024.

The upstream repo (potential_spatial_baselines_code/code/MCLT-master) hardcodes a
specific dataset (San Diego, 189 bands, fixed patch sizes, image-tied queue size).
This adapter implements the same ideas in a dataset-agnostic way so it slots into
the framework contract (works on Pavia-103, synthetic-6D, etc.):

  - overlapping spectral patch embedding  (g(s): bands -> overlapping windows -> tokens)
  - small transformer encoder + projection head, L2-normalised feature
  - MoCo: base + momentum encoder + negative queue, InfoNCE contrastive loss,
    trained UNSUPERVISED on background spectra (two augmented views)
  - detection: cosine similarity between embedded test spectrum and embedded prior

Consumes RAW bands. Per-pixel (spectral), so needs_spatial=False — runs on every
dataset. Higher score = more target-like.
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


class _OverlapSpectralEmbed(nn.Module):
    """spectrum (B, D) -> overlapping band windows -> linear tokens (B, T, E) + cls/pos."""
    def __init__(self, D, win=9, stride=4, embed=64):
        super().__init__()
        if win > D:
            win = D
        self.win, self.stride = win, max(1, stride)
        starts = list(range(0, max(1, D - win + 1), self.stride))
        if starts[-1] != D - win:
            starts.append(D - win)
        self.register_buffer("starts", torch.tensor(starts, dtype=torch.long))
        T = len(starts)
        self.proj = nn.Linear(win, embed)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, embed))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, s):                          # s: (B, D)
        wins = torch.stack([s[:, i:i + self.win] for i in self.starts.tolist()], 1)  # (B,T,win)
        z = self.proj(wins)                        # (B, T, E)
        cls = self.cls.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], 1) + self.pos
        return z


class _Encoder(nn.Module):
    def __init__(self, D, embed=64, depth=2, heads=4, out=64):
        super().__init__()
        self.embed = _OverlapSpectralEmbed(D, embed=embed)
        layer = nn.TransformerEncoderLayer(embed, heads, embed * 2, batch_first=True,
                                           dropout=0.0, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.Linear(embed, embed), nn.GELU(), nn.Linear(embed, out))

    def forward(self, s):
        z = self.tr(self.embed(s))[:, 0]           # cls token
        return F.normalize(self.head(z), dim=-1)


def _augment(x, rng_noise=0.05):
    """Two spectral views: gaussian jitter + random per-band gain."""
    gain = 1.0 + (torch.rand_like(x) - 0.5) * 0.2
    return x * gain + torch.randn_like(x) * rng_noise


@register("MCLT")
class MCLT(Detector):
    needs_spatial = False
    space = "raw"

    def _build(self, D):
        c = self.cfg
        enc = _Encoder(D, embed=int(c.get("embed", 64)), depth=int(c.get("depth", 2)),
                       heads=int(c.get("heads", 4)), out=int(c.get("dim", 64)))
        return enc

    def fit(self, ctx: DetectorInput) -> "Detector":
        c = self.cfg
        dev = ctx.device
        X = torch.tensor(ctx.train_raw, dtype=torch.float32, device=dev)
        D = X.shape[1]
        self._D = D
        torch.manual_seed(ctx.seed)
        self.base = self._build(D).to(dev)
        self.mom = copy.deepcopy(self.base).to(dev)
        for p in self.mom.parameters():
            p.requires_grad = False
        dim = int(c.get("dim", 64)); K = int(c.get("queue", 4096)); T = float(c.get("temp", 0.2))
        m = float(c.get("momentum", 0.99))
        queue = F.normalize(torch.randn(dim, K, device=dev), dim=0)
        opt = torch.optim.AdamW(self.base.parameters(), lr=float(c.get("lr", 1e-3)),
                                weight_decay=1e-4)
        epochs = int(c.get("mclt_epochs", 30)); bs = int(c.get("batch", 256))
        noise = float(c.get("aug_noise", 0.05))
        ptr = 0; n = len(X); log = {}
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev); tot = 0.0; nb = 0
            for i in range(0, n, bs):
                xb = X[perm[i:i + bs]]
                q = self.base(_augment(xb, noise))
                with torch.no_grad():
                    for pb, pm in zip(self.base.parameters(), self.mom.parameters()):
                        pm.data = pm.data * m + pb.data * (1 - m)
                    k = self.mom(_augment(xb, noise))
                l_pos = (q * k).sum(-1, keepdim=True)                  # (b,1)
                l_neg = q @ queue.clone().detach()                    # (b,K)
                logits = torch.cat([l_pos, l_neg], 1) / T
                loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():                                  # enqueue
                    bsz = k.size(0)
                    if ptr + bsz <= K:
                        queue[:, ptr:ptr + bsz] = k.T
                    else:
                        queue[:, ptr:] = k.T[:, :K - ptr]
                        queue[:, :bsz - (K - ptr)] = k.T[:, K - ptr:]
                    ptr = (ptr + bsz) % K
                tot += float(loss.detach()); nb += 1
            log[ep] = tot / max(nb, 1)
        self.base.eval(); self._log = log
        return self

    @torch.no_grad()
    def _embed(self, x, dev, bs=2048):
        out = []
        for i in range(0, len(x), bs):
            out.append(self.base(torch.tensor(x[i:i + bs], dtype=torch.float32, device=dev)).cpu())
        return torch.cat(out).numpy()

    def score(self, ctx: DetectorInput) -> np.ndarray:
        dev = ctx.device
        f = self._embed(ctx.test_raw, dev)                            # (n, dim) normalised
        p = self._embed(ctx.signature_raw[None], dev)[0]              # (dim,)
        return f @ p                                                  # cosine (already L2-norm)

    # -- persistence --
    def state(self):
        return {"cfg": self.cfg, "log": self._log, "D": self._D,
                "sd": {k: v.cpu() for k, v in self.base.state_dict().items()}}

    def load_state(self, s):
        super().load_state(s)
        self._D = s["D"]
        self.base = self._build(self._D)
        self.base.load_state_dict(s["sd"]); self.base.eval()
