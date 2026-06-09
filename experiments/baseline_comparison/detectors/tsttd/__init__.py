"""
TSTTD adapter — faithful to the authors' Train_eval.py (identical model, ISIA
triplet+BCE loss, R2TM target synthesis, warmup+cosine schedule, hyperparams).
Only the I/O is adapted: background = our training pixels, prior = our signature,
detection = cosine similarity of the learned feature to the prior feature.

Reference: Jiao, Gong, Zhong, IEEE TGRS 2023. Upstream:
  potential_spatial_baselines_code/code/TSTTD-main/{Model,Train_eval,Scheduler}.py
"""

from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.optim as optim

from ...framework.detector_api import Detector, DetectorInput
from ...framework.registry import register
from .model import SpectralGroupAttention
from .scheduler import GradualWarmupScheduler


def _seed(seed=1):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _spectral_group(x, n, m):
    """(B, c) -> (B, n, m) overlapping spectral windows (verbatim from Train_eval)."""
    pad = m // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode='symmetric')
    out = np.zeros([x.shape[0], n, m], dtype=np.float32)
    for i in range(n):
        out[:, i, :] = xp[:, i:i + m]
    return torch.from_numpy(out).float()


def _cos(x, y):
    xn = torch.sqrt(torch.sum(x ** 2, dim=1))
    yn = torch.sqrt(torch.sum(y ** 2, dim=1))
    return torch.sum(x * y, dim=1) / (xn * yn + 1e-8)


def _isia_loss(x, B, margin=1.0, lambd=1):
    positive, negative, prior = x[:B], x[B:2 * B], x[2 * B:]
    p_sim = _cos(positive, prior)
    n_sim1 = _cos(negative, prior)
    n_sim2 = _cos(negative, positive)
    max_n = torch.maximum(n_sim1, n_sim2)
    triplet = torch.mean(torch.relu(margin + max_n - p_sim))
    p = torch.sigmoid(p_sim); n = torch.sigmoid(1 - n_sim1)
    bce = -0.5 * torch.mean(torch.log(p + 1e-8) + torch.log(n + 1e-8))
    return triplet + lambd * bce


@register("TSTTD")
class TSTTD(Detector):
    needs_spatial = False
    space = "raw"

    def _build(self, band):
        c = self.cfg
        return SpectralGroupAttention(
            band=band, m=int(c.get("group_length", 20)), d=int(c.get("channel", 128)),
            depth=int(c.get("depth", 4)), heads=int(c.get("heads", 4)),
            dim_head=int(c.get("dim_head", 64)), mlp_dim=int(c.get("mlp_dim", 64)),
            adjust=bool(c.get("adjust", False)))

    def fit(self, ctx: DetectorInput) -> "Detector":
        c = self.cfg
        dev = ctx.device if torch.cuda.is_available() else "cpu"
        _seed(int(c.get("seed", 1)))
        epoch = int(c.get("epoch", 20)); B = int(c.get("batch_size", 64))
        m = int(c.get("group_length", 20)); band = ctx.train_raw.shape[1]
        self._band = band
        # standardize like Tools.standard (store for scoring)
        bg = ctx.train_raw.astype(np.float32)
        self._mn = float(bg.min()); self._rng = float(bg.max() - bg.min()) or 1.0
        bg = (bg - self._mn) / self._rng
        sig = ((ctx.signature_raw.astype(np.float32) - self._mn) / self._rng)[None, :]
        # R2TM positive (target-like) samples
        alphas = np.random.uniform(0, 0.1, len(bg))[:, None].astype(np.float32)
        tgt = alphas * bg + (1 - alphas) * sig

        self._model = self._build(band).to(dev); self._model.train()
        opt = torch.optim.AdamW(self._model.parameters(), lr=float(c.get("lr", 1e-4)),
                                weight_decay=1e-4)
        cos = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch, eta_min=0)
        warm = GradualWarmupScheduler(opt, multiplier=int(c.get("multiplier", 2)),
                                      warm_epoch=max(epoch // 10, 1), after_scheduler=cos)
        gc = float(c.get("grad_clip", 1.0))
        n = (len(bg) // B) * B
        for e in range(epoch):
            perm = np.random.permutation(len(bg))[:n]
            for i in range(0, n, B):
                idx = perm[i:i + B]
                combined = np.concatenate([tgt[idx], bg[idx], sig], axis=0)
                x0 = _spectral_group(combined, band, m).to(dev)
                opt.zero_grad()
                feats = self._model(x0)
                loss = _isia_loss(feats, B)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), gc)
                opt.step()
            warm.step()
        self._model.eval(); self._log = {"final_loss": float(loss.detach())}
        return self

    @torch.no_grad()
    def score(self, ctx: DetectorInput) -> np.ndarray:
        dev = ctx.device if torch.cuda.is_available() else "cpu"
        m = int(self.cfg.get("group_length", 20)); band = self._band
        sig = ((ctx.signature_raw.astype(np.float32) - self._mn) / self._rng)[None, :]
        tgt_feat = self._model(_spectral_group(sig, band, m).to(dev)).cpu().numpy()
        X = ((ctx.test_raw.astype(np.float32) - self._mn) / self._rng)
        out, bs = [], 512
        for i in range(0, len(X), bs):
            f = self._model(_spectral_group(X[i:i + bs], band, m).to(dev)).cpu().numpy()
            xn = np.sqrt((f ** 2).sum(1)); yn = np.sqrt((tgt_feat ** 2).sum(1))
            out.append((f * tgt_feat).sum(1) / (xn * yn + 1e-8))
        return np.concatenate(out)

    def state(self):
        return {"cfg": self.cfg, "log": self._log, "band": self._band,
                "mn": self._mn, "rng": self._rng,
                "sd": {k: v.cpu() for k, v in self._model.state_dict().items()}}

    def load_state(self, s):
        super().load_state(s)
        self._band, self._mn, self._rng = s["band"], s["mn"], s["rng"]
        self._model = self._build(self._band); self._model.load_state_dict(s["sd"])
        self._model.eval()
