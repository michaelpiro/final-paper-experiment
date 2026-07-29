"""THANTD wrapper. Model/trainer: thantd_model.py (verbatim port).

Seed-order fix: torch/numpy are seeded BEFORE THANTD() is constructed (the
camera-ready `_fit_thantd` built the model first, so its init depended on
ambient session state)."""
import os

import numpy as np
import torch

from repro.core.seeding import seed_all


def _fit(tr, sig, cfg, seed, ckpt, device):
    from .thantd_model import THANTD, build_thantd_samples, train_thantd
    if ckpt and os.path.exists(ckpt):
        m = THANTD(b=tr.shape[1])
        m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
        m.to(device).eval()
        print('    resumed', ckpt, flush=True)
        return m
    seed_all(seed)                                   # BEFORE construction
    rng = np.random.default_rng(seed)
    m = THANTD(b=tr.shape[1])
    a, p, n = build_thantd_samples(tr, sig, alpha=0.5,
                                   n_samples=int(cfg['thantd_samples']),
                                   rng=rng, bkg_pool=tr)
    m.to(device)
    train_thantd(m, a, p, n, epochs=int(cfg['thantd_epochs']), batch_size=64,
                 lr=1e-4, margin=0.3, device=device)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(m.state_dict(), ckpt)
    return m


def _score(m, planted, sig, device):
    from .thantd_model import score_thantd
    return score_thantd(m, sig, planted, device=device)


class THANTD:
    """Uniform detector API: THANTD(cfg).fit(tr, sig, seed, device, ckpt).score(...)."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.state = None

    def fit(self, tr, sig, seed, device, ckpt=None):
        self.state = _fit(tr, sig, self.cfg, seed, ckpt, device)
        return self

    def score(self, planted, sig, device):
        return _score(self.state, planted, sig, device)
