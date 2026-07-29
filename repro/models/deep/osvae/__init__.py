"""OS-VAE wrapper. Model/trainer: osvae_model.py (verbatim port; fit_osvae
seeds torch before constructing the CVAE — already contract-compliant)."""
import os

import torch


def _fit(tr, sig, cfg, seed, ckpt, device):
    from .osvae_model import CVAE, fit_osvae
    if ckpt and os.path.exists(ckpt):
        m = CVAE(tr.shape[1])
        m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
        m.to(device).eval(); m._device = str(device)
        print('    resumed', ckpt, flush=True)
        return m
    m = fit_osvae(tr, sig, seed=seed, epochs=int(cfg['osvae_epochs']),
                  device=str(device))
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(m.state_dict(), ckpt)
    return m


def _score(m, planted, sig, device):
    from .osvae_model import score_osvae
    return score_osvae(m, planted, sig)


class OSVAE:
    """Uniform detector API: OSVAE(cfg).fit(tr, sig, seed, device, ckpt).score(...)."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.state = None

    def fit(self, tr, sig, seed, device, ckpt=None):
        self.state = _fit(tr, sig, self.cfg, seed, ckpt, device)
        return self

    def score(self, planted, sig, device):
        return _score(self.state, planted, sig, device)
