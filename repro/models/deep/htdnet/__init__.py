"""HTD-Net wrapper. Model/trainers: htdnet_model.py (verbatim port)."""
import os

import numpy as np
import torch

from repro.core.seeding import seed_all


def _fit(tr, sig, cfg, seed, ckpt, device):
    from . import htdnet_model as H
    if ckpt and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = H.SDCNN(); sd.load_state_dict(blob['sdcnn'])
        sd._scale = blob['scale']; sd.to(device).eval()
        print('    resumed', ckpt, flush=True)
        return dict(sd=sd, gen_t=blob['gen_t'], bkg=blob['bkg'])
    seed_all(seed)
    rng = np.random.default_rng(seed)
    uae, sc = H.train_uae(tr, epochs=int(cfg['uae_epochs']), device=device,
                          seed=seed)
    gen_t = H.generate_targets(uae, sc, sig, tr, n_samples=1000,
                               device=device, rng=rng)
    bkg = H.lp_background_selection(tr, sig, n_direct=60, n_total=500)
    sd = H.train_sdcnn(gen_t, bkg, H.ACELabeler(tr),
                       epochs=int(cfg['sdcnn_epochs']),
                       pairs_per_epoch=int(cfg['sdcnn_pairs']),
                       device=device, seed=seed, log_every=10)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(dict(sdcnn=sd.state_dict(), scale=sd._scale,
                        gen_t=gen_t, bkg=bkg), ckpt)
    return dict(sd=sd, gen_t=gen_t, bkg=bkg)


def _score(state, planted, sig, device):
    from . import htdnet_model as H
    return H.htdnet_detect(state['sd'], planted, state['gen_t'],
                           state['bkg'], device=device)


class HTDNet:
    """Uniform detector API: HTDNet(cfg).fit(tr, sig, seed, device, ckpt).score(...)."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.state = None

    def fit(self, tr, sig, seed, device, ckpt=None):
        self.state = _fit(tr, sig, self.cfg, seed, ckpt, device)
        return self

    def score(self, planted, sig, device):
        return _score(self.state, planted, sig, device)
