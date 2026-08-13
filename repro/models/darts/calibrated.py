"""calibrated.py — CalibratedDARTS: the CalibratedDART recipe applied to DARTS.

Same three changes as `repro.models.dart.calibrated.CalibratedDART` — a
configurable front-end, a data-driven sigma, and AdamW + cosine + a held-out
best-epoch monitor — applied to the neighbour-conditioned score net. It
SUBCLASSES DARTS, so the `_NeighborDenoiser` architecture (d_lat, K, enc_hidden,
score_hidden), the detection statistic and DARTS-CFAR are inherited unchanged.
Only `fit` is overridden.

READ THIS BEFORE USING A NON-ZCA FRONT-END HERE. DARTS is measured to PREFER the
published ZCA front-end (WMW handoff Sec. 2.6): its latent-kNN conditioning
already carries the background's mode information, so a front-end that leaves
that structure in the input corrupts the neighbour metric. Harm grew
monotonically with the residual structure and no k beat ZCA at 5000 epochs.
`normalize` / `shrink` leave it fully intact — so unlike DART, where dropping
full whitening was the win, here it may be the loss. Run `whiten_mode: zca`
alongside to separate the training-recipe effect from the front-end effect.
"""
import os

import numpy as np
import torch

from repro.core.data import Whitening
from repro.core.normalization import make_frontend
from repro.core.seeding import seed_all
from repro.core.sigma import resolve_sigma
from repro.models.dart.calibrated import run_dsm_training

from .model import DARTS, _NeighborDenoiser


class CalibratedDARTS(DARTS):
    """DARTS with a configurable front-end, data-driven sigma and a validated
    training loop. Architecture and scoring are inherited from DARTS."""

    def fit(self, tr_raw, tr_nbr, seed, device, ckpt=None, reseed=True,
            val_raw=None, val_nbr=None):
        cfg = self.cfg
        D = tr_raw.shape[1]
        if reseed:
            seed_all(seed)                               # BEFORE construction
        Xr = np.asarray(tr_raw, np.float32)

        def _build(whitening, sigma):                    # DARTS's architecture
            return _NeighborDenoiser(
                D, int(cfg['d_lat']), int(cfg['K']), list(cfg['enc_hidden']),
                list(cfg['score_hidden']), sigma, cfg['activation'],
                whitening).to(device)

        if ckpt and os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location='cpu', weights_only=False)
            Wh = Whitening(np.asarray(blob['whiten_mu'], np.float32),
                           np.asarray(blob['whiten_W'], np.float32)).to(device)
            self.net = _build(Wh, float(blob['sigma']))
            self.net.load_state_dict(blob['net'])
            self.net.eval()
            print(f'    [CalibratedDARTS] resumed {ckpt}', flush=True)
            return self

        Wh = make_frontend(Xr, cfg, seed=seed).to(device)
        Zw = Wh(torch.tensor(Xr, device=device)).detach().cpu().numpy()
        # delta (train->val mean displacement) widens a rule-name sigma to cover
        # the shift faced at test time; see repro.core.sigma.transfer_gap.
        Zval = None
        if val_raw is not None and len(val_raw) >= 8:
            Zval = Wh(torch.tensor(np.asarray(val_raw, np.float32), device=device)
                      ).detach().cpu().numpy()
        sigma = resolve_sigma(Zw, cfg, seed=seed, label='DARTS', D=D, Zval=Zval)
        self.net = _build(Wh, sigma)

        n_fit = None
        Nr = np.asarray(tr_nbr, np.float32)
        if val_raw is not None and val_nbr is not None and len(val_raw) >= 8:
            n_fit = len(Xr)
            Xr = np.concatenate([Xr, np.asarray(val_raw, np.float32)], 0)
            Nr = np.concatenate([Nr, np.asarray(val_nbr, np.float32)], 0)
            print(f'    [CalibratedDARTS] validating on a DISJOINT region '
                  f'({len(Xr) - n_fit} px)', flush=True)
        X = torch.tensor(Xr, device=device)
        N = torch.tensor(Nr, device=device)

        def loss_fn(idx, fixed_noise):
            x_w = self.net.whitening(X[idx])
            nbr_w = self.net.whitening(N[idx])
            eps = (fixed_noise.to(x_w.dtype) if fixed_noise is not None
                   else torch.randn_like(x_w)) * sigma
            score = self.net._forward_inner(x_w + eps, nbr_w)
            return ((score + eps / (sigma ** 2)) ** 2).sum(-1).mean()

        run_dsm_training(self.net, Wh, X, cfg, seed, f'CalDARTS s{seed}',
                         sigma, device, loss_fn, n_fit=n_fit)
        self.net.eval()
        if ckpt:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({'net': {k: v.cpu() for k, v in self.net.state_dict().items()},
                        'whiten_mu': Wh.mu.detach().cpu().numpy(),
                        'whiten_W': Wh.W.detach().cpu().numpy(),
                        'sigma': sigma}, ckpt)
        return self
