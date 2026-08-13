"""configurable.py — ConfigurableLRao: the published spatial LRao with its four
hard-coded choices exposed as ablation axes.

`repro.models.lrao.model.LRao` fixes all four:

    front-end      per-band robust median/MAD, always
    score net      an MLP, always
    objective      signal-AGNOSTIC  tr(G^T Sigma^-1 G), always
    Sigma^-1       full SVD pseudo-inverse, always

which means the spatial protocol cannot run any of the LRao ablations the IID
protocol has. This subclass exposes exactly those four, with the SAME names and
semantics as repro.protocols.iid.train_lrao_local, so a spatial ablation and an
IID ablation of the same name mean the same thing:

    lrao_preproc       robust (published spatial) | mad (the paper's global
                       scalar) | whiten (our ZCA) | iqr | dart (whatever
                       whiten_mode the DART block uses — for a matched compare)
    lrao_net           mlp (ours) | cnn (the paper's 1-D CNN, TrafoScore)
    lrao_signal_aware  false (agnostic tr(J*)) | true (maximise J_s along the
                       known signature)
    lfi_sigma_reg      none (pseudo-inverse, published) | truncate | shrink
                       (Ledoit-Wolf; n-adaptive and the only one whose backward
                       is stable when the gradient flows through Sigma)

It is built on the SHARED core (repro.core.models.lfi_loss_mode2 /
lfi_loss_mode2_signal_aware / compute_lfi_detector_scores_mode2) rather than a
second copy of the maths, so the two protocols cannot silently drift apart.

TWO THINGS CARRIED OVER FROM THE IID IMPLEMENTATION, both load-bearing:

  * signal-aware MUST let the gradient flow through Sigma. J_s is scale-invariant
    in psi, so with Sigma detached the gradient just inflates psi (which inflates
    Sigma too) and training stalls after epoch 1.
  * signal-aware must NOT be model-selected on J_s. For the low-capacity paper
    CNN, J_s and detection AUC anti-correlate, so maximising J_s actively hurts.
    With `lrao_det_val` (default on for signal-aware) we instead hold out a
    background split, plant the known signature on it, and early-stop on that
    held-out detection AUC.
"""
import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from repro.core.models import (
    ScoreNet, lfi_loss_mode2, lfi_loss_mode2_signal_aware,
    compute_lfi_detector_scores_mode2,
)
from repro.core.normalization import (
    make_frontend, global_mad_whitening, robust_whitening_mad,
    robust_whitening_iqr,
)
from repro.core.seeding import seed_all

from .model import LRao


def _lrao_frontend(train_raw, cfg, seed=0):
    """The `lrao_preproc` axis. Every branch returns a frozen Whitening."""
    mode = str(cfg.get('lrao_preproc', 'robust'))
    if mode == 'mad':                       # the LRao paper's single global scalar
        return global_mad_whitening(train_raw)
    if mode == 'iqr':
        return robust_whitening_iqr(train_raw)
    if mode == 'dart':                      # match the DART block's front-end
        return make_frontend(train_raw, cfg, seed=seed)
    if mode in ('whiten', 'zca'):
        # IID resolves 'whiten' via lrao_input_norm: with `lrao_input_norm:
        # robust` (what its plans set in `common`) 'whiten' means robust-IQR, NOT
        # ZCA. Match that, or an entry of the same name means different things in
        # the two protocols (measured per-coordinate std: robust-IQR 1.3, ZCA 0.74).
        if mode == 'whiten' and str(cfg.get('lrao_input_norm', 'zca')) == 'robust':
            return robust_whitening_iqr(train_raw)
        return make_frontend(train_raw, {**cfg, 'whiten_mode': 'zca'}, seed=seed)
    return robust_whitening_mad(train_raw)  # 'robust' — the published spatial front


class ConfigurableLRao(LRao):
    """LRao with front-end / net / objective / covariance-inverse as ablation
    axes. Keeps the published fit contract (val early stopping, checkpoint
    resume, `fit_idx`) so the spatial protocol is unchanged around it."""

    def _make_model(self, D, W, device):
        if str(self.cfg.get('lrao_net', 'mlp')) == 'cnn':
            from .paper_model import TrafoScore          # the paper's 1-D CNN
            return TrafoScore(W, self.cfg, D).to(device)
        return ScoreNet(D, list(self.cfg['hidden']), self.cfg['activation'],
                        whitening=W).to(device)

    # psi() and the statistic both go through the shared core
    def psi(self, x):
        return self.model(x)

    def _sigma_reg(self, signal_aware):
        """The covariance-inverse regulariser, with one guard.

        signal-aware differentiates THROUGH Sigma (it must — see the module
        docstring), and the backward of an SVD pseudo-inverse is ill-defined at
        near-zero singular values. 'none'/'truncate' are only safe detached. The
        published spatial default is 'none', so a signal-aware spatial ablation
        silently inherits the unstable combination.

        Measured (Pavia-4, MLP, whiten front, test AUC by epoch):
            reg='none'    .753 .712 .761 .758 .713 .740   <- swings +-.05, no trend
            reg='shrink'  .746 .764 .771 .772 .770 .768   <- smooth, and best
        J_s itself climbs fine in both cases (-2.5e-7 -> -6.4e-7), so this is not
        a stalled optimisation: it is a noisy gradient making the objective stop
        tracking detection.

        So signal-aware promotes 'none' to 'shrink' and says so. Set
        `lfi_allow_unstable_sigma: true` to keep 'none' (e.g. to reproduce the
        instability deliberately).
        """
        reg = str(self.cfg.get('lfi_sigma_reg', 'none'))
        if signal_aware and reg in ('none', 'truncate') and \
                not bool(self.cfg.get('lfi_allow_unstable_sigma', False)):
            if not getattr(self, '_reg_warned', False):
                print(f"    [ConfigurableLRao] lrao_signal_aware=True with "
                      f"lfi_sigma_reg={reg!r}: the gradient flows through Sigma and "
                      f"that inverse has an ill-defined backward -> using 'shrink' "
                      f"instead (set lfi_allow_unstable_sigma: true to override).",
                      flush=True)
                self._reg_warned = True
            return 'shrink'
        return reg

    def _cost(self, batch, s_dir):
        cfg = self.cfg
        reg = self._sigma_reg(s_dir is not None)
        cut = float(cfg.get('lfi_sigma_cutoff', 1e-3))
        if s_dir is not None:
            # detach_sigma=False is REQUIRED here (see the module docstring)
            return lfi_loss_mode2_signal_aware(self.model, batch, s_dir,
                                               float(cfg['delta_theta']),
                                               detach_sigma=False,
                                               sigma_reg=reg, sigma_cutoff=cut)
        return lfi_loss_mode2(self.model, batch, float(cfg['delta_theta']),
                              detach_sigma=bool(cfg['detach_sigma']),
                              sigma_reg=reg, sigma_cutoff=cut)

    def fit(self, tr_raw, seed, device, run_dir, s_raw=None, val_raw=None):
        cfg = self.cfg
        os.makedirs(run_dir, exist_ok=True)
        tr = np.asarray(tr_raw, np.float32)
        D = tr.shape[1]
        idx = np.random.default_rng(seed).permutation(len(tr))
        nv = max(1, int(len(tr) * float(cfg['val_fraction'])))
        self.fit_idx, val_idx = idx[nv:], idx[:nv]

        seed_all(seed)                                   # BEFORE construction
        W = _lrao_frontend(tr[self.fit_idx], cfg, seed=seed).to(device)
        self.model = self._make_model(D, W, device)
        self.net = self.model                            # parity with LRao

        signal_aware = bool(cfg.get('lrao_signal_aware', False))
        if signal_aware and s_raw is None:
            raise ValueError('lrao_signal_aware=True needs s_raw (the target '
                             'signature); the spatial protocol passes scene["sig"]')
        s_dir = None
        if signal_aware:
            s = np.asarray(s_raw, np.float32)
            s_dir = torch.tensor(s / (float(np.linalg.norm(s)) + 1e-12), device=device)

        best_p = os.path.join(run_dir, 'best.pt')
        if os.path.exists(best_p):
            blob = torch.load(best_p, map_location='cpu', weights_only=False)
            self.model.load_state_dict(blob['net'])
            self.model.to(device).eval()
            self.fit_idx = np.asarray(blob['fit_idx'])
            print(f'    [ConfigurableLRao] resumed {best_p}', flush=True)
            return self

        # signal-aware selects on held-out DETECTION AUC, not on J_s
        det_val = signal_aware and bool(cfg.get('lrao_det_val', True))
        if det_val:
            from repro.core.data import plant_targets
            from repro.core.metrics import auc_safe
            vp, vlab, _ = plant_targets(
                tr[val_idx], np.asarray(s_raw, np.float32),
                float(cfg.get('lrao_det_val_amplitude', 0.15)),
                float(cfg.get('lrao_det_val_fraction', 0.10)),
                model='additive', seed=seed)
            vp = vp.astype(np.float32)

        opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg['lr']),
                               weight_decay=float(cfg['weight_decay']))
        Xf = torch.tensor(tr[self.fit_idx], device=device)
        Xv = torch.tensor(tr[val_idx], device=device)
        P, B = len(Xf), int(cfg['batch_size'])
        best, best_state, best_epoch, bad = float('inf'), None, 0, 0
        patience = int(cfg.get('patience', 20))
        # Match IID: the detection check is EXPENSIVE and noisy epoch-to-epoch, so
        # it runs every `lrao_det_val_every` epochs and `patience` counts CHECKS,
        # not epochs. Checking every epoch (what this did before) meant patience
        # 20 killed a signal-aware run at epoch 21, long before J_s has moved --
        # J_s climbs only ~2.5x over 40 epochs, so the run was being stopped
        # during its noise floor. IID's default is 20 epochs per check.
        det_every = int(cfg.get('lrao_det_val_every', 20)) if det_val else 1
        # ...and only a MEANINGFUL relative gain resets the counter, so tiny
        # noise-level wobbles neither reset nor prematurely exhaust it.
        min_delta = float(cfg.get('lrao_min_delta', 0.005))
        pbar = tqdm(range(1, int(cfg['max_epochs']) + 1),
                    desc=f'CfgLRao s{seed}', dynamic_ncols=True, leave=False)
        for ep in pbar:
            perm = torch.randperm(P, device=device)
            run, nb, skipped = 0.0, 0, 0
            for i in range(0, P, B):
                try:
                    loss = self._cost(Xf[perm[i:i + B]], s_dir)
                except Exception:
                    skipped += 1; continue
                if not torch.isfinite(loss):
                    skipped += 1; continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               float(cfg['grad_clip']))
                opt.step()
                run += float(loss.detach()); nb += 1
            if nb == 0:
                print(f'      [warn] ConfigurableLRao stalled at epoch {ep} '
                      f'(all batches skipped)', flush=True)
                break
            tl = run / max(nb, 1)
            if ep % det_every and ep != int(cfg['max_epochs']):
                pbar.set_postfix(train=format(tl, '.3e' if signal_aware else '.4f'),
                                 bad=bad, skip=skipped)
                continue                                 # not a check epoch
            if det_val:                                  # lower is better -> -AUC
                self.model.eval()
                try:
                    vl = -float(auc_safe(vlab, self.score(vp, tr[self.fit_idx],
                                                          np.asarray(s_raw, np.float32))))
                except Exception:
                    vl = float('inf')
                self.model.train()
            else:
                with torch.no_grad():
                    vl = float(self._cost(Xv, s_dir).detach())
            # J_s is a ~1e-7 scalar for the signal-aware objective, so print it
            # in scientific notation: '.4f' renders it as -0.0000 and reads like a
            # dead loss when it is in fact climbing normally (IID prints '.2f' and
            # has the same cosmetic problem).
            _fmt = '.3e' if signal_aware else '.4f'
            pbar.set_postfix(train=format(tl, _fmt),
                             **({'valAUC': f'{-vl:.3f}'} if det_val
                                else {'val': format(vl, _fmt)}),
                             bad=bad, skip=skipped)
            prev_best = best
            if vl < best:                                # always keep the best weights
                best, best_epoch = vl, ep
                best_state = copy.deepcopy(self.model.state_dict())
            margin = min_delta * abs(prev_best) if np.isfinite(prev_best) else 0.0
            if (not np.isfinite(prev_best)) or (vl < prev_best - margin):
                bad = 0                                  # a meaningful gain
            else:
                bad += 1
            if True:
                if patience > 0 and bad >= patience:
                    print(f'      [early-stop] ConfigurableLRao at epoch {ep}', flush=True)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        torch.save({'net': {k: v.cpu() for k, v in self.model.state_dict().items()},
                    'fit_idx': self.fit_idx, 'epoch': best_epoch,
                    'val_loss': best}, best_p)
        print(f"    [ConfigurableLRao] best epoch {best_epoch} "
              f"({'val AUC ' + format(-best, '.4f') if det_val else 'val ' + format(best, '.4f')}"
              f", fit {len(self.fit_idx)}/val {len(val_idx)})", flush=True)
        return self

    @torch.no_grad()
    def score(self, test_pixels, ref_pixels, sig):
        """ref_pixels MUST be the fit subset: tr_raw[self.fit_idx]."""
        cfg = self.cfg
        return compute_lfi_detector_scores_mode2(
            self.model, np.asarray(ref_pixels, np.float32),
            np.asarray(test_pixels, np.float32), np.asarray(sig, np.float32),
            delta_theta=float(cfg['delta_theta']),
            # same regulariser the net was TRAINED under (so the signal-aware
            # promotion above applies here too, not just in the loss)
            sigma_reg=self._sigma_reg(bool(cfg.get('lrao_signal_aware', False))),
            sigma_cutoff=float(cfg.get('lfi_sigma_cutoff', 1e-3)))
