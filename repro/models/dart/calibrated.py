"""calibrated.py — CalibratedDART: DART with the fixed hyper-parameters replaced
by data-driven choices.

The published DART (`repro.models.dart.model.DART`) hard-codes three things:
a ZCA front-end, a fixed noise level sigma = sqrt(rho), and a plain Adam loop
whose best epoch is the lowest TRAINING loss. CalibratedDART changes exactly
those and nothing else — it SUBCLASSES DART, so the score network, psi(), the
detection statistic and DART-CFAR are all inherited verbatim.

    front-end   `whiten_mode`     zca | normalize | shrink | vicreg | wmw | ...
                                  (repro.core.normalization.make_frontend)
    sigma       `dsm_sigma_rho`   a number, or a rule name resolved from the
                                  whitened pixels (repro.core.sigma.resolve_sigma)
    training    AdamW(amsgrad, betas .95/.999) + cosine schedule, an optional
                sample-size-adaptive budget, and a HELD-OUT split with fixed
                pre-sampled noise so the best epoch is chosen on validation loss

Why it exists: on Pavia-U multi-class these three changes take DART from
AUC .713 -> .811 at n=20 and .855 -> .933 at n=2000, and move it past the
learned-Rao baselines at every training size. The front-end is the biggest term
(~.10 AUC), sigma the smaller one (~.01-.04).

`train_calibrated_scorenet` is the same recipe as a plain function, for the IID
protocol, which builds a `ScoreNet` rather than using the DART class.
"""
import copy
import os
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from repro.core.data import Whitening
from repro.core.models import ScoreNet, dsm_loss
from repro.core.normalization import make_frontend
from repro.core.seeding import seed_all
from repro.core.sigma import resolve_sigma

from .model import DART


# ---------------------------------------------------------------------------
# The shared pieces: budget, and the DSM loop with a held-out monitor
# ---------------------------------------------------------------------------

def resolve_budget(n_train: int, cfg: dict, label: str = '') -> Tuple[int, int, float]:
    """(batch, epochs, lr). With `dsm_auto_budget` the three are derived from n so
    one config spans n=20..thousands at a roughly constant number of optimizer
    steps; the lr is batch-scaled by the sqrt rule.

    NOTE the lr this returns changes the optimal sigma (measured: same 8000
    steps, lr 5e-4 -> sigma* 0.245, lr 1e-3 -> sigma* 0.141), so a sigma constant
    calibrated under one budget does not transfer to another.
    """
    if not bool(cfg.get('dsm_auto_budget', False)):
        return (min(int(cfg['batch_size']), n_train),
                int(cfg.get('dsm_epochs', cfg.get('epochs', 2000))),
                float(cfg['lr']))
    bs = int(min(np.clip(n_train, int(cfg.get('dsm_min_bs', 16)),
                         int(cfg.get('dsm_max_bs', 256))), n_train))
    spe = max(1, int(np.ceil(n_train / bs)))
    epochs = int(np.clip(round(int(cfg.get('dsm_target_steps', 20000)) / spe),
                         int(cfg.get('dsm_min_epochs', 100)),
                         int(cfg.get('dsm_max_epochs', 3000))))
    lr = float(cfg['lr']) * float(np.sqrt(bs / max(int(cfg.get('dsm_ref_bs', 64)), 1)))
    print(f'    [budget] {label}: n={n_train} bs={bs} epochs={epochs} '
          f'(~{epochs * spe} steps) lr={lr:.2e}', flush=True)
    return bs, epochs, lr


def run_dsm_training(inner, whitening, X, cfg, seed, label, sigma, device,
                     loss_fn=None, n_fit=None) -> Tuple[list, int]:
    """Train `inner` (a whitened-space -> whitened-space module) by DSM.

    Shared by CalibratedDART and CalibratedDARTS; `loss_fn(batch_idx, noise)` lets
    DARTS inject its neighbour context. Returns (history, best_epoch) and leaves
    `inner` holding the best-epoch weights.

    The held-out monitor uses noise pre-sampled ONCE from a local generator, so
    the validation loss is deterministic across epochs and never draws from the
    global RNG (which would perturb the training noise stream — this was silently
    wrong on MPS, whose RNG is not covered by get/set_rng_state).
    """
    N = len(X)
    if n_fit is not None and n_fit < N:
        # EXPLICIT validation set: rows [n_fit:] are a spatially DISJOINT region,
        # not a random split of the training pixels. That matters — selecting the
        # best epoch on an in-box split selects for fitting the training box,
        # while the task is to transfer to a different region. Measured on
        # Pavia-4: the sigma optimum is g=0.53 on a same-box hold-out but g=1.69
        # on the real test box, so same-box selection is wrong by ~3x.
        idx_tr = torch.arange(n_fit, device=device)
        idx_va = torch.arange(n_fit, N, device=device)
    else:
        vf = float(cfg.get('dsm_val_frac', 0.1))
        n_val = int(round(vf * N))
        use_val = n_val >= 8 and (N - n_val) >= 8
        if use_val:
            vp = torch.randperm(N, generator=torch.Generator().manual_seed(1)).to(device)
            idx_tr, idx_va = vp[n_val:], vp[:n_val]
        else:
            idx_tr, idx_va = torch.arange(N, device=device), None
    n_tr = len(idx_tr)
    bs, epochs, lr = resolve_budget(n_tr, cfg, label)

    opt = torch.optim.AdamW(inner.parameters(), lr=lr, amsgrad=True,
                            betas=(0.95, 0.999),
                            weight_decay=float(cfg['weight_decay']))
    sch = str(cfg.get('dsm_scheduler', 'cosine')).lower()
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1),
                                                        eta_min=1e-12)
             if sch == 'cosine' else None)
    Dw = int(whitening.W.shape[0])
    val_noise = None
    if idx_va is not None:
        g = torch.Generator().manual_seed(seed + 777)
        val_noise = torch.randn(len(idx_va), Dw, generator=g).to(device)

    hist, best, best_state, best_ep = [], float('inf'), None, -1
    pbar = tqdm(range(1, epochs + 1), desc=label, dynamic_ncols=True, leave=False)
    for ep in pbar:
        inner.train()
        perm = idx_tr[torch.randperm(n_tr, device=device)]
        tot, nb = 0.0, 0
        for i in range(0, n_tr, bs):
            loss = loss_fn(perm[i:i + bs], None)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        tr_loss = tot / max(nb, 1)
        hist.append(tr_loss)
        if idx_va is not None:
            inner.eval()
            with torch.no_grad():
                mon = float(loss_fn(idx_va, val_noise))
            inner.train()
        else:
            mon = tr_loss
        if sched is not None:
            sched.step()
        if mon < best:
            best, best_ep = mon, ep
            best_state = copy.deepcopy(inner.state_dict())
        pbar.set_postfix(loss=f'{tr_loss:.3f}', val=f'{mon:.3f}', best=best_ep)
    if best_state is not None:
        inner.load_state_dict(best_state)
    print(f"    {label}: best epoch {best_ep}/{epochs} "
          f"({'val' if idx_va is not None else 'train'} {best:.4f})", flush=True)
    return hist, best_ep


# ---------------------------------------------------------------------------
# The spatial-protocol class
# ---------------------------------------------------------------------------

class CalibratedDART(DART):
    """DART with a configurable front-end, a data-driven sigma and a validated
    training loop. Only `fit` is overridden — `_build_net`, `psi`, `score` and
    `local_moment_normalize` are inherited from DART unchanged."""

    def fit(self, tr_raw, seed, device, ckpt=None, val_raw=None):
        cfg = self.cfg
        D = tr_raw.shape[1]
        seed_all(seed)                                   # BEFORE construction
        Xr = np.asarray(tr_raw, np.float32)

        if ckpt and os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location='cpu', weights_only=False)
            self.whitening = Whitening(np.asarray(blob['whiten_mu'], np.float32),
                                       np.asarray(blob['whiten_W'], np.float32)).to(device)
            self.sigma = float(blob['sigma'])
            self.net = self._build_net(D).to(device)     # DART's architecture
            self.net.load_state_dict(blob['net'])
            self.net.eval()
            self.resumed = True
            print(f'    [CalibratedDART] resumed {ckpt}', flush=True)
            return self
        self.resumed = False

        # The front-end is fit on TRAIN ONLY. sigma is too, except that when a
        # disjoint val region exists we also read its MEAN off it (delta, the
        # train->val displacement) so a rule-name sigma can widen to cover the
        # shift it will face at test time — see repro.core.sigma.transfer_gap.
        # Means only: no labels and no signature, and a numeric dsm_sigma_rho is
        # unaffected. Disable with dsm_sigma_transfer: false.
        self.whitening = make_frontend(Xr, cfg, seed=seed).to(device)
        Zw = self.whitening(torch.tensor(Xr, device=device)).detach().cpu().numpy()
        Zval = None
        if val_raw is not None and len(val_raw) >= 8:
            Zval = self.whitening(
                torch.tensor(np.asarray(val_raw, np.float32), device=device)
            ).detach().cpu().numpy()
        self.sigma = resolve_sigma(Zw, cfg, seed=seed, label='DART', D=D, Zval=Zval)
        self.net = self._build_net(D).to(device)         # <- inherited architecture

        n_fit = None
        if val_raw is not None and len(val_raw) >= 8:
            Xr = np.concatenate([Xr, np.asarray(val_raw, np.float32)], axis=0)
            n_fit = len(np.asarray(tr_raw))
            print(f'    [CalibratedDART] validating on a DISJOINT region '
                  f'({len(Xr) - n_fit} px), not an in-box split', flush=True)
        X = torch.tensor(Xr, device=device)
        Xw = self.whitening(X).detach()
        sigma = self.sigma

        def loss_fn(idx, fixed_noise):
            w = Xw[idx]
            eps = torch.randn_like(w) * self.sigma
            target = -eps / (self.sigma ** 2)
            loss = ((self.net(w + eps) - target) ** 2).sum(-1).mean()
            return loss

        run_dsm_training(self.net, self.whitening, X, cfg, seed,
                         f'CalDART s{seed}', sigma, device, loss_fn, n_fit=n_fit)
        self.net.eval()
        if ckpt:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({'net': {k: v.cpu() for k, v in self.net.state_dict().items()},
                        'whiten_mu': self.whitening.mu.detach().cpu().numpy(),
                        'whiten_W': self.whitening.W.detach().cpu().numpy(),
                        'sigma': self.sigma}, ckpt)
        return self


# ---------------------------------------------------------------------------
# The IID-protocol function (same recipe, on a ScoreNet)
# ---------------------------------------------------------------------------

def train_calibrated_scorenet(train_raw: np.ndarray, cfg: dict, seed: int,
                              label: str, s_raw: np.ndarray = None,
                              val_raw: np.ndarray = None
                              ) -> Tuple[ScoreNet, List[float]]:
    """The CalibratedDART recipe applied to a `ScoreNet` — what the IID protocol
    trains. Returns (net, per-epoch train loss), on CPU and in eval mode."""
    torch.manual_seed(seed)
    device = torch.device(cfg.get('device', 'cpu'))
    D = train_raw.shape[1]
    Xr = np.asarray(train_raw, np.float32)
    W = make_frontend(Xr, cfg, seed=seed).to(device)
    Zw = W(torch.tensor(Xr, device=device)).detach().cpu().numpy()

    def _trainer(fit_np, c, sd, lb):                     # only used by 'detval'
        return train_calibrated_scorenet(fit_np, c, sd, lb)[0]

    def _scorer(model, fit_np, planted, s):
        from repro.core.detectors import dsm_additive
        return dsm_additive(planted, fit_np, model, s)

    Zval = None
    if val_raw is not None and len(val_raw) >= 8:
        Zval = W(torch.tensor(np.asarray(val_raw, np.float32), device=device)
                 ).detach().cpu().numpy()
    sigma = resolve_sigma(Zw, cfg, seed=seed, label=label, D=D,
                          train_raw=Xr, s_raw=s_raw,
                          trainer=_trainer, scorer=_scorer, Zval=Zval)

    model = ScoreNet(D, list(cfg['hidden_dims']), cfg['activation'], whitening=W,
                     arch=str(cfg.get('dsm_arch', 'mlp')),
                     n_experts=int(cfg.get('dsm_n_experts', cfg.get('gmm_K', 16)))
                     ).to(device)
    X = torch.tensor(Xr).to(device)

    def loss_fn(idx, fixed_noise):
        return dsm_loss(model, X[idx], sigma, noise=fixed_noise)

    hist, _ = run_dsm_training(model, W, X, cfg, seed, f'DSM {label}',
                               sigma, device, loss_fn)
    model.cpu().eval()
    return model, hist
