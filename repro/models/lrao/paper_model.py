"""class LRaoPaper — the ORIGINAL LRao-detector model (Zschetzsche et al.),
wrapped to the same interface our experiments use for the native ``LRao``.

This module vendors the authors' own model from the paper's repository
(https://github.com/JonasZschetzsche/LRao-detector, ``CNN_LRao_functions.py``):
the 1-D convolutional transform ``Trafo`` (tanh, no bias) trained by maximizing
the trace of the linear Fisher information in the DFT domain

    cost = -tr(J),   J = G^H Sigma^-1 G   with   Sigma^-1 built from 1/PSD,

and the LRao detection statistic ``T(y) = g(y)^T J0^-1 g(y)`` with
``g(y) = dmu0^T Sigma^-1 (Psi(y) - mu0)``. The ``Trafo`` class below is kept
faithful to the upstream implementation; only the driver around it is new.

To make it a drop-in for our spatial pipeline it exposes the native LRao
contract:  ``fit(tr_raw, seed, device, run_dir) -> self`` (with ``self.fit_idx``
and resume-from-checkpoint), and ``score(test_pixels, ref_pixels, sig)``.

Adaptation notes (how the paper's *time-series* model is mapped onto our HSI
*pixel* task, kept as close to the original as possible):
  * each pixel (a D-band spectrum) is treated as one length-D sequence — the
    input the CNN convolves over, exactly as the paper feeds a length-N record;
  * a single GLOBAL robust scale (median / 1.4826*MAD, the paper's real-data
    normalization) is applied as preprocessing — one scalar, so the CNN's
    translation-equivariance across the sequence is preserved (unlike the
    native LRao's per-band diagonal normalization);
  * training is signal-agnostic (H = identity) because ``fit`` gets no
    signature — matching how the native ``LRao`` is trained; the target
    signature ``sig`` enters only at ``score`` time, as in the paper's
    ``calc_test_statistics``;
  * the stationarity / DFT-PSD covariance and mu0 = 0 assumptions are the
    paper's; they are not obviously valid for HSI spectra — this class exists to
    run the paper's model verbatim in our harness, not to claim it is the right
    prior for this data.

``L-LRao``-style linear variant: set ``cfg['n_conv_layers'] = 1`` (a single
conv, no tanh) — analogous to ``hidden = []`` for the native class.
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from repro.core.seeding import seed_all


# ===========================================================================
# Vendored model + helpers (faithful to the paper's CNN_LRao_functions.py)
# ===========================================================================
def est_psd_averaged_periodogram(x):
    """Nonparametric PSD estimate by averaging the periodogram over the batch."""
    return torch.mean(torch.abs(torch.fft.fft(x, dim=1, norm="ortho")) ** 2, dim=0)


def yule_walker(x, order):
    """Solve Yule-Walker equations for AR coefficients and noise variance."""
    r = torch.zeros((len(x), order + 1), dtype=torch.float64, device=x.device)
    r[:, 0] = torch.sum(x ** 2, dim=1)
    for k in range(1, order + 1):
        r[:, k] = torch.sum(x[:, 0:-k] * x[:, k:], dim=1)
    r = torch.mean(r, dim=0) / x.shape[1]
    R = pytorch_toeplitz(r[:-1].reshape(1, -1))
    a = torch.linalg.solve(R, r[1:])
    sigma2 = r[0] - (r[1:] * a).sum()
    return a, sigma2


def est_psd_ar_yule_walker(x, p):
    """Autoregressive parametric PSD estimate using the Yule-Walker method."""
    a, sigma2 = yule_walker(x, p)
    w = 2 * np.pi * torch.arange(x.shape[1], device=x.device).reshape(1, -1) / x.shape[1]
    psda = sigma2 / (torch.abs(1 - torch.sum(
        a.reshape(-1, 1) * torch.exp(
            -1j * torch.arange(1, len(a) + 1, device=x.device).reshape(-1, 1) * w),
        dim=0)) ** 2)
    return torch.tensor(psda.clone(), dtype=torch.complex64)


def pytorch_toeplitz(V):
    """Construct a Toeplitz matrix from a given vector."""
    d = V.shape[1]
    A = V.unsqueeze(1).unsqueeze(2)
    A_nofirst_flipped = torch.flip(A[:, :, :, 1:], dims=[3])
    A_concat = torch.concatenate([A_nofirst_flipped, A], dim=3)
    unfold = torch.nn.Unfold(kernel_size=(1, d))
    T = unfold(A_concat)
    T = torch.flip(T, dims=[2])
    return T.squeeze()


def init_weights(m):
    """Xavier-uniform init for Conv1d layers (gain tuned for tanh)."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(
            m.weight, gain=torch.nn.init.calculate_gain("tanh"))


def trafo_config(cfg, D):
    """Map an experiment cfg onto the paper's Trafo config, filling the paper's
    defaults for keys we don't already carry. ``da`` (the Jacobian finite-diff
    step) falls back to our detection step so training and scoring match."""
    return {
        "dim_inp": int(D),
        "da": float(cfg.get("da", cfg.get("lfi_delta_theta",
                                          cfg.get("delta_theta", 0.01)))),
        "filt_size": int(cfg.get("filt_size", 3)),
        "n_hidden_channels": int(cfg.get("n_hidden_channels", 20)),
        "n_conv_layers": int(cfg.get("n_conv_layers", 3)),
        "psd_method_params": cfg.get("psd_method_params", {"name": "periodogram"}),
        "psd_floor": float(cfg.get("psd_floor", 1e-12)),
    }


class Trafo(torch.nn.Module):
    """The paper's convolutional transform that maximizes the LFI at its output.

    Faithful port of ``CNN_LRao_functions.Trafo`` — the only edits are cosmetic
    (device comes from the tensors, not a module global) and the removal of the
    upstream method that referenced an undefined helper.
    """

    def __init__(self, config):
        super().__init__()
        self.dim_inp = config.get("dim_inp_train", config["dim_inp"])
        self.da = config["da"]
        bias = False
        layers = []
        n_layers = config["n_conv_layers"]
        if n_layers <= 1:
            # single-conv linear-ish transform (the L-LRao analogue)
            layers.append(torch.nn.Conv1d(1, 1, config["filt_size"],
                                          bias=bias, padding="same"))
        else:
            layers.append(torch.nn.Conv1d(1, config["n_hidden_channels"],
                                          config["filt_size"], bias=bias,
                                          padding="same"))
            layers.append(torch.nn.Tanh())
            for _ in range(n_layers - 2):
                layers.append(torch.nn.Conv1d(
                    config["n_hidden_channels"], config["n_hidden_channels"],
                    config["filt_size"], bias=bias, padding="same"))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Conv1d(config["n_hidden_channels"], 1,
                                          config["filt_size"], bias=bias,
                                          padding="same"))
        layers.append(torch.nn.Flatten())
        self.filt = torch.nn.Sequential(*layers)
        self.filt.apply(init_weights)
        self.psd_method_params = config["psd_method_params"]
        self.psd_floor = float(config.get("psd_floor", 1e-12))

    def forward(self, x):
        return self.filt(x[:, None, :])

    def est_psd(self, x):
        match self.psd_method_params["name"]:
            case "periodogram":
                psd = est_psd_averaged_periodogram(x)
            case "yule":
                psd = est_psd_ar_yule_walker(x, self.psd_method_params["order"])
        # floor to keep the 1/PSD terms finite for near-constant bands
        return torch.clamp(torch.real(psd), min=self.psd_floor).to(psd.dtype) \
            if torch.is_complex(psd) else torch.clamp(psd, min=self.psd_floor)

    def lfi_diag_dft(self, x, puffer_sides, H=None):
        """Estimate the diagonal LFI of Trafo(x) w.r.t. the additive signals in
        the columns of H, using the DFT method (upstream ``lfi_diag_dft``)."""
        if H is None:
            H = torch.eye(self.dim_inp, dtype=x.dtype, device=x.device)
        Pyy = self.est_psd(self.forward(x))
        lfi_diag = torch.zeros(H.shape[1], device=x.device)
        for i in range(H.shape[1]):
            ds = self.da * H[:, i].reshape(1, -1)
            if puffer_sides > 0:
                ds[0, :puffer_sides] = 0
                ds[0, -puffer_sides:] = 0
            x1 = x.clone() - ds
            x2 = x.clone() + ds
            dmu = torch.mean((self.forward(x2) - self.forward(x1)) / (2 * self.da),
                             dim=0)
            dMu = torch.fft.fft(dmu, norm="ortho")
            lfi_diag[i] = torch.sum(torch.abs(dMu) ** 2 / Pyy)
        return lfi_diag

    def set_statistics_dft(self, x, H=None):
        """Compute & store statistics (Pyy, inv_cov, dmu0, j0, j0_inv) of
        Trafo(x). Upstream ``set_statistics_dft`` — mu0 is fixed to 0."""
        if H is None:
            H = torch.eye(self.dim_inp, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            y = self.forward(x)
            self.Pyy = self.est_psd(y)
            self.mu0 = 0
            self.dmu0 = torch.zeros(y.shape[1], H.shape[1], device=x.device)
            for i in range(H.shape[1]):
                ds = self.da * H[:, i].reshape(1, -1)
                x1 = x.clone() - ds
                x2 = x.clone() + ds
                self.dmu0[:, i] = torch.mean(
                    (self.forward(x2) - self.forward(x1)) / (2 * self.da), dim=0)
        dft_mat = torch.fft.fft(torch.eye(y.shape[1], device=x.device),
                                norm="ortho")
        self.inv_cov = torch.real(
            torch.conj(dft_mat.T) @ (dft_mat / self.Pyy.reshape(-1, 1)))
        self.dMu0 = torch.fft.fft(self.dmu0, dim=0, norm="ortho")
        self.j0 = torch.real(
            torch.conj(self.dMu0).T @ (self.dMu0 / self.Pyy.reshape(-1, 1)))
        self.j0_inv = torch.inverse(self.j0)

    def g(self, x):
        """Pessimistic (linear) score:  dmu0^T Sigma^-1 (Psi(x) - mu0)."""
        return self.dmu0.T @ self.inv_cov @ (self.forward(x) - self.mu0).T

    def detect(self, x):
        """LRao statistic  T = g(x)^T J0^-1 g(x)  (vectorized form of the
        upstream per-sample ``detect``)."""
        with torch.no_grad():
            G = self.g(x).T                                  # (N, l)
            return ((G @ self.j0_inv) * G).sum(dim=1)        # (N,)


class TrafoScore(nn.Module):
    """The paper's 1-D CNN as a GENERAL-covariance score net (no DFT), wrapped
    as a plain score module ``psi(x) = Trafo(preproc(x))`` so it is a drop-in
    for the IID learned-Rao machinery (``lfi_loss_mode2`` /
    ``lfi_loss_mode2_signal_aware`` / ``compute_lfi_detector_scores_mode2``,
    which only need a callable (N,D)->(N,D) with ``.parameters()``).

    ``preproc`` is a frozen Whitening-compatible front-end (the paper's global
    MAD, or our ZCA / robust-IQR — the preprocessing ablation axis). The DFT/PSD
    methods on ``Trafo`` are intentionally unused here: the covariance is formed
    and inverted generally by the caller. ``.whiten`` / ``.net`` are exposed for
    parity with ``ScoreNet`` (e.g. if ever driven by ``dsm_loss``)."""

    def __init__(self, preproc, cfg, D):
        super().__init__()
        self.whitening = preproc
        self.core = Trafo(trafo_config(cfg, D))

    def whiten(self, x):
        return self.whitening(x)

    @property
    def net(self):
        return self.core

    def forward(self, x):
        return self.core(self.whitening(x))


# ===========================================================================
# Driver: our-experiment interface around the paper's model
# ===========================================================================
class LRaoPaper:
    """Paper's CNN+LRao detector, exposing the native ``LRao`` contract."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.med = None          # global robust location (scalar)
        self.scale = None        # global robust scale 1.4826*MAD (scalar)
        self.net = None
        self.fit_idx = None

    # ---- config mapping --------------------------------------------------
    def _trafo_cfg(self, D):
        return trafo_config(self.cfg, D)

    # ---- robust global normalization (paper's real-data preprocessing) ---
    @staticmethod
    def _robust_norm(pixels):
        X = np.asarray(pixels, np.float64)
        med = float(np.median(X))
        mad = float(np.median(np.abs(X - med)) * 1.4826)
        scale = float(np.sqrt(max(mad ** 2, 1e-22)))
        return np.float32(med), np.float32(scale)

    def _normalize(self, x_t):
        return (x_t - self.med) / self.scale

    # ---- training (validation early stopping, resume-safe) ---------------
    def fit(self, tr_raw, seed, device, run_dir):
        cfg = self.cfg
        os.makedirs(run_dir, exist_ok=True)
        tr = np.asarray(tr_raw, np.float32)
        D = tr.shape[1]

        idx = np.random.default_rng(seed).permutation(len(tr))
        nv = max(1, int(len(tr) * float(cfg.get('val_fraction', 0.2))))
        self.fit_idx, val_idx = idx[nv:], idx[:nv]

        med, scale = self._robust_norm(tr[self.fit_idx])
        self.med = torch.tensor(med, device=device)
        self.scale = torch.tensor(scale, device=device)

        seed_all(seed)                                   # BEFORE construction
        self.net = Trafo(self._trafo_cfg(D)).to(device)

        best_p = os.path.join(run_dir, 'best.pt')
        if os.path.exists(best_p):
            blob = torch.load(best_p, map_location='cpu', weights_only=False)
            self.net.load_state_dict(blob['net'])
            self.net.to(device).eval()
            self.fit_idx = np.asarray(blob['fit_idx'])
            self.med = torch.tensor(np.float32(blob['med']), device=device)
            self.scale = torch.tensor(np.float32(blob['scale']), device=device)
            print(f'    [LRaoPaper] resumed {best_p} (epoch {blob["epoch"]}, '
                  f'val {blob["val_loss"]:.4f})', flush=True)
            return self

        opt = torch.optim.Adam(self.net.parameters(),
                               lr=float(cfg.get('lr', 5e-4)),
                               weight_decay=float(cfg.get('weight_decay', 5e-5)))
        Xf = self._normalize(torch.tensor(tr[self.fit_idx], device=device))
        Xv = self._normalize(torch.tensor(tr[val_idx], device=device))
        # signal-agnostic training: maximize tr(LFI) over all input directions.
        # Each direction costs two forward passes per batch (the paper's finite
        # differences), so full training scans all D. 'train_directions' = k
        # samples k random identity columns per batch instead — an UNBIASED
        # estimate of tr(LFI)/D, k<D trades fidelity for speed. None -> all D.
        eyeD = torch.eye(D, dtype=Xf.dtype, device=device)
        n_dir = cfg.get('train_directions')
        n_dir = None if n_dir is None else int(n_dir)
        H_train_full = eyeD                       # validation always uses all D
        puffer = int(cfg.get('puffer_sides', 0))
        clip = float(cfg.get('grad_clip', 1.0))
        P, B = len(Xf), int(cfg.get('batch_size', 2048))
        max_epochs = int(cfg.get('max_epochs', 1000))
        patience = int(cfg.get('patience', 20))
        ckpt_every = int(cfg.get('ckpt_every', 0))

        tr_losses, val_losses = [], []
        best_val, best_state, best_epoch, bad = float('inf'), None, 0, 0
        stopped = max_epochs
        pbar = tqdm(range(1, max_epochs + 1), desc=f'LRaoPaper s{seed}',
                    dynamic_ncols=True, leave=False)
        for ep in pbar:
            self.net.train()
            perm = torch.randperm(P, device=device)
            run, nb = 0.0, 0
            for i in range(0, P, B):
                if n_dir is not None and n_dir < D:
                    cols = torch.randperm(D, device=device)[:n_dir]
                    H_train = eyeD[:, cols]
                else:
                    H_train = H_train_full
                lfi = self.net.lfi_diag_dft(Xf[perm[i:i + B]], puffer, H_train)
                loss = -lfi.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                opt.step()
                run += float(loss.detach()); nb += 1
            tl = run / max(nb, 1)
            self.net.eval()
            with torch.no_grad():
                vl = float(-self.net.lfi_diag_dft(Xv, 0, H_train_full).mean())
            tr_losses.append(tl); val_losses.append(vl)
            pbar.set_postfix(train=f'{tl:.4f}', val=f'{vl:.4f}', bad=bad)

            if vl < best_val:
                best_val, best_epoch, bad = vl, ep, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
            if ckpt_every and ep % ckpt_every == 0:
                torch.save({'net': {k: v.cpu() for k, v in
                                    self.net.state_dict().items()},
                            'epoch': ep, 'train_loss': tl, 'val_loss': vl},
                           os.path.join(run_dir, f'epoch_{ep:04d}.pt'))
            if bad >= patience:
                stopped = ep
                print(f'    [LRaoPaper] early stop at epoch {ep} (no new val '
                      f'minimum for {patience} epochs)', flush=True)
                break

        if best_state is None:                        # max_epochs <= 0 guard
            best_state = copy.deepcopy(self.net.state_dict())
            best_epoch = stopped = len(tr_losses)
        torch.save({'net': {k: v.cpu() for k, v in best_state.items()},
                    'epoch': best_epoch, 'val_loss': best_val,
                    'fit_idx': self.fit_idx, 'val_idx': val_idx,
                    'med': float(med), 'scale': float(scale)}, best_p)
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(dict(seed=seed, n_fit=int(len(self.fit_idx)),
                           n_val=int(len(val_idx)), train_loss=tr_losses,
                           val_loss=val_losses, best_epoch=best_epoch,
                           best_val_loss=best_val, stopped_epoch=stopped), f)
        self.net.load_state_dict(best_state)
        self.net.eval()
        print(f'    [LRaoPaper] best val epoch {best_epoch} (val {best_val:.4f}, '
              f'stopped {stopped}, fit {len(self.fit_idx)}/val {len(val_idx)})',
              flush=True)
        return self

    # ---- detection -------------------------------------------------------
    @torch.no_grad()
    def score(self, test_pixels, ref_pixels, sig):
        """LRao statistic for target signature ``sig``. ``ref_pixels`` MUST be
        the fit subset (``tr_raw[self.fit_idx]``). Returns a per-pixel array.

        cfg['statistic']: 'lrao' (default, two-sided quadratic, the paper's
        ``detect``) or 'llmp' (one-sided normalized, matching the native
        ``LRao.score`` sign convention)."""
        cfg = self.cfg
        dev = next(self.net.parameters()).device
        Xref = self._normalize(torch.tensor(np.asarray(ref_pixels, np.float32),
                                            device=dev))
        Xte = self._normalize(torch.tensor(np.asarray(test_pixels, np.float32),
                                           device=dev))
        # signature expressed in normalized units (global scalar scale)
        sig_t = torch.tensor(np.asarray(sig, np.float32), device=dev) / self.scale
        H = sig_t.reshape(-1, 1)                          # single column (l=1)

        self.net.set_statistics_dft(Xref, H)
        if not torch.isfinite(self.net.j0).all() or float(self.net.j0) <= 0:
            return np.zeros(len(test_pixels), dtype=np.float32)

        if str(cfg.get('statistic', 'lrao')) == 'llmp':
            # one-sided: g(y)/sqrt(J0)  (l == 1)
            g = self.net.g(Xte).squeeze(0)               # (N,)
            T = g / torch.sqrt(torch.clamp(self.net.j0.reshape(()), min=1e-12))
        else:
            T = self.net.detect(Xte)                     # quadratic LRao
        return T.detach().cpu().numpy().astype(np.float32)
