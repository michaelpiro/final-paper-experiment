"""class DART — everything the DART detector is, in one file.

The model (paper): an MLP score network psi(x) with a frozen ZCA whitening
first layer, trained by denoising score matching (DSM) on background pixels:
    L = E || psi(w + sigma*eps) + eps/sigma^2 ||^2      (noise in whitened space)
Detection (additive LMP statistic):
    T(y) = -( (psi(y) - psi_bar)^T s ) / sqrt( s^T C_psi s )
DART-CFAR: the same score MAP, locally normalized (local moment normalization):
    T_i = (q_i - mu_local) / sqrt( (1-lam)*var_local + lam*var_global )

L-DART is this class with cfg['hidden'] = [] (a linear/affine score).
All hyperparameters come from the config dict (configs/spatial_clean.yaml
`dart:`). fit() seeds torch/numpy BEFORE building any weights.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from repro.core.data import Whitening
from repro.core.seeding import seed_all


class DART:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.whitening = None
        self.net = None          # nn.Sequential MLP, operates in whitened space
        self.sigma = None

    # ---- architecture ----------------------------------------------------
    def _build_net(self, D):
        act = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh}[
            self.cfg['activation']]
        dims = [D] + list(self.cfg['hidden']) + [D]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        return nn.Sequential(*layers)

    def psi(self, x):
        """The score in DATA space: psi(x) = W^T net(whiten(x))."""
        return self.net(self.whitening(x)) @ self.whitening.W

    # ---- training --------------------------------------------------------
    def fit(self, tr_raw, seed, device, ckpt=None):
        cfg = self.cfg
        D = tr_raw.shape[1]
        seed_all(seed)                                   # BEFORE construction
        self.whitening = Whitening.from_data(
            np.asarray(tr_raw, np.float32),
            eig_floor=float(cfg['whiten_eig_floor'])).to(device)
        self.sigma = float(np.sqrt(cfg['dsm_sigma_rho']))  # whitened cov ~ I
        self.net = self._build_net(D).to(device)
        if ckpt and os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location='cpu', weights_only=False)
            self.net.load_state_dict(blob['net'])
            self.net.to(device).eval()
            print(f'    [DART] resumed {ckpt}', flush=True)
            return self
        opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg['lr']),
                               weight_decay=float(cfg['weight_decay']))
        X = torch.tensor(np.asarray(tr_raw, np.float32), device=device)
        Xw = self.whitening(X).detach()                  # train in whitened space
        P, B, E = len(Xw), int(cfg['batch_size']), int(cfg['epochs'])
        best_loss, best_state = float('inf'), None
        pbar = tqdm(range(E), desc=f'DART s{seed}', dynamic_ncols=True, leave=False)
        for ep in pbar:
            perm = torch.randperm(P, device=device)
            tot, nb = 0.0, 0
            for i in range(0, P, B):
                w = Xw[perm[i:i + B]]
                eps = torch.randn_like(w) * self.sigma
                target = -eps / (self.sigma ** 2)
                loss = ((self.net(w + eps) - target) ** 2).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item()); nb += 1
            last = tot / max(nb, 1)
            pbar.set_postfix(loss=f'{last:.4f}')
            if last < best_loss:
                best_loss, best_state = last, copy.deepcopy(self.net.state_dict())
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        if ckpt:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({'net': {k: v.cpu() for k, v in self.net.state_dict().items()},
                        'sigma': self.sigma}, ckpt)
        return self

    # ---- detection -------------------------------------------------------
    @torch.no_grad()
    def _scores(self, pixels):
        dev = next(self.net.parameters()).device
        X = torch.tensor(np.asarray(pixels, np.float32), device=dev)
        return self.psi(X).cpu().numpy()

    def score(self, test_pixels, tr_raw, sig):
        """Additive LMP: T(y) = -((psi(y)-psi_bar) @ s) / sqrt(s @ C_psi @ s)."""
        z_train = self._scores(tr_raw)
        z_bar = z_train.mean(axis=0)
        C_psi = np.cov(z_train, rowvar=False)
        if C_psi.ndim == 0:
            C_psi = np.array([[float(C_psi)]])
        z_test = self._scores(test_pixels)
        norm = np.sqrt(max(float(sig @ C_psi @ sig), 1e-12))
        return -((z_test - z_bar) @ sig) / norm

    @staticmethod
    def local_moment_normalize(flat_scores, shape, bg, guard, cfar_lam=0.0,
                               eps=1e-6):
        """DART-CFAR: local moment normalization of the score map (paper eq.):
        T_i = (q_i - mu_local) / sqrt((1-lam)*var_local + lam*var_global),
        window stats over bg x bg minus a guard x guard block (box filters)."""
        H, W = shape
        q = np.asarray(flat_scores, dtype=np.float64).reshape(H, W)
        nb, ng = bg * bg, (guard * guard if guard and guard > 0 else 0)
        m_bg = uniform_filter(q, size=bg, mode='wrap')
        s2_bg = uniform_filter(q * q, size=bg, mode='wrap')
        m_g = uniform_filter(q, size=guard, mode='wrap') if ng > 0 else 0.0
        s2_g = uniform_filter(q * q, size=guard, mode='wrap') if ng > 0 else 0.0
        denom = max(nb - ng, 1)
        mean_local = (nb * m_bg - ng * m_g) / denom
        e2_local = (nb * s2_bg - ng * s2_g) / denom
        std_local = np.sqrt(np.maximum(e2_local - mean_local ** 2, 0.0))
        var_global = float(q.var()) + eps
        lam = float(cfar_lam)
        var_eff = (1.0 - lam) * (std_local ** 2) + lam * var_global
        std_eff = np.sqrt(np.maximum(var_eff, 0.0)) + eps
        return ((q - mean_local) / std_eff).reshape(-1).astype(np.float32)
