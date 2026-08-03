"""class LRao — everything the learned-Rao detector is, in one file.

The model (LRao paper, our published configuration): an MLP score net psi with
a frozen ROBUST normalization first layer — per-band median and 1.4826*MAD, so
the normalization is DIAGONAL (no decorrelation, unlike DART's ZCA) and is kept
as two vectors rather than a matrix — trained signal-agnostically by maximizing
the linear Fisher information:
    cost = -tr(J*) = -tr( G^T Sigma^-1 G ),  G = E[d psi/d x],
    Sigma = cov(psi) (full SVD pseudo-inverse — no eigenvalue truncation,
    in training and at detection alike)
with the LRao paper's prescribed usage: a val_fraction held-out split, early
stopping on the VALIDATION cost (patience epochs without a new minimum), the
best-validation model kept. The model, its normalization, and its scoring
reference use only the fit subset (self.fit_idx).
Detection (mode-2 learned-Rao statistic; the signature enters only here):
    g_s = G s,  J_s = g_s^T Sigma^-1 g_s,
    T(y) = g_s^T Sigma^-1 (psi(y) - mu) / sqrt(J_s)
L-LRao is this class with cfg['hidden'] = []. fit() seeds BEFORE construction;
a checkpoint is written every ckpt_every epochs. Config: `lrao:` section.
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from repro.core.models import _robust_svd_np
from repro.core.seeding import seed_all


class LRao:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.med = None          # per-band median      (D,)
        self.inv_scale = None    # 1 / (1.4826 * MAD)   (D,)
        self.net = None
        self.fit_idx = None

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

    @staticmethod
    def _robust_norm(pixels):
        """Per-band robust location/scale: median and 1/(1.4826*MAD)."""
        X = np.asarray(pixels, np.float64)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826
        scale = np.sqrt(np.maximum(mad ** 2, 1e-22))
        return med.astype(np.float32), (1.0 / scale).astype(np.float32)

    def normalize(self, x):
        return (x - self.med) * self.inv_scale

    def psi(self, x):
        """Score in DATA space. The normalization is diagonal, so the chain
        rule back to data space is the same per-band scaling."""
        return self.net(self.normalize(x)) * self.inv_scale

    def _lfi_cost(self, batch):
        """-tr(G^T Sigma^-1 G) on a RAW batch (Jacobian through the frozen
        normalization; Sigma under no-grad when detach_sigma)."""
        cfg = self.cfg
        with torch.no_grad() if bool(cfg['detach_sigma']) else torch.enable_grad():
            psi0 = self.psi(batch)
            mu = psi0.mean(dim=0)
            c = psi0 - mu
            Sigma = (c.T @ c) / max(len(batch) - 1, 1)
            U, S, Vh = torch.linalg.svd(Sigma)
            S_inv = torch.where(S > 0, 1.0 / S, torch.zeros_like(S))
            Sigma_inv = Vh.T @ torch.diag(S_inv) @ U.T
        from torch.func import jacrev, vmap
        J_all = vmap(jacrev(lambda x: self.psi(x.unsqueeze(0)).squeeze(0)))(batch)
        G = J_all.mean(dim=0)
        return -(G.T @ Sigma_inv @ G).trace()

    # ---- training (validation early stopping) ----------------------------
    def fit(self, tr_raw, seed, device, run_dir):
        cfg = self.cfg
        os.makedirs(run_dir, exist_ok=True)
        tr = np.asarray(tr_raw, np.float32)
        idx = np.random.default_rng(seed).permutation(len(tr))
        nv = max(1, int(len(tr) * float(cfg['val_fraction'])))
        self.fit_idx, val_idx = idx[nv:], idx[:nv]
        seed_all(seed)                                   # BEFORE construction
        med, inv_scale = self._robust_norm(tr[self.fit_idx])
        self.med = torch.tensor(med, device=device)
        self.inv_scale = torch.tensor(inv_scale, device=device)
        self.net = self._build_net(tr.shape[1]).to(device)
        best_p = os.path.join(run_dir, 'best.pt')
        if os.path.exists(best_p):
            blob = torch.load(best_p, map_location='cpu', weights_only=False)
            self.net.load_state_dict(blob['net'])
            self.net.to(device).eval()
            self.fit_idx = np.asarray(blob['fit_idx'])
            print(f'    [LRao] resumed {best_p} (epoch {blob["epoch"]}, '
                  f'val {blob["val_loss"]:.4f})', flush=True)
            return self
        opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg['lr']),
                               weight_decay=float(cfg['weight_decay']))
        Xf = torch.tensor(tr[self.fit_idx], device=device)
        Xv = torch.tensor(tr[val_idx], device=device)
        P, B = len(Xf), int(cfg['batch_size'])
        tr_losses, val_losses = [], []
        best_val, best_state, best_epoch, bad = float('inf'), None, 0, 0
        stopped = int(cfg['max_epochs'])
        pbar = tqdm(range(1, int(cfg['max_epochs']) + 1), desc=f'LRao s{seed}',
                    dynamic_ncols=True, leave=False)
        for ep in pbar:
            perm = torch.randperm(P, device=device)
            run, nb = 0.0, 0
            for i in range(0, P, B):
                loss = self._lfi_cost(Xf[perm[i:i + B]])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               float(cfg['grad_clip']))
                opt.step()
                run += float(loss.detach()); nb += 1
            tl = run / max(nb, 1)
            vl = float(self._lfi_cost(Xv).detach())
            tr_losses.append(tl); val_losses.append(vl)
            pbar.set_postfix(train=f'{tl:.4f}', val=f'{vl:.4f}', bad=bad)
            if vl < best_val:
                best_val, best_epoch, bad = vl, ep, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
            if int(cfg.get('ckpt_every', 0)) and ep % int(cfg['ckpt_every']) == 0:
                torch.save({'net': {k: v.cpu() for k, v in
                                    self.net.state_dict().items()},
                            'epoch': ep, 'train_loss': tl, 'val_loss': vl},
                           os.path.join(run_dir, f'epoch_{ep:04d}.pt'))
            if bad >= int(cfg['patience']):
                stopped = ep
                print(f'    [LRao] early stop at epoch {ep} (no new val '
                      f'minimum for {cfg["patience"]} epochs)', flush=True)
                break
        torch.save({'net': {k: v.cpu() for k, v in best_state.items()},
                    'epoch': best_epoch, 'train_loss': tr_losses[best_epoch - 1],
                    'val_loss': best_val, 'fit_idx': self.fit_idx,
                    'val_idx': val_idx}, best_p)
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump(dict(seed=seed, n_fit=int(len(self.fit_idx)),
                           n_val=int(len(val_idx)), train_loss=tr_losses,
                           val_loss=val_losses, best_epoch=best_epoch,
                           best_val_loss=best_val, stopped_epoch=stopped), f)
        self.net.load_state_dict(best_state)
        self.net.eval()
        print(f'    [LRao] best val epoch {best_epoch} (val {best_val:.4f}, '
              f'stopped {stopped}, fit {len(self.fit_idx)}/val {len(val_idx)})',
              flush=True)
        return self

    # ---- detection (mode-2 statistic) ------------------------------------
    @torch.no_grad()
    def score(self, test_pixels, ref_pixels, sig):
        """ref_pixels MUST be the fit subset: tr_raw[self.fit_idx]."""
        cfg = self.cfg
        dev = next(self.net.parameters()).device
        d = ref_pixels.shape[1]
        X_tr = torch.tensor(np.asarray(ref_pixels, np.float32), device=dev)
        X_te = torch.tensor(np.asarray(test_pixels, np.float32), device=dev)
        I_d = torch.eye(d, device=dev)
        psi_tr = self.psi(X_tr).cpu().numpy()
        if not np.all(np.isfinite(psi_tr)):
            return np.zeros(len(test_pixels), dtype=np.float32)
        mu = psi_tr.mean(axis=0)
        Sigma = (psi_tr - mu).T @ (psi_tr - mu) / max(len(ref_pixels) - 1, 1)
        U, S, Vh = _robust_svd_np(Sigma)
        S_inv = np.where(S > 0, 1.0 / S, 0.0)
        Sigma_inv = Vh.T @ np.diag(S_inv) @ U.T
        dth = float(cfg['delta_theta'])
        G = np.zeros((psi_tr.shape[1], d))
        for j in range(d):
            plus = self.psi(X_tr + dth * I_d[j]).cpu().numpy()
            minus = self.psi(X_tr - dth * I_d[j]).cpu().numpy()
            G[:, j] = ((plus - minus) / (2.0 * dth)).mean(axis=0)
        g_s = G @ sig
        J_s = float(g_s @ Sigma_inv @ g_s)
        psi_te = self.psi(X_te).cpu().numpy()
        return (psi_te - mu) @ (Sigma_inv @ g_s) / np.sqrt(max(J_s, 1e-12))
