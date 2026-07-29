"""class DARTS — everything the DARTS detector is, in one file.

The model (paper): a spatially-adapted score estimator. A shared MLP encoder
phi embeds the query pixel and its window neighbours into a latent space; the
K latent-nearest neighbours condition an MLP denoiser f; the score follows
from Tweedie's identity psi(y) = (x_hat - y)/sigma^2. Trained by DSM (noise
only the query pixel), with a frozen ZCA whitening first layer.
Detection: the same additive LMP statistic as DART.
DARTS-CFAR: local moment normalization of the score MAP over the pixel's
ENTIRE window minus a guard block (no top-K in the normalization):
    T_i = (q_i - mu_local) / sqrt( (1-lam)*var_local + lam*var_global )
fit() seeds torch/numpy BEFORE building any weights. Config: `darts:` section.
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from repro.core.data import Whitening
from repro.core.seeding import seed_all


class _NeighborDenoiser(nn.Module):
    """The torch module (architecture exactly as published)."""

    def __init__(self, D, d_lat, K, enc_hidden, score_hidden, sigma,
                 activation, whitening):
        super().__init__()
        self.D, self.d_lat, self.K, self.sigma = D, d_lat, K, sigma
        self.whitening = whitening
        act = {'silu': nn.SiLU, 'relu': nn.ReLU}[activation]

        def mlp(in_dim, hidden, out_dim):
            dims = [in_dim] + list(hidden)
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), act()]
            layers.append(nn.Linear(dims[-1], out_dim))
            return nn.Sequential(*layers)

        self.phi = mlp(D, enc_hidden, d_lat)
        self.f = mlp(D + d_lat * (1 + K), score_hidden, D)

    def _forward_inner(self, y, neighbors):
        B, M, D = neighbors.shape
        z_i = self.phi(y)
        z_j = self.phi(neighbors.reshape(B * M, D)).reshape(B, M, self.d_lat)
        with torch.no_grad():                    # top-K latent-nearest (paper)
            dists = ((z_j - z_i.unsqueeze(1)) ** 2).sum(-1)
            K_eff = min(self.K, M)
            topk = dists.topk(K_eff, dim=1, largest=False).indices
        z_topk = z_j.gather(1, topk.unsqueeze(-1).expand(-1, -1, self.d_lat))
        if K_eff < self.K:
            z_topk = torch.cat([z_topk, z_topk.new_zeros(
                B, self.K - K_eff, self.d_lat)], dim=1)
        u = torch.cat([y, z_i, z_topk.reshape(B, self.K * self.d_lat)], dim=-1)
        return (self.f(u) - y) / (self.sigma ** 2)          # Tweedie score

    def forward(self, y, neighbors):
        """Whitened-space score mapped back to DATA space."""
        s_w = self._forward_inner(self.whitening(y), self.whitening(neighbors))
        return s_w @ self.whitening.W


class DARTS:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.net = None

    def fit(self, tr_raw, tr_nbr, seed, device, ckpt=None, reseed=True):
        """reseed=False replicates the PUBLISHED RNG protocol: the paper's
        pipeline seeded once per run and trained DART first, so DARTS's init
        drew from the post-DART stream. The spatial protocol passes
        reseed=False (with DART trained in-session) to reproduce the published
        models exactly; standalone use keeps the fresh per-model seed."""
        cfg = self.cfg
        D = tr_raw.shape[1]
        if reseed:
            seed_all(seed)                               # BEFORE construction
        Wh = Whitening.from_data(np.asarray(tr_raw, np.float32),
                                 eig_floor=float(cfg['whiten_eig_floor'])
                                 ).to(device)
        sigma = float(np.sqrt(cfg['dsm_sigma_rho']))
        self.net = _NeighborDenoiser(
            D, int(cfg['d_lat']), int(cfg['K']), list(cfg['enc_hidden']),
            list(cfg['score_hidden']), sigma, cfg['activation'], Wh).to(device)
        if ckpt and os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location='cpu', weights_only=False)
            self.net.load_state_dict(blob['net'])
            self.net.to(device).eval()
            print(f'    [DARTS] resumed {ckpt}', flush=True)
            return self
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(cfg['lr']),
                                weight_decay=float(cfg['weight_decay']))
        X = torch.tensor(np.asarray(tr_raw, np.float32), device=device)
        N = torch.tensor(np.asarray(tr_nbr, np.float32), device=device)
        P, B, E = len(X), int(cfg['batch_size']), int(cfg['epochs'])
        best_loss, best_state = float('inf'), None
        pbar = tqdm(range(E), desc=f'DARTS s{seed}', dynamic_ncols=True,
                    leave=False)
        for ep in pbar:
            perm = torch.randperm(P, device=device)
            tot, nb = 0.0, 0
            for i in range(0, P, B):
                sel = perm[i:i + B]
                x_w = self.net.whitening(X[sel])
                nbr_w = self.net.whitening(N[sel])
                eps = torch.randn_like(x_w) * sigma
                target = -eps / (sigma ** 2)
                score = self.net._forward_inner(x_w + eps, nbr_w)
                loss = ((score - target) ** 2).sum(-1).mean()
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
            torch.save({'net': {k: v.cpu() for k, v in
                                self.net.state_dict().items()}}, ckpt)
        return self

    @torch.no_grad()
    def _scores(self, pix, nbr, batch_size=512):
        dev = next(self.net.parameters()).device
        out = []
        for i in range(0, len(pix), batch_size):
            p = torch.tensor(np.asarray(pix[i:i + batch_size], np.float32),
                             device=dev)
            n = torch.tensor(np.asarray(nbr[i:i + batch_size], np.float32),
                             device=dev)
            out.append(self.net(p, n).cpu().numpy())
        return np.concatenate(out, axis=0)

    def score(self, test_pix, test_nbr, tr_pix, tr_nbr, sig):
        """Additive LMP (same normalization convention as DART)."""
        z_train = self._scores(tr_pix, tr_nbr)
        z_test = self._scores(test_pix, test_nbr)
        z_bar = z_train.mean(axis=0)
        C_psi = np.cov(z_train, rowvar=False)
        if C_psi.ndim == 0:
            C_psi = np.array([[float(C_psi)]])
        norm = float(np.sqrt(max(float(sig @ C_psi @ sig), 1e-12)))
        return -((z_test - z_bar) @ sig) / norm

    @staticmethod
    def local_moment_normalize(score_flat, shape, win, guard=1, cfar_lam=0.0,
                               eps=1e-6):
        """DARTS-CFAR: whole-window local moment normalization (map-only)."""
        H, W = shape
        q_i = torch.tensor(np.asarray(score_flat, np.float32))
        wsize = int(win)
        p = wsize // 2
        qmap = q_i.reshape(1, 1, H, W)
        patches = F.unfold(F.pad(qmap, (p, p, p, p), mode='circular'),
                           kernel_size=wsize)
        patches = patches.reshape(wsize * wsize, H * W).t()
        cc = wsize // 2
        gr = max(int(guard), 1) // 2
        keep = [r * wsize + c for r in range(wsize) for c in range(wsize)
                if not (abs(r - cc) <= gr and abs(c - cc) <= gr)]
        q_set = patches[:, keep]
        mu_local = q_set.mean(dim=1)
        std_local = q_set.var(dim=1, unbiased=False).sqrt()
        lam = float(cfar_lam)
        var_global = (q_i.var() + eps)
        var_eff = (1.0 - lam) * (std_local ** 2) + lam * var_global
        std_eff = torch.sqrt(torch.clamp(var_eff, min=0.0)) + eps
        return ((q_i - mu_local) / std_eff).cpu().numpy().astype(np.float32)
