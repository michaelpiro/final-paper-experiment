"""colab_deep/osvae_model.py — OS-VAE port (Tian et al., IEEE TGRS 2024).

Lifted verbatim from SDSM/results/osvae_strong/run_weak_sweep.py — the port
validated against the authors' release at AUC 0.9956 vs published 0.9993 on
ABU-Airport-2 (2026-07-01) — and split into fit/score so a single training
serves a whole amplitude sweep. All math is unchanged, including the port's
faithful quirks (training input 2*standard(x), test input 2*standard(x)-1).

Detector = SAM(pixel, reconstruction) x (1 - exp(-0.1 * CEM(pixel))); the
fused variant is the paper's headline detector. The CEM coarse term is
test-adaptive (autocorrelation of the scored set), recomputed per call.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


def standard(X):
    return (X - X.min()) / (X.max() - X.min() + 1e-12)


def sam_ang(Xln, pv):
    nx = np.linalg.norm(Xln, axis=1) + 1e-12
    npv = np.linalg.norm(pv) + 1e-12
    return np.arccos(np.clip((Xln @ pv) / (nx * npv), -1, 1))


class CVAE(nn.Module):
    def __init__(self, l, z=30):
        super().__init__(); a = nn.LeakyReLU()
        self.enc = nn.Sequential(nn.Linear(l, 500), a, nn.Linear(500, 400), a,
                                 nn.Linear(400, 300), a, nn.Linear(300, 200), a,
                                 nn.Linear(200, 100), a, nn.Linear(100, 2 * z))
        self.dec = nn.Sequential(nn.Linear(z, 100), a, nn.Linear(100, 200), a,
                                 nn.Linear(200, 300), a, nn.Linear(300, 400), a,
                                 nn.Linear(400, 500), a, nn.Linear(500, l),
                                 nn.Tanh())
        self.z = z

    def forward(self, x):
        h = self.enc(x); mn, lv = h[:, :self.z], h[:, self.z:]
        return self.dec(mn + torch.exp(0.5 * lv) * torch.randn_like(mn)), mn, lv


def dictionary(train, d, m=25, n=5, Kt=20):
    hs = standard(train); p = standard(d); L = train.shape[1]
    keep = np.ones(len(hs), bool)
    keep[np.argsort(sam_ang(hs, p))[:Kt]] = False
    Xb = hs[keep]; mm = min(m, max(2, len(Xb) // (L + 1)))
    lab = KMeans(mm, n_init=5, random_state=0).fit(Xb).labels_; atoms = []
    for i in range(mm):
        ci = np.where(lab == i)[0]
        if len(ci) < L:
            continue
        Xi = Xb[ci]; rXi = Xi - Xi.mean(0)
        incov = np.linalg.pinv((rXi.T @ rXi) / (len(ci) - 1))
        md = np.einsum("ij,jk,ik->i", rXi, incov, rXi)
        atoms.append(Xi[np.argsort(md)[:n]])
    return (np.concatenate(atoms, 0).astype(np.float32) if atoms
            else standard(train)[:m].astype(np.float32))


def fit_osvae(train, d, seed=0, epochs=200, device='cpu'):
    """Train the CVAE on background pixels with the OSP-regularized loss."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    L = train.shape[1]
    E = dictionary(train, d)
    Et = torch.tensor(E, device=dev)
    dt = torch.tensor((2 * standard(d)).astype(np.float32), device=dev)
    P_U = torch.eye(L, device=dev) - Et.t() @ torch.linalg.pinv(Et.t())
    tmp = dt @ (P_U @ dt)

    def osp_loss(xr):
        sim = 1.0 / (torch.norm(xr - dt.unsqueeze(0), dim=1) + 1e-9)
        return ((xr @ (P_U @ dt)) / tmp * sim).mean() * 10000.0

    Xtr = torch.tensor((2 * standard(train)).astype(np.float32), device=dev)
    model = CVAE(L).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    BS = min(1000, len(Xtr))
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for b in range(max(1, len(Xtr) // BS)):
            xb = Xtr[perm[b * BS:(b + 1) * BS]]
            if len(xb) < 2:
                continue
            xr, mn, lv = model(xb)
            mse = ((xr - xb) ** 2).mean(-1)
            kl = -0.5 * torch.sum(1 + lv - mn ** 2 - torch.exp(lv), -1)
            loss = (kl + mse + 1e-5 * osp_loss(xr)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model._device = device
    return model


def score_osvae(model, test, d):
    """Fused OS-VAE statistic (higher = more target)."""
    dev = torch.device(getattr(model, '_device', 'cpu'))
    with torch.no_grad():
        rec, _, _ = model(torch.tensor((2 * standard(test) - 1)
                                       .astype(np.float32), device=dev))
    rec = rec.cpu().numpy()
    tn = standard(test); rn = standard(rec)
    sam = np.arccos(np.clip(
        (tn * rn).sum(1) / ((np.linalg.norm(tn, axis=1) + 1e-12)
                            * (np.linalg.norm(rn, axis=1) + 1e-12)), -1, 1))
    sam_n = standard(sam)
    R = (test.T @ test) / len(test)
    Rid = np.linalg.solve(R, d)
    w = Rid / (d @ Rid)
    coarse = np.abs(test @ w)                      # CEM coarse detection
    return sam_n * (1 - np.exp(-0.1 * coarse))     # fused (paper variant)
