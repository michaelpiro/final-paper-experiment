"""Front probe (user, 08-14): band-drop + mild-PCA fronts vs std control.
Motivation: isotropic-raw noise wastes capacity on tiny-variance directions;
both fronts REMOVE those directions instead of rescaling them (PCA is an
orthonormal rotation, so isotropic noise in the reduced coords is still
isotropic in the raw subspace — convention-consistent).

Fronts:
  std               control (all bands, per-band std)
  drop99 / drop98   drop lowest-variance bands holding the bottom 1% / 2%
                    of total variance, then std on the kept bands
  pca99.99 / pca99.999 / pca99.9999
                    project onto top PCs keeping that % of variance,
                    then per-component std (NOTE: = truncated PCA-whitening)
  pcaN99.99 / pcaN99.999 / pcaN99.9999
                    same projection, NO per-component rescale (mean-center
                    + one global scalar) = rigid rotation + truncation;
                    preserves the variance hierarchy / mixture geometry

Noise conventions: 'before' (isotropic in the reduced raw space) and
'after' (isotropic in the FRONT space = diagonal-relative in reduced space:
sigma_i^2 = rho * Var_i, target eps_i/sigma_i^2 — general diagonal DSM).
For pcaN (scalar front) the two coincide -> after arms skipped.

Settings: multi / single (n=2048, IID pools) + pavia4 (scene pool,
bitumen theta=.15). 1 seed (42), 5000 epochs, rhos {0.03, 0.1}, eval/100.
DART [128] relu, Adam 5e-4 wd=0 clip 1.0 batch 512, noise-before in the
REDUCED space (sigma^2 = rho * mean reduced variance).
Run from pythonProject cwd:  .venv/bin/python <this file>"""
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ov", os.path.join(HERE, 'run_overnight.py'))
OV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OV)

from repro.protocols.iid import _pd_at_fa, _auc
from repro.core.data import Whitening
from repro.core.models import ScoreNet

OUT_JSON = os.path.join(HERE, 'results_fronts_probe.json')
SETTINGS = ['multi', 'single', 'pavia4']
FRONTS = ['std', 'drop99', 'drop98', 'pca99.99', 'pca99.999', 'pca99.9999',
          'pcaN99.99', 'pcaN99.999', 'pcaN99.9999']
RHOS = [0.03, 0.1]
SEED = 42
N_IID = 2048
EPOCHS = 5000
EVAL_EVERY = 100
THETA_P4 = 0.15

torch.set_num_threads(5)


def pools(setting):
    if setting == 'pavia4':
        sc, planted, labels = OV.scene_pools('pavia4', SEED, THETA_P4)
        return (np.asarray(sc['tr'], np.float32),
                np.asarray(planted, np.float32), np.asarray(labels),
                np.asarray(sc['sig'], np.float32))
    tr, planted, y, s = OV.iid_pools(setting, SEED, N_IID)
    return (np.asarray(tr, np.float32), np.asarray(planted, np.float32),
            np.asarray(y), np.asarray(s, np.float32))


def reduce_front(front, tr, planted, s):
    """Returns (tr', planted', s', kept_dim_info) in the reduced space."""
    X = np.asarray(tr, np.float64)
    D = X.shape[1]
    if front == 'std':
        return tr, planted, s, f'{D} bands (all)'
    if front.startswith('drop'):
        keep_frac = float(front[4:]) / 100.0          # .99 or .98
        v = X.var(0)
        order = np.argsort(v)                          # ascending
        cum = np.cumsum(v[order])
        drop_mask = cum <= (1.0 - keep_frac) * v.sum()
        drop = set(order[drop_mask].tolist())
        keep = np.array([i for i in range(D) if i not in drop])
        return (tr[:, keep], planted[:, keep], s[keep],
                f'{len(keep)}/{D} bands (dropped {D - len(keep)})')
    noscale = front.startswith('pcaN')
    frac = float(front[4 if noscale else 3:]) / 100.0  # pca / pcaN fronts
    mu = X.mean(0)
    lam, V = np.linalg.eigh(np.cov(X, rowvar=False))
    lam, V = lam[::-1], V[:, ::-1]                     # descending
    m = int(np.searchsorted(np.cumsum(lam) / lam.sum(), frac) + 1)
    P = V[:, :m].astype(np.float32)
    tag = 'noscale ' if noscale else ''
    return ((tr - mu.astype(np.float32)) @ P,
            (planted - mu.astype(np.float32)) @ P,
            s @ P, f'{m}/{D} PCs ({tag}{frac * 100:.4f}%)')


def run_one(setting, front, rho, mode='before'):
    key = f'{setting}_{front}_r{rho}' + ('_after' if mode == 'after'
                                         else '')
    t0 = time.time()
    tr0, planted0, y, s0 = pools(setting)
    tr, planted, s, info = reduce_front(front, tr0, planted0, s0)
    d = tr.shape[1]
    print(f'[{key}] front keeps: {info}', flush=True)
    X64 = np.asarray(tr, np.float64)
    if front.startswith('pcaN'):
        # rotation-only: mean-center + ONE global scalar (geometry intact)
        c = float(np.sqrt(X64.var(0).mean()))
        W = Whitening(X64.mean(0).astype(np.float32),
                      (np.eye(d) / c).astype(np.float32))
    else:
        W = Whitening(X64.mean(0).astype(np.float32),
                      np.diag(1.0 / X64.std(0)).astype(np.float32))
    if mode == 'after':      # isotropic in front space = relative here
        sig_vec = np.sqrt(rho) * X64.std(0)
    else:                    # isotropic in the reduced raw space
        sig_vec = np.full(d, np.sqrt(rho * X64.var(0).mean()))
    sv = torch.tensor(sig_vec.astype(np.float32))
    torch.manual_seed(SEED)
    net = ScoreNet(d, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=0.0)
    gen = torch.Generator(); gen.manual_seed(97 * SEED)
    X = torch.tensor(np.asarray(tr, np.float32))
    Pt = torch.tensor(np.asarray(planted, np.float32))
    n = len(X)

    def psi(A):
        out = []
        with torch.no_grad():
            for i in range(0, len(A), 4096):
                out.append(net(A[i:i + 4096]).numpy())
        return np.concatenate(out, 0)

    curve, best = [], {'pd': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=120,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sv
            loss = ((net(b + eps) + eps / sv ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            net.eval()
            z_tr, z_te = psi(X), psi(Pt)
            zb = z_tr.mean(0)
            C = np.cov(z_tr, rowvar=False)
            T = -((z_te - zb) @ s) / np.sqrt(float(s @ C @ s))
            pd = float(_pd_at_fa(y, T, 0.1))
            auc = float(_auc(y, T))
            curve.append({'epoch': ep, 'pd': round(pd, 4),
                          'auc': round(auc, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'auc': auc, 'epoch': ep}
            bar.set_postfix_str(f'loss={float(loss.detach()):.3g} '
                                f'pd={pd:.3f} best={best["pd"]:.3f}')
    bar.close()
    out = {'setting': setting, 'front': front, 'rho': rho, 'dim': d,
           'noise': mode,
           'info': info, 'seed': SEED, 'epochs': EPOCHS,
           'pd_final': curve[-1]['pd'], 'auc_final': curve[-1]['auc'],
           'pd_best': round(best['pd'], 4), 'best_epoch': best['epoch'],
           'curve': curve, 'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] {info}  final={curve[-1]["pd"]:.3f} '
          f'best={best["pd"]:.3f}@{best["epoch"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(st, f, r, m) for st in SETTINGS for r in RHOS
             for m in ('before', 'after') for f in FRONTS
             if not (m == 'after' and f.startswith('pcaN'))
             and (f'{st}_{f}_r{r}' + ('_after' if m == 'after' else ''))
             not in done]
    print(f'{len(tasks)} probe runs, {EPOCHS} ep, seed {SEED}', flush=True)
    for st, f, r, m in tasks:
        run_one(st, f, r, m)
    print('ALL DONE', flush=True)
