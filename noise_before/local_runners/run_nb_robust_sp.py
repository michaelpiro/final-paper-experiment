"""LRao's ROBUST normalization (median / 1.4826*MAD, diagonal, frozen,
no linear after) as the DART front under NOISE-BEFORE — SINGLE-class and
PAVIA4 arms (user, 08-12; multi arm = run_nb_robust.py, seed-42 partial:
.860/.855/.825 vs std .835). single@2048 (Pd@.1) + pavia4 theta=.15
(AUC), rho {.003,.01,.03,.1}, 15000 ep, eval every 100 from 200, seeds
42-44 (seed-major: full seed-42 picture first). SEQUENTIAL, live tqdm.
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
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_robust_sp.json')
RHOS = [0.003, 0.01, 0.03, 0.1]
SEEDS = [42, 43, 44]
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 200, 100

torch.set_num_threads(6)


def robust_front(tr):
    X = np.asarray(tr, np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = mad   # 08-12 user: no non-gradient clipping anywhere
    return Whitening(med.astype(np.float32),
                     np.diag(1.0 / scale).astype(np.float32))


def run_one(setting, rho, seed):
    key = f'{setting}_robust_r{rho}_s{seed}'
    t0 = time.time()
    if setting == 'single':
        tr, planted, labels, s = OV.iid_pools('single', seed, 2048)
        spatial = False
    else:
        sc, planted, labels = OV.scene_pools('pavia4', seed, 0.15)
        tr, s = sc['tr'], sc['sig']
        spatial = True
    D = tr.shape[1]
    W = robust_front(tr)
    sigma = float(np.sqrt(rho * np.asarray(tr, np.float64).var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    n = len(X)
    curve = []
    best = {'crit': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=110,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            sc_ = dsm_additive(planted, tr, net, s)
            pd = float(_pd_at_fa(labels, sc_, 0.1))
            auc = float(_auc(labels, sc_))
            crit = auc if spatial else pd
            curve.append({'epoch': ep, 'pd': round(pd, 4),
                          'auc': round(auc, 4)})
            if crit > best['crit']:
                best = {'crit': crit, 'epoch': ep, 'pd': pd, 'auc': auc}
            bar.set_postfix(loss=f'{float(loss):.3g}',
                            crit=f'{crit:.3f}',
                            best=f'{best["crit"]:.3f}@{best["epoch"]}')
    bar.close()
    out = {'crit_best': round(best['crit'], 4), 'best_epoch': best['epoch'],
           'pd_best': round(best['pd'], 4), 'auc_best': round(best['auc'], 4),
           'final': curve[-1], 'rho': rho, 'sigma_raw': round(sigma, 1),
           'curve': curve, 'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["crit"]:.3f}@{best["epoch"]} '
          f'(pd {best["pd"]:.3f} auc {best["auc"]:.3f}) '
          f'final={curve[-1]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(st, r, sd) for sd in SEEDS for st in ('single', 'pavia4')
             for r in RHOS if f'{st}_robust_r{r}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, sequential (6 threads)', flush=True)
    for st, r, sd in tasks:
        run_one(st, r, sd)
    print('ALL DONE', flush=True)
