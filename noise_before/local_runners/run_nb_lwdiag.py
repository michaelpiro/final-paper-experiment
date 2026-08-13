"""LW-DIAGONAL standardization, no whitening (user, 08-11): front =
diag(1/sqrt(diag(Sigma_LW))) — per-band scales from the Ledoit-Wolf
covariance diagonal instead of raw std. Pure diagonal, no off-diagonal
terms anywhere. NOTE: at n=2048 LW's shrinkage ~0.4%, so diag(S_lw) ~=
raw variances — expected to tie plain std here; the variant differentiates
at small n. multi@2048, noise-before, rho {0.01, 0.003}, 15000 ep, eval
every 100 from 500, 3 seeds, 2 workers (quick streaming results).
Refs: std-before best .787 (rho .01) / .793 (rho .003).
Run from pythonProject cwd:  .venv/bin/python <this file>"""
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(os.path.dirname(HERE), 'zcainit_ladder')
spec = importlib.util.spec_from_file_location(
    "ug", os.path.join(LAD, 'run_unfreeze_grid.py'))
UG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(UG)
PA = UG.PA

from repro.protocols.iid import _pd_at_fa, _auc
from repro.core.data import Whitening
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_lwdiag.json')
N_TR = 2048
RHOS = [0.01, 0.003]
SEEDS = [42, 43, 44]
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 500, 100
WORKERS = 2

JSON_LOCK = threading.Lock()


def run_one(task):
    rho, seed = task
    key = f'multi_nblw_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    lw = LedoitWolf().fit(X64)
    d_lw = np.sqrt(np.clip(np.diag(lw.covariance_), 1e-12, None))
    W = Whitening(mu.astype(np.float32),
                  np.diag(1.0 / d_lw).astype(np.float32))
    with PA.KF.INIT_LOCK:
        torch.manual_seed(seed)
        net = ScoreNet(D, [128], 'relu', whitening=W)
    mean_var = float(np.mean(np.var(X64, axis=0)))
    sigma = float(np.sqrt(rho * mean_var))
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    curve = []
    best = {'pd': -1.0}
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(N_TR, generator=gen)
        for i in range(0, N_TR, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            sc = dsm_additive(planted, tr, net, s)
            pd = _pd_at_fa(labels, sc, icfg['pfa'])
            curve.append({'epoch': ep, 'pd': pd})
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
    out = {'pd_best': best['pd'], 'best_epoch': best['epoch'],
           'pd_final': curve[-1]['pd'],
           'lw_alpha': round(float(lw.shrinkage_), 4),
           'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]:.3f} alpha={out["lw_alpha"]} '
          f'({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(r, s) for r in RHOS for s in SEEDS
             if f'multi_nblw_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
