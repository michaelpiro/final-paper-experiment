"""Fixed-alpha decorrelation sweep (user, 08-11): standardize by the
DIAGONAL of the Ledoit-Wolf covariance (per-band scales sqrt(diag(S_lw))),
then whiten by the blended correlation ((1-alpha)C + alpha I)^(-1/2), with
alpha FORCED (LW's own alpha ~.004 = full decorrelation = collapse).
alpha in {0.7, 0.9, 0.97, 0.99}; alpha->1 recovers pure LW-diag std.
multi@2048, noise-before, rho=.01 (the winner), 15000 ep, eval every 100
from 500, 3 seeds. Refs: std-before .787; stdlin .798.
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

OUT_JSON = os.path.join(HERE, 'results_nb_alpha.json')
N_TR = 2048
RHO = 0.01
ALPHAS = [0.7, 0.9, 0.97, 0.99]
SEEDS = [42, 43, 44]
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 500, 100
WORKERS = 6

JSON_LOCK = threading.Lock()


def run_one(task):
    alpha, seed = task
    key = f'multi_nba_a{alpha}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    lw = LedoitWolf().fit(X64)
    S_lw = (lw.covariance_ + lw.covariance_.T) / 2
    d_lw = np.sqrt(np.clip(np.diag(S_lw), 1e-12, None))
    C = S_lw / np.outer(d_lw, d_lw)                   # LW correlation
    C_a = (1 - alpha) * C + alpha * np.eye(D)
    ev, V = np.linalg.eigh((C_a + C_a.T) / 2)
    ev = np.clip(ev, 1e-6, None)
    W0 = (V @ np.diag(1.0 / np.sqrt(ev)) @ V.T) @ np.diag(1.0 / d_lw)
    W = Whitening(mu.astype(np.float32), W0.astype(np.float32))
    with PA.KF.INIT_LOCK:
        torch.manual_seed(seed)
        net = ScoreNet(D, [128], 'relu', whitening=W)
    mean_var = float(np.mean(np.var(X64, axis=0)))
    sigma = float(np.sqrt(RHO * mean_var))
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
           'pd_final': curve[-1]['pd'], 'alpha': alpha,
           'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]:.3f} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(a, s) for a in ALPHAS for s in SEEDS
             if f'multi_nba_a{a}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
