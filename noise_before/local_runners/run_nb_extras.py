"""Two std-variants under NOISE-BEFORE (user, 08-11):
  stdlw   frozen: diag(1/std) then LW-shrunk-correlation^(-1/2) — LW's
          data-chosen alpha decides how much decorrelation is supported
          (alpha->1 pure std, ->0 full corr-ZCA). alpha recorded.
  stdlin  std (frozen) -> trainable Linear(D,D, init=I) -> trunk — the
          learnable front re-asked in this convention; drift + dist
          tracked; best-epoch-by-detection.
multi@2048, raw noise rho in {0.01, 0.003}, 15000 ep, eval every 100 from
500, 3 seeds. Refs: std-before .787/.793 best (rho .01/.003 @15k).
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
import torch.nn as nn
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

OUT_JSON = os.path.join(HERE, 'results_nb_extras.json')
N_TR = 2048
SEEDS = [42, 43, 44]
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 500, 100
RHOS = [0.01, 0.003]
WORKERS = 6

JSON_LOCK = threading.Lock()


def run_one(task):
    arm, rho, seed = task
    key = f'multi_nbx_{arm}_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    std = X64.std(0) + 1e-9
    extra = {}
    if arm == 'stdlw':
        Zs = (X64 - mu) / std
        lw = LedoitWolf().fit(Zs)
        C = (lw.covariance_ + lw.covariance_.T) / 2
        ev, V = np.linalg.eigh(C)
        ev = np.clip(ev, 1e-6, None)
        W0 = (V @ np.diag(1.0 / np.sqrt(ev)) @ V.T) @ np.diag(1.0 / std)
        extra['lw_alpha'] = round(float(lw.shrinkage_), 4)
        W = Whitening(mu.astype(np.float32), W0.astype(np.float32))
        with PA.KF.INIT_LOCK:
            torch.manual_seed(seed)
            net = ScoreNet(D, [128], 'relu', whitening=W)
        front = None
    else:                                       # stdlin
        W = Whitening(mu.astype(np.float32),
                      np.diag(1.0 / std).astype(np.float32))
        with PA.KF.INIT_LOCK:
            torch.manual_seed(seed)
            net = ScoreNet(D, [128], 'relu', whitening=W)
            front = nn.Linear(D, D, bias=True)
            with torch.no_grad():
                front.weight.copy_(torch.eye(D))
                front.bias.zero_()
            net.net = nn.Sequential(front, *list(net.net))
    net = net.to('cpu')
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
            row = {'epoch': ep, 'pd': pd}
            if front is not None:
                row['drift'] = round(float(
                    (front.weight.detach() - torch.eye(D)).norm())
                    / np.sqrt(D), 3)
            curve.append(row)
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
    out = {'pd_best': best['pd'], 'best_epoch': best['epoch'],
           'pd_final': curve[-1]['pd'], **extra,
           'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    if front is not None:
        out['drift_final'] = curve[-1].get('drift')
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]:.3f} {extra} ({out["sec"]}s)',
          flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(a, r, s) for a in ('stdlw', 'stdlin') for r in RHOS
             for s in SEEDS if f'multi_nbx_{a}_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
