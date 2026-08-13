"""NOISE-BEFORE-FRONT rho sweep with best-epoch-by-detection (user, 08-11).

Standing convention from now on: raw isotropic noise added BEFORE the
front, sigma^2 = rho * mean band variance. multi@2048, fronts {std, zca,
wmw}, rho {0.001,0.003,0.01,0.03,0.1,0.3}, 6000 epochs (high budget),
detection (Pd@0.1) evaluated every 100 epochs starting at 500; the run
reports the BEST-DETECTION epoch/value (+ final + the curve).
Refs (noise-before @3000ep final): std .66@rho.1, zca .29, wmw .18.
Run from pythonProject cwd:  .venv/bin/python <this file>
"""
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

OUT_JSON = os.path.join(HERE, 'results_nb_rho.json')
N_TR = 2048
SEEDS = [42, 43, 44]
EPOCHS = int(os.environ.get('NB_EPOCHS', 6000))
EVAL_START, EVAL_EVERY = 500, 100
FRONTS = ['std', 'zca', 'wmw']
RHOS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
WORKERS = 5

JSON_LOCK = threading.Lock()


def run_one(task):
    front_name, rho, seed = task
    key = f'multi_nbr_{front_name}_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    W0 = {'wmw': Ww, 'zca': Wz, 'std': Ws}[front_name]
    W = Whitening(mu.astype(np.float32), W0.astype(np.float32))
    with PA.KF.INIT_LOCK:
        torch.manual_seed(seed)
        net = ScoreNet(D, [128], 'relu', whitening=W)
    mean_var = float(np.mean(np.var(np.asarray(tr, np.float64), axis=0)))
    sigma = float(np.sqrt(rho * mean_var))
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(tr)
    curve = []
    best = {'pd': -1.0, 'epoch': 0}
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
                best = {'pd': pd, 'epoch': ep,
                        'auc': _auc(labels, sc)}
    out = {'pd_best': best['pd'], 'best_epoch': best['epoch'],
           'auc_best': best.get('auc'), 'pd_final': curve[-1]['pd'],
           'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best_pd={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]:.3f} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(f, r, s) for f in FRONTS for r in RHOS for s in SEEDS
             if f'multi_nbr_{f}_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, eval every {EVAL_EVERY} '
          f'from {EVAL_START}, {WORKERS} workers', flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
