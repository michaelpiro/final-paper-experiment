"""Extended-budget check for the right-censored small-rho cells (user,
08-11): noise-before, std front, 15000 epochs, eval every 100 from 500.
Cells: multi rho {0.003, 0.01}, single rho {0.01} — the ones whose best
epoch sat at/near the 6000 boundary. 3 seeds.
Refs @6000: multi .003 -> .447 (climbing), .01 -> .772@~5600;
single .01 -> .673@~5700. Run from pythonProject cwd."""
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
spec = importlib.util.spec_from_file_location(
    "sp", os.path.join(HERE, 'run_nb_rho_sp.py'))
SP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SP)
KF = SP.KF

from repro.protocols.iid import _pd_at_fa, _auc
from repro.core.data import Whitening
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_long.json')
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 500, 100
SEEDS = [42, 43, 44]
CELLS = [('multi', 0.003), ('multi', 0.01), ('single', 0.01),
         ('single', 0.003), ('single', 0.001)]   # user 08-11: small-rho single
WORKERS = 6

JSON_LOCK = threading.Lock()

LAD = os.path.join(os.path.dirname(HERE), 'zcainit_ladder')
spec2 = importlib.util.spec_from_file_location(
    "ug", os.path.join(LAD, 'run_unfreeze_grid.py'))
UG = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(UG)


def data_for(setting, seed):
    if setting == 'multi':
        tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
        return tr, planted, labels, s, mu, Ws
    tr, planted, labels, s, mu, Ws, Wz, Ww = SP.data_for('single', seed)
    return tr, planted, labels, s, mu, Ws


def run_one(task):
    setting, rho, seed = task
    key = f'{setting}_nblong_std_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, mu, Ws = data_for(setting, seed)
    D = tr.shape[1]
    W = Whitening(mu.astype(np.float32), np.asarray(Ws, np.float32))
    with KF.INIT_LOCK:
        torch.manual_seed(seed)
        net = ScoreNet(D, [128], 'relu', whitening=W)
    mean_var = float(np.mean(np.var(np.asarray(tr, np.float64), axis=0)))
    sigma = float(np.sqrt(rho * mean_var))
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    n = len(X)
    curve = []
    best = {'pd': -1.0}
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            sc = dsm_additive(planted, tr, net, s)
            pd = _pd_at_fa(labels, sc, 0.1)
            curve.append({'epoch': ep, 'pd': pd})
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
    out = {'pd_best': best['pd'], 'best_epoch': best['epoch'],
           'pd_final': curve[-1]['pd'], 'sigma_raw': round(sigma, 1),
           'curve': curve, 'sec': round(time.time() - t0)}
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]:.3f} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(st, r, s) for (st, r) in CELLS for s in SEEDS
             if f'{st}_nblong_std_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
