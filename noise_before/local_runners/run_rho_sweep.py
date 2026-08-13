"""RHO sweep @ n=2048 (user, 08-13): multi + single, std front,
NOISE-BEFORE, wd=0, grad clip 1.0 (only clip), Adam 5e-4, batch 512,
5000 ep, eval every 50 from 100, best epoch by Pd@Pfa=.1, 5 seeds.
16 rho values spanning 1e-4 .. 2.0. Full curves saved.
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

OUT_JSON = os.path.join(HERE, 'results_rho_sweep_30k.json')
N_TR = 2048
RHOS = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03,
        0.05, 0.071, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 30000
EVAL_START, EVAL_EVERY = 200, 100
CLIP = 1.0

torch.set_num_threads(8)


def std_front(tr):
    X = np.asarray(tr, np.float64)
    return Whitening(X.mean(0).astype(np.float32),
                     np.diag(1.0 / X.std(0)).astype(np.float32))


def run_one(setting, rho, seed):
    key = f'{setting}_std_rho{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s = OV.iid_pools(setting, seed, N_TR)
    D = tr.shape[1]
    W = std_front(tr)
    sigma = float(np.sqrt(rho * np.asarray(tr, np.float64).var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    curve = []
    best = {'pd': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=130,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(N_TR, generator=gen)
        for i in range(0, N_TR, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            sc = dsm_additive(planted, tr, net, s)
            pd = float(_pd_at_fa(labels, sc, 0.1))
            curve.append({'epoch': ep, 'pd': round(pd, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
            bar.set_postfix_str(f'loss={float(loss):.3g} pd={pd:.3f} '
                                f'best={best["pd"]:.3f}@{best["epoch"]}')
    bar.close()
    out = {'pd_best': round(best['pd'], 4), 'best_epoch': best['epoch'],
           'pd_final': curve[-1]['pd'], 'rho': rho, 'setting': setting,
           'sigma_raw': round(sigma, 2), 'curve': curve,
           'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={curve[-1]["pd"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(st, r, sd) for st in ('multi', 'single') for r in RHOS
             for sd in SEEDS if f'{st}_std_rho{r}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, clip={CLIP}, wd=0, sequential',
          flush=True)
    for st, r, sd in tasks:
        run_one(st, r, sd)
    print('ALL DONE', flush=True)
