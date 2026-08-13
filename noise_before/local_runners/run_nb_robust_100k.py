"""ROBUST front, NOISE-BEFORE, rho=.001, MAX budget (user, 08-12):
100000 epochs, batch 1024 (2 steps/ep @ n=2048), Adam 5e-4 with COSINE
annealing to 5e-6 over the full run. multi@2048 only, seeds 42-44,
eval every 500 from 500. 40k-ep reference (s42, no schedule): .690@32000
still climbing. SEQUENTIAL, live tqdm.
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

from repro.protocols.iid import _pd_at_fa
from repro.core.data import Whitening
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_robust_100k.json')
RHO = 0.005   # 08-12: .001 too small (multi .690, single .670); user moved here
SEEDS = [42, 43, 44]
EPOCHS = 100000
BATCH = 1024
LR, LR_MIN = 5e-4, 5e-6
EVAL_START, EVAL_EVERY = 500, 500

torch.set_num_threads(6)


def robust_front(tr):
    X = np.asarray(tr, np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = mad   # 08-12 user: no non-gradient clipping anywhere
    return Whitening(med.astype(np.float32),
                     np.diag(1.0 / scale).astype(np.float32))


def run_one(seed):
    key = f'multi_robust100k_r{RHO}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s = OV.iid_pools('multi', seed, 2048)
    D = tr.shape[1]
    W = robust_front(tr)
    sigma = float(np.sqrt(RHO * np.asarray(tr, np.float64).var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR_MIN)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    n = len(X)
    curve = []
    best = {'pd': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=110,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            b = X[perm[i:i + BATCH]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            sc_ = dsm_additive(planted, tr, net, s)
            pd = float(_pd_at_fa(labels, sc_, 0.1))
            curve.append({'epoch': ep, 'pd': round(pd, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
            bar.set_postfix(loss=f'{float(loss):.3g}', pd=f'{pd:.3f}',
                            best=f'{best["pd"]:.3f}@{best["epoch"]}',
                            lr=f'{sched.get_last_lr()[0]:.2g}')
    bar.close()
    out = {'pd_best': round(best['pd'], 4), 'best_epoch': best['epoch'],
           'pd_final': curve[-1]['pd'], 'rho': RHO, 'batch': BATCH,
           'lr': LR, 'lr_min': LR_MIN, 'sigma_raw': round(sigma, 1),
           'curve': curve, 'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [sd for sd in SEEDS if f'multi_robust100k_r{RHO}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, rho={RHO}, {EPOCHS} ep, batch {BATCH}, '
          f'cosine {LR}->{LR_MIN}, sequential', flush=True)
    for sd in tasks:
        run_one(sd)
    print('ALL DONE', flush=True)
