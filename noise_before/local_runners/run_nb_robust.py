"""LRao's ROBUST normalization as the DART front under NOISE-BEFORE
(user, 08-12): per-band median + 1/(1.4826*MAD) — the frozen diagonal
front from repro/models/lrao/model.py — with NO linear layer after.
Standard whitened ScoreNet architecture (Wt back-map). multi@2048,
rho sweep {.003,.01,.03,.1}, 15000 ep, eval every 100 from 200, seeds
42-44 (rho-major). SEQUENTIAL, live tqdm (pd postfix).
Reference (overnight): std rho=.003 .797, stdlin rho=.1 .817.
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
LAD = os.path.join(os.path.dirname(HERE), 'zcainit_ladder')
spec = importlib.util.spec_from_file_location(
    "ug", os.path.join(LAD, 'run_unfreeze_grid.py'))
UG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(UG)

from repro.protocols.iid import _pd_at_fa
from repro.core.data import Whitening
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_robust.json')
N_TR = 2048
RHOS = [0.003, 0.01, 0.03, 0.1]
SEEDS = [42, 43, 44]
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 200, 100

torch.set_num_threads(8)


def run_one(rho, seed):
    key = f'multi_robust_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    med = np.median(X64, axis=0)
    mad = np.median(np.abs(X64 - med), axis=0) * 1.4826
    scale = mad   # 08-12 user: no non-gradient clipping anywhere
    W = Whitening(med.astype(np.float32),
                  np.diag(1.0 / scale).astype(np.float32))
    sigma = float(np.sqrt(rho * X64.var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    curve = []
    best = {'pd': -1.0}
    nan_batches = 0
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=110,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(N_TR, generator=gen)
        for i in range(0, N_TR, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            try:
                sc = dsm_additive(planted, tr, net, s)
                pd = float(_pd_at_fa(labels, sc, icfg['pfa']))
            except Exception:
                pd = float('nan')
            curve.append({'epoch': ep, 'pd': pd})
            if np.isfinite(pd) and pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
            bar.set_postfix(loss=f'{float(loss):.3g}', pd=f'{pd:.3f}',
                            best=f'{best["pd"]:.3f}@{best["epoch"]}'
                            if best['pd'] >= 0 else 'n/a')
    bar.close()
    out = {'pd_best': best.get('pd'), 'best_epoch': best.get('epoch'),
           'pd_final': curve[-1]['pd'] if curve else None,
           'rho': rho, 'sigma_raw': round(sigma, 1),
           'nan_batches': nan_batches, 'curve': curve,
           'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best.get("epoch")} '
          f'final={out["pd_final"]} nan={nan_batches} ({out["sec"]}s)',
          flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(r, s) for s in SEEDS for r in RHOS
             if f'multi_robust_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, sequential (8 threads)', flush=True)
    for rho, seed in tasks:
        run_one(rho, seed)
    print('ALL DONE', flush=True)
