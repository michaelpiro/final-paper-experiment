"""LINEAR-ONLY front under NOISE-BEFORE (user, 08-12): NO frozen std —
raw data -> trainable Linear(D,D) -> trunk. INIT arms:
  identity : W=I, b=0 (VERDICT s42: chance .100 at every rho — collapses
             to the trivial zero-score solution, final loss == D/sigma^2)
  stdinit  : W=diag(1/std), b=-mu/std — the linear IS the std front at
             epoch 0, then free to train (user follow-up).
multi@2048, rho sweep {.003,.01,.03,.1}, 15000 ep, eval every 100 from
200, seeds 42-44 (rho-major order). SEQUENTIAL, live tqdm (pd postfix).
drift = ||W - W_init||_F / ||W_init||_F.
Reference (overnight): std rho=.003 .797, stdlin rho=.1 .817.
Run from pythonProject cwd:  INIT=std .venv/bin/python <this file>"""
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(os.path.dirname(HERE), 'zcainit_ladder')
spec = importlib.util.spec_from_file_location(
    "ug", os.path.join(LAD, 'run_unfreeze_grid.py'))
UG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(UG)

from repro.protocols.iid import _pd_at_fa
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_linonly.json')
N_TR = 2048
RHOS = [0.003, 0.01, 0.03, 0.1]
SEEDS = [42, 43, 44]
EPOCHS = 15000
EVAL_START, EVAL_EVERY = 200, 100
INIT = os.environ.get('INIT', 'identity')   # 'identity' | 'std'

torch.set_num_threads(8)


def run_one(rho, seed):
    tag = '' if INIT == 'identity' else '_stdinit'
    key = f'multi_linonly{tag}_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    v = X64.var(0)
    sigma = float(np.sqrt(rho * v.mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=None)
    lin = nn.Linear(D, D, bias=True)
    if INIT == 'std':
        std = np.sqrt(v) + 1e-9
        W0 = torch.diag(torch.tensor((1.0 / std).astype(np.float32)))
        b0 = torch.tensor((-(X64.mean(0) / std)).astype(np.float32))
    else:
        W0 = torch.eye(D)
        b0 = torch.zeros(D)
    with torch.no_grad():
        lin.weight.copy_(W0)
        lin.bias.copy_(b0)
    net.net = nn.Sequential(lin, *net.net)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    w0norm = float(W0.norm())
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
            drift = round(float(
                (lin.weight.detach() - W0).norm()) / w0norm, 3)
            curve.append({'epoch': ep, 'pd': pd, 'drift': drift})
            if np.isfinite(pd) and pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
            bar.set_postfix(loss=f'{float(loss):.3g}', pd=f'{pd:.3f}',
                            best=f'{best["pd"]:.3f}@{best["epoch"]}'
                            if best['pd'] >= 0 else 'n/a', drift=drift)
    bar.close()
    out = {'pd_best': best.get('pd'), 'best_epoch': best.get('epoch'),
           'pd_final': curve[-1]['pd'] if curve else None,
           'drift_final': curve[-1]['drift'] if curve else None,
           'rho': rho, 'sigma_raw': round(sigma, 1),
           'nan_batches': nan_batches, 'curve': curve,
           'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best.get("epoch")} '
          f'final={out["pd_final"]} drift={out["drift_final"]} '
          f'nan={nan_batches} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tag = '' if INIT == 'identity' else '_stdinit'
    tasks = [(r, s) for s in SEEDS for r in RHOS
             if f'multi_linonly{tag}_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks (init={INIT}), {EPOCHS} ep, sequential (8 threads)',
          flush=True)
    for rho, seed in tasks:
        run_one(rho, seed)
    print('ALL DONE', flush=True)
