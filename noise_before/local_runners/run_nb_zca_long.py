"""ZCA front (NO eigen floor) on Pavia MULTI@2048, NOISE-BEFORE, long
budget (user, 08-13): rho=0.003, 50000 ep, 3 seeds, grad clip 2.0,
weight decay 1e-5 (paper dart value). Question: does budget + clip + wd
rescue ZCA-before on multi (3000-ep reference: .29 final; std/robust
best-epoch .797/.860)? Eval every 200 from 200, Pd@Pfa=.1 criterion.
SEQUENTIAL, live tqdm. Only clip anywhere = the gradient clip.
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

FRONT = os.environ.get('FRONT', 'zca')   # zca | std (08-13 user A/B)
OUT_JSON = os.path.join(HERE, os.environ.get('OUT') or f'results_nb_{FRONT}_long.json')
N_TR = 2048
RHO = 0.003
SEEDS = [42, 43, 44]
EPOCHS = 50000
EVAL_START, EVAL_EVERY = 200, 200
CLIP = float(os.environ.get('CLIP', 2.0))
WD = float(os.environ.get('WD', 1e-5))
SEEDS = [int(x) for x in os.environ['SEEDS'].split(',')] if os.environ.get('SEEDS') else SEEDS

torch.set_num_threads(8)


def zca_front(tr):
    X = np.asarray(tr, np.float64)
    mu = X.mean(0)
    C = np.cov(X, rowvar=False)
    lam, V = np.linalg.eigh(C)          # no floor — raw spectrum
    Wz = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T
    return Whitening(mu.astype(np.float32), Wz.astype(np.float32))


def std_front(tr):
    X = np.asarray(tr, np.float64)
    mu = X.mean(0)
    d = X.std(0)
    return Whitening(mu.astype(np.float32),
                     np.diag(1.0 / d).astype(np.float32))


def run_one(seed):
    key = f'multi_{FRONT}_r{RHO}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, icfg, mu, Wz, Ws, Ww = UG.fronts_for(seed)
    D = tr.shape[1]
    W = zca_front(tr) if FRONT == 'zca' else std_front(tr)
    sigma = float(np.sqrt(RHO * np.asarray(tr, np.float64).var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=WD)
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
            pd = float(_pd_at_fa(labels, sc, icfg['pfa']))
            curve.append({'epoch': ep, 'pd': round(pd, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
            bar.set_postfix_str(f'loss={float(loss):.3g} pd={pd:.3f} '
                                f'best={best["pd"]:.3f}@{best["epoch"]}')
    bar.close()
    out = {'pd_best': round(best['pd'], 4), 'best_epoch': best['epoch'],
           'pd_final': curve[-1]['pd'], 'rho': RHO, 'clip': CLIP, 'wd': WD,
           'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [sd for sd in SEEDS if f'multi_{FRONT}_r{RHO}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, front={FRONT}, rho={RHO}, {EPOCHS} ep, clip={CLIP}, '
          f'wd={WD}, sequential (8 threads)', flush=True)
    for sd in tasks:
        run_one(sd)
    print('ALL DONE', flush=True)
