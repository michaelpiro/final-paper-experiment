"""LRao batch A/B (user, 08-13): multi@1024, FULL-batch vs batch-512,
everything else identical (no cutoff, robust IQR norm, detach sigma,
wd=0, no clip, lr 5e-4, 2500 epochs, eval every 50, best + final).
Diagnoses why GrandSweep LRao-multi peaks ~.49 at n=1024 vs the
lrao_noreg/lrao_batch references (~.71-.76).
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
from repro.core.models import (ScoreNet, lfi_loss_mode2,
                               compute_lfi_detector_scores_mode2)
from repro.core.normalization import robust_whitening_iqr

OUT_JSON = os.path.join(HERE, 'results_lrao_ab.json')
N = 1024
SEEDS = [42, 43, 44]
EPOCHS = 2500
EVAL_EVERY = 50
LR, WD = 5e-4, 0.0
DTH = 0.01           # lfi_delta_theta (config value)
DETACH = True        # lfi_detach_sigma (config value)

torch.set_num_threads(6)


def run_one(arm, seed):
    key = f'multi_lraoAB_{arm}_n{N}_s{seed}'
    t0 = time.time()
    tr, planted, y, s = OV.iid_pools('multi', seed, N)
    bsz = N if arm == 'full' else 512
    Wl = robust_whitening_iqr(tr)
    torch.manual_seed(seed)
    model = ScoreNet(tr.shape[1], [128], 'relu', whitening=Wl)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    Xl = torch.tensor(np.asarray(tr, np.float32))
    curve, best = [], {'pd': -1.0, 'epoch': None}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=120,
               mininterval=5.0, file=sys.stdout, ascii=True)
    trJ = float('nan')
    for ep in bar:
        model.train()
        perm = torch.randperm(N)
        tot, nb = 0.0, 0
        for i in range(0, N, bsz):
            b = Xl[perm[i:i + bsz]]
            try:
                loss = lfi_loss_mode2(model, b, DTH, detach_sigma=DETACH)
            except Exception:
                continue
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        trJ = -tot / max(nb, 1)
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            model.eval()
            T = np.asarray(compute_lfi_detector_scores_mode2(
                model, tr, planted, s, DTH))
            pd = float(_pd_at_fa(y, T, 0.1))
            curve.append({'epoch': ep, 'pd': round(pd, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'epoch': ep}
            bar.set_postfix_str(f'trJ={trJ:.4g} pd={pd:.3f} '
                                f'best={best["pd"]:.3f}@{best["epoch"]}')
    bar.close()
    out = {'arm': arm, 'seed': seed, 'pd_best': round(best['pd'], 4),
           'best_epoch': best['epoch'], 'pd_final': curve[-1]['pd'],
           'curve': curve, 'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={out["pd_final"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(a, sd) for sd in SEEDS for a in ('full', 'b512')
             if f'multi_lraoAB_{a}_n{N}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, eval/{EVAL_EVERY}, sequential',
          flush=True)
    for a, sd in tasks:
        run_one(a, sd)
    print('ALL DONE', flush=True)
