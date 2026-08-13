"""REPRODUCTION runs (user, 08-13): single + multi IID experiments with
the CHOSEN recipe — std front, NOISE-BEFORE, Adam 5e-4, NO weight decay,
grad clip 1.0 (the only clip anywhere), batch 512, 15000 ep, eval every
100 from 200, best epoch by Pd@Pfa=.1. Paper n-grid, paper amplitudes,
5 paper seeds. Saves per-run: full curve (JSON), best-epoch checkpoint +
raw detection scores at best epoch (ckpt_repro/<key>.pt).
rho fixed at the per-setting std optimum: multi .003 / single .01.
Run from pythonProject cwd:  .venv/bin/python <this file>
Env overrides: SETTINGS, NS, SEEDS, EPOCHS."""
import copy
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

OUT_JSON = os.path.join(HERE, 'results_repro_std.json')
CKPT_DIR = os.path.join(HERE, 'ckpt_repro')
RHO_BY = {'multi': 0.003, 'single': 0.01}
SETTINGS = os.environ.get('SETTINGS', 'multi,single').split(',')
NS = [int(x) for x in os.environ.get(
    'NS', '20,40,60,100,200,500,1000,2000').split(',')]
SEEDS = [int(x) for x in os.environ.get('SEEDS', '42,43,44,45,46').split(',')]
EPOCHS = int(os.environ.get('EPOCHS', 15000))
EVAL_START, EVAL_EVERY = 200, 100
CLIP = 1.0

torch.set_num_threads(8)
os.makedirs(CKPT_DIR, exist_ok=True)


def std_front(tr):
    X = np.asarray(tr, np.float64)
    return Whitening(X.mean(0).astype(np.float32),
                     np.diag(1.0 / X.std(0)).astype(np.float32))


def run_one(setting, n, seed):
    key = f'{setting}_std_n{n}_s{seed}'
    t0 = time.time()
    rho = RHO_BY[setting]
    tr, planted, labels, s = OV.iid_pools(setting, seed, n)
    D = tr.shape[1]
    W = std_front(tr)
    sigma = float(np.sqrt(rho * np.asarray(tr, np.float64).var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)   # NO weight decay
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    curve = []
    best = {'pd': -1.0}
    best_state, best_scores = None, None
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=130,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), 512):
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
            auc = float(_auc(labels, sc))
            curve.append({'epoch': ep, 'pd': round(pd, 4),
                          'auc': round(auc, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'auc': auc, 'epoch': ep}
                best_state = copy.deepcopy(net.state_dict())
                best_scores = np.asarray(sc, np.float32)
            bar.set_postfix_str(f'loss={float(loss):.3g} pd={pd:.3f} '
                                f'best={best["pd"]:.3f}@{best["epoch"]}')
    bar.close()
    torch.save({'net': best_state, 'epoch': best['epoch'],
                'pd': best['pd'], 'auc': best['auc'],
                'scores': best_scores, 'labels': np.asarray(labels),
                'setting': setting, 'n': n, 'seed': seed, 'rho': rho,
                'front': 'std', 'clip': CLIP, 'wd': 0.0,
                'sigma_raw': sigma},
               os.path.join(CKPT_DIR, f'{key}.pt'))
    out = {'pd_best': round(best['pd'], 4), 'auc_at_best': round(best['auc'], 4),
           'best_epoch': best['epoch'], 'pd_final': curve[-1]['pd'],
           'rho': rho, 'n': n, 'sigma_raw': round(sigma, 1),
           'curve': curve, 'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["pd"]:.3f}@{best["epoch"]} '
          f'final={curve[-1]["pd"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(st, n, sd) for st in SETTINGS for n in NS for sd in SEEDS
             if f'{st}_std_n{n}_s{sd}' not in done]
    print(f'{len(tasks)} tasks (settings={SETTINGS}, n={NS}, seeds={SEEDS}), '
          f'{EPOCHS} ep, clip={CLIP}, wd=0, sequential', flush=True)
    for st, n, sd in tasks:
        run_one(st, n, sd)
    print('ALL DONE', flush=True)
