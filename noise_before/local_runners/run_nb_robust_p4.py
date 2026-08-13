"""ROBUST front (median / 1.4826*MAD) on PAVIA4 — DART, NOISE-BEFORE
(user, 08-12). theta=.15, AUC criterion, rho {.003,.01,.03,.1}, seeds
42-44 (seed-major), 10000 ep budget, eval every 100 from 200.
SEQUENTIAL, live tqdm. References: std ~.769 (flat in rho), published
ZCA-after .793.
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

OUT_JSON = os.path.join(HERE, 'results_nb_robust_p4.json')
# 08-12 low-rho program: .0005@10k still climbing (.738@cap) -> 60k;
# next rho .0001 at ~200k (best-epoch ~ 1/rho; .003 peaked ~3-6k).
RHOS = [float(x) for x in os.environ.get('RHOS', '0.0005').split(',')]
SEEDS = [42, 43]
EPOCHS = int(os.environ.get('EPOCHS', 60000))
EVAL_START, EVAL_EVERY = 200, 100
# 08-12 metal arm (user): TARGET=metal -> painted-metal-sheets (GT cls 5)
# signature at theta .075 (key prefix pavia4metal); default = paper
# bitumen (cls 7) at theta .15.
METAL = os.environ.get('TARGET', '') == 'metal'
THETA = 0.075 if METAL else 0.15
TARGET_CLS = 5 if METAL else 7
PREFIX = 'pavia4metal' if METAL else 'pavia4'

torch.set_num_threads(6)


def robust_front(tr):
    X = np.asarray(tr, np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = mad   # 08-12 user: no non-gradient clipping anywhere
    return Whitening(med.astype(np.float32),
                     np.diag(1.0 / scale).astype(np.float32))


def run_one(rho, seed):
    key = f'{PREFIX}_robust_r{rho}_s{seed}'
    t0 = time.time()
    if METAL:
        from repro.scenes import pavia_protocol as PP
        from repro.core.data import plant_targets
        sc = OV.KF.get_scene('pavia4')
        s = PP.foreign_signature(sc['data'], sc['gt'], sc['te'],
                                 cls=TARGET_CLS).astype(np.float32)
        planted, labels, _ = plant_targets(
            sc['te'], s, THETA, float(OV.KF.SP_CFG['target_fraction']),
            model='additive', seed=seed, spatial_shape=sc['te_shape'],
            edge_guard=int(OV.KF.SP_CFG['edge_guard']))
        planted = planted.astype(np.float32)
        labels = np.asarray(labels)
        tr = sc['tr']
    else:
        sc, planted, labels = OV.scene_pools('pavia4', seed, THETA)
        tr, s = sc['tr'], sc['sig']
    D = tr.shape[1]
    W = robust_front(tr)
    sigma = float(np.sqrt(rho * np.asarray(tr, np.float64).var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(D, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    n = len(X)
    curve = []
    best = {'auc': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=130,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 08-12 user: LRao-parity clip
            opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            sc_ = dsm_additive(planted, tr, net, s)
            auc = float(_auc(labels, sc_))
            pd = float(_pd_at_fa(labels, sc_, 0.1))
            curve.append({'epoch': ep, 'auc': round(auc, 4),
                          'pd': round(pd, 4)})
            if auc > best['auc']:
                best = {'auc': auc, 'epoch': ep, 'pd': pd}
            bar.set_postfix_str(
                f'loss={float(loss):.3g} auc={auc:.3f} '
                f'best={best["auc"]:.3f}@{best["epoch"]}')
    bar.close()
    out = {'auc_best': round(best['auc'], 4), 'best_epoch': best['epoch'],
           'pd_at_best': round(best['pd'], 4),
           'final': curve[-1], 'rho': rho, 'sigma_raw': round(sigma, 1),
           'curve': curve, 'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best_auc={best["auc"]:.3f}@{best["epoch"]} '
          f'final={curve[-1]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(r, sd) for sd in SEEDS for r in RHOS
             if f'{PREFIX}_robust_r{r}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, sequential (6 threads)', flush=True)
    for r, sd in tasks:
        run_one(r, sd)
    print('ALL DONE', flush=True)
