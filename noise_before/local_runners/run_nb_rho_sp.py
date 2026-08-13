"""Noise-before rho sweep on SINGLE-class and PAVIA4 (user, 08-11).
Same design as run_nb_rho: raw isotropic noise BEFORE the front, fronts
{std, zca, wmw}, rho {0.001..0.3}, 6000 ep, detection evaluated every 100
epochs from 500, best-detection epoch reported. Selection metric: single =
Pd@0.1; pavia4 = AUC (theta=.15, pd05 recorded at the best-AUC epoch).
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
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location(
    "kf", os.path.join(DIAG, 'kfree_wmw', 'run_kfree.py'))
KF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KF)
spec2 = importlib.util.spec_from_file_location(
    "sw", os.path.join(DIAG, 'single_wmw', 'run_single_wmw.py'))
SW = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(SW)

from repro.protocols.iid import _pd_at_fa, _auc
from repro.core.data import Whitening, plant_targets
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive

OUT_JSON = os.path.join(HERE, 'results_nb_rho_sp.json')
SEEDS = [42, 43, 44]
EPOCHS = int(os.environ.get('NB_EPOCHS', 6000))
EVAL_START, EVAL_EVERY = 500, 100
FRONTS = ['std']          # user 08-11: std only (zca/wmw dropped mid-run)
RHOS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
WORKERS = 5

JSON_LOCK = threading.Lock()
_C = {}


def fronts_of(X, seed):
    X = np.asarray(X, np.float64)
    mu = X.mean(0); std = X.std(0) + 1e-9
    S = np.cov(X, rowvar=False); S = (S + S.T) / 2
    ev, V = np.linalg.eigh(S)
    inv = 1.0 / np.sqrt(np.clip(ev, max(float(ev[-1]) * 1e-5, 1e2), None))
    Wz = V @ np.diag(inv) @ V.T
    Wfront, _, _, _ = KF.build_front('elbow', X, seed)
    Ww = Wfront.W.numpy().astype(np.float64)
    return mu, np.diag(1.0 / std), Wz, Ww


def data_for(setting, seed):
    k = (setting, seed)
    if k in _C:
        return _C[k]
    if setting == 'single':
        tr, planted, labels, s = SW.build_single(seed)
        extra = None
    else:
        sc = KF.get_scene('pavia4')
        tr = sc['tr']
        planted, labels, _ = plant_targets(
            sc['te'], sc['sig'], 0.15, float(KF.SP_CFG['target_fraction']),
            model='additive', seed=seed, spatial_shape=sc['te_shape'],
            edge_guard=int(KF.SP_CFG['edge_guard']))
        planted = planted.astype(np.float32)
        labels = np.asarray(labels)
        s = sc['sig']
        extra = None
    mu, Ws, Wz, Ww = fronts_of(tr, seed)
    _C[k] = (tr, planted, labels, s, mu, Ws, Wz, Ww)
    return _C[k]


def run_one(task):
    setting, front_name, rho, seed = task
    key = f'{setting}_nbr_{front_name}_r{rho}_s{seed}'
    t0 = time.time()
    tr, planted, labels, s, mu, Ws, Wz, Ww = data_for(setting, seed)
    D = tr.shape[1]
    W0 = {'wmw': Ww, 'zca': Wz, 'std': Ws}[front_name]
    W = Whitening(mu.astype(np.float32), W0.astype(np.float32))
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
    best = {'crit': -1.0}
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
            sc_ = dsm_additive(planted, tr, net, s)
            pd = _pd_at_fa(labels, sc_, 0.1)
            auc = _auc(labels, sc_)
            crit = pd if setting == 'single' else auc
            curve.append({'epoch': ep, 'pd': pd, 'auc': auc})
            if crit > best['crit']:
                best = {'crit': crit, 'epoch': ep, 'pd': pd, 'auc': auc}
    out = {'best_epoch': best['epoch'], 'pd_best': best['pd'],
           'auc_best': best['auc'], 'final_pd': curve[-1]['pd'],
           'final_auc': curve[-1]['auc'], 'sigma_raw': round(sigma, 1),
           'curve': curve, 'sec': round(time.time() - t0)}
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best={best["crit"]:.3f}@{best["epoch"]} '
          f'final_pd={out["final_pd"]:.3f} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(st, f, r, s) for st in ('single', 'pavia4') for f in FRONTS
             for r in RHOS for s in SEEDS
             if f'{st}_nbr_{f}_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    KF.get_scene('pavia4')
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
