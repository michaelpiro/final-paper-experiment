"""OVERNIGHT campaign (user, 08-11/12). Noise-BEFORE convention, eval every
50 epochs, best-epoch-by-detection, resume-safe per phase.

Phase A pavia_rho : multi/single@2048 + pavia4; fronts std/lwdiag/stdlin;
                    rho {.003,.01,.03,.1}; 3 seeds; 15000 ep.
Phase B pavia_n   : multi/single, n {20..2000}; fronts std/stdlin;
                    rho {.01,.1}; 3 seeds; 15000 ep.
Phase C sd_dart   : sandiego/sandiego2 (theta=.075); std/stdlin;
                    rho {.003,.01,.03,.1}; 3 seeds; 15000 ep.
Phase D sd_darts  : both scenes; DARTS std / stdlin-graft;
                    rho {.01,.03,.1}; 2 seeds; 3000 ep; eval every 200.

Run from pythonProject cwd (wrap with systemd-inhibit):
  systemd-inhibit --what=sleep:idle:handle-lid-switch \
    .venv/bin/python <this file>
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
import torch.nn as nn
from sklearn.covariance import LedoitWolf

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location(
    "kf", os.path.join(DIAG, 'kfree_wmw', 'run_kfree.py'))
KF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KF)
spec2 = importlib.util.spec_from_file_location(
    "ndl", os.path.join(HERE, 'run_nb_darts_lin.py'))
NDL = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(NDL)

from repro.protocols.iid import (load_hsi, build_pools, _pd_at_fa, _auc)
from repro.core.data import Whitening, plant_targets
from repro.core.metrics import auc_safe, dr_at_fpr
from repro.core.models import ScoreNet
from repro.core.detectors import dsm_additive
from repro.core.seeding import seed_all
from repro.models.darts.model import _NeighborDenoiser

WORKERS = 5
SEEDS3 = [42, 43, 44]
IID_EPOCHS = 15000
EVAL_START, EVAL_EVERY = 200, 50
JSON_LOCK = threading.Lock()
torch.set_num_threads(3)

IID_CFG = KF.PA.IID_CFG if hasattr(KF, 'PA') else None
import yaml
MCFG = yaml.safe_load(open('repro/configs/iid_multi.yaml'))
MCFG.update(dataset='repro/data/pavia-u.mat')
SCFG = yaml.safe_load(open('repro/configs/iid_single.yaml'))
SCFG.update(dataset='repro/data/pavia-u.mat')

_DATA = {}


def iid_pools(mode, seed, n):
    key = (mode, seed, n)
    if key in _DATA:
        return _DATA[key]
    cfg = MCFG if mode == 'multi' else SCFG
    rng = np.random.default_rng(seed)
    data, gt = load_hsi(cfg['dataset'])
    bkg, tgt = build_pools(data, gt.flatten(), cfg, mode)
    s = tgt.mean(axis=0).astype(np.float32)
    idx = np.arange(len(bkg)); rng.shuffle(idx)
    shuf = bkg[idx]
    tr = shuf[:n].astype(np.float32)
    te = shuf[-int(cfg['test_size']):].astype(np.float32)
    planted, labels, _ = plant_targets(te, s, cfg['amplitude'],
                                       cfg['target_fraction'],
                                       model='additive', seed=seed)
    _DATA[key] = (tr, planted.astype(np.float32), np.asarray(labels), s)
    return _DATA[key]


def scene_pools(name, seed, theta):
    key = (name, seed, theta)
    if key in _DATA:
        return _DATA[key]
    sc = KF.get_scene(name, need_nbr=True)
    planted, labels, _ = plant_targets(
        sc['te'], sc['sig'], theta, float(KF.SP_CFG['target_fraction']),
        model='additive', seed=seed, spatial_shape=sc['te_shape'],
        edge_guard=int(KF.SP_CFG['edge_guard']))
    _DATA[key] = (sc, planted.astype(np.float32), np.asarray(labels))
    return _DATA[key]


def make_front(front, tr):
    X = np.asarray(tr, np.float64)
    mu = X.mean(0)
    if front == 'lwdiag':
        lw = LedoitWolf().fit(X)
        d = np.sqrt(np.clip(np.diag(lw.covariance_), 1e-12, None))
    else:                                    # std / stdlin base
        d = X.std(0) + 1e-9
    return Whitening(mu.astype(np.float32),
                     np.diag(1.0 / d).astype(np.float32))


def train_dart(front, tr, planted, labels, s, rho, seed, epochs,
               spatial=False):
    D = tr.shape[1]
    W = make_front(front, tr)
    with KF.INIT_LOCK:
        torch.manual_seed(seed)
        net = ScoreNet(D, [128] if D == 103 else [200], 'relu', whitening=W)
        if front == 'stdlin':
            lin = nn.Linear(D, D, bias=True)
            with torch.no_grad():
                lin.weight.copy_(torch.eye(D)); lin.bias.zero_()
            net.net = nn.Sequential(lin, *list(net.net))
    v = float(np.mean(np.var(np.asarray(tr, np.float64), axis=0)))
    sigma = float(np.sqrt(rho * v))
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    n = len(X)
    curve = []
    best = {'crit': -1.0}
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == epochs):
            net.eval()
            sc = dsm_additive(planted, tr, net, s)
            pd = _pd_at_fa(labels, sc, 0.1)
            auc = _auc(labels, sc)
            crit = auc if spatial else pd
            curve.append({'epoch': ep, 'pd': round(pd, 4),
                          'auc': round(auc, 4)})
            if crit > best['crit']:
                best = {'crit': crit, 'epoch': ep, 'pd': pd, 'auc': auc}
    return best, curve


def save(out_json, key, out):
    with JSON_LOCK:
        res = json.load(open(out_json)) if os.path.exists(out_json) else {}
        res[key] = out
        json.dump(res, open(out_json, 'w'), indent=1)
    print(f'[{key}] best={out["best"]}', flush=True)


def run_phase(name, tasks, fn):
    out_json = os.path.join(HERE, f'results_ov_{name}.json')
    done = set(json.load(open(out_json)).keys()) if os.path.exists(out_json) else set()
    todo = [t for t in tasks if t[0] not in done]
    print(f'=== PHASE {name}: {len(todo)}/{len(tasks)} to run ===',
          flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(lambda t: fn(out_json, *t), todo):
            pass


# ---------------- Phase A: pavia rho ----------------

def a_task(out_json, key, setting, front, rho, seed):
    t0 = time.time()
    if setting in ('multi', 'single'):
        tr, planted, labels, s = iid_pools(setting, seed, 2048)
        best, curve = train_dart(front, tr, planted, labels, s, rho, seed,
                                 IID_EPOCHS, spatial=False)
    else:
        sc, planted, labels = scene_pools('pavia4', seed, 0.15)
        best, curve = train_dart(front, sc['tr'], planted, labels,
                                 sc['sig'], rho, seed, IID_EPOCHS,
                                 spatial=True)
    save(out_json, key, {'best': {k: round(v, 4) if isinstance(v, float)
                                  else v for k, v in best.items()},
                         'final': curve[-1], 'curve': curve,
                         'sec': round(time.time() - t0)})


# ---------------- Phase C: SD dart ----------------

def c_task(out_json, key, scene, front, rho, seed):
    t0 = time.time()
    sc, planted, labels = scene_pools(scene, seed, 0.075)
    best, curve = train_dart(front, sc['tr'], planted, labels, sc['sig'],
                             rho, seed, IID_EPOCHS, spatial=True)
    save(out_json, key, {'best': {k: round(v, 4) if isinstance(v, float)
                                  else v for k, v in best.items()},
                         'final': curve[-1], 'curve': curve,
                         'sec': round(time.time() - t0)})


# ---------------- Phase D: SD darts ----------------

def d_task(out_json, key, scene, arm, rho, seed):
    t0 = time.time()
    theta = 0.075
    sc, planted, labels = scene_pools(scene, seed, theta)
    tr, s = sc['tr'], sc['sig']
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    mu = X64.mean(0); std = X64.std(0) + 1e-9
    W = Whitening(mu.astype(np.float32),
                  np.diag(1.0 / std).astype(np.float32))
    cfg = dict(KF.SP_CFG['darts'])
    sigma = float(np.sqrt(rho * float(np.mean(np.var(X64, axis=0)))))
    seed_all(seed)
    cls = NDL._LinDenoiser if arm == 'stdlin' else _NeighborDenoiser
    net = cls(D, int(cfg['d_lat']), int(cfg['K']), list(cfg['enc_hidden']),
              list(cfg['score_hidden']),
              float(np.sqrt(cfg['dsm_sigma_rho'])), cfg['activation'],
              W).to('cpu')
    if arm == 'stdlin':
        net.add_front(D); net.to('cpu')
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg['lr']),
                            weight_decay=float(cfg['weight_decay']))
    X = torch.tensor(np.asarray(tr, np.float32))
    N = torch.tensor(np.asarray(sc['_tr_nbr'], np.float32))
    y_lab = labels
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    P, B = len(X), int(cfg['batch_size'])

    def evaluate():
        with torch.no_grad():
            def scores(pix, nbr):
                out = []
                for i in range(0, len(pix), 512):
                    p = torch.tensor(np.asarray(pix[i:i+512], np.float32))
                    nb = torch.tensor(np.asarray(nbr[i:i+512], np.float32))
                    out.append(net(p, nb).numpy())
                return np.concatenate(out, 0)
            z_tr = scores(tr, sc['_tr_nbr'])
            z_te = scores(planted, sc['_te_nbr'])
        zb = z_tr.mean(0)
        C = np.cov(z_tr, rowvar=False)
        T = -((z_te - zb) @ s) / np.sqrt(max(float(s @ C @ s), 1e-12))
        return auc_safe(y_lab, T), float(
            dr_at_fpr(y_lab, T, fpr_list=(0.05,))['0.05'])

    curve = []
    best = {'auc': -1.0}
    for ep in range(1, 3001):
        net.train()
        perm = torch.randperm(P, generator=gen)
        for i in range(0, P, B):
            sel = perm[i:i + B]
            eps = torch.randn((len(sel), D), generator=gen) * sigma
            psi = net(X[sel] + eps, N[sel])
            loss = ((psi + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= 200 and (ep % 200 == 0 or ep == 3000):
            net.eval()
            auc, pd05 = evaluate()
            curve.append({'epoch': ep, 'auc': round(auc, 4),
                          'pd05': round(pd05, 4)})
            if auc > best['auc']:
                best = {'auc': auc, 'pd05': pd05, 'epoch': ep}
    save(out_json, key, {'best': {k: round(v, 4) if isinstance(v, float)
                                  else v for k, v in best.items()},
                         'final': curve[-1], 'curve': curve,
                         'sec': round(time.time() - t0)})


if __name__ == '__main__':
    # A: pavia rho
    tasks = []
    for st in ('multi', 'single', 'pavia4'):
        for f in ('std', 'lwdiag', 'stdlin'):
            for r in (0.003, 0.01, 0.03, 0.1):
                for sd in SEEDS3:
                    tasks.append((f'{st}_{f}_r{r}_s{sd}', st, f, r, sd))
    run_phase('pavia_rho', tasks, a_task)

    # B: pavia n-sweep
    tasks = []
    for st in ('multi', 'single'):
        for f in ('std', 'stdlin'):
            for n in (20, 40, 60, 100, 200, 500, 1000, 2000):
                for r in (0.01, 0.1):
                    for sd in SEEDS3:
                        tasks.append((f'{st}_{f}_n{n}_r{r}_s{sd}',
                                      st, f, n, r, sd))

    def b_task(out_json, key, st, f, n, r, sd):
        t0 = time.time()
        tr, planted, labels, s = iid_pools(st, sd, n)
        best, curve = train_dart(f, tr, planted, labels, s, r, sd,
                                 IID_EPOCHS, spatial=False)
        save(out_json, key,
             {'best': {k: round(v, 4) if isinstance(v, float) else v
                       for k, v in best.items()},
              'final': curve[-1], 'curve': curve,
              'sec': round(time.time() - t0)})

    run_phase('pavia_n', tasks, b_task)

    # C: SD dart
    tasks = []
    for scn in ('sandiego', 'sandiego2'):
        for f in ('std', 'stdlin'):
            for r in (0.003, 0.01, 0.03, 0.1):
                for sd in SEEDS3:
                    tasks.append((f'{scn}_{f}_r{r}_s{sd}', scn, f, r, sd))
    run_phase('sd_dart', tasks, c_task)

    # D: SD darts
    tasks = []
    for scn in ('sandiego', 'sandiego2'):
        for arm in ('std', 'stdlin'):
            for r in (0.01, 0.03, 0.1):
                for sd in (42, 43):
                    tasks.append((f'{scn}_darts_{arm}_r{r}_s{sd}',
                                  scn, arm, r, sd))
    run_phase('sd_darts', tasks, d_task)

    print('OVERNIGHT ALL DONE', flush=True)
