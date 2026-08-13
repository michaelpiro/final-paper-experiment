"""DARTS + trainable linear front under NOISE-BEFORE (user, 08-11): the
stdlin graft. Arms: std (baseline) and stdlin (std -> trainable
Linear(D,D), init=I, shared by query+neighbors, trained jointly).
pavia4, theta=.15, rho=0.03 (DARTS-before best so far), 5000 ep, eval
every 200 from 200 (AUC), 3 seeds, drift tracked.
1000-ep archive ref: std rho=.03 best_auc .819. Noise-after canon @5000:
.896. Run from pythonProject cwd:  .venv/bin/python <this file>"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location(
    "kf", os.path.join(DIAG, 'kfree_wmw', 'run_kfree.py'))
KF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KF)

from repro.core.data import Whitening, plant_targets
from repro.core.metrics import auc_safe, dr_at_fpr
from repro.core.seeding import seed_all
from repro.models.darts.model import _NeighborDenoiser

OUT_JSON = os.path.join(HERE, 'results_nb_darts_lin.json')
SEEDS = [42, 43, 44]
EPOCHS = int(os.environ.get('NDL_EPOCHS', 5000))
EVAL_START, EVAL_EVERY = 200, 200
RHO = 0.03
THETA = 0.15
WORKERS = 3

JSON_LOCK = threading.Lock()


class _LinDenoiser(_NeighborDenoiser):
    """Trainable Linear after the frozen whitening, shared by query and
    neighbors (the stdlin front inside DARTS)."""

    def add_front(self, D):
        self.lin = nn.Linear(D, D, bias=True)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(D))
            self.lin.bias.zero_()

    def _forward_inner(self, y, neighbors):
        y = self.lin(y)
        neighbors = self.lin(neighbors)
        return super()._forward_inner(y, neighbors)


def run_one(task):
    arm, seed = task
    key = f'pavia4_ndl_{arm}_r{RHO}_s{seed}'
    t0 = time.time()
    sc = KF.get_scene('pavia4', need_nbr=True)
    tr, te, s = sc['tr'], sc['te'], sc['sig']
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    mu = X64.mean(0); std = X64.std(0) + 1e-9
    W = Whitening(mu.astype(np.float32),
                  np.diag(1.0 / std).astype(np.float32))
    cfg = dict(KF.SP_CFG['darts'])
    sigma = float(np.sqrt(RHO * float(np.mean(np.var(X64, axis=0)))))
    seed_all(seed)
    cls = _LinDenoiser if arm == 'stdlin' else _NeighborDenoiser
    net = cls(D, int(cfg['d_lat']), int(cfg['K']), list(cfg['enc_hidden']),
              list(cfg['score_hidden']), float(np.sqrt(cfg['dsm_sigma_rho'])),
              cfg['activation'], W).to('cpu')
    if arm == 'stdlin':
        net.add_front(D)
        net.to('cpu')
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg['lr']),
                            weight_decay=float(cfg['weight_decay']))
    X = torch.tensor(np.asarray(tr, np.float32))
    N = torch.tensor(np.asarray(sc['_tr_nbr'], np.float32))
    planted, labels, _ = plant_targets(
        te, s, THETA, float(KF.SP_CFG['target_fraction']), model='additive',
        seed=seed, spatial_shape=sc['te_shape'],
        edge_guard=int(KF.SP_CFG['edge_guard']))
    planted = planted.astype(np.float32)
    y_lab = np.asarray(labels)
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
    sqD = float(np.sqrt(D))
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(P, generator=gen)
        for i in range(0, P, B):
            sel = perm[i:i + B]
            eps = torch.randn((len(sel), D), generator=gen) * sigma
            psi = net(X[sel] + eps, N[sel])
            loss = ((psi + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            auc, pd05 = evaluate()
            row = {'epoch': ep, 'auc': auc, 'pd05': pd05}
            if arm == 'stdlin':
                row['drift'] = round(float(
                    (net.lin.weight.detach() - torch.eye(D)).norm()) / sqD, 3)
            curve.append(row)
            if auc > best['auc']:
                best = {'auc': auc, 'pd05': pd05, 'epoch': ep}
    out = {'auc_best': best['auc'], 'pd05_best': best['pd05'],
           'best_epoch': best['epoch'], 'auc_final': curve[-1]['auc'],
           'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    with JSON_LOCK:
        res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
        res[key] = out
        json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best_auc={best["auc"]:.3f}@{best["epoch"]} '
          f'pd05={best["pd05"]:.3f} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(a, s) for a in ('std', 'stdlin') for s in SEEDS
             if f'pavia4_ndl_{a}_r{RHO}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    KF.get_scene('pavia4', need_nbr=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
