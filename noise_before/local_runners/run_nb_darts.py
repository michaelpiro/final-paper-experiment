"""DARTS under the NOISE-BEFORE convention (user, 08-11): raw isotropic
noise on the query pixel BEFORE whitening, loss on the data-space score
(psi(x+eps, nbr) target -eps/sigma^2). Front = std (the convention's
winner); pavia4 (th=.15) and sandiego2 (th=.075); rho {0.003, 0.01, 0.03}
(around the std optimum); 1000 ep (published budget), best-epoch-by-
detection every 200 ep from 200 (AUC criterion), 3 seeds, checkpointed
best state not needed (curve recorded). Run from pythonProject cwd."""
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

OUT_JSON = os.path.join(HERE, 'results_nb_darts.json')
SEEDS = [42, 43, 44]
EPOCHS = int(os.environ.get('NBD_EPOCHS', 1000))
EVAL_START, EVAL_EVERY = 200, 200
RHOS = [0.003, 0.01, 0.03, 0.1, 0.3]   # extended: .03 was the grid-edge best
SCENES = {'pavia4': 0.15}      # user 08-11: no SD training for now
WORKERS = 6

JSON_LOCK = threading.Lock()


def run_one(task):
    scene_name, rho, seed = task
    key = f'{scene_name}_nbdarts_std_r{rho}_s{seed}'
    t0 = time.time()
    sc = KF.get_scene(scene_name, need_nbr=True)
    tr, te, s = sc['tr'], sc['te'], sc['sig']
    D = tr.shape[1]
    X64 = np.asarray(tr, np.float64)
    mu = X64.mean(0); std = X64.std(0) + 1e-9
    W = Whitening(mu.astype(np.float32),
                  np.diag(1.0 / std).astype(np.float32))
    cfg = dict(KF.SP_CFG['darts'])
    sigma = float(np.sqrt(rho * float(np.mean(np.var(X64, axis=0)))))
    seed_all(seed)
    net = _NeighborDenoiser(
        D, int(cfg['d_lat']), int(cfg['K']), list(cfg['enc_hidden']),
        list(cfg['score_hidden']), float(np.sqrt(cfg['dsm_sigma_rho'])),
        cfg['activation'], W).to('cpu')
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg['lr']),
                            weight_decay=float(cfg['weight_decay']))
    X = torch.tensor(np.asarray(tr, np.float32))
    N = torch.tensor(np.asarray(sc['_tr_nbr'], np.float32))
    theta = SCENES[scene_name]
    planted, labels, _ = plant_targets(
        te, s, theta, float(KF.SP_CFG['target_fraction']), model='additive',
        seed=seed, spatial_shape=sc['te_shape'],
        edge_guard=int(KF.SP_CFG['edge_guard']))
    planted = planted.astype(np.float32)
    y = np.asarray(labels)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    P, B = len(X), int(cfg['batch_size'])

    def lmp():
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
        return auc_safe(y, T), float(dr_at_fpr(y, T,
                                               fpr_list=(0.05,))['0.05'])

    curve = []
    best = {'auc': -1.0}
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(P, generator=gen)
        for i in range(0, P, B):
            sel = perm[i:i + B]
            eps = torch.randn((len(sel), D), generator=gen) * sigma
            psi = net(X[sel] + eps, N[sel])       # data-space score
            loss = ((psi + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            auc, pd05 = lmp()
            curve.append({'epoch': ep, 'auc': auc, 'pd05': pd05})
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
    tasks = [(sc_, r, s) for sc_ in SCENES for r in RHOS for s in SEEDS
             if f'{sc_}_nbdarts_std_r{r}_s{s}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, {WORKERS} workers', flush=True)
    for sc_ in SCENES:
        KF.get_scene(sc_, need_nbr=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for _ in ex.map(run_one, tasks):
            pass
    print('ALL DONE', flush=True)
