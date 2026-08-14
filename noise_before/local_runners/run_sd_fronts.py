"""SD1/SD2 DART front probe (user, 08-14): the aggressive pcaN front
(rotation-only truncation + global scalar) vs std, noise-before, theta=.075.
2 seeds, 5000 ep, rhos {.03,.01}. Same eval as the fronts probe
(pd@Pfa=.1 + AUC, z-projection statistic). Reuses reduce_front from
run_fronts_probe.py. Run from pythonProject cwd."""
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
    "fp", os.path.join(HERE, 'run_fronts_probe.py'))
FP = importlib.util.module_from_spec(spec)
sys.modules['fp'] = FP
spec.loader.exec_module(FP)

from repro import scenes
from repro.protocols.spatial import load_cfg
from repro.core.data import Whitening, plant_targets
from repro.core.models import ScoreNet
from repro.protocols.iid import _pd_at_fa, _auc

OUT_JSON = os.path.join(HERE, 'results_sd_fronts.json')
SCENES = ['sandiego', 'sandiego2']
FRONTS = ['std', 'pcaN99.9', 'pcaN99.5', 'pcaN99.0']
RHOS = [0.03, 0.01]
SEEDS = [42, 43]
EPOCHS = 5000
EVAL_EVERY = 100
THETA = 0.075

torch.set_num_threads(5)
SP_CFG = load_cfg()
_SC = {}


def pools(scene, seed):
    if scene not in _SC:
        _SC[scene] = scenes.build(scene, SP_CFG)
    sc = _SC[scene]
    s = np.asarray(sc['sig'], np.float32)
    planted, labels, _ = plant_targets(
        sc['te'], s, THETA, float(SP_CFG['target_fraction']),
        model='additive', seed=seed, spatial_shape=sc['te_shape'],
        edge_guard=int(SP_CFG['edge_guard']))
    return (np.asarray(sc['tr'], np.float32),
            planted.astype(np.float32), np.asarray(labels), s)


def run_one(scene, front, rho, seed):
    key = f'{scene}_{front}_r{rho}_s{seed}'
    t0 = time.time()
    tr0, planted0, y, s0 = pools(scene, seed)
    tr, planted, s, info = FP.reduce_front(front, tr0, planted0, s0)
    d = tr.shape[1]
    print(f'[{key}] front keeps: {info}', flush=True)
    X64 = np.asarray(tr, np.float64)
    if front.startswith('pcaN'):
        c = float(np.sqrt(X64.var(0).mean()))
        W = Whitening(X64.mean(0).astype(np.float32),
                      (np.eye(d) / c).astype(np.float32))
    else:
        W = Whitening(X64.mean(0).astype(np.float32),
                      np.diag(1.0 / X64.std(0)).astype(np.float32))
    sigma = float(np.sqrt(rho * X64.var(0).mean()))
    torch.manual_seed(seed)
    net = ScoreNet(d, [128], 'relu', whitening=W)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=0.0)
    gen = torch.Generator(); gen.manual_seed(97 * seed)
    X = torch.tensor(np.asarray(tr, np.float32))
    Pt = torch.tensor(np.asarray(planted, np.float32))
    n = len(X)

    def psi(A):
        out = []
        with torch.no_grad():
            for i in range(0, len(A), 4096):
                out.append(net(A[i:i + 4096]).numpy())
        return np.concatenate(out, 0)

    curve, best = [], {'pd': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=120,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, 512):
            b = X[perm[i:i + 512]]
            eps = torch.randn(b.shape, generator=gen) * sigma
            loss = ((net(b + eps) + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            net.eval()
            z_tr, z_te = psi(X), psi(Pt)
            zb = z_tr.mean(0)
            C = np.cov(z_tr, rowvar=False)
            T = -((z_te - zb) @ s) / np.sqrt(float(s @ C @ s))
            pd = float(_pd_at_fa(y, T, 0.1))
            pd05 = float(_pd_at_fa(y, T, 0.05))
            auc = float(_auc(y, T))
            curve.append({'epoch': ep, 'pd': round(pd, 4),
                          'pd05': round(pd05, 4), 'auc': round(auc, 4)})
            if pd > best['pd']:
                best = {'pd': pd, 'pd05': pd05, 'auc': auc, 'epoch': ep}
            bar.set_postfix_str(f'loss={float(loss.detach()):.3g} '
                                f'pd={pd:.3f} auc={auc:.3f} '
                                f'best={best["pd"]:.3f}')
    bar.close()
    out = {'scene': scene, 'front': front, 'rho': rho, 'seed': seed,
           'dim': d, 'info': info, 'theta': THETA, 'epochs': EPOCHS,
           'pd_final': curve[-1]['pd'], 'pd05_final': curve[-1]['pd05'],
           'auc_final': curve[-1]['auc'], 'pd_best': round(best['pd'], 4),
           'best_epoch': best['epoch'], 'curve': curve,
           'sec': round(time.time() - t0)}
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] {info}  final pd={curve[-1]["pd"]:.3f} '
          f'pd05={curve[-1]["pd05"]:.3f} auc={curve[-1]["auc"]:.3f} '
          f'best={best["pd"]:.3f}@{best["epoch"]} ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(sc, f, r, sd) for sc in SCENES for f in FRONTS for r in RHOS
             for sd in SEEDS if f'{sc}_{f}_r{r}_s{sd}' not in done]
    print(f'{len(tasks)} SD DART front runs, {EPOCHS} ep', flush=True)
    for sc, f, r, sd in tasks:
        run_one(sc, f, r, sd)
    print('ALL DONE', flush=True)
