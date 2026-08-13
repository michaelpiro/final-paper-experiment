"""DARTS + ROBUST front (median / 1.4826*MAD) on PAVIA4, NOISE-BEFORE
(user, 08-12). Plain _NeighborDenoiser (no linear graft), robust
Whitening instead of std. theta=.15, 10000 ep budget, eval every 200
from 200 (AUC), rho {.03, .01, .1} x seeds 42,43 (rho .03 first — the
DARTS-before incumbent). SEQUENTIAL, live tqdm.
References @5000ep: darts-std .856, darts-stdlin .908, noise-after
canon .896 / gmm .913.
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
DIAG = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location(
    "kf", os.path.join(DIAG, 'kfree_wmw', 'run_kfree.py'))
KF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KF)

from repro.core.data import Whitening, plant_targets
from repro.core.metrics import auc_safe, dr_at_fpr
from repro.core.seeding import seed_all
from repro.models.darts.model import _NeighborDenoiser

OUT_JSON = os.path.join(HERE, os.environ.get(
    'OUT', 'results_nb_darts_robust.json'))
# 08-12 EMA arm (user): raw weights oscillate wildly late (.852->.538->
# .812 within hundreds of epochs at r.5/.03). Track weight-EMA(0.999,
# per step) alongside raw at every eval. rhos = the two peaks seen:
# .5 (probe best .852@4200) and .03 (incumbent .842@6400).
RHOS = [float(x) for x in os.environ.get('RHOS', '0.5,0.03').split(',')]
SEEDS = [42, 43]
EPOCHS = 10000
EVAL_START, EVAL_EVERY = 200, 200
# 08-12 target change (user): metal (painted metal sheets, GT class 5)
# instead of the paper's bitumen (class 7); amplitude down .15 -> .075.
THETA = 0.075
TARGET_CLS = 5
EMA_DECAY = 0.999
EMA_ON = os.environ.get('EMA', '0') == '1'   # 08-12: metal arm runs WITHOUT EMA (user)

torch.set_num_threads(6)


def robust_front(tr):
    X = np.asarray(tr, np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = mad   # 08-12 user: no non-gradient clipping anywhere
    return Whitening(med.astype(np.float32),
                     np.diag(1.0 / scale).astype(np.float32))


def run_one(rho, seed):
    key = f'pavia4metal_darts_robust_r{rho}_s{seed}'
    t0 = time.time()
    sc = KF.get_scene('pavia4', need_nbr=True)
    tr, te = sc['tr'], sc['te']
    from repro.scenes import pavia_protocol as PP
    s = PP.foreign_signature(sc['data'], sc['gt'], te,
                             cls=TARGET_CLS).astype(np.float32)
    D = tr.shape[1]
    W = robust_front(tr)
    cfg = dict(KF.SP_CFG['darts'])
    sigma = float(np.sqrt(rho * float(np.mean(np.var(
        np.asarray(tr, np.float64), axis=0)))))
    seed_all(seed)
    net = _NeighborDenoiser(D, int(cfg['d_lat']), int(cfg['K']),
                            list(cfg['enc_hidden']),
                            list(cfg['score_hidden']),
                            float(np.sqrt(cfg['dsm_sigma_rho'])),
                            cfg['activation'], W).to('cpu')
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
        T = -((z_te - zb) @ s) / np.sqrt(float(s @ C @ s))
        return auc_safe(y_lab, T), float(
            dr_at_fpr(y_lab, T, fpr_list=(0.05,))['0.05'])

    ema = {k: p.detach().clone() for k, p in net.named_parameters()}

    def eval_ema():
        backup = {k: p.detach().clone() for k, p in net.named_parameters()}
        with torch.no_grad():
            for k, p in net.named_parameters():
                p.copy_(ema[k])
        auc, pd05 = evaluate()
        with torch.no_grad():
            for k, p in net.named_parameters():
                p.copy_(backup[k])
        return auc, pd05

    curve = []
    best = {'auc': -1.0}
    best_e = {'auc': -1.0}
    bar = tqdm(range(1, EPOCHS + 1), desc=key, ncols=130,
               mininterval=5.0, file=sys.stdout, ascii=True)
    for ep in bar:
        net.train()
        perm = torch.randperm(P, generator=gen)
        for i in range(0, P, B):
            sel = perm[i:i + B]
            eps = torch.randn((len(sel), D), generator=gen) * sigma
            psi = net(X[sel] + eps, N[sel])
            loss = ((psi + eps / sigma ** 2) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 08-12 user: LRao-parity clip
            opt.step()
            if EMA_ON:
                with torch.no_grad():
                    for k, p in net.named_parameters():
                        ema[k].mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
        if ep >= EVAL_START and (ep % EVAL_EVERY == 0 or ep == EPOCHS):
            net.eval()
            auc, pd05 = evaluate()
            row = {'epoch': ep, 'auc': round(auc, 4),
                   'pd05': round(pd05, 4)}
            if auc > best['auc']:
                best = {'auc': auc, 'pd05': pd05, 'epoch': ep}
            post = (f'loss={float(loss):.3g} auc={auc:.3f} '
                    f'best={best["auc"]:.3f}@{best["epoch"]}')
            if EMA_ON:
                auc_e, pd05_e = eval_ema()
                row.update(auc_ema=round(auc_e, 4),
                           pd05_ema=round(pd05_e, 4))
                if auc_e > best_e['auc']:
                    best_e = {'auc': auc_e, 'pd05': pd05_e, 'epoch': ep}
                post += (f' ema={auc_e:.3f} '
                         f'bestE={best_e["auc"]:.3f}@{best_e["epoch"]}')
            curve.append(row)
            bar.set_postfix_str(post)
    bar.close()
    out = {'auc_best': round(best['auc'], 4),
           'pd05_best': round(best['pd05'], 4),
           'best_epoch': best['epoch'],
           'theta': THETA, 'target_cls': TARGET_CLS,
           'final': curve[-1],
           'rho': rho, 'sigma_raw': round(sigma, 1), 'curve': curve,
           'sec': round(time.time() - t0)}
    if EMA_ON:
        out.update(auc_ema_best=round(best_e['auc'], 4),
                   pd05_ema_best=round(best_e['pd05'], 4),
                   best_ema_epoch=best_e['epoch'], ema_decay=EMA_DECAY)
    res = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    res[key] = out
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(f'[{key}] best_auc={best["auc"]:.3f}@{best["epoch"]} '
          f'pd05={best["pd05"]:.3f} final={curve[-1]} ({out["sec"]}s)',
          flush=True)


if __name__ == '__main__':
    done = set(json.load(open(OUT_JSON)).keys()) if os.path.exists(OUT_JSON) else set()
    tasks = [(r, sd) for r in RHOS for sd in SEEDS
             if f'pavia4metal_darts_robust_r{r}_s{sd}' not in done]
    print(f'{len(tasks)} tasks, {EPOCHS} ep, sequential (6 threads)', flush=True)
    KF.get_scene('pavia4', need_nbr=True)
    for r, sd in tasks:
        run_one(r, sd)
    print('ALL DONE', flush=True)
