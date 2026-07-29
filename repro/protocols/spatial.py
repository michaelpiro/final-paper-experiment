"""Spatial protocol: scene x seed x theta sweep over all 12 detectors.

For each (scene, seed): train/resume DART + DARTS (per seed), LRao (per seed,
validation early stopping), the four deep baselines (per seed); score every
planted test set of the theta grid plus the clean TRAIN box (for CFAR
thresholds); save raw scores npz per (seed, theta), per-seed metrics (AUC,
pAUC, Pd@alpha, Pd_cfar, Pfa — per-class Pfa on Pavia), and theta_results.json.

Planting is the published convention (plant_targets, per-seed, edge_guard) so
labels are bit-identical to the archived camera-ready sweep npz files.

Everything configurable comes from configs/spatial.yaml; nothing is hardcoded.
"""
import json
import os

import numpy as np
import torch
import yaml

from repro import scenes
from repro.core.data import extract_neighborhoods, plant_targets
from repro.core.metrics import (cfar_threshold, dr_at_fpr, partial_auc,
                                per_class_fpr)
from repro.core.metrics import auc_safe
from repro.models.classical import AMF, AMFLocal, GMMLevin
from repro.models.dart import DART
from repro.models.darts import DARTS
from repro.models.lrao import LRao
from repro.models.deep import REGISTRY as DEEP

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(os.path.dirname(_HERE), 'configs', 'spatial.yaml')


def load_cfg(path=None, **overrides):
    cfg = yaml.safe_load(open(path or DEFAULT_CFG))
    cfg.update(overrides)
    return cfg


def _windows(flat, shape, k, device):
    H, W = shape
    img = torch.tensor(np.asarray(flat, np.float32).reshape(H, W, -1),
                       device=device)
    pix, nbr = extract_neighborhoods(img, k)
    return pix.cpu().numpy(), nbr.cpu().numpy()


def _fit_all(scene, cfg, seed, ckpt_dir, device):
    """Train or resume every trainable detector for one (scene, seed)."""
    name = scene['name']
    models = {}
    models['DART'] = DART(cfg['dart']).fit(
        scene['tr'], seed, device,
        ckpt=os.path.join(ckpt_dir, f'dart_{name}_seed{seed}.pt'))
    if '_tr_nbr' not in scene:
        _, scene['_tr_nbr'] = _windows(scene['tr'], scene['tr_shape'],
                                       int(cfg['k']), device)
    darts_ck = os.path.join(ckpt_dir, f'darts_{name}_seed{seed}.pt')
    if getattr(models['DART'], 'resumed', False) and not os.path.exists(darts_ck):
        print('    [warn] DARTS will train but DART was resumed: the RNG '
              'stream cannot match the published run (clear dart_* ckpts '
              'to retrain both in the published order).', flush=True)
    # reseed=False: published RNG protocol (DARTS inherits the post-DART
    # stream; window extraction in between consumes no RNG)
    models['DARTS'] = DARTS(cfg['darts']).fit(
        scene['tr'], scene['_tr_nbr'], seed, device, ckpt=darts_ck,
        reseed=False)
    models['LRao'] = LRao(cfg['lrao']).fit(
        scene['tr'], seed, device,
        run_dir=os.path.join(ckpt_dir, f'lrao_{name}_seed{seed}'))
    for dname in cfg.get('deep_detectors', []):
        # deep baselines: load the bundled published checkpoints unless
        # retrain_deep is set (training them takes hours)
        ck = os.path.join(ckpt_dir, f'{dname}_{name}_seed{seed}.pt')
        pre = cfg.get('deep_pretrained')
        if pre and not bool(cfg.get('retrain_deep', False)):
            cand = os.path.join(pre, f'{dname}_{name}_seed{seed}.pt')
            if os.path.exists(cand):
                ck = cand
        models[dname] = DEEP[dname](cfg['deep']).fit(
            scene['tr'].astype(np.float64), scene['sig'], seed, device,
            ckpt=ck)
    return models


def score_all(scene, models, planted, cfg, device, is_train_box=False):
    """Every detector on one pixel set. Neighbor windows come from the CLEAN
    image (the published convention); planted values enter as center pixels."""
    tr, sig = scene['tr'], scene['sig']
    shape = scene['tr_shape'] if is_train_box else scene['te_shape']
    key = '_tr' if is_train_box else '_te'
    k = int(cfg['k'])
    lam = float(cfg['cfar_lam'])
    clean = tr if is_train_box else scene['te']
    if key + '_nbr' not in scene:
        _, scene[key + '_nbr'] = _windows(clean, shape, k, device)
    amf_local = AMFLocal(cfg)
    wA = amf_local.resolved_window(tr.shape[1])
    if key + '_nbr_amf' not in scene:
        _, scene[key + '_nbr_amf'] = _windows(clean, shape, wA, device)
    nbr = scene[key + '_nbr']

    out = {}
    out['DART'] = models['DART'].score(planted, tr, sig)
    out['DART-CFAR'] = DART.local_moment_normalize(
        out['DART'], shape, int(cfg['dart_cfar_window']),
        int(cfg['dart_cfar_guard']), cfar_lam=lam)
    out['DARTS'] = models['DARTS'].score(planted, nbr, tr,
                                         scene['_tr_nbr'], sig)
    out['DARTS-CFAR'] = DARTS.local_moment_normalize(
        out['DARTS'], shape, win=int(cfg.get('darts_cfar_window') or k),
        guard=int(cfg.get('darts_cfar_guard', 1)), cfar_lam=lam)
    out['AMF-global'] = AMF(cfg).fit(tr).score(planted, sig)
    out['AMF-local'] = amf_local.score(planted, scene[key + '_nbr_amf'],
                                       sig, device=device)
    out['GMM-Levin'] = GMMLevin(cfg).fit(tr).score(planted, sig)
    tr_fit = tr[models['LRao'].fit_idx]            # LRao's own fit subset
    out['LRao'] = models['LRao'].score(planted, tr_fit, sig)
    for dname in cfg.get('deep_detectors', []):
        out[dname] = models[dname].score(planted.astype(np.float64), sig,
                                         device)
    return out


def _row(labels, sc, thr, alpha, te_gt=None):
    y, sc = np.asarray(labels), np.asarray(sc, float)
    r = dict(auc=auc_safe(y, sc),
             pauc=float(partial_auc(y, sc, fpr_max=alpha)),
             pd05=float(dr_at_fpr(y, sc, fpr_list=(alpha,))[str(alpha)]),
             pd_cfar=float((sc[y == 1] > thr).mean()),
             pfa=float((sc[y == 0] > thr).mean()))
    if te_gt is not None:
        pcf = per_class_fpr(sc, y, te_gt, thr)
        vals = list(pcf.values())
        r.update(pfa_avg=float(np.nanmean(vals)), pfa_max=float(np.nanmax(vals)),
                 unlab=float(pcf.get('unlabeled', 0.0)),
                 asph=float(pcf.get('asphalt', 0.0)),
                 trees=float(pcf.get('trees', 0.0)))
    return r


def run_scene(scene_name, cfg=None, out_root='results/spatial', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = cfg if isinstance(cfg, dict) else load_cfg(cfg)
    out_dir = os.path.join(out_root, scene_name)
    ckpt_dir = os.path.join(out_root, 'ckpt')
    sc_dir = os.path.join(out_dir, 'scores')
    os.makedirs(sc_dir, exist_ok=True)
    scene = scenes.build(scene_name, cfg)
    te_gt = None
    if scene_name == 'pavia4':
        r0, r1, c0, c1 = scene['test_box']
        te_gt = np.asarray(scene['gt'], int)[r0:r1, c0:c1].ravel()
    alpha = float(cfg['alpha'])
    frac, guard = float(cfg['target_fraction']), int(cfg['edge_guard'])
    thetas = [float(t) for t in cfg['thetas']]

    metrics = {}
    for seed in [int(s) for s in cfg['seeds']]:
        print(f'[{scene_name}] seed {seed}', flush=True)
        models = _fit_all(scene, cfg, seed, ckpt_dir, device)
        tr_scores = score_all(scene, models, scene['tr'], cfg, device,
                              is_train_box=True)
        np.savez_compressed(
            os.path.join(sc_dir, f'train__{scene_name}__seed{seed}.npz'),
            **{d: np.asarray(s, np.float32) for d, s in tr_scores.items()})
        thr = {d: cfar_threshold(np.asarray(s, float), target_fpr=alpha)
               for d, s in tr_scores.items()}
        for th in thetas:
            planted, labels, _ = plant_targets(
                scene['te'], scene['sig'], th, frac, model='additive',
                seed=seed, spatial_shape=scene['te_shape'], edge_guard=guard)
            planted = planted.astype(np.float32)
            det = score_all(scene, models, planted, cfg, device)
            np.savez_compressed(
                os.path.join(sc_dir,
                             f'scores__{scene_name}__seed{seed}__additive__{th}.npz'),
                labels=np.asarray(labels, np.int8),
                **{d: np.asarray(s, np.float32) for d, s in det.items()})
            for d, s in det.items():
                row = _row(labels, s, thr[d], alpha, te_gt)
                metrics.setdefault(str(th), {}).setdefault(str(seed), {})[d] = row
                print(f'[{scene_name}] seed{seed} th={th} {d:10s} '
                      f'AUC={row["auc"]:.4f}', flush=True)
            with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
                json.dump(dict(scene=scene_name, thetas=thetas, alpha=alpha,
                               rows=metrics), f, indent=1)
    return metrics


def summarize(out_root='results/spatial', scene_names=None, table_theta=0.15):
    """Markdown AUC grid per scene + the table row at table_theta."""
    for scn in scene_names or ('pavia4', 'sandiego', 'sandiego2'):
        p = os.path.join(out_root, scn, 'metrics.json')
        if not os.path.exists(p):
            print(f'[{scn}] no metrics yet')
            continue
        m = json.load(open(p))
        rows = m['rows']
        ths = [str(t) for t in m['thetas'] if str(t) in rows]
        dets = list(next(iter(next(iter(rows.values())).values())).keys())
        print(f'\n=== {scn} — AUC mean±std over seeds ===')
        print('| Detector | ' + ' | '.join(f'θ={t}' for t in ths) + ' |')
        print('|' + '---|' * (len(ths) + 1))
        for d in dets:
            cells = []
            for t in ths:
                v = [rows[t][s][d]['auc'] for s in rows[t]]
                cells.append(f'{np.mean(v):.3f}±{np.std(v):.3f}')
            print(f'| {d} | ' + ' | '.join(cells) + ' |')
        t = str(table_theta)
        if t in rows:
            print(f'--- table rows (θ={table_theta}) ---')
            for d in dets:
                keys = rows[t][next(iter(rows[t]))][d].keys()
                agg = {k: float(np.mean([rows[t][s][d][k] for s in rows[t]]))
                       for k in keys}
                print(f'{d:10s} ' + '  '.join(f'{k}={v:.3f}'
                                              for k, v in agg.items()))
