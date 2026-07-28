"""tsp_repro.camera_ready_tables — Pavia + San Diego performance tables from
the camera-ready checkpoints (Colab- and local-friendly).

Inputs: an artifacts directory with the layout of the `camera-ready-v1`
release asset (github.com/michaelpiro/final-paper-experiment):

    ckpt_spatial/ours_<scene>_seed<k>.pt, lrao_<scene>.pt
    spatial_sweep_<scene>/ckpt_deep/<DET>__<scene>__seed<k>.pt
    spatial_sweep_<scene>/scores__<scene>__seed<k>__additive__0.15.npz
    spatial_pavia/multiseed_*/seeds/compare_*/foreign/metrics.json

Criteria (the paper's algorithm, unchanged): theta=0.15, 5 seeds, thresholds
= empirical upper quantiles of the TRAIN-background scores at nominal 0.05.
Pavia: the 6 core rows are reused from the run_multiseed metrics (identical
to the printed Table 1, incl. per-class Pfa); AMF-global, LRao, and the 4
deep rows are computed from checkpoints. San Diego: all 12 rows computed
(no land-cover labels -> no per-class columns).

Open decision: the fixed-LRao row is THIS run's reproducible recipe
(Pavia AUC ~0.695); the archived July recipe gave 0.744.

Usage (Colab):
    from tsp_repro.camera_ready_tables import make_all
    make_all(art_dir='camera_ready_ckpts', out_dir='tables_out')
"""

import glob
import json
import os

import numpy as np

import tsp_repro  # noqa: F401
import src.spatial as SP  # noqa: F401  (normalizers via spatial_camera_ready)
from sklearn.metrics import roc_auc_score
from src.detectors import amf, amf_local, dsm_additive, gmm_glrt_levin_additive
from src.metrics import cfar_threshold, dr_at_fpr, partial_auc, per_class_fpr
from src.models import score_nmlp_additive
from tsp_repro import registry as RG
from tsp_repro import spatial_camera_ready as SC
from tsp_repro.iid_camera_ready import _DEEP

SEEDS = [42, 43, 44, 45, 46]
TH = 0.15
ALPHA = 0.05
DEEP = ['THANTD', 'HTDNet', 'TSTTD', 'OSVAE']
ORDER = ['DART', 'DART-CFAR', 'DARTS', 'DARTS-CFAR', 'AMF-global',
         'AMF-local', 'GMM-Levin', 'LRao'] + DEEP


def _train_scores(scene, models, lrao, deep_states, cfg, device):
    tr, sig = scene['tr'], scene['sig']
    k = int(cfg['k']); lam = float(cfg.get('cfar_lam', 0.1))
    if '_trw' not in scene:
        _, scene['_trw'] = SC._windows(tr, scene['tr_shape'], k, device)
        wA = (int(cfg.get('amf_local_window') or 15) if scene['name'] == 'pavia4'
              else RG.amf_local_window(tr.shape[1]))
        _, scene['_trw_amf'] = SC._windows(tr, scene['tr_shape'], wA, device)
    out = {}
    out['DART'] = dsm_additive(tr, tr, models['dsm'], sig)
    out['DART-CFAR'] = SP._cfar_normalize_map(
        out['DART'], scene['tr_shape'], bg=int(cfg.get('dsm_cfar_window', 5)),
        guard=int(cfg.get('dsm_cfar_guard', 1)), cfar_lam=lam)
    darts = score_nmlp_additive(models['nmlp'], tr, scene['_trw'], tr,
                                scene['_trw'], sig)
    out['DARTS'] = darts
    out['DARTS-CFAR'] = SP._knn_fisher_normalize(
        darts, models['nmlp'], tr, scene['_trw'], scene['tr_shape'], k,
        cfar_lam=lam, use_topk=bool(cfg.get('cfar_fisher_use_topk', False)),
        win=cfg.get('sdsm_cfar_window') or None,
        guard=int(cfg.get('sdsm_cfar_guard', 1)))
    out['AMF-global'] = amf(tr, tr, sig, eig_floor=0.0)
    out['AMF-local'] = amf_local(tr, scene['_trw_amf'], sig, device=device,
                                 loading=0.0)
    out['GMM-Levin'] = gmm_glrt_levin_additive(tr, tr, sig, p_steps=50)
    out['LRao'] = RG.score_lrao(lrao, dict(tr=tr, _sig=sig), tr, device)
    for n, st in deep_states.items():
        out[n] = _DEEP[n][1](st, tr.astype(np.float64), sig, device)
    return out


def _row(te_sc, labels, tr_sc, te_gt=None):
    y = labels
    thr = cfar_threshold(np.asarray(tr_sc, float), target_fpr=ALPHA)
    sc = np.asarray(te_sc, float)
    r = dict(pauc=partial_auc(y, sc, fpr_max=ALPHA),
             auc=roc_auc_score(y, sc),
             pd05=dr_at_fpr(y, sc, fpr_list=(ALPHA,))[str(ALPHA)],
             pd_cfar=float((sc[y == 1] > thr).mean()),
             pfa=float((sc[y == 0] > thr).mean()))
    if te_gt is not None:
        pcf = per_class_fpr(sc, y, te_gt, thr)
        vals = list(pcf.values())
        r.update(pfa_avg=float(np.nanmean(vals)),
                 pfa_max=float(np.nanmax(vals)),
                 unlab=pcf.get('unlabeled', 0.0),
                 asph=pcf.get('asphalt', 0.0),
                 trees=pcf.get('trees', 0.0))
    return r


def compute_scene(art, scn, dets, te_gt=None, device=None):
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = SC.spatial_cfg(device)
    scene = SC.build_scene(scn)
    ck_dir = os.path.join(art, 'ckpt_spatial')
    sweep = os.path.join(art, f'spatial_sweep_{scn}')
    lrao = SC.fit_scene_lrao(scene, ck_dir, device)
    acc = {d: [] for d in dets}
    for seed in SEEDS:
        models = SC.fit_scene_models(scene, cfg, seed, ck_dir, device)
        deep_states = {n: _DEEP[n][0](scene['tr'].astype(np.float64),
                                      scene['sig'], seed,
                                      os.path.join(sweep, 'ckpt_deep',
                                                   f'{n}__{scn}__seed{seed}.pt'),
                                      device)
                       for n in DEEP if n in dets}
        trs = _train_scores(scene, models, lrao, deep_states, cfg, device)
        ref = np.load(os.path.join(
            sweep, f'scores__{scn}__seed{seed}__additive__{TH}.npz'))
        for d in dets:
            acc[d].append(_row(ref[d], ref['labels'], trs[d], te_gt))
        print(f'[{scn}] seed {seed} done', flush=True)
    return {d: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            for d, rs in acc.items()}


def pavia_core_rows(art):
    rows = {}
    for p in sorted(glob.glob(os.path.join(
            art, 'spatial_pavia', 'multiseed_*', 'seeds', 'compare_*',
            'foreign', 'metrics.json'))):
        for r in json.load(open(p))['rows']:
            rows.setdefault(r['Detector'], []).append(r)
    out = {}
    remap = {'AMF': 'AMF-local'}
    for det, rs in rows.items():
        g = lambda k: float(np.mean([x[k] for x in rs]))
        out[remap.get(det, det)] = dict(
            pauc=g('pAUC@0.05'), auc=g('AUC'), pd05=g('Pd@Pfa=0.05'),
            pd_cfar=g('Pd_cfar'), pfa=g('Pfa_avg'), pfa_avg=g('Pfa_avg'),
            pfa_max=g('Pfa_max'), unlab=g('Pfa[unlabeled]'),
            asph=g('Pfa[asphalt]'), trees=g('Pfa[trees]'))
    return out


def _emit(rows, dets, cols, heads, caption, label, path):
    L = ['% Auto-generated by tsp_repro.camera_ready_tables',
         '\\begin{table*}[t]', '\\centering', f'\\caption{{{caption}}}',
         f'\\label{{{label}}}', '\\small', '\\setlength{\\tabcolsep}{4pt}',
         '\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
         'Detector & ' + ' & '.join(heads) + '\\\\', '\\midrule']
    for d in dets:
        if d not in rows:
            continue
        L.append(d + ' & ' + ' & '.join(f'{rows[d][c]:.3f}' for c in cols)
                 + '\\\\')
        if d == 'LRao':
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    open(path, 'w').write('\n'.join(L))
    md = ['| Detector | ' + ' | '.join(heads) + ' |',
          '|' + '---|' * (len(cols) + 1)]
    for d in dets:
        if d in rows:
            md.append('| ' + d + ' | '
                      + ' | '.join(f'{rows[d][c]:.3f}' for c in cols) + ' |')
    open(path.replace('.tex', '.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('wrote', path, '\n')


def make_all(art_dir, out_dir='tables_out', device=None):
    os.makedirs(out_dir, exist_ok=True)
    from tsp_repro.protocol import _load_pavia
    _, gt = _load_pavia()
    te_gt = np.asarray(gt, int)[419:508, 250:334].ravel()

    pav = pavia_core_rows(art_dir)
    pav.update(compute_scene(art_dir, 'pavia4', ['AMF-global', 'LRao'] + DEEP,
                             te_gt=te_gt, device=device))
    cols = ['pauc', 'auc', 'pd05', 'pd_cfar', 'pfa_avg', 'pfa_max',
            'unlab', 'asph', 'trees']
    heads = ['pAUC$_{0.05}$', 'AUC', '$P_d$@0.05', '$P_d^{\\mathrm{cfar}}$',
             '$\\hat P_{fa}$', '$\\hat P_{fa}^{\\max}$', 'unlab', 'asph',
             'trees']
    _emit(pav, ORDER, cols, heads,
          'Spatial detection on Pavia University (scenario 4, bitumen, '
          '$\\theta=0.15$, 5 seeds). Thresholds are empirical upper quantiles '
          'of the training-background scores (nominal $0.05$).',
          'tab:spatial', os.path.join(out_dir, 'table_pavia.tex'))

    for scn, lab in (('sandiego', 'tab:sd1'), ('sandiego2', 'tab:sd2')):
        rows = compute_scene(art_dir, scn, ORDER, device=device)
        nm = 'San Diego I' if scn == 'sandiego' else 'San Diego II'
        _emit(rows, ORDER, ['pauc', 'auc', 'pd05', 'pd_cfar', 'pfa'],
              ['pAUC$_{0.05}$', 'AUC', '$P_d$@0.05',
               '$P_d^{\\mathrm{cfar}}$', '$\\hat P_{fa}$'],
              f'Spatial detection on {nm} (real aircraft signature, '
              '$\\theta=0.15$, 5 seeds). Same criteria as the Pavia table; '
              'no land-cover labels, hence no per-class rates.',
              lab, os.path.join(out_dir, f'table_{scn}.tex'))
