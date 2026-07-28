"""tsp_repro.spatial_camera_ready — camera-ready SPATIAL experiments.

Division of labor:
  - The Pavia Table-1 run itself is the ORIGINAL pipeline: call
    src.spatial.run_multiseed with configs/spatial.yaml (this module does not
    reimplement it; its per-seed summary tables are printed by run_detection).
  - This module adds what the original pipeline does not have:
      * San Diego I-II spatial scenes (archive-exact boxes + median-norm
        aircraft signature via colab_deep.sd_protocol),
      * the amplitude sweep (same THETAS as the IID sweep),
      * the fixed LRao row (robust normalization, trained once per scene),
      * a global (non-local) AMF row, labeled 'AMF-global' (Table 1's row
        called 'AMF' is the LOCAL one - labels here are unambiguous),
      * the four deep baselines (same secondary pixels + signature),
    all scored with the ORIGINAL src.spatial scoring/normalization functions
    (dsm_additive, score_nmlp_additive, _cfar_normalize_map,
    _knn_fisher_normalize, amf_local, gmm_glrt_levin_additive) under the
    ORIGINAL protocol conventions: targets planted at test-box centers with
    src.data.plant_targets (pool-based count, edge_guard), neighbor windows
    taken from the clean image (as in run_detection).

AMF-local window: the config's 15x15 on Pavia (224 >= 2D for D=103); the
dimension-aware 21x21 on San Diego (D=189), where an unloaded 15x15 local SCM
is sample-starved (n/D=1.2) and collapses for reasons unrelated to the
detector concept.

Pavia models are REUSED from the run_multiseed checkpoints (models.pt per
seed) when a results dir is given, so nothing trains twice.
"""

import glob
import json
import os

import numpy as np
import torch
import yaml

import tsp_repro  # noqa: F401
import src.iid as iid                      # plotting + metric helpers
import src.spatial as SP
from src.data import plant_targets
from src.detectors import (amf, amf_local, dsm_additive,
                           gmm_glrt_levin_additive)
from src.models import score_nmlp_additive

from colab_deep import paper_protocol as PP
from colab_deep import sd_protocol as SDP
from tsp_repro import registry as RG
from tsp_repro.iid_camera_ready import _DEEP, THETAS, zip_and_download  # noqa: F401
from tsp_repro.protocol import _load_pavia

_HERE = os.path.dirname(os.path.abspath(__file__))
SPATIAL_CFG_PATH = os.path.join(_HERE, 'configs', 'spatial.yaml')

PAVIA_TRAIN_BOX = [85, 193, 207, 306]
PAVIA_TEST_BOX = [419, 508, 250, 334]
SEEDS = [42, 43, 44, 45, 46]

OUR_DETS = ['DART', 'DART-CFAR', 'DARTS', 'DARTS-CFAR',
            'AMF-global', 'AMF-local', 'GMM-Levin', 'LRao']


def spatial_cfg(device='cpu', **overrides):
    cfg = yaml.safe_load(open(SPATIAL_CFG_PATH))
    cfg['device'] = device
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Scenes (archive-exact; fingerprints asserted)
# ---------------------------------------------------------------------------
def build_scene(name):
    """dict(tr, tr_shape, te, te_shape, sig, data) — raw float32 arrays."""
    if name == 'pavia4':
        data, gt = _load_pavia()
        tb = PP.side_crop(PAVIA_TRAIN_BOX, PP.N_BUDGET)
        tr = PP.crop(data, tb)
        te = PP.crop(data, PAVIA_TEST_BOX)
        sig = PP.foreign_signature(data, gt, te)
        tr_shape = (tb[1] - tb[0], tb[3] - tb[2])
        te_shape = (PAVIA_TEST_BOX[1] - PAVIA_TEST_BOX[0],
                    PAVIA_TEST_BOX[3] - PAVIA_TEST_BOX[2])
        assert len(tr) == 4026 and abs(np.linalg.norm(sig) - 14357.9) < 1.0
    else:
        sc = SDP.build(name)               # asserts its own fingerprints
        tr, te, sig, te_shape = sc['tr'], sc['te'], sc['sig'], sc['te_shape']
        rj = json.load(open(SDP._find(f'{name}_regions.json')))
        tb = rj['train_box']
        tr_shape = (tb[1] - tb[0], tb[3] - tb[2])
        data = sc['data']
    print(f'[{name}] train {len(tr)} px {tr_shape}  test {len(te)} px '
          f'{te_shape}  ||s||={np.linalg.norm(sig):.1f}', flush=True)
    return dict(name=name, tr=np.asarray(tr, np.float32), tr_shape=tr_shape,
                te=np.asarray(te, np.float32), te_shape=te_shape,
                sig=np.asarray(sig, np.float32), data=data)


def _windows(flat, shape, k, device):
    from src.data import extract_neighborhoods
    H, W = shape
    img = torch.tensor(flat.reshape(H, W, -1), dtype=torch.float32,
                       device=device)
    pix, nbr = extract_neighborhoods(img, k)
    return pix.cpu().numpy(), nbr.cpu().numpy()


# ---------------------------------------------------------------------------
# Model fitting (original trainers, checkpoint resume; Pavia can reuse
# run_multiseed's models.pt)
# ---------------------------------------------------------------------------
def _find_multiseed_ckpt(reuse_dir, seed):
    for cfg_p in glob.glob(os.path.join(reuse_dir, '**', 'config.yaml'),
                           recursive=True):
        try:
            c = yaml.safe_load(open(cfg_p))
        except Exception:
            continue
        if int(c.get('seed', -1)) == int(seed):
            mp = os.path.join(os.path.dirname(cfg_p), 'models.pt')
            if os.path.exists(mp):
                return mp
    return None


def fit_scene_models(scene, cfg, seed, ckpt_dir, device, reuse_dir=None):
    """Train (or load) DART + DARTS for one (scene, seed) with the ORIGINAL
    trainers and the paper budgets from configs/spatial.yaml."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ck = os.path.join(ckpt_dir, f"ours_{scene['name']}_seed{seed}.pt")
    D = scene['tr'].shape[1]
    if os.path.exists(ck):
        blob = torch.load(ck, map_location='cpu', weights_only=False)
        models = SP._build_models_from_ckpt(blob, D, cfg, device)
        print(f'  resumed {ck}', flush=True)
        return models
    if reuse_dir:
        mp = _find_multiseed_ckpt(reuse_dir, seed)
        if mp:
            blob = torch.load(mp, map_location='cpu', weights_only=False)
            models = SP._build_models_from_ckpt(blob, D, blob.get('cfg', cfg),
                                                device)
            torch.save({k: models[k].state_dict() for k in models}, ck)
            print(f'  reusing run_multiseed models {mp}', flush=True)
            return models
    torch.manual_seed(seed); np.random.seed(seed)
    dsm = SP._train_dsm_best(scene['tr'], cfg, device)
    _, tr_nbr = _windows(scene['tr'], scene['tr_shape'], int(cfg['k']), device)
    torch.manual_seed(seed)
    nmlp = SP._train_nmlp_best(scene['tr'], tr_nbr, cfg, device)
    torch.save({'dsm': dsm.state_dict(), 'nmlp': nmlp.state_dict()}, ck)
    return {'dsm': dsm, 'nmlp': nmlp}


def fit_scene_lrao(scene, ckpt_dir, device, seed=42):
    """Fixed LRao (robust normalization), trained ONCE per scene."""
    ck = os.path.join(ckpt_dir, f"lrao_{scene['name']}.pt")
    pseudo = dict(tr=scene['tr'], _sig=scene['sig'])
    return RG.fit_lrao(pseudo, seed, ck, device)


# ---------------------------------------------------------------------------
# Scoring (original functions, original conventions)
# ---------------------------------------------------------------------------
def score_all(scene, models, lrao, planted, cfg, device, deep_states=()):
    """All detectors on one planted test set. Neighbor windows come from the
    CLEAN image; planted values enter as center pixels (run_detection
    convention)."""
    tr, sig = scene['tr'], scene['sig']
    te_shape = scene['te_shape']
    k = int(cfg['k'])
    lam = float(cfg.get('cfar_lam', 0.1))
    out = {}

    # neighbor windows (clean image) — cached on the scene dict
    if '_te_nbr' not in scene:
        _, scene['_te_nbr'] = _windows(scene['te'], te_shape, k, device)
        _, scene['_tr_nbr'] = _windows(tr, scene['tr_shape'], k, device)
        wA = (int(cfg.get('amf_local_window') or 15)
              if scene['name'] == 'pavia4'
              else RG.amf_local_window(tr.shape[1]))
        scene['_amf_win'] = wA
        _, scene['_te_nbr_amf'] = _windows(scene['te'], te_shape, wA, device)

    out['DART'] = dsm_additive(planted, tr, models['dsm'], sig)
    out['DART-CFAR'] = SP._cfar_normalize_map(
        out['DART'], te_shape, bg=int(cfg.get('dsm_cfar_window', 5)),
        guard=int(cfg.get('dsm_cfar_guard', 1)), cfar_lam=lam)
    darts = score_nmlp_additive(models['nmlp'], planted, scene['_te_nbr'],
                                tr, scene['_tr_nbr'], sig)
    out['DARTS'] = darts
    out['DARTS-CFAR'] = SP._knn_fisher_normalize(
        darts, models['nmlp'], planted, scene['_te_nbr'], te_shape, k,
        cfar_lam=lam, use_topk=bool(cfg.get('cfar_fisher_use_topk', False)),
        win=cfg.get('sdsm_cfar_window') or None,
        guard=int(cfg.get('sdsm_cfar_guard', 1)))
    # NOTE naming: Table 1's row called 'AMF' is the LOCAL one (run_multiseed's
    # only AMF). The sweep labels are unambiguous: AMF-global vs AMF-local.
    out['AMF-global'] = amf(planted, tr, sig,
                            eig_floor=float(cfg.get('baseline_eig_floor', 0.0)))
    out['AMF-local'] = amf_local(planted, scene['_te_nbr_amf'], sig,
                                 device=device,
                                 loading=float(cfg.get('local_scm_loading', 0.0)))
    out['GMM-Levin'] = gmm_glrt_levin_additive(
        planted, tr, sig, p_steps=int(cfg.get('gmm_steps', 50)))
    out['LRao'] = RG.score_lrao(lrao, dict(tr=tr, _sig=sig), planted, device)
    for name, state in deep_states.items():
        _, score_fn = _DEEP[name]
        out[name] = score_fn(state, planted.astype(np.float64), sig, device)
    return out


# ---------------------------------------------------------------------------
# Full scene runner: fits + theta sweep + archives + figures
# ---------------------------------------------------------------------------
def run_scene(scene_name, cfg=None, seeds=SEEDS, thetas=None, out_dir=None,
              ckpt_dir='ckpt_spatial', device=None, deep=(), reuse_dir=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = cfg or spatial_cfg(device)
    thetas = list(thetas or THETAS)
    out_dir = out_dir or f'results/spatial_sweep_{scene_name}'
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    deep = [d for d in deep if d in _DEEP]
    if deep:
        os.makedirs(os.path.join(out_dir, 'ckpt_deep'), exist_ok=True)

    scene = build_scene(scene_name)
    lrao = fit_scene_lrao(scene, ckpt_dir, device)
    results = {}
    for seed in seeds:
        models = fit_scene_models(scene, cfg, seed, ckpt_dir, device,
                                  reuse_dir=reuse_dir)
        deep_states = {}
        for name in deep:
            fit_fn, _ = _DEEP[name]
            ck = os.path.join(out_dir, 'ckpt_deep',
                              f'{name}__{scene_name}__seed{seed}.pt')
            print(f'[{scene_name}] seed{seed} fitting {name} ...', flush=True)
            deep_states[name] = fit_fn(scene['tr'].astype(np.float64),
                                       scene['sig'], int(seed), ck, device)
        for th in thetas:
            planted, labels, _ = plant_targets(
                scene['te'], scene['sig'], float(th),
                float(cfg.get('target_fraction', 0.10)), model='additive',
                seed=int(seed), spatial_shape=scene['te_shape'],
                edge_guard=int(cfg.get('edge_guard', 3)))
            planted = planted.astype(np.float32)
            det_scores = score_all(scene, models, lrao, planted, cfg, device,
                                   deep_states)
            np.savez_compressed(
                os.path.join(out_dir,
                             f'scores__{scene_name}__seed{seed}__additive__{th}.npz'),
                labels=labels.astype(np.int8),
                **{d: np.asarray(s, np.float32) for d, s in det_scores.items()})
            for det, sc in det_scores.items():
                au = iid._auc(labels, sc)
                results.setdefault(det, {}).setdefault(th, []).append(au)
                print(f'[{scene_name}] seed{seed} {det:10s} th={th}: '
                      f'AUC={au:.4f}', flush=True)
            with open(os.path.join(out_dir, 'theta_results.json'), 'w') as f:
                json.dump(dict(scene=scene_name, thetas=thetas,
                               auc={d: {str(t): v for t, v in r.items()}
                                    for d, r in results.items()}), f, indent=1)

    mu = {d: [float(np.mean(results[d][t])) for t in thetas] for d in results}
    sd = {d: [float(np.std(results[d][t])) for t in thetas] for d in results}
    iid._plot_vs(thetas, mu, r'target amplitude  $\theta$', 'AUC',
                 f'AUC vs amplitude  ({scene_name}, spatial)',
                 os.path.join(fig_dir, 'auc_vs_theta.pdf'), logx=True,
                 series_std=sd)
    print(f'\n[{scene_name}] sweep done -> {out_dir}', flush=True)
    return results


# ---------------------------------------------------------------------------
# The "print all results" cell
# ---------------------------------------------------------------------------
def print_all(all_results, thetas=None):
    """all_results: {scene: results dict from run_scene}. Prints a full
    markdown-style AUC grid (mean+-std over seeds) per scene."""
    thetas = list(thetas or THETAS)
    for scene, res in all_results.items():
        print(f'\n=== {scene} — AUC (mean±std over seeds) ===')
        header = '| Detector | ' + ' | '.join(f'θ={t}' for t in thetas) + ' |'
        print(header)
        print('|' + '---|' * (len(thetas) + 1))
        order = [d for d in OUR_DETS if d in res] + \
                [d for d in res if d not in OUR_DETS]
        for det in order:
            cells = []
            for t in thetas:
                v = res[det].get(t) or res[det].get(str(t))
                cells.append(f'{np.mean(v):.3f}±{np.std(v):.3f}'
                             if v else '--')
            print(f'| {det} | ' + ' | '.join(cells) + ' |')
