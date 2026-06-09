"""
make_pauc_figures.py — Paper-quality P_det / pAUC figures from saved checkpoints.

Reloads models + pipelines saved during run_sweep.py, re-scores the test data
(using the SAME random seeds → identical splits / plants), and produces:

  1. P_det at P_fa = 0.1 vs n_train  (replaces full AUC per Ami's email)
  2. pAUC (integral 0→0.1 of ROC / 0.1) vs n_train
  3. P_det at multiple P_fa values vs n_train  (small-multiples)
  4. Test-statistic distributions per background class  (shows CFAR property)

Usage (from repo root):
    .venv/bin/python experiments/honest_pipeline/make_pauc_figures.py \\
        --single_n  experiments/honest_pipeline/results/sweep_single_20260608_165625 \\
        --multi_n   experiments/honest_pipeline/results/sweep_multi_n_latest \\
        --single_rho experiments/honest_pipeline/results/sweep_single_20260608_165812 \\
        --multi_rho  experiments/honest_pipeline/results/sweep_multi_20260608_161636 \\
        --out        experiments/honest_pipeline/results/pauc_figures
"""

import argparse, os, sys, json, pickle
from pathlib import Path

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP); sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import yaml
import scipy.io
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
from tqdm import tqdm

from pipeline import HonestDetectionPipeline, amf_score, amf_replacement_score
from final_paper_experiments.data_utils import compute_sigma_from_data
from dsm_model import ScoreNet, compute_scores, compute_lfi_detector_scores_mode2

# Try to import GMM (multiclass only)
try:
    # GMM-Levin (cluster-based product-GMM GLRT). The honest detector estimates
    # the fill factor by ML grid search; the oracle reference uses the SAME
    # fitted density with the true fill factor plugged in (score(..., oracle_p=)).
    from final_paper_experiments.baselines.gmm_glrt_levin import GMMGLRTLevin
    HAS_GMM = True
except ImportError:
    HAS_GMM = False
    print("WARNING: GMM modules not found, skipping GMM-GLRT detectors.")

CLS_NAMES = {1:'asphalt', 2:'meadows', 3:'gravel', 4:'trees', 5:'metal_sheets',
             6:'bare_soil', 7:'bitumen', 8:'bricks', 9:'shadows'}

# ── Paper-quality style ───────────────────────────────────────────────────────
_STYLE = {
    'font.family':     'serif',
    'font.size':       9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid':       True,
    'grid.alpha':      0.3,
    'grid.linestyle':  '--',
    'grid.linewidth':  0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi':      200,
}

DET_COLORS = {
    'AMF':         '#1f77b4',
    'AMF-rep':     '#aec7e8',
    'DSM':         '#ff7f0e',
    'LRao':        '#2ca02c',
    'GMM-GLRT':    '#9467bd',
    'GMM-GLRT-G':  '#e377c2',
}
DET_MARKERS = {
    'AMF':         'o',
    'AMF-rep':     's',
    'DSM':         's',
    'LRao':        'D',
    'GMM-GLRT':    'v',
    'GMM-GLRT-G':  'P',
}
DET_LABELS = {
    'AMF':         'AMF',
    'AMF-rep':     'AMF-rep',
    'DSM':         'DSM (ours)',
    'LRao':        'LRao-IID',
    'GMM-GLRT':    'GMM-Levin',            # honest: ML grid-search of fill factor
    'GMM-GLRT-G':  'GMM-Levin (oracle)',   # clairvoyant: true fill factor plugged in
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def partial_auc_normalized(fpr: np.ndarray, tpr: np.ndarray,
                            fpr_max: float = 0.1) -> float:
    """Partial AUC from P_fa=0 to fpr_max, normalized to [0,1].

    Returns pAUC / fpr_max (= 1.0 for perfect detector, 0.5 for random).
    Uses trapezoidal integration on the interpolated ROC.
    """
    # Keep only points up to fpr_max (plus the interpolated point at fpr_max)
    mask = fpr <= fpr_max
    fpr_clip = fpr[mask]
    tpr_clip = tpr[mask]
    # If ROC doesn't reach fpr_max, interpolate the endpoint
    if fpr[-1] < fpr_max:
        fpr_clip = np.append(fpr_clip, fpr_max)
        tpr_clip = np.append(tpr_clip, tpr[-1])
    elif not np.any(fpr == fpr_max):
        # interpolate
        idx = np.searchsorted(fpr, fpr_max)
        if idx > 0 and idx < len(fpr):
            t = (fpr_max - fpr[idx-1]) / max(fpr[idx] - fpr[idx-1], 1e-12)
            tpr_at = tpr[idx-1] + t * (tpr[idx] - tpr[idx-1])
        else:
            tpr_at = tpr[min(idx, len(tpr)-1)]
        fpr_clip = np.append(fpr_clip, fpr_max)
        tpr_clip = np.append(tpr_clip, tpr_at)
    area = np.trapz(tpr_clip, fpr_clip)
    return float(area / fpr_max)


def dr_at_fpr(fpr: np.ndarray, tpr: np.ndarray,
              target_fpr: float) -> float:
    """Interpolated detection rate at a given false alarm rate."""
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    if target_fpr >= fpr[-1]:
        return float(tpr[-1])
    idx = np.searchsorted(fpr, target_fpr)
    if idx == 0:
        return float(tpr[0])
    t = (target_fpr - fpr[idx-1]) / max(fpr[idx] - fpr[idx-1], 1e-12)
    return float(tpr[idx-1] + t * (tpr[idx] - tpr[idx-1]))


def roc_safe(labels, scores):
    """Compute ROC curve, returning (fpr, tpr, auc)."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr, tpr, float(sk_auc(fpr, tpr))
    except Exception:
        return np.array([0., 1.]), np.array([0., 1.]), float('nan')


# ── Data helpers (mirror run_sweep.py exactly) ────────────────────────────────

def plant(bkg, s, amp, frac, model, seed):
    rng = np.random.RandomState(seed)
    n = len(bkg); k = int(frac * n)
    pos = rng.choice(n, k, replace=False)
    y = bkg.copy().astype(np.float32)
    lab = np.zeros(n, dtype=int); lab[pos] = 1
    if model == 'additive':
        y[pos] += amp * s
    else:
        y[pos] = (1 - amp) * y[pos] + amp * s
    return y, lab


def _score_dsm(model, tr_pca, te_pca, s, tm):
    z_tr = compute_scores(model, tr_pca)
    z_te = compute_scores(model, te_pca)
    if tm == 'additive':
        z_bar = z_tr.mean(0); C = np.cov(z_tr, rowvar=False)
        if C.ndim == 0: C = np.array([[float(C)]])
        norm = float(np.sqrt(max(float(s @ C @ s), 1e-12)))
        return -((z_te - z_bar) @ s) / norm
    else:
        psi_bar = z_tr.mean(0); d = tr_pca.shape[1]
        r_tr = ((z_tr - psi_bar) * (tr_pca - s)).sum(1)
        I_rep = max(float((r_tr**2).mean()) - d**2, 1e-12)
        r_te  = ((z_te - psi_bar) * (te_pca - s)).sum(1)
        return (r_te + d) / np.sqrt(I_rep)


def _score_lrao(model, tr_pca, te_pca, s, cfg):
    return compute_lfi_detector_scores_mode2(
        model, tr_pca, te_pca, s,
        delta_theta=cfg.get('lfi_delta_theta', 0.01),
        sigma_cutoff=cfg.get('lfi_sigma_cutoff', 1e-3))


# ── Core: rescore one run directory ──────────────────────────────────────────

def rescore_n_sweep(run_dir: str,
                    fpr_targets=(0.001, 0.01, 0.05, 0.1),
                    fpr_max_pauc: float = 0.1,
                    verbose: bool = True):
    """
    Load all saved models in `run_dir`, re-score test data, and compute
    P_det at fixed FPR values + partial AUC for each (seed, n, detector, tm).

    Returns
    -------
    results : dict with structure:
        results[tm][det][n] = {
            'pd':   {fpr: [seed_value, ...]},   # raw per-seed values
            'pauc': [seed_value, ...],
            'auc':  [seed_value, ...],
        }
    meta : dict — config, n_list, seeds, bkg_desc, rho_d (optional)
    """
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    met = json.load(open(run_dir / 'metrics.json'))

    n_list   = sorted(met['n_list'])
    seeds    = met.get('seeds', [42])
    d_list   = sorted(met.get('d_list', cfg.get('latent_dim_list', [20])))
    rho_list = sorted(met.get('rho_list', cfg.get('rho_list', [0.01])))
    amp      = float(met.get('amp_list', [cfg.get('amplitude', 0.15)])[0])
    bkg_desc = met.get('bkg_desc', '')
    d        = d_list[0]    # single d for n-sweep
    rho      = rho_list[0]  # single rho for n-sweep

    # Load dataset
    mat  = scipy.io.loadmat(cfg['dataset'])
    data = mat['data'].astype(np.float32)
    gt   = mat['map'].astype(int)
    H, W, D_raw = data.shape
    flat    = data.reshape(-1, D_raw)
    gt_flat = gt.reshape(-1)

    tcls     = cfg['target_cls']
    bkg_cls_raw = cfg.get('bkg_cls')
    if bkg_cls_raw is None:
        bkg_mask = (gt_flat != 0) & (gt_flat != tcls)
    else:
        if isinstance(bkg_cls_raw, str):
            bkg_cls_list = [int(x.strip()) for x in bkg_cls_raw.split(',')]
        elif hasattr(bkg_cls_raw, '__iter__'):
            bkg_cls_list = [int(x) for x in bkg_cls_raw]
        else:
            bkg_cls_list = [int(bkg_cls_raw)]
        bkg_mask = np.isin(gt_flat, bkg_cls_list)
    bkg_all    = flat[bkg_mask]
    bkg_labels = gt_flat[bkg_mask]
    tgt_all    = flat[gt_flat == tcls]
    t_raw      = tgt_all.mean(0).astype(np.float32)

    n_max  = max(n_list)
    n_test = cfg['test_n']
    frac   = cfg['target_fraction']

    # Determine which detectors to compute
    det_keys = list(met.keys() - {'n_list','d_list','rho_list','amp_list',
                                   'seeds','target_cls','bkg_desc','rho_d',
                                   'target_models'})
    has_gmm  = 'gmm' in det_keys and HAS_GMM
    has_gmmg = 'gmm_g' in det_keys and HAS_GMM
    if verbose:
        print(f"  n_list={n_list}  d={d}  rho={rho}  seeds={seeds}")
        print(f"  Detectors in file: {sorted(det_keys)}")

    # Initialise results (respect additive-only runs via target_models)
    tms   = tuple(met.get('target_models', ['additive', 'replacement']))
    # DSM PCA-dim variants: extra d's in d_list produce DSM-d{dd} curves
    # (Ami: 2-3 DSM variants to approach Levin). Primary d = d_list[0] = 'DSM'.
    dsm_variants = [f'DSM-d{dd}' for dd in d_list if dd != d]
    dets  = ['AMF', 'DSM', 'LRao'] + dsm_variants
    if has_gmm:  dets.append('GMM-GLRT')
    if has_gmmg: dets.append('GMM-GLRT-G')
    results = {tm: {det: {n: {'pd': {f: [] for f in fpr_targets},
                               'pauc': [], 'auc': []}
                           for n in n_list}
                    for det in dets}
               for tm in tms}

    mdl_dir = run_dir / 'models'
    # Raw per-detector scores + ROC dumped here for full reproducibility.
    raw_dir = run_dir / 'raw_scores'
    raw_dir.mkdir(exist_ok=True)
    raw_index = []   # list of {seed,key,kind,detector,tm,auc,file}

    for seed in seeds:
        if verbose: print(f"  seed={seed} ...", end='', flush=True)
        seed_mdl = mdl_dir / f'seed_{seed}'

        # Reproduce exact same bkg split (mirror run_sweep.py run_one_seed)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(bkg_all))
        bkg_tr = bkg_all[idx[:n_max]]
        bkg_te = bkg_all[idx[n_max:n_max + n_test]]
        te_labels = bkg_labels[idx[n_max:n_max + n_test]]

        # AMF global_max normaliser (computed from full bkg_tr, mirror run_sweep.py)
        gm = float(bkg_tr.max() + 1e-12)
        t_gm = t_raw / gm
        bkg_te_gm = bkg_te / gm

        # Load pipelines for every PCA dim (primary d + variants), fit on bkg_tr[:n_max]
        pipe_by_d = {}
        for dd in d_list:
            with open(seed_mdl / f'pipeline_d{dd}.pkl', 'rb') as fh:
                pipe_by_d[dd] = pickle.load(fh)
        pipe = pipe_by_d[d]                       # primary pipeline (LRao space)
        te_pca_by_d = {dd: pipe_by_d[dd].project(bkg_te) for dd in d_list}
        sadd_by_d   = {dd: pipe_by_d[dd].signature_additive(t_raw) for dd in d_list}
        srep_by_d   = {dd: pipe_by_d[dd].signature_replacement(t_raw) for dd in d_list}

        for n in tqdm(n_list, desc=f'    n', leave=False, disable=not verbose):
            tr_gm  = bkg_tr[:n] / gm
            rho_str = str(rho).replace('.', 'p')

            # --- DSM models per PCA dim (primary 'DSM' + 'DSM-d{dd}' variants) ---
            dsm_by_d, trpca_by_d = {}, {}
            for dd in d_list:
                trpca_by_d[dd] = pipe_by_d[dd].project(bkg_tr[:n])
                ck = torch.load(seed_mdl / f'dsm_rho{rho_str}_d{dd}_n{n}.pt',
                                map_location='cpu', weights_only=False)
                m = ScoreNet(dd, list(cfg['hidden_dims']), cfg['activation'])
                m.load_state_dict(ck['state_dict']); m.eval()
                dsm_by_d[dd] = m
            tr_pca = trpca_by_d[d]                 # primary-d train scores (LRao)

            # --- Load LRao model (primary d) ---
            lrao_ckpt = torch.load(seed_mdl / f'lrao_d{d}_n{n}.pt',
                                   map_location='cpu', weights_only=False)
            lrao_m = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
            lrao_m.load_state_dict(lrao_ckpt['state_dict'])
            lrao_m.eval()

            # --- GMM-GLRT: refit on bkg_tr[:n] / gm ---
            if has_gmm or has_gmmg:
                det_gmm = GMMGLRTLevin(
                    cond_tol=cfg.get('gmm_cond_tol', 1e3),
                    max_dim=cfg.get('gmm_max_dim', 5),
                    k_max=cfg.get('gmm_k_max', 5),
                    seed=seed).fit(tr_gm)

            for tm in tms:
                sig = sadd_by_d[d] if tm == 'additive' else srep_by_d[d]
                te_pca, lab = plant(te_pca_by_d[d], sig, amp, frac, tm, seed)
                te_gm,  _   = plant(bkg_te_gm,  t_gm, amp, frac, tm, seed)

                # Scores
                sc_amf  = (amf_score(tr_gm, te_gm, t_gm)
                           if tm == 'additive'
                           else amf_replacement_score(tr_gm, te_gm, t_gm))
                sc_lrao = _score_lrao(lrao_m, tr_pca, te_pca, sig, cfg)

                sc_map = {'AMF': sc_amf, 'LRao': sc_lrao}
                # DSM primary + PCA-dim variants
                for dd in d_list:
                    sg = sadd_by_d[dd] if tm == 'additive' else srep_by_d[dd]
                    te_dd, _ = plant(te_pca_by_d[dd], sg, amp, frac, tm, seed)
                    key = 'DSM' if dd == d else f'DSM-d{dd}'
                    sc_map[key] = _score_dsm(dsm_by_d[dd], trpca_by_d[dd], te_dd, sg, tm)
                if has_gmm:
                    # Honest GMM-Levin: fill factor p estimated by ML grid search.
                    sc_map['GMM-GLRT'] = det_gmm.score(
                        te_gm, t_gm, model=tm,
                        p_steps=cfg.get('gmm_p_steps', 50),
                        p_max=cfg.get('gmm_p_max', 1.0))
                if has_gmmg:
                    # Oracle reference: SAME fitted Levin product-GMM density,
                    # but the true fill factor is plugged in instead of being
                    # grid-searched. This is a genuine upper bound on GMM-Levin
                    # (identical density model, clairvoyant amplitude). Reported
                    # explicitly as an oracle — never as a fair baseline.
                    sc_map['GMM-GLRT-G'] = det_gmm.score(
                        te_gm, t_gm, model=tm, oracle_p=amp)

                for det, sc in sc_map.items():
                    fpr_r, tpr_r, auc_r = roc_safe(lab, sc)
                    bucket = results[tm][det][n]
                    bucket['auc'].append(auc_r)
                    bucket['pauc'].append(partial_auc_normalized(fpr_r, tpr_r, fpr_max_pauc))
                    for f in fpr_targets:
                        bucket['pd'][f].append(dr_at_fpr(fpr_r, tpr_r, f))

                    # --- dump raw scores + ROC (full reproducibility) ---
                    fname = f'seed{seed}_n{n}_{det}_{tm}.npz'
                    np.savez_compressed(
                        raw_dir / fname,
                        scores=np.asarray(sc, dtype=np.float32),
                        labels=np.asarray(lab, dtype=np.int8),
                        fpr=np.asarray(fpr_r, dtype=np.float32),
                        tpr=np.asarray(tpr_r, dtype=np.float32),
                        auc=np.float32(auc_r))
                    raw_index.append({'seed': int(seed), 'kind': 'n', 'key': int(n),
                                      'detector': det, 'tm': tm,
                                      'auc': float(auc_r), 'file': fname})

        if verbose: print(' done', flush=True)

    json.dump(raw_index, open(raw_dir / 'raw_index.json', 'w'), indent=2)

    meta = dict(n_list=n_list, d=d, d_list=d_list, rho=rho, seeds=seeds,
                bkg_desc=bkg_desc, target_cls=tcls, amp=amp,
                fpr_targets=list(fpr_targets), fpr_max_pauc=fpr_max_pauc,
                rho_d=met.get('rho_d', {}))
    return results, meta


def rescore_rho_sweep(run_dir: str,
                      fpr_targets=(0.001, 0.01, 0.05, 0.1),
                      fpr_max_pauc: float = 0.1,
                      verbose: bool = True):
    """Like rescore_n_sweep but sweeps over rho (x-axis = rho, fixed n)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    met = json.load(open(run_dir / 'metrics.json'))

    n_list   = sorted(met['n_list'])
    seeds    = met.get('seeds', [42])
    d_list   = sorted(met.get('d_list', cfg.get('latent_dim_list', [20])))
    rho_list = sorted(met.get('rho_list', cfg.get('rho_list', [0.01])))
    amp      = float(met.get('amp_list', [cfg.get('amplitude', 0.15)])[0])
    bkg_desc = met.get('bkg_desc', '')
    d = d_list[0]; n = n_list[0]

    mat  = scipy.io.loadmat(cfg['dataset'])
    data = mat['data'].astype(np.float32)
    gt   = mat['map'].astype(int)
    H, W, D_raw = data.shape
    flat = data.reshape(-1, D_raw); gt_flat = gt.reshape(-1)

    tcls     = cfg['target_cls']
    bkg_cls_raw = cfg.get('bkg_cls')
    if bkg_cls_raw is None:
        bkg_mask = (gt_flat != 0) & (gt_flat != tcls)
    else:
        if isinstance(bkg_cls_raw, str):
            bkg_cls_list = [int(x.strip()) for x in bkg_cls_raw.split(',')]
        elif hasattr(bkg_cls_raw, '__iter__'):
            bkg_cls_list = [int(x) for x in bkg_cls_raw]
        else:
            bkg_cls_list = [int(bkg_cls_raw)]
        bkg_mask = np.isin(gt_flat, bkg_cls_list)
    bkg_all = flat[bkg_mask]
    tgt_all = flat[gt_flat == tcls]
    t_raw   = tgt_all.mean(0).astype(np.float32)

    n_max = n; n_test = cfg['test_n']; frac = cfg['target_fraction']

    tms  = tuple(met.get('target_models', ['additive', 'replacement']))
    dets = ['AMF', 'DSM', 'LRao']
    results = {tm: {det: {rho: {'pd': {f: [] for f in fpr_targets},
                                 'pauc': [], 'auc': []}
                           for rho in rho_list}
                    for det in dets}
               for tm in tms}

    mdl_dir = run_dir / 'models'
    raw_dir = run_dir / 'raw_scores'
    raw_dir.mkdir(exist_ok=True)
    raw_index = []

    for seed in seeds:
        if verbose: print(f"  seed={seed} ...", end='', flush=True)
        seed_mdl = mdl_dir / f'seed_{seed}'

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(bkg_all))
        bkg_tr = bkg_all[idx[:n_max]]
        bkg_te = bkg_all[idx[n_max:n_max + n_test]]

        gm = float(bkg_tr.max() + 1e-12)
        t_gm = t_raw / gm
        bkg_te_gm = bkg_te / gm

        pkl_path = seed_mdl / f'pipeline_d{d}.pkl'
        with open(pkl_path, 'rb') as fh:
            pipe = pickle.load(fh)
        bkg_te_pca = pipe.project(bkg_te)
        s_add = pipe.signature_additive(t_raw)
        s_rep = pipe.signature_replacement(t_raw)
        tr_pca = pipe.project(bkg_tr[:n])
        tr_gm  = bkg_tr[:n] / gm

        # AMF only depends on n (not rho); compute once
        amf_scores_cache = {}
        for tm in tms:
            sig = s_add if tm == 'additive' else s_rep
            te_pca_pl, lab = plant(bkg_te_pca, sig, amp, frac, tm, seed)
            te_gm_pl,  _   = plant(bkg_te_gm,  t_gm, amp, frac, tm, seed)
            amf_scores_cache[(tm, 'te_pca')] = te_pca_pl
            amf_scores_cache[(tm, 'te_gm')]  = te_gm_pl
            amf_scores_cache[(tm, 'lab')]     = lab
            sc_amf = (amf_score(tr_gm, te_gm_pl, t_gm) if tm == 'additive'
                      else amf_replacement_score(tr_gm, te_gm_pl, t_gm))
            amf_scores_cache[(tm, 'amf')] = sc_amf

        # LRao only depends on n (not rho); load once
        lrao_path = seed_mdl / f'lrao_d{d}_n{n}.pt'
        lrao_ckpt = torch.load(lrao_path, map_location='cpu', weights_only=False)
        lrao_m = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
        lrao_m.load_state_dict(lrao_ckpt['state_dict'])
        lrao_m.eval()

        for rho in tqdm(rho_list, desc=f'    rho', leave=False, disable=not verbose):
            rho_str = str(rho).replace('.', 'p')
            dsm_path = seed_mdl / f'dsm_rho{rho_str}_d{d}_n{n}.pt'
            dsm_ckpt = torch.load(dsm_path, map_location='cpu', weights_only=False)
            dsm_m = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
            dsm_m.load_state_dict(dsm_ckpt['state_dict'])
            dsm_m.eval()

            for tm in tms:
                sig     = s_add if tm == 'additive' else s_rep
                te_pca  = amf_scores_cache[(tm, 'te_pca')]
                te_gm   = amf_scores_cache[(tm, 'te_gm')]
                lab     = amf_scores_cache[(tm, 'lab')]
                sc_amf  = amf_scores_cache[(tm, 'amf')]
                sc_dsm  = _score_dsm(dsm_m, tr_pca, te_pca, sig, tm)
                sc_lrao = _score_lrao(lrao_m, tr_pca, te_pca, sig, cfg)

                for det, sc in [('AMF', sc_amf), ('DSM', sc_dsm), ('LRao', sc_lrao)]:
                    fpr_r, tpr_r, auc_r = roc_safe(lab, sc)
                    bucket = results[tm][det][rho]
                    bucket['auc'].append(auc_r)
                    bucket['pauc'].append(partial_auc_normalized(fpr_r, tpr_r, fpr_max_pauc))
                    for f in fpr_targets:
                        bucket['pd'][f].append(dr_at_fpr(fpr_r, tpr_r, f))

                    fname = f'seed{seed}_rho{rho_str}_{det}_{tm}.npz'
                    np.savez_compressed(
                        raw_dir / fname,
                        scores=np.asarray(sc, dtype=np.float32),
                        labels=np.asarray(lab, dtype=np.int8),
                        fpr=np.asarray(fpr_r, dtype=np.float32),
                        tpr=np.asarray(tpr_r, dtype=np.float32),
                        auc=np.float32(auc_r))
                    raw_index.append({'seed': int(seed), 'kind': 'rho', 'key': float(rho),
                                      'detector': det, 'tm': tm,
                                      'auc': float(auc_r), 'file': fname})

        if verbose: print(' done', flush=True)

    json.dump(raw_index, open(raw_dir / 'raw_index.json', 'w'), indent=2)

    meta = dict(n=n, d=d, rho_list=rho_list, seeds=seeds,
                bkg_desc=bkg_desc, target_cls=tcls, amp=amp,
                fpr_targets=list(fpr_targets), fpr_max_pauc=fpr_max_pauc,
                rho_d=met.get('rho_d', {}))
    return results, meta


# ── Statistics helpers ────────────────────────────────────────────────────────

def _agg(vals):
    a = np.array([v for v in vals if not np.isnan(v)])
    if len(a) == 0: return float('nan'), float('nan')
    return float(a.mean()), float(a.std())


def summarise_n_sweep(results, meta, metric='pd', fpr_for_pd=0.1):
    """Aggregate per-seed values → mean ± std dict.

    Returns
    -------
    agg[tm][det][n] = {'pd': {fpr: (mean, std)}, 'pauc': (mean, std), 'auc': (mean, std)}
    """
    n_list = meta['n_list']
    tms    = list(results.keys())
    dets   = list(results[tms[0]].keys())
    agg = {}
    for tm in tms:
        agg[tm] = {}
        for det in dets:
            agg[tm][det] = {}
            for n in n_list:
                b = results[tm][det][n]
                agg[tm][det][n] = {
                    'pauc': _agg(b['pauc']),
                    'auc':  _agg(b['auc']),
                    'pd':   {f: _agg(b['pd'][f]) for f in meta['fpr_targets']},
                }
    return agg


def summarise_rho_sweep(results, meta):
    rho_list = meta['rho_list']
    tms      = list(results.keys())
    dets     = list(results[tms[0]].keys())
    agg = {}
    for tm in tms:
        agg[tm] = {}
        for det in dets:
            agg[tm][det] = {}
            for rho in rho_list:
                b = results[tm][det][rho]
                agg[tm][det][rho] = {
                    'pauc': _agg(b['pauc']),
                    'auc':  _agg(b['auc']),
                    'pd':   {f: _agg(b['pd'][f]) for f in meta['fpr_targets']},
                }
    return agg


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _band(ax, x, mu, sd, det, logx=True):
    c  = DET_COLORS.get(det, '#555')
    mk = DET_MARKERS.get(det, 'o')
    lb = DET_LABELS.get(det, det)
    pf = ax.semilogx if logx else ax.plot
    pf(x, mu, marker=mk, color=c, lw=1.8, ms=5,
       markerfacecolor=c, markeredgewidth=0.5, markeredgecolor='white', label=lb)
    ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)


def _setup_ax(ax, ylabel=''):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(direction='in')
    ax.xaxis.set_tick_params(direction='in')
    ax.set_ylim(0, 1.05)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def plot_pd_vs_n(agg, meta, out_path, main_fpr=0.1, title_prefix=''):
    """2×1 figure: P_det at main_fpr vs n (additive | replacement)."""
    n_list = meta['n_list']
    present = [tm for tm in ('additive', 'replacement') if tm in agg]
    dets   = list(agg[present[0]].keys())
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, len(present), figsize=(3.6*len(present), 3.2),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ci, tm in enumerate(present):
            ax = axes[ci]
            for det in dets:
                mu = np.array([agg[tm][det][n]['pd'][main_fpr][0] for n in n_list])
                sd = np.array([agg[tm][det][n]['pd'][main_fpr][1] for n in n_list])
                _band(ax, n_list, mu, sd, det)
            _setup_ax(ax, ylabel=f'$P_{{det}}$ @ $P_{{fa}}={main_fpr}$' if ci == 0 else '')
            ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=9)
            ax.set_title(f'{"Additive" if tm == "additive" else "Replacement"} model',
                         fontsize=9)
            if ci == len(present)-1:
                ax.legend(fontsize=7.5, framealpha=0.9,
                          loc='center left', bbox_to_anchor=(1.02, 0.5))
        bkg = meta.get('bkg_desc', '')
        tcls = CLS_NAMES.get(meta.get('target_cls', 0), '')
        rho_str = f"$\\rho$={meta.get('rho', '?')}"
        fig.suptitle(
            f"{title_prefix}  tgt={tcls}, bkg={bkg}, $\\theta$={meta.get('amp','?')}, {rho_str}",
            fontsize=8.5, y=1.01)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def plot_pauc_vs_n(agg, meta, out_path, title_prefix=''):
    """2×1 figure: pAUC (0→0.1) vs n (additive | replacement)."""
    n_list = meta['n_list']
    present = [tm for tm in ('additive', 'replacement') if tm in agg]
    dets   = list(agg[present[0]].keys())
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, len(present), figsize=(3.6*len(present), 3.2),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ci, tm in enumerate(present):
            ax = axes[ci]
            for det in dets:
                mu = np.array([agg[tm][det][n]['pauc'][0] for n in n_list])
                sd = np.array([agg[tm][det][n]['pauc'][1] for n in n_list])
                _band(ax, n_list, mu, sd, det)
            _setup_ax(ax, ylabel='pAUC (0→0.1)' if ci == 0 else '')
            ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=9)
            ax.set_title(f'{"Additive" if tm == "additive" else "Replacement"} model',
                         fontsize=9)
            if ci == len(present)-1:
                ax.legend(fontsize=7.5, framealpha=0.9,
                          loc='center left', bbox_to_anchor=(1.02, 0.5))
        bkg = meta.get('bkg_desc', '')
        tcls = CLS_NAMES.get(meta.get('target_cls', 0), '')
        fig.suptitle(f"{title_prefix}  tgt={tcls}, bkg={bkg}, $\\theta$={meta.get('amp','?')}",
                     fontsize=8.5, y=1.01)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def plot_pd_multi_fpr(agg, meta, out_path, tm='additive', title_prefix=''):
    """4-panel figure: P_det at each FPR level vs n, all detectors."""
    n_list      = meta['n_list']
    fpr_targets = meta['fpr_targets']
    dets        = list(agg[tm].keys())
    with plt.rc_context(_STYLE):
        nc = len(fpr_targets)
        fig, axes = plt.subplots(1, nc, figsize=(3.5*nc, 3.2), sharey=True)
        for ci, fpr in enumerate(fpr_targets):
            ax = axes[ci]
            for det in dets:
                mu = np.array([agg[tm][det][n]['pd'][fpr][0] for n in n_list])
                sd = np.array([agg[tm][det][n]['pd'][fpr][1] for n in n_list])
                _band(ax, n_list, mu, sd, det)
            _setup_ax(ax, ylabel='$P_{det}$' if ci == 0 else '')
            ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=9)
            ax.set_title(f'$P_{{fa}}={fpr}$', fontsize=9)
            if ci == nc-1:
                ax.legend(fontsize=7.5, framealpha=0.9,
                          loc='center left', bbox_to_anchor=(1.02, 0.5))
        bkg = meta.get('bkg_desc', '')
        tcls = CLS_NAMES.get(meta.get('target_cls', 0), '')
        fig.suptitle(f"{title_prefix}  {tm}  tgt={tcls}, bkg={bkg}",
                     fontsize=8.5, y=1.01)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def plot_pd_vs_rho(agg, meta, out_path, main_fpr=0.1, title_prefix=''):
    """2×1 figure: P_det at main_fpr vs rho (additive | replacement)."""
    rho_list = meta['rho_list']
    present  = [tm for tm in ('additive', 'replacement') if tm in agg]
    dets     = list(agg[present[0]].keys())
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, len(present), figsize=(3.6*len(present), 3.2),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ci, tm in enumerate(present):
            ax = axes[ci]
            for det in dets:
                mu = np.array([agg[tm][det][rho]['pd'][main_fpr][0] for rho in rho_list])
                sd = np.array([agg[tm][det][rho]['pd'][main_fpr][1] for rho in rho_list])
                if det == 'AMF':
                    # AMF is rho-independent; show as dashed reference
                    ax.axhline(mu[0], color=DET_COLORS[det], lw=1.8, ls='--',
                               label=DET_LABELS[det])
                    ax.axhspan(mu[0]-sd[0], mu[0]+sd[0], color=DET_COLORS[det], alpha=0.08)
                elif det == 'LRao':
                    ax.axhline(mu[0], color=DET_COLORS[det], lw=1.8, ls='-.',
                               label=DET_LABELS[det])
                    ax.axhspan(mu[0]-sd[0], mu[0]+sd[0], color=DET_COLORS[det], alpha=0.08)
                else:
                    _band(ax, rho_list, mu, sd, det)
            ax.set_xscale('log')
            _setup_ax(ax, ylabel=f'$P_{{det}}$ @ $P_{{fa}}={main_fpr}$' if ci == 0 else '')
            ax.set_xlabel('$\\rho$ (DSM noise level)', fontsize=9)
            ax.set_title(f'{"Additive" if tm == "additive" else "Replacement"} model',
                         fontsize=9)
            if ci == len(present)-1:
                ax.legend(fontsize=7.5, framealpha=0.9,
                          loc='center left', bbox_to_anchor=(1.02, 0.5))
        bkg  = meta.get('bkg_desc', '')
        tcls = CLS_NAMES.get(meta.get('target_cls', 0), '')
        n    = meta.get('n', '?')
        fig.suptitle(
            f"{title_prefix}  tgt={tcls}, bkg={bkg}, n={n}, $\\theta$={meta.get('amp','?')}",
            fontsize=8.5, y=1.01)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


# ── Test-statistic distribution per background class ─────────────────────────

def compute_class_stat_distributions(run_dir: str, n_for_dist: int = 1000,
                                     seed: int = 42, verbose: bool = True):
    """
    For each background class in the test set, compute DSM and AMF statistics
    on PURE background (no targets planted) and return per-class score arrays.

    This illustrates the CFAR property: if the statistic is near-N(0,1) for
    all background classes, CFAR holds.
    """
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    met = json.load(open(run_dir / 'metrics.json'))

    n_list   = sorted(met['n_list'])
    d_list   = sorted(met.get('d_list', [20]))
    rho_list = sorted(met.get('rho_list', [0.01]))
    amp      = float(met.get('amp_list', [0.15])[0])
    d = d_list[0]; rho = rho_list[0]
    n = n_for_dist if n_for_dist in n_list else max(n_list)

    mat  = scipy.io.loadmat(cfg['dataset'])
    data = mat['data'].astype(np.float32)
    gt   = mat['map'].astype(int)
    flat = data.reshape(-1, data.shape[2]); gt_flat = gt.reshape(-1)

    tcls     = cfg['target_cls']
    bkg_cls_raw = cfg.get('bkg_cls')
    if bkg_cls_raw is None:
        bkg_mask = (gt_flat != 0) & (gt_flat != tcls)
    else:
        if isinstance(bkg_cls_raw, str):
            bkg_cls_list = [int(x.strip()) for x in bkg_cls_raw.split(',')]
        elif hasattr(bkg_cls_raw, '__iter__'):
            bkg_cls_list = [int(x) for x in bkg_cls_raw]
        else:
            bkg_cls_list = [int(bkg_cls_raw)]
        bkg_mask = np.isin(gt_flat, bkg_cls_list)
    bkg_all    = flat[bkg_mask]
    bkg_labels = gt_flat[bkg_mask]
    tgt_all    = flat[gt_flat == tcls]
    t_raw      = tgt_all.mean(0).astype(np.float32)

    n_max = max(n_list); n_test = cfg['test_n']

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(bkg_all))
    bkg_tr = bkg_all[idx[:n_max]]
    bkg_te = bkg_all[idx[n_max:n_max + n_test]]
    te_labels = bkg_labels[idx[n_max:n_max + n_test]]

    seed_mdl = run_dir / 'models' / f'seed_{seed}'
    pkl_path  = seed_mdl / f'pipeline_d{d}.pkl'
    with open(pkl_path, 'rb') as fh:
        pipe = pickle.load(fh)
    bkg_te_pca = pipe.project(bkg_te)
    s_add = pipe.signature_additive(t_raw)
    tr_pca = pipe.project(bkg_tr[:n])

    rho_str = str(rho).replace('.', 'p')
    dsm_path = seed_mdl / f'dsm_rho{rho_str}_d{d}_n{n}.pt'
    dsm_ckpt = torch.load(dsm_path, map_location='cpu', weights_only=False)
    dsm_m = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
    dsm_m.load_state_dict(dsm_ckpt['state_dict'])
    dsm_m.eval()

    gm     = float(bkg_tr.max() + 1e-12)
    tr_gm  = bkg_tr[:n] / gm
    te_gm  = bkg_te / gm
    t_gm   = t_raw / gm

    # Raw DSM additive statistic (standardised by training)
    z_tr   = compute_scores(dsm_m, tr_pca)
    z_te   = compute_scores(dsm_m, bkg_te_pca)
    z_bar  = z_tr.mean(0)
    C      = np.cov(z_tr, rowvar=False)
    if C.ndim == 0: C = np.array([[float(C)]])
    norm   = float(np.sqrt(max(float(s_add @ C @ s_add), 1e-12)))
    sc_dsm = -((z_te - z_bar) @ s_add) / norm

    # AMF statistic
    sc_amf = amf_score(tr_gm, te_gm, t_gm)

    # Per-class distributions
    unique_cls = sorted(np.unique(te_labels))
    class_scores = {}
    for cls_id in unique_cls:
        mask = te_labels == cls_id
        class_scores[int(cls_id)] = {
            'DSM':  sc_dsm[mask],
            'AMF':  sc_amf[mask],
            'name': CLS_NAMES.get(int(cls_id), f'cls{cls_id}'),
            'n':    int(mask.sum()),
        }
    return class_scores


def plot_class_distributions(class_scores: dict, out_path: str,
                              title: str = 'Test-statistic distribution per class'):
    """Box + violin plot of DSM/AMF statistic per background class."""
    cls_ids = sorted(class_scores.keys())
    names   = [class_scores[c]['name'] for c in cls_ids]
    n_cls   = len(cls_ids)

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(max(7, 2.5*n_cls), 3.8), sharey=False)
        for ci, (det, ax) in enumerate([('DSM', axes[0]), ('AMF', axes[1])]):
            data = [class_scores[c][det] for c in cls_ids]
            vp = ax.violinplot(data, positions=range(n_cls), showmedians=True,
                               widths=0.6)
            for pc in vp['bodies']:
                pc.set_facecolor(DET_COLORS.get(det, '#888'))
                pc.set_alpha(0.5)
            ax.set_xticks(range(n_cls))
            ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(f'{DET_LABELS.get(det, det)} statistic', fontsize=9)
            ax.set_title(f'{DET_LABELS.get(det, det)} stat. per background class', fontsize=9)
            ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.5)
        fig.suptitle(title, fontsize=9, y=1.01)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


# ── Paper 2×2 combined figure ─────────────────────────────────────────────────

def plot_combined_2x2(agg_single, meta_single,
                      agg_multi,  meta_multi,
                      out_path: str,
                      main_fpr: float = 0.1,
                      metric: str = 'pd',
                      metric_label: str = '$P_{det}$'):
    """
    2-row × 2-col figure for the paper's IID section.
      Row 0: single-class (additive | replacement)
      Row 1: multi-class  (additive | replacement)
    """
    n_list_s = meta_single['n_list']
    n_list_m = meta_multi['n_list']
    tms = [tm for tm in ('additive', 'replacement') if tm in agg_single]

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, len(tms), figsize=(3.6*len(tms), 6),
                                  gridspec_kw={'hspace': 0.45, 'wspace': 0.15},
                                  squeeze=False)
        for row, (agg, meta, n_list, tag) in enumerate([
                (agg_single, meta_single, n_list_s, 'single-class'),
                (agg_multi,  meta_multi,  n_list_m, 'multi-class')]):
            dets = list(agg[tms[0]].keys())
            for col, tm in enumerate(tms):
                ax = axes[row][col]
                for det in dets:
                    if metric == 'pd':
                        mu = np.array([agg[tm][det][n]['pd'][main_fpr][0] for n in n_list])
                        sd = np.array([agg[tm][det][n]['pd'][main_fpr][1] for n in n_list])
                    else:
                        mu = np.array([agg[tm][det][n]['pauc'][0] for n in n_list])
                        sd = np.array([agg[tm][det][n]['pauc'][1] for n in n_list])
                    _band(ax, n_list, mu, sd, det)
                _setup_ax(ax, ylabel=metric_label if col == 0 else '')
                ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=9)
                tm_str = 'Additive' if tm == 'additive' else 'Replacement'
                bkg  = meta.get('bkg_desc', '')
                tcls = CLS_NAMES.get(meta.get('target_cls', 0), '')
                ax.set_title(f'{tag}: {tm_str}\n'
                             f'tgt={tcls}, bkg={bkg}', fontsize=8)
                if col == len(tms)-1 and row == 1:
                    ax.legend(fontsize=7, framealpha=0.9, loc='lower right',
                              ncol=2)
        if metric == 'pd':
            sup = f'$P_{{det}}$ at $P_{{fa}}={main_fpr}$ vs training set size'
        else:
            sup = f'Partial AUC ($P_{{fa}} \\in [0, {main_fpr}]$, normalized) vs training set size'
        fig.suptitle(sup, fontsize=10, y=1.01)
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--single_n',  default='experiments/honest_pipeline/results/sweep_single_20260608_165625')
    p.add_argument('--multi_n',   default='experiments/honest_pipeline/results/sweep_multi_n_latest')
    p.add_argument('--single_rho',default='experiments/honest_pipeline/results/sweep_single_20260608_165812')
    p.add_argument('--multi_rho', default='experiments/honest_pipeline/results/sweep_multi_20260608_161636')
    p.add_argument('--out',       default='experiments/honest_pipeline/results/pauc_figures')
    p.add_argument('--skip_rho',  action='store_true', help='Skip rho-sweep rescoring (faster)')
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fpr_targets = (0.001, 0.01, 0.05, 0.1)

    # ─── Single-class n-sweep ───────────────────────────────────────────────
    print("\n=== Single-class n-sweep ===")
    res_s, meta_s = rescore_n_sweep(args.single_n, fpr_targets=fpr_targets)
    agg_s         = summarise_n_sweep(res_s, meta_s)
    # Save cache
    import pickle as _pkl
    _pkl.dump({'results': res_s, 'meta': meta_s, 'agg': agg_s},
              open(out / 'cache_single_n.pkl', 'wb'))

    plot_pd_vs_n(agg_s, meta_s,
                 out / 'single_pd_vs_n.pdf',
                 main_fpr=0.1, title_prefix='Single-class')
    plot_pauc_vs_n(agg_s, meta_s,
                   out / 'single_pauc_vs_n.pdf',
                   title_prefix='Single-class')
    for tm in agg_s:
        plot_pd_multi_fpr(agg_s, meta_s,
                          out / f'single_pd_multi_fpr_{tm}.pdf',
                          tm=tm, title_prefix='Single-class')

    # ─── Multi-class n-sweep ────────────────────────────────────────────────
    print("\n=== Multi-class n-sweep ===")
    res_m, meta_m = rescore_n_sweep(args.multi_n, fpr_targets=fpr_targets)
    agg_m         = summarise_n_sweep(res_m, meta_m)
    _pkl.dump({'results': res_m, 'meta': meta_m, 'agg': agg_m},
              open(out / 'cache_multi_n.pkl', 'wb'))

    plot_pd_vs_n(agg_m, meta_m,
                 out / 'multi_pd_vs_n.pdf',
                 main_fpr=0.1, title_prefix='Multi-class')
    plot_pauc_vs_n(agg_m, meta_m,
                   out / 'multi_pauc_vs_n.pdf',
                   title_prefix='Multi-class')
    for tm in agg_m:
        plot_pd_multi_fpr(agg_m, meta_m,
                          out / f'multi_pd_multi_fpr_{tm}.pdf',
                          tm=tm, title_prefix='Multi-class')

    # ─── Combined 2×2 paper figure ─────────────────────────────────────────
    plot_combined_2x2(agg_s, meta_s, agg_m, meta_m,
                      out / 'paper_pd_vs_n.pdf',
                      main_fpr=0.1, metric='pd',
                      metric_label='$P_{det}$ @ $P_{fa}=0.1$')
    plot_combined_2x2(agg_s, meta_s, agg_m, meta_m,
                      out / 'paper_pauc_vs_n.pdf',
                      main_fpr=0.1, metric='pauc',
                      metric_label='pAUC (0→0.1)')

    # ─── Class-distribution figure (multi-class only) ───────────────────────
    print("\n=== Class stat distributions (multi-class) ===")
    cls_scores = compute_class_stat_distributions(args.multi_n, n_for_dist=1000, seed=42)
    tcls = CLS_NAMES.get(meta_m.get('target_cls', 1), '')
    plot_class_distributions(
        cls_scores,
        out / 'multi_class_distributions.pdf',
        title=f'DSM / AMF statistic per background class  (tgt={tcls}, n=1000)')

    # ─── Rho sweeps ─────────────────────────────────────────────────────────
    if not args.skip_rho:
        print("\n=== Single-class rho-sweep ===")
        res_sr, meta_sr = rescore_rho_sweep(args.single_rho, fpr_targets=fpr_targets)
        agg_sr          = summarise_rho_sweep(res_sr, meta_sr)
        _pkl.dump({'results': res_sr, 'meta': meta_sr, 'agg': agg_sr},
                  open(out / 'cache_single_rho.pkl', 'wb'))
        plot_pd_vs_rho(agg_sr, meta_sr,
                       out / 'single_pd_vs_rho.pdf',
                       main_fpr=0.1, title_prefix='Single-class')

        print("\n=== Multi-class rho-sweep ===")
        res_mr, meta_mr = rescore_rho_sweep(args.multi_rho, fpr_targets=fpr_targets)
        agg_mr          = summarise_rho_sweep(res_mr, meta_mr)
        _pkl.dump({'results': res_mr, 'meta': meta_mr, 'agg': agg_mr},
                  open(out / 'cache_multi_rho.pkl', 'wb'))
        plot_pd_vs_rho(agg_mr, meta_mr,
                       out / 'multi_pd_vs_rho.pdf',
                       main_fpr=0.1, title_prefix='Multi-class')

    print(f"\nAll figures saved to {out}/")


if __name__ == '__main__':
    main()
