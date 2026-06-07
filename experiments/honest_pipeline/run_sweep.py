"""
Honest pipeline sweep — DSM vs AMF(full-D) vs LRao, multiple seeds.

Sweeps (n_train, latent_dim, rho) over multiple seeds and reports
mean ± std AUC.

  AMF   — global_max, full 103-D, no training
  DSM   — per_band_std, PCA-d, trained (sigma = f(rho))
  LRao  — per_band_std, PCA-d, trained + early stopping (rho-independent)

Key design:
  - LRao trained once per (seed, d, n) — does not depend on rho.
  - DSM trained per (seed, rho, d, n) — sigma changes with rho.
  - AMF uses global_max, signature via direction rule.
  - rho_d diagnostic reported per (d, seed) so validity is always visible.

Works for single-class (bkg_cls set) and multiclass (bkg_cls: null →
all non-target labeled pixels form the background).

Usage:
    .venv/bin/python -u experiments/honest_pipeline/run_sweep.py
    .venv/bin/python -u experiments/honest_pipeline/run_sweep.py \\
        --config experiments/honest_pipeline/sweep_multi_n.yaml
"""

import argparse, os, sys, json, time
from datetime import datetime

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP); sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pipeline import HonestDetectionPipeline, amf_score, amf_replacement_score
from final_paper_experiments.data_utils import compute_sigma_from_data
from final_paper_experiments.baselines.gmm_glrt_levin import GMMGLRTLevin
from final_paper_experiments.baselines.detectors import (
    gmm_glrt_oracle, gmm_glrt_replacement_oracle,
)
from dsm_model import (ScoreNet, dsm_loss, compute_scores, lfi_loss_mode2,
                       compute_lfi_detector_scores_mode2)

CLS_NAMES = {1:'asphalt',2:'meadows',3:'gravel',4:'trees',5:'metal_sheets',
             6:'bare_soil',7:'bitumen',8:'bricks',9:'shadows'}
DET_COLORS   = {'AMF': '#1f77b4', 'DSM': '#ff7f0e', 'LRao': '#2ca02c',
                'DSM-lin': '#bcbd22',           # yellow-green: linear score model
                'G-rep-LMP': '#08306b',
                'GMM-GLRT': '#9467bd',         # Levin product-of-GMMs oracle
                'GMM-GLRT-G': '#e377c2'}        # simple grid-based GMM oracle
DET_MARKERS  = {'AMF': 'o',       'DSM': 's',       'LRao': 'D',
                'DSM-lin': 'x',
                'G-rep-LMP': '^', 'GMM-GLRT': 'v', 'GMM-GLRT-G': 'P'}
DET_LABELS   = {'AMF': 'AMF',     'DSM': 'DSM',     'LRao': 'LRao-IID',
                'DSM-lin': 'DSM-linear',
                'G-rep-LMP': 'G-rep-LMP',
                'GMM-GLRT':   'GMM-GLRT (Levin)',
                'GMM-GLRT-G': 'GMM-GLRT (Gauss, oracle)'}

_STYLE = {
    'font.family':        'serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.6,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'figure.dpi':         150,
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def auc_safe(lab, sc):
    try:    return float(roc_auc_score(lab, sc))
    except: return float('nan')


def plant(bkg, s, amp, frac, model, seed):
    rng = np.random.RandomState(seed)
    n = len(bkg); k = int(frac * n)
    pos = rng.choice(n, k, replace=False)
    y = bkg.copy().astype(np.float32); lab = np.zeros(n, dtype=int); lab[pos] = 1
    if model == 'additive':   y[pos] += amp * s
    else:                     y[pos] = (1 - amp) * y[pos] + amp * s
    return y, lab


def train_dsm(d, tr_pca, sigma, cfg, seed, label=''):
    torch.manual_seed(seed)
    model = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    X = torch.tensor(tr_pca, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch_size'], len(tr_pca))
    pbar = tqdm(range(cfg['dsm_epochs']), desc=f'  DSM {label}',
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N); tot = 0.
        for i in range(0, N, bs):
            b = X[perm[i:i+bs]]; loss = dsm_loss(model, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        pbar.set_postfix(loss=f"{tot/max(N//bs,1):.3f}")
    model.eval(); return model


def train_dsm_lin(d, tr_pca, sigma, cfg, seed, label=''):
    """Linear score model (hidden_dims=[]) — analytic Gaussian-score baseline."""
    torch.manual_seed(seed)
    model = ScoreNet(d, [], cfg['activation'])   # single Linear(d,d), no hidden
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                              weight_decay=cfg['weight_decay'])
    X = torch.tensor(tr_pca, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch_size'], len(tr_pca))
    pbar = tqdm(range(cfg['dsm_epochs']), desc=f'  DSM-lin {label}',
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N); tot = 0.
        for i in range(0, N, bs):
            b = X[perm[i:i+bs]]; loss = dsm_loss(model, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        pbar.set_postfix(loss=f"{tot/max(N//bs,1):.3f}")
    model.eval(); return model


def train_lrao(d, tr_pca, cfg, seed, label=''):
    """Train LRao for a fixed number of epochs (no early stopping)."""
    torch.manual_seed(seed)
    model  = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
    opt    = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                               weight_decay=cfg['weight_decay'])
    epochs = int(cfg.get('lrao_epochs', 3000))
    delta  = cfg.get('lfi_delta_theta', 0.01)
    cutoff = cfg.get('lfi_sigma_cutoff', 1e-3)
    X  = torch.tensor(tr_pca, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch_size'], len(tr_pca))
    pbar = tqdm(range(epochs), desc=f'  LRao {label}',
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N)
        try:
            for i in range(0, N, bs):
                b    = X[perm[i:i+bs]]
                loss = lfi_loss_mode2(model, b, delta, cutoff, detach_sigma=False)
                if not torch.isfinite(loss): raise FloatingPointError()
                opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(trJ=f"{float(-loss):.2f}")
        except Exception:
            break
    model.eval(); return model


def _score_dsm(model, tr_pca, te_pca, s, tm):
    """DSM detection statistic (uses the raw score output directly)."""
    z_tr = compute_scores(model, tr_pca); z_te = compute_scores(model, te_pca)
    if tm == 'additive':
        z_bar = z_tr.mean(0); C = np.cov(z_tr, rowvar=False)
        if C.ndim == 0: C = np.array([[float(C)]])
        norm = float(np.sqrt(max(float(s @ C @ s), 1e-12)))
        return -((z_te - z_bar) @ s) / norm
    else:
        psi_bar = z_tr.mean(0)
        r_tr = ((z_tr - psi_bar) * (tr_pca - s)).sum(1)
        r_bar, r_std = r_tr.mean(), r_tr.std() + 1e-12
        return (((z_te - psi_bar) * (te_pca - s)).sum(1) - r_bar) / r_std


def _score_lrao(model, tr_pca, te_pca, s, cfg):
    """LRao Mode-2 detection statistic.

    Uses compute_lfi_detector_scores_mode2 which computes:
        g  = dE[psi(y)]/dtheta|_{theta=0}  (finite-diff sensitivity)
        T  = g^T C_psi^{-1} (psi(y) - psi_bar) / sqrt(J*)

    This is the correct LFI statistic — NOT the DSM formula with s.
    Both additive and replacement use the same Mode-2 stat (signal-agnostic).
    """
    return compute_lfi_detector_scores_mode2(
        model, tr_pca, te_pca, s,
        delta_theta=cfg.get('lfi_delta_theta', 0.01),
        sigma_cutoff=cfg.get('lfi_sigma_cutoff', 1e-3))


# ---------------------------------------------------------------------------
# One-seed sweep — returns nested dict of AUC values
# ---------------------------------------------------------------------------

def run_one_seed(seed, bkg_all, t_raw, cfg, n_list, d_list, rho_list, D_raw,
                 models_dir=None):
    """Run the full (n, d, rho) sweep for one seed.

    Parameters
    ----------
    models_dir : str or None
        If given, save all trained models and pipelines here under
        seed_{seed}/ subdirectory.

    Returns
    -------
    amf_auc   : {tm: {n: auc}}
    lrao_auc  : {d: {n: {tm: auc}}}
    dsm_auc   : {rho: {d: {tm: {n: auc}}}}
    rho_d_map : {d: rho_d_value}
    """
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(bkg_all))
    n_max = max(n_list); n_test = cfg['test_n']
    assert len(bkg_all) >= n_max + n_test, \
        f"Need {n_max+n_test} bkg pixels, have {len(bkg_all)}"
    bkg_tr = bkg_all[idx[:n_max]]
    bkg_te = bkg_all[idx[n_max:n_max + n_test]]

    # Create per-seed model directory
    seed_mdl = None
    if models_dir is not None:
        seed_mdl = os.path.join(models_dir, f'seed_{seed}')
        os.makedirs(seed_mdl, exist_ok=True)

    n_dsm_total  = len(rho_list) * len(d_list) * len(n_list)
    n_lrao_total = len(d_list) * len(n_list)
    print(f"  plan: {len(n_list)} AMF  |  "
          f"{n_lrao_total} LRao (rho-independent)  |  "
          f"{n_dsm_total} DSM ({len(rho_list)} rho × {len(d_list)} d × {len(n_list)} n)",
          flush=True)

    # --- AMF: global_max, full 103-D (amplitude loop handled after DSM) ---
    print(f"  [AMF] setup ...", flush=True)
    gm = float(bkg_tr.max() + 1e-12)
    t_gm = t_raw / gm
    amf_auc = {'additive': {}, 'replacement': {}}

    # --- fit pipelines once per d ---
    print(f"  [PCA] fitting {len(d_list)} pipelines ...", flush=True)
    pipes = {}
    for d in d_list:
        p = HonestDetectionPipeline(latent_dim=d, norm=cfg['score_norm'])
        p.fit(bkg_tr); pipes[d] = p
        if seed_mdl:
            import pickle
            with open(os.path.join(seed_mdl, f'pipeline_d{d}.pkl'), 'wb') as fh:
                pickle.dump(p, fh)

    rho_d_map = {d: pipes[d].rho_d(t_raw) for d in d_list}
    print(f"  rho_d: { {d: f'{rho_d_map[d]:.3f}' for d in d_list} }", flush=True)

    amp_list = sorted(cfg.get('amp_list', [cfg['amplitude']]))

    # --- LRao: once per (d, n), evaluated at all amplitudes ---
    print(f"\n  [LRao] {n_lrao_total} trainings ({len(d_list)} d × {len(n_list)} n) "
          f"× {len(amp_list)} amplitudes ...", flush=True)
    lrao_auc  = {}   # lrao_auc[d][n][tm][amp]
    lrao_models = {}  # cache trained models: (d, n) -> model
    lrao_count = 0
    t_lrao_start = time.time()
    for d in d_list:
        pipe  = pipes[d]; lrao_auc[d] = {}
        s_add = pipe.signature_additive(t_raw)
        s_rep = pipe.signature_replacement(t_raw)
        bkg_te_pca = pipe.project(bkg_te)
        for n in n_list:
            lrao_count += 1
            t0 = time.time()
            print(f"  [LRao {lrao_count}/{n_lrao_total}] d={d} n={n} ...",
                  end='', flush=True)
            tr_pca = pipe.project(bkg_tr[:n])
            lrao_m = train_lrao(d, tr_pca, cfg, seed, label=f'd={d} n={n}')
            lrao_models[(d, n)] = lrao_m
            if seed_mdl:
                torch.save({'state_dict': lrao_m.state_dict(),
                            'cfg': cfg, 'd': d, 'n': n, 'seed': seed},
                           os.path.join(seed_mdl, f'lrao_d{d}_n{n}.pt'))
            lrao_auc[d][n] = {}
            for tm in ('additive', 'replacement'):
                sig = s_add if tm == 'additive' else s_rep
                lrao_auc[d][n][tm] = {}
                for amp in amp_list:
                    te, lab = plant(bkg_te_pca, sig, amp,
                                    cfg['target_fraction'], tm, seed)
                    lrao_auc[d][n][tm][amp] = auc_safe(
                        lab, _score_lrao(lrao_m, tr_pca, te, sig, cfg))
            ref_amp = amp_list[len(amp_list)//2]
            print(f" add={lrao_auc[d][n]['additive'][ref_amp]:.3f} "
                  f"rep={lrao_auc[d][n]['replacement'][ref_amp]:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"  [LRao] done in {time.time()-t_lrao_start:.0f}s", flush=True)

    # --- DSM + DSM-lin: per (rho, d, n); evaluate at all amplitudes ---
    print(f"\n  [DSM+DSM-lin] {n_dsm_total} trainings × 2  ×  {len(amp_list)} amplitudes ...",
          flush=True)
    # dsm_auc[rho][d][tm][n][amp]  /  dsm_lin_auc[rho][d][tm][n][amp]
    dsm_auc = {}; dsm_lin_auc = {}
    dsm_count = 0
    t_dsm_start = time.time()
    for rho in rho_list:
        dsm_auc[rho] = {}; dsm_lin_auc[rho] = {}
        for d in d_list:
            pipe  = pipes[d]
            dsm_auc[rho][d]     = {'additive': {}, 'replacement': {}}
            dsm_lin_auc[rho][d] = {'additive': {}, 'replacement': {}}
            s_add = pipe.signature_additive(t_raw)
            s_rep = pipe.signature_replacement(t_raw)
            bkg_te_pca = pipe.project(bkg_te)
            for n in n_list:
                dsm_count += 1
                t0 = time.time()
                print(f"  [DSM {dsm_count}/{n_dsm_total}] rho={rho} d={d} n={n} ...",
                      end='', flush=True)
                tr_pca = pipe.project(bkg_tr[:n])
                sigma  = compute_sigma_from_data(tr_pca, rho)
                dsm_m     = train_dsm(d, tr_pca, sigma, cfg, seed,
                                      label=f'rho={rho} d={d} n={n}')
                dsm_lin_m = train_dsm_lin(d, tr_pca, sigma, cfg, seed,
                                          label=f'rho={rho} d={d} n={n}')
                if seed_mdl:
                    rho_str = str(rho).replace('.', 'p')
                    torch.save({'state_dict': dsm_m.state_dict(),
                                'cfg': cfg, 'd': d, 'n': n,
                                'rho': rho, 'sigma': sigma, 'seed': seed},
                               os.path.join(seed_mdl,
                                            f'dsm_rho{rho_str}_d{d}_n{n}.pt'))
                for tm in ('additive', 'replacement'):
                    sig = s_add if tm == 'additive' else s_rep
                    dsm_auc[rho][d][tm][n] = {}
                    dsm_lin_auc[rho][d][tm][n] = {}
                    for amp in amp_list:
                        te, lab = plant(bkg_te_pca, sig, amp,
                                        cfg['target_fraction'], tm, seed)
                        dsm_auc[rho][d][tm][n][amp] = auc_safe(
                            lab, _score_dsm(dsm_m, tr_pca, te, sig, tm))
                        dsm_lin_auc[rho][d][tm][n][amp] = auc_safe(
                            lab, _score_dsm(dsm_lin_m, tr_pca, te, sig, tm))
                print(f" add={dsm_auc[rho][d]['additive'][n][amp_list[0]]:.3f}"
                      f"(lin={dsm_lin_auc[rho][d]['additive'][n][amp_list[0]]:.3f}) "
                      f"rep={dsm_auc[rho][d]['replacement'][n][amp_list[0]]:.3f}"
                      f"(lin={dsm_lin_auc[rho][d]['replacement'][n][amp_list[0]]:.3f})"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [DSM+DSM-lin] done in {time.time()-t_dsm_start:.0f}s", flush=True)

    # --- AMF: evaluate at all amplitudes ---
    print(f"\n  [AMF] {len(n_list)} n-values × {len(amp_list)} amplitudes ...", flush=True)
    for n in n_list:
        tr_gm = bkg_tr[:n] / gm
        for tm in ('additive', 'replacement'):
            amf_auc[tm][n] = {}
            for amp in amp_list:
                te, lab = plant(bkg_te / gm, t_gm, amp,
                                cfg['target_fraction'], tm, seed)
                sc = amf_score(tr_gm, te, t_gm) if tm == 'additive' \
                     else amf_replacement_score(tr_gm, te, t_gm)
                amf_auc[tm][n][amp] = auc_safe(lab, sc)
    ref_amp = amp_list[len(amp_list)//2]
    row = "  ".join(f"n={n}:{amf_auc['additive'][n][ref_amp]:.3f}" for n in n_list)
    print(f"  [AMF add @ amp={ref_amp}] {row}", flush=True)

    # --- GMM-GLRT (Levin, oracle) — full-D global_max, like AMF ---
    # Background GMM fit once per n; oracle scoring with KNOWN amplitude.
    # MULTICLASS-ONLY by default: a GMM background is motivated by multimodal
    # clutter; for a single near-Gaussian class it is unnecessary. Override
    # with `run_gmm_glrt: true/false` in the config.
    is_multi = cfg.get('bkg_cls') is None
    gmm_auc = {'additive': {}, 'replacement': {}}
    if cfg.get('run_gmm_glrt', is_multi):
        print(f"\n  [GMM-GLRT] {len(n_list)} n-values × {len(amp_list)} amplitudes ...",
              flush=True)
        for n in n_list:
            t0 = time.time()
            det = GMMGLRTLevin(
                cond_tol=cfg.get('gmm_cond_tol', 1e3),
                max_dim=cfg.get('gmm_max_dim', 5),
                k_max=cfg.get('gmm_k_max', 5),
                seed=seed).fit(bkg_tr[:n] / gm)
            for tm in ('additive', 'replacement'):
                gmm_auc[tm][n] = {}
                for amp in amp_list:
                    te, lab = plant(bkg_te / gm, t_gm, amp,
                                    cfg['target_fraction'], tm, seed)
                    sc = det.score(te, t_gm, model=tm,
                                   p_steps=cfg.get('gmm_p_steps', 50),
                                   p_max=cfg.get('gmm_p_max', 1.0))
                    gmm_auc[tm][n][amp] = auc_safe(lab, sc)
            print(f"  [GMM-GLRT] n={n}  add={gmm_auc['additive'][n][ref_amp]:.3f} "
                  f"rep={gmm_auc['replacement'][n][ref_amp]:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # --- GMM-GLRT-G (simple K-Gaussian mixture, oracle θ) — multiclass only ---
    # Same family as gmm_glrt() but with KNOWN amplitude (no grid search).
    # Lower-capacity than Levin's product-of-GMMs; serves as a fair-to-Gaussian
    # GMM oracle that uses the same K-component mixture as our older baseline.
    gmm_g_auc = {'additive': {}, 'replacement': {}}
    if cfg.get('run_gmm_glrt_g', is_multi):
        K_g = cfg.get('gmm_g_K', 3)
        print(f"\n  [GMM-GLRT-G] K={K_g}  {len(n_list)} n × {len(amp_list)} amp ...",
              flush=True)
        for n in n_list:
            t0 = time.time()
            tr_gm = bkg_tr[:n] / gm
            for tm in ('additive', 'replacement'):
                gmm_g_auc[tm][n] = {}
                for amp in amp_list:
                    te, lab = plant(bkg_te / gm, t_gm, amp,
                                    cfg['target_fraction'], tm, seed)
                    if tm == 'additive':
                        sc = gmm_glrt_oracle(te, tr_gm, t_gm, theta=amp, K=K_g)
                    else:
                        sc = gmm_glrt_replacement_oracle(te, tr_gm, t_gm,
                                                         theta=amp, K=K_g)
                    gmm_g_auc[tm][n][amp] = auc_safe(lab, sc)
            print(f"  [GMM-GLRT-G] n={n}  add={gmm_g_auc['additive'][n][ref_amp]:.3f} "
                  f"rep={gmm_g_auc['replacement'][n][ref_amp]:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    return amf_auc, lrao_auc, dsm_auc, dsm_lin_auc, gmm_auc, gmm_g_auc, rho_d_map, amp_list


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg(arrays):
    """arrays: list of floats -> (mean, std)"""
    a = np.array([v for v in arrays if not np.isnan(v)])
    if len(a) == 0: return float('nan'), float('nan')
    return float(a.mean()), float(a.std())


# ---------------------------------------------------------------------------
# Plotting  (paper-quality style matching reference figure)
# ---------------------------------------------------------------------------

def _setup_ax(ax):
    """Apply paper style to a single axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_tick_params(direction='in')
    ax.xaxis.set_tick_params(direction='in')
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel('AUC', fontsize=10)
    ax.yaxis.label.set_fontfamily('serif')
    ax.xaxis.label.set_fontfamily('serif')


def _band(ax, x, mu_list, sd_list, det, logx=True, ms=6):
    color  = DET_COLORS[det]
    marker = DET_MARKERS[det]
    label  = DET_LABELS[det]
    plot_fn = ax.semilogx if logx else ax.plot
    mu = np.array(mu_list); sd = np.array(sd_list)
    plot_fn(x, mu, marker=marker, color=color, lw=1.8, ms=ms,
            markerfacecolor=color, markeredgewidth=0.5,
            markeredgecolor='white', label=label)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15)


def _get(src, amp):
    if isinstance(src, dict): return src[amp]
    return src


def plot_sweep(agg_amf, agg_lrao, agg_dsm, rho_d_map,
               n_list, d_list, rho_list, D_raw, tm, title, path, ref_amp=None,
               agg_gmm=None, agg_gmmg=None, agg_dsm_lin=None):
    """AUC vs n_train — paper style. One panel per (rho, d) combination."""
    with plt.rc_context(_STYLE):
        nr, nc = len(rho_list), len(d_list)
        fig, axes = plt.subplots(nr, nc, figsize=(4.2*nc, 3.8*nr),
                                  sharex=True, sharey=True, squeeze=False)
        for ri, rho in enumerate(rho_list):
            for ci, d in enumerate(d_list):
                ax = axes[ri, ci]; x = n_list
                rho_d = rho_d_map[d]
                _band(ax, x, [_get(agg_amf[tm][n], ref_amp)[0] for n in x],
                              [_get(agg_amf[tm][n], ref_amp)[1] for n in x], 'AMF')
                if agg_gmm is not None:
                    _band(ax, x, [_get(agg_gmm[tm][n], ref_amp)[0] for n in x],
                                  [_get(agg_gmm[tm][n], ref_amp)[1] for n in x], 'GMM-GLRT')
                if agg_gmmg is not None:
                    _band(ax, x, [_get(agg_gmmg[tm][n], ref_amp)[0] for n in x],
                                  [_get(agg_gmmg[tm][n], ref_amp)[1] for n in x], 'GMM-GLRT-G')
                _band(ax, x, [_get(agg_lrao[d][n][tm], ref_amp)[0] for n in x],
                              [_get(agg_lrao[d][n][tm], ref_amp)[1] for n in x], 'LRao')
                _band(ax, x, [_get(agg_dsm[rho][d][tm][n], ref_amp)[0] for n in x],
                              [_get(agg_dsm[rho][d][tm][n], ref_amp)[1] for n in x], 'DSM')
                if agg_dsm_lin is not None:
                    _band(ax, x, [_get(agg_dsm_lin[rho][d][tm][n], ref_amp)[0] for n in x],
                                  [_get(agg_dsm_lin[rho][d][tm][n], ref_amp)[1] for n in x], 'DSM-lin')
                _setup_ax(ax)
                ax.set_title(f'AUC vs $n$\n'
                             f'$(d={d},\\,\\theta={ref_amp},\\,\\rho={rho})$',
                             fontsize=9, fontfamily='serif')
                if ri == nr-1: ax.set_xlabel('$n_\\mathrm{train}$', fontsize=10)
                if ci == nc-1 and ri == 0: ax.legend(fontsize=8, framealpha=0.9)
        fig.suptitle(title, fontsize=11, fontfamily='serif', y=1.01)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"  saved {path}", flush=True)


def plot_auc_vs_d(agg_amf, agg_lrao, agg_dsm, rho_d_map,
                  n_list, d_list, rho_list, amp_list, D_raw, tm,
                  ref_n, ref_amp, title, path, agg_gmm=None, agg_gmmg=None,
                  agg_dsm_lin=None):
    """AUC vs PCA dim d — one panel per rho."""
    with plt.rc_context(_STYLE):
        nr = len(rho_list)
        fig, axes = plt.subplots(1, nr, figsize=(4.2*nr, 3.8),
                                  sharey=True, squeeze=False)
        for ci, rho in enumerate(rho_list):
            ax = axes[0, ci]
            _band(ax, d_list,
                  [agg_amf[tm][ref_n][ref_amp][0]]*len(d_list),
                  [agg_amf[tm][ref_n][ref_amp][1]]*len(d_list), 'AMF', logx=False)
            if agg_gmm is not None:   # GMM-GLRT is d-independent (own PCA) → flat
                _band(ax, d_list,
                      [agg_gmm[tm][ref_n][ref_amp][0]]*len(d_list),
                      [agg_gmm[tm][ref_n][ref_amp][1]]*len(d_list), 'GMM-GLRT', logx=False)
            if agg_gmmg is not None:
                _band(ax, d_list,
                      [agg_gmmg[tm][ref_n][ref_amp][0]]*len(d_list),
                      [agg_gmmg[tm][ref_n][ref_amp][1]]*len(d_list), 'GMM-GLRT-G', logx=False)
            _band(ax, d_list,
                  [agg_lrao[d][ref_n][tm][ref_amp][0] for d in d_list],
                  [agg_lrao[d][ref_n][tm][ref_amp][1] for d in d_list], 'LRao', logx=False)
            _band(ax, d_list,
                  [agg_dsm[rho][d][tm][ref_n][ref_amp][0] for d in d_list],
                  [agg_dsm[rho][d][tm][ref_n][ref_amp][1] for d in d_list], 'DSM', logx=False)
            if agg_dsm_lin is not None:
                _band(ax, d_list,
                      [agg_dsm_lin[rho][d][tm][ref_n][ref_amp][0] for d in d_list],
                      [agg_dsm_lin[rho][d][tm][ref_n][ref_amp][1] for d in d_list], 'DSM-lin', logx=False)
            # annotate rho_d values below x-ticks
            ax.set_xticks(d_list)
            ax.set_xticklabels(
                [f'$d={d}$\n$\\rho_d={rho_d_map[d]:.2f}$' for d in d_list],
                fontsize=8)
            _setup_ax(ax)
            ax.set_title(f'PCA dim $d$\n'
                         f'$(n={ref_n},\\,\\theta={ref_amp},\\,\\rho={rho})$',
                         fontsize=9, fontfamily='serif')
            ax.set_xlabel('PCA dim $d$', fontsize=10)
            if ci == nr-1: ax.legend(fontsize=8, framealpha=0.9)
        fig.suptitle(title, fontsize=11, fontfamily='serif', y=1.01)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"  saved {path}", flush=True)


def plot_auc_vs_rho(agg_amf, agg_lrao, agg_dsm, rho_d_map,
                    n_list, d_list, rho_list, amp_list, D_raw, tm,
                    ref_n, ref_d, ref_amp, title, path,
                    agg_gmm=None, agg_gmmg=None, agg_dsm_lin=None):
    """AUC vs DSM noise level rho. AMF/LRao/GMM shown as dashed reference."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        _band(ax, rho_list,
              [agg_dsm[rho][ref_d][tm][ref_n][ref_amp][0] for rho in rho_list],
              [agg_dsm[rho][ref_d][tm][ref_n][ref_amp][1] for rho in rho_list],
              'DSM')
        if agg_dsm_lin is not None:
            _band(ax, rho_list,
                  [agg_dsm_lin[rho][ref_d][tm][ref_n][ref_amp][0] for rho in rho_list],
                  [agg_dsm_lin[rho][ref_d][tm][ref_n][ref_amp][1] for rho in rho_list],
                  'DSM-lin')
        # rho-independent detectors as dashed horizontal reference lines
        refs = [('AMF', agg_amf[tm][ref_n][ref_amp]),
                ('LRao', agg_lrao[ref_d][ref_n][tm][ref_amp])]
        if agg_gmm is not None:
            refs.append(('GMM-GLRT', agg_gmm[tm][ref_n][ref_amp]))
        if agg_gmmg is not None:
            refs.append(('GMM-GLRT-G', agg_gmmg[tm][ref_n][ref_amp]))
        for det, (mu, sd) in refs:
            ax.axhline(mu, color=DET_COLORS[det], lw=1.8, ls='--',
                       label=f'{DET_LABELS[det]} (ref)')
            ax.axhspan(mu-sd, mu+sd, color=DET_COLORS[det], alpha=0.08)
        ax.set_xscale('log')
        ax.set_xlabel('$\\rho$ (DSM noise level)', fontsize=10)
        _setup_ax(ax)
        ax.set_title(f'DSM noise level $\\rho$\n'
                     f'$(n={ref_n},\\,d={ref_d},\\,\\theta={ref_amp})$',
                     fontsize=9, fontfamily='serif')
        ax.legend(fontsize=8, framealpha=0.9)
        fig.suptitle(title, fontsize=11, fontfamily='serif', y=1.01)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"  saved {path}", flush=True)


def plot_auc_vs_amp(agg_amf, agg_lrao, agg_dsm, rho_d_map,
                    n_list, d_list, rho_list, amp_list, D_raw, tm,
                    ref_n, ref_d, ref_rho, title, path,
                    agg_gmm=None, agg_gmmg=None, agg_dsm_lin=None):
    """AUC vs target amplitude theta."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        _band(ax, amp_list,
              [agg_amf[tm][ref_n][amp][0] for amp in amp_list],
              [agg_amf[tm][ref_n][amp][1] for amp in amp_list], 'AMF', logx=False)
        if agg_gmm is not None:
            _band(ax, amp_list,
                  [agg_gmm[tm][ref_n][amp][0] for amp in amp_list],
                  [agg_gmm[tm][ref_n][amp][1] for amp in amp_list], 'GMM-GLRT', logx=False)
        if agg_gmmg is not None:
            _band(ax, amp_list,
                  [agg_gmmg[tm][ref_n][amp][0] for amp in amp_list],
                  [agg_gmmg[tm][ref_n][amp][1] for amp in amp_list], 'GMM-GLRT-G', logx=False)
        _band(ax, amp_list,
              [agg_lrao[ref_d][ref_n][tm][amp][0] for amp in amp_list],
              [agg_lrao[ref_d][ref_n][tm][amp][1] for amp in amp_list], 'LRao', logx=False)
        _band(ax, amp_list,
              [agg_dsm[ref_rho][ref_d][tm][ref_n][amp][0] for amp in amp_list],
              [agg_dsm[ref_rho][ref_d][tm][ref_n][amp][1] for amp in amp_list], 'DSM', logx=False)
        if agg_dsm_lin is not None:
            _band(ax, amp_list,
                  [agg_dsm_lin[ref_rho][ref_d][tm][ref_n][amp][0] for amp in amp_list],
                  [agg_dsm_lin[ref_rho][ref_d][tm][ref_n][amp][1] for amp in amp_list], 'DSM-lin', logx=False)
        ax.set_xlabel('Target amplitude $\\theta$', fontsize=10)
        _setup_ax(ax)
        ax.set_title(f'Target amplitude $\\theta$\n'
                     f'$(n={ref_n},\\,d={ref_d},\\,\\rho={ref_rho})$',
                     fontsize=9, fontfamily='serif')
        ax.legend(fontsize=8, framealpha=0.9)
        fig.suptitle(title, fontsize=11, fontfamily='serif', y=1.01)
        fig.tight_layout()
        fig.savefig(path, bbox_inches='tight', dpi=200); plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(_EXP, 'sweep_rho.yaml'))
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'multi' if cfg.get('bkg_cls') is None else 'single'
    run_dir  = os.path.join(cfg['results_dir'], f'sweep_{tag}_{ts}')
    fig_dir  = os.path.join(run_dir, 'figures')
    mdl_dir  = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(mdl_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    seeds    = cfg.get('seeds', [int(cfg.get('seed', 42))])
    n_list   = sorted(cfg['n_train_list'])
    d_list   = sorted(cfg['latent_dim_list'])
    rho_list = sorted(cfg['rho_list'])

    mat  = scipy.io.loadmat(cfg['dataset'])
    data = mat['data'].astype(np.float32); gt = mat['map'].astype(int)
    H, W, D_raw = data.shape
    flat = data.reshape(-1, D_raw); gt_flat = gt.reshape(-1)
    print(f"Image {H}x{W}x{D_raw}", flush=True)

    tcls = cfg['target_cls']
    if cfg.get('bkg_cls') is not None:
        bkg_all = flat[gt_flat == cfg['bkg_cls']]
        bkg_desc = CLS_NAMES.get(cfg['bkg_cls'], f"cls{cfg['bkg_cls']}")
    else:
        bkg_all = flat[(gt_flat != 0) & (gt_flat != tcls)]
        bkg_desc = 'all-non-target'
    tgt_all = flat[gt_flat == tcls]
    t_raw   = tgt_all.mean(0).astype(np.float32)
    print(f"bkg={len(bkg_all)}px ({bkg_desc})  "
          f"tgt cls {tcls}({CLS_NAMES.get(tcls,'?')})={len(tgt_all)}px REMOVED\n",
          flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"n_train: {n_list}  d: {d_list}  rho: {rho_list}\n", flush=True)

    # ---- per-seed results ----
    all_amf  = []   # list of amf_auc dicts
    all_lrao = []
    all_dsm  = []
    all_dsm_lin = []  # linear score model (no hidden layers)
    all_gmm  = []     # Levin product-of-GMMs (oracle)
    all_gmmg = []     # simple K-Gaussian mixture (oracle)
    all_rho_d = []

    for k, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Seed {k+1}/{len(seeds)}  (seed={seed})", flush=True)
        print('='*60, flush=True)
        (amf_auc, lrao_auc, dsm_auc, dsm_lin_auc, gmm_auc, gmm_g_auc,
         rho_d_map, amp_list) = run_one_seed(
            seed, bkg_all, t_raw, cfg, n_list, d_list, rho_list, D_raw,
            models_dir=mdl_dir)
        all_amf.append(amf_auc); all_lrao.append(lrao_auc)
        all_dsm.append(dsm_auc); all_dsm_lin.append(dsm_lin_auc)
        all_gmm.append(gmm_auc)
        all_gmmg.append(gmm_g_auc); all_rho_d.append(rho_d_map)

        # quick summary per seed (at median amplitude)
        ref_amp = amp_list[len(amp_list)//2]
        for tm in ('additive', 'replacement'):
            best_rho = rho_list[len(rho_list)//2]
            best_d   = d_list[-1]
            row = "  ".join(
                f"n={n}: DSM={dsm_auc[best_rho][best_d][tm][n][ref_amp]:.3f} "
                f"LRao={lrao_auc[best_d][n][tm][ref_amp]:.3f} "
                f"AMF={amf_auc[tm][n][ref_amp]:.3f}"
                for n in n_list)
            print(f"  [{tm}] rho={best_rho} d={best_d} amp={ref_amp}: {row}", flush=True)

        print(f"  ({time.time()-t0:.0f}s)", flush=True)

        # incremental save
        json.dump({'seeds_done': k+1, 'seeds': seeds,
                   'n_list': n_list, 'd_list': d_list, 'rho_list': rho_list,
                   'target_cls': tcls, 'bkg_desc': bkg_desc},
                  open(os.path.join(run_dir, 'progress.json'), 'w'), indent=2)

    # ---- aggregate mean ± std ----
    print(f"\n{'='*60}", flush=True)
    print("Aggregating ...", flush=True)

    # AMF: {tm: {n: {amp: (mean, std)}}}
    agg_amf = {tm: {n: {amp: _agg([a[tm][n][amp] for a in all_amf])
                        for amp in amp_list}
                    for n in n_list}
               for tm in ('additive', 'replacement')}

    # LRao: {d: {n: {tm: {amp: (mean, std)}}}}
    agg_lrao = {d: {n: {tm: {amp: _agg([a[d][n][tm][amp] for a in all_lrao])
                              for amp in amp_list}
                         for tm in ('additive', 'replacement')}
                    for n in n_list} for d in d_list}

    # DSM: {rho: {d: {tm: {n: {amp: (mean, std)}}}}}
    agg_dsm = {rho: {d: {tm: {n: {amp: _agg([a[rho][d][tm][n][amp] for a in all_dsm])
                                   for amp in amp_list}
                               for n in n_list}
                          for tm in ('additive', 'replacement')}
                     for d in d_list} for rho in rho_list}

    # DSM-lin (linear score model): same shape as agg_dsm
    agg_dsm_lin = {rho: {d: {tm: {n: {amp: _agg([a[rho][d][tm][n][amp] for a in all_dsm_lin])
                                       for amp in amp_list}
                                   for n in n_list}
                              for tm in ('additive', 'replacement')}
                         for d in d_list} for rho in rho_list}

    # GMM-GLRT (Levin): {tm: {n: {amp: (mean, std)}}}
    has_gmm = all_gmm and all_gmm[0].get('additive')
    agg_gmm = None
    if has_gmm:
        agg_gmm = {tm: {n: {amp: _agg([a[tm][n][amp] for a in all_gmm])
                            for amp in amp_list}
                        for n in n_list}
                   for tm in ('additive', 'replacement')}

    # GMM-GLRT-G (simple K-Gauss, oracle): {tm: {n: {amp: (mean, std)}}}
    has_gmmg = all_gmmg and all_gmmg[0].get('additive')
    agg_gmmg = None
    if has_gmmg:
        agg_gmmg = {tm: {n: {amp: _agg([a[tm][n][amp] for a in all_gmmg])
                              for amp in amp_list}
                         for n in n_list}
                    for tm in ('additive', 'replacement')}

    agg_rho_d = {d: float(np.mean([r[d] for r in all_rho_d])) for d in d_list}

    # ---- print final summary table ----
    ref_amp = amp_list[len(amp_list)//2]
    for tm in ('additive', 'replacement'):
        print(f"\n=== {tm.upper()} (n={n_list[-1]}, d={d_list[-1]}, amp={ref_amp}) ===")
        print(f"  AMF-{D_raw}D: {agg_amf[tm][n_list[-1]][ref_amp][0]:.3f} ± {agg_amf[tm][n_list[-1]][ref_amp][1]:.3f}")
        print(f"  LRao d={d_list[-1]}: {agg_lrao[d_list[-1]][n_list[-1]][tm][ref_amp][0]:.3f} ± "
              f"{agg_lrao[d_list[-1]][n_list[-1]][tm][ref_amp][1]:.3f}")
        for rho in rho_list:
            v    = agg_dsm[rho][d_list[-1]][tm][n_list[-1]][ref_amp]
            vlin = agg_dsm_lin[rho][d_list[-1]][tm][n_list[-1]][ref_amp]
            print(f"  DSM rho={rho}: {v[0]:.3f} ± {v[1]:.3f}  |  DSM-lin: {vlin[0]:.3f} ± {vlin[1]:.3f}")

    # ---- save ----
    def _ser(v): return list(v) if isinstance(v, (tuple, list)) else v
    out = {
        'n_list': n_list, 'd_list': d_list, 'rho_list': rho_list,
        'amp_list': amp_list, 'seeds': seeds,
        'target_cls': tcls, 'bkg_desc': bkg_desc, 'rho_d': agg_rho_d,
        'amf':  {tm: {n: {amp: _ser(agg_amf[tm][n][amp])
                           for amp in amp_list}
                      for n in n_list}
                 for tm in ('additive','replacement')},
        'lrao': {str(d): {n: {tm: {amp: _ser(agg_lrao[d][n][tm][amp])
                                    for amp in amp_list}
                               for tm in ('additive','replacement')}
                           for n in n_list}
                 for d in d_list},
        'dsm':  {str(rho): {str(d): {tm: {n: {amp: _ser(agg_dsm[rho][d][tm][n][amp])
                                               for amp in amp_list}
                                            for n in n_list}
                                      for tm in ('additive','replacement')}
                             for d in d_list}
                 for rho in rho_list},
        'dsm_lin': {str(rho): {str(d): {tm: {n: {amp: _ser(agg_dsm_lin[rho][d][tm][n][amp])
                                                   for amp in amp_list}
                                              for n in n_list}
                                          for tm in ('additive','replacement')}
                               for d in d_list}
                   for rho in rho_list},
    }
    if agg_gmm is not None:
        out['gmm'] = {tm: {n: {amp: _ser(agg_gmm[tm][n][amp])
                                for amp in amp_list}
                           for n in n_list}
                      for tm in ('additive','replacement')}
    if agg_gmmg is not None:
        out['gmm_g'] = {tm: {n: {amp: _ser(agg_gmmg[tm][n][amp])
                                  for amp in amp_list}
                             for n in n_list}
                        for tm in ('additive','replacement')}
    json.dump(out, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)

    # ---- reference values for slice plots ----
    ref_n   = n_list[-1]
    ref_d   = d_list[-1]
    ref_rho = rho_list[len(rho_list)//2]
    ref_amp = amp_list[len(amp_list)//2]

    # ---- figures ----
    tgt_name = CLS_NAMES.get(tcls, f'cls{tcls}')
    base = (f"target={tgt_name}, bkg={bkg_desc}, "
            f"AMF:global_max 103-D | DSM/LRao:{cfg['score_norm']} PCA-d  "
            f"[{len(seeds)} seeds]")
    for tm in ('additive', 'replacement'):
        if len(n_list) > 1:
            plot_sweep(agg_amf, agg_lrao, agg_dsm, agg_rho_d,
                       n_list, d_list, rho_list, D_raw, tm,
                       f"{base} — {tm}", os.path.join(fig_dir, f'auc_vs_n_{tm}.pdf'),
                       ref_amp=ref_amp, agg_gmm=agg_gmm, agg_gmmg=agg_gmmg,
                       agg_dsm_lin=agg_dsm_lin)
        if len(d_list) > 1:
            plot_auc_vs_d(agg_amf, agg_lrao, agg_dsm, agg_rho_d,
                          n_list, d_list, rho_list, amp_list, D_raw, tm,
                          ref_n, ref_amp, f"{base} — {tm}",
                          os.path.join(fig_dir, f'auc_vs_d_{tm}.pdf'), agg_gmm=agg_gmm,
                          agg_gmmg=agg_gmmg, agg_dsm_lin=agg_dsm_lin)
        if len(rho_list) > 1:
            plot_auc_vs_rho(agg_amf, agg_lrao, agg_dsm, agg_rho_d,
                            n_list, d_list, rho_list, amp_list, D_raw, tm,
                            ref_n, ref_d, ref_amp, f"{base} — {tm}",
                            os.path.join(fig_dir, f'auc_vs_rho_{tm}.pdf'), agg_gmm=agg_gmm,
                            agg_gmmg=agg_gmmg, agg_dsm_lin=agg_dsm_lin)
        if len(amp_list) > 1:
            plot_auc_vs_amp(agg_amf, agg_lrao, agg_dsm, agg_rho_d,
                            n_list, d_list, rho_list, amp_list, D_raw, tm,
                            ref_n, ref_d, ref_rho, f"{base} — {tm}",
                            os.path.join(fig_dir, f'auc_vs_amp_{tm}.pdf'), agg_gmm=agg_gmm,
                            agg_gmmg=agg_gmmg, agg_dsm_lin=agg_dsm_lin)

    print(f"\nDone.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
