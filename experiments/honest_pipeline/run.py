"""
Honest pipeline experiment — demonstrates the derived validity conditions.

Operational setting:
  - Background = one or more classes; the TARGET class is REMOVED from the
    background entirely (the target is "not in the image").
  - Normalization is calibrated on background only (per-band std by default).
  - Target signature = mean of the held-out target class (external, raw units),
    carried through the pipeline with the model-correct rule.

What it shows:
  - Sweeps PCA dimension d.
  - For each d: the THEORETICAL retained-deflection fraction rho_d AND the
    EMPIRICAL detection AUC of DSM / AMF in PCA-d space.
  - The full-D AMF (no PCA) as the detectability ceiling.
  - Validity prediction: AUC tracks rho_d, and PCA-d AUC -> full-D AUC as
    rho_d -> 1.  This is condition (2) of the derivation made visible.

Usage:
    .venv/bin/python -u experiments/honest_pipeline/run.py
    .venv/bin/python -u experiments/honest_pipeline/run.py --config experiments/honest_pipeline/config.yaml
"""

import argparse, os, sys, json, time
from datetime import datetime

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP); sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import scipy.io

from pipeline import (HonestDetectionPipeline, amf_score, amf_replacement_score)
from final_paper_experiments.data_utils import compute_sigma_from_data

CLS_NAMES = {1:'asphalt',2:'meadows',3:'gravel',4:'trees',5:'metal_sheets',
             6:'bare_soil',7:'bitumen',8:'bricks',9:'shadows'}


def auc(lab, sc):
    try:    return float(roc_auc_score(lab, sc))
    except: return float('nan')


def plant(test_bkg, t_raw, amp, frac, model, seed):
    rng = np.random.RandomState(seed)
    n = len(test_bkg); k = int(frac * n)
    pos = rng.choice(n, k, replace=False)
    y = test_bkg.copy().astype(np.float32)
    lab = np.zeros(n, dtype=int); lab[pos] = 1
    if model == 'additive':
        y[pos] = y[pos] + amp * t_raw
    else:  # replacement
        y[pos] = (1 - amp) * y[pos] + amp * t_raw
    return y, lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(_EXP, 'config.yaml'))
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'honest_{ts}')
    fig_dir = os.path.join(run_dir, 'figures'); os.makedirs(fig_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    seed = int(cfg['seed']); rng = np.random.default_rng(seed)

    # ---- load RAW data (pipeline does its own normalization) ----
    mat = scipy.io.loadmat(cfg['dataset'])
    data = mat['data'].astype(np.float32); gt = mat['map'].astype(int)
    H, W, D = data.shape
    flat = data.reshape(-1, D); gt_flat = gt.reshape(-1)
    print(f"Image {H}x{W}x{D}  (raw radiance)", flush=True)

    # ---- operational split: TARGET class removed from background ----
    tcls = cfg['target_cls']
    if cfg.get('bkg_cls', None) is not None:
        bkg_all = flat[gt_flat == cfg['bkg_cls']]
    else:
        bkg_all = flat[(gt_flat != 0) & (gt_flat != tcls)]   # all non-target labeled
    tgt_all = flat[gt_flat == tcls]
    t_raw   = tgt_all.mean(axis=0).astype(np.float32)        # external signature (raw)
    print(f"background: {len(bkg_all)} px  |  target cls {tcls} "
          f"({CLS_NAMES.get(tcls,'?')}): {len(tgt_all)} px (REMOVED from bkg)\n",
          flush=True)

    # ---- shuffle bkg, split train / test ----
    idx = rng.permutation(len(bkg_all))
    n_tr, n_te = cfg['train_n'], cfg['test_n']
    assert len(bkg_all) >= n_tr + n_te, "not enough background pixels"
    tr_raw = bkg_all[idx[:n_tr]]
    te_raw = bkg_all[idx[n_tr:n_tr + n_te]]

    d_list = sorted(set(cfg['latent_dim_list']))
    norm   = cfg['norm']
    results = {'additive': {}, 'replacement': {}, 'rho_d': {}}

    # ---- full-D AMF ceiling (no PCA), in normalized space ----
    pipe_full = HonestDetectionPipeline(latent_dim=D, norm=norm).fit(tr_raw)
    tr_n   = pipe_full.normalize(tr_raw)
    full_amf = {}
    for tm in ('additive', 'replacement'):
        te_plant, lab = plant(te_raw, t_raw, cfg['amplitude'],
                              cfg['target_fraction'], tm, seed)
        te_n = pipe_full.normalize(te_plant)
        if tm == 'additive':
            s_full = (t_raw * pipe_full.A).astype(np.float32)        # A t
            sc = amf_score(tr_n, te_n, s_full)
        else:
            s_full = ((t_raw - pipe_full.c) * pipe_full.A).astype(np.float32)  # A(t-c)
            sc = amf_replacement_score(tr_n, te_n, s_full)
        full_amf[tm] = auc(lab, sc)
    print(f"[full-D AMF ceiling]  add={full_amf['additive']:.3f}  "
          f"rep={full_amf['replacement']:.3f}\n", flush=True)

    # ---- sweep over d ----
    for d in d_list:
        t0 = time.time()
        pipe = HonestDetectionPipeline(latent_dim=d, norm=norm).fit(tr_raw)

        rho = pipe.rho_d(t_raw)
        results['rho_d'][d] = rho

        tr_pca = pipe.project(tr_raw)
        sigma  = compute_sigma_from_data(tr_pca, cfg['dsm_sigma_rho'])
        pipe.train_dsm(tr_pca, sigma,
                       hidden=cfg['hidden_dims'], activation=cfg['activation'],
                       epochs=cfg['dsm_epochs'], lr=cfg['lr'],
                       weight_decay=cfg['weight_decay'],
                       batch_size=cfg['batch_size'], seed=seed)

        for tm in ('additive', 'replacement'):
            te_plant, lab = plant(te_raw, t_raw, cfg['amplitude'],
                                  cfg['target_fraction'], tm, seed)
            te_pca = pipe.project(te_plant)
            if tm == 'additive':
                s_pca = pipe.signature_additive(t_raw)
                sc_dsm = pipe.score_dsm_additive(tr_pca, te_pca, s_pca)
                sc_amf = amf_score(tr_pca, te_pca, s_pca)
            else:
                s_pca = pipe.signature_replacement(t_raw)
                sc_dsm = pipe.score_dsm_replacement(tr_pca, te_pca, s_pca)
                sc_amf = amf_replacement_score(tr_pca, te_pca, s_pca)
            results[tm].setdefault('DSM', {})[d] = auc(lab, sc_dsm)
            results[tm].setdefault('AMF-PCA', {})[d] = auc(lab, sc_amf)

        print(f"  d={d:>3}  rho_d={rho:.3f}  "
              f"[add] DSM={results['additive']['DSM'][d]:.3f} "
              f"AMF={results['additive']['AMF-PCA'][d]:.3f}  "
              f"[rep] DSM={results['replacement']['DSM'][d]:.3f} "
              f"AMF={results['replacement']['AMF-PCA'][d]:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- save metrics ----
    out = {'norm': norm, 'target_cls': tcls,
           'target_name': CLS_NAMES.get(tcls, '?'),
           'full_amf': full_amf, 'd_list': d_list, **results}
    json.dump(out, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)

    # ---- figure: rho_d (theory) overlaid with AUC (empirical) ----
    for tm in ('additive', 'replacement'):
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ds = d_list
        rho = [results['rho_d'][d] for d in ds]
        ax1.plot(ds, rho, 'k--o', lw=2, label=r'$\rho_d$ (retained deflection, theory)')
        ax1.set_xlabel('PCA dimension  d'); ax1.set_ylabel(r'$\rho_d$')
        ax1.set_ylim(0, 1.05); ax1.set_xscale('log')

        ax2 = ax1.twinx()
        ax2.plot(ds, [results[tm]['DSM'][d] for d in ds], 'D-',
                 color='#d62728', lw=2, label='DSM AUC')
        ax2.plot(ds, [results[tm]['AMF-PCA'][d] for d in ds], 's-',
                 color='#1f77b4', lw=2, label='AMF-PCA AUC')
        ax2.axhline(full_amf[tm], color='gray', ls=':', lw=1.5,
                    label=f'full-D AMF ceiling ({full_amf[tm]:.3f})')
        ax2.set_ylabel('AUC'); ax2.set_ylim(0.4, 1.02)

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='lower right')
        ax1.set_title(f'{tm.capitalize()} — target={CLS_NAMES.get(tcls,"?")}, '
                      f'norm={norm}\nAUC tracks $\\rho_d$; PCA AUC → ceiling as '
                      f'$\\rho_d\\to1$')
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f'validity_{tm}.pdf'), bbox_inches='tight')
        plt.close(fig)

    print(f"\nDone.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
