"""
Paper reproduction sweep — IID single-class experiment.

Runs the full n_train sweep with N seeds (default 5) and produces:
  - Per-seed metrics saved in   results/iid_single_paper_<ts>/seed_<k>/
  - Aggregated metrics.json     results/iid_single_paper_<ts>/metrics_agg.json
  - AUC vs n plots with ±1 std  results/iid_single_paper_<ts>/figures/

Exact paper settings (Section 4.1 of main.pdf):
  norm_mode   = per_band        (per-band [0,1] normalization)
  latent_dim  = 5               (d = 5, PCA on all pixels)
  signature   = l2-normalized mean of target-class PCA projections
  rho         = 5e-3            (DSM noise level)
  dsm_epochs  = 8000
  lrao_epochs = 3000 (with early stopping, patience=3)
  n_train     = {50, 100, 200, 300, 500, 750, 1000, 2000}
  test_size   = 2000 (1800 background + 200 planted targets at amplitude=0.15)
  seeds       = 5 independent repeats

Usage:
    .venv/bin/python -u experiments/iid_single/run_paper_sweep.py
    .venv/bin/python -u experiments/iid_single/run_paper_sweep.py --config experiments/iid_single/paper_sweep.yaml
"""

import argparse, os, sys, json, time, copy
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from iid_core import run_iid, _det_color

# Detectors that appear in the paper figures
PAPER_DETS_ADD = ['AMF', 'Reg-AMF', 'DSM-PCA', 'LRao-PCA']
PAPER_DETS_REP = ['AMF-rep', 'DSM-PCA-rep', 'LRao-PCA']


def _collect_auc_curves(all_metrics, n_list, branch, tm, det):
    """Extract per-seed AUC list for one detector."""
    out = []
    for m in all_metrics:
        vals = m.get(branch, {}).get(tm, {}).get(det)
        if vals is not None and len(vals) == len(n_list):
            out.append(vals)
    return np.array(out) if out else None   # (n_seeds, n_n_train)


def plot_sweep(all_metrics, n_list, d, tm, dets_classical, dets_score, out_pdf):
    """AUC vs n_train with ±1 std bands — paper Fig. 2 style."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = list(range(len(n_list)))

    for det in dets_classical:
        curves = _collect_auc_curves(all_metrics, n_list, 'classical', tm, det)
        if curves is None: continue
        mu = curves.mean(0); sd = curves.std(0)
        c  = _det_color(det)
        ax.plot(x, mu, 'o-', color=c, lw=2, label=det)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)

    for det in dets_score:
        curves = _collect_auc_curves(all_metrics, n_list,
                                     f'score.d_{d}' if '.' in 'score.d_5'
                                     else 'score', tm, det)
        # score metrics are nested: metrics['score'][f'd_{d}'][tm][det]
        # re-extract properly
        out = []
        for m in all_metrics:
            vals = (m.get('score', {})
                      .get(f'd_{d}', {})
                      .get(tm, {})
                      .get(det))
            if vals is not None and len(vals) == len(n_list):
                out.append(vals)
        if not out: continue
        curves = np.array(out)
        mu = curves.mean(0); sd = curves.std(0)
        c  = _det_color(det)
        ax.plot(x, mu, 'D-', color=c, lw=2, label=det)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_list], rotation=30)
    ax.set_xlabel('Training samples  n')
    ax.set_ylabel('AUC')
    ax.set_title(f'd={d}  {tm}  (mean ± 1 std, {len(all_metrics)} seeds)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_pdf}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'paper_sweep.yaml'))
    args = p.parse_args()
    cfg_base = yaml.safe_load(open(args.config))

    seeds  = cfg_base.pop('seeds', [42, 43, 44, 45, 46])
    n_list = sorted(set(cfg_base['n_train_list']))
    d_list = sorted(set(cfg_base['latent_dim_list']))

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    agg_dir = os.path.join(cfg_base['results_dir'], f'iid_single_paper_{ts}')
    fig_dir = os.path.join(agg_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    yaml.dump({'seeds': seeds, **cfg_base},
              open(os.path.join(agg_dir, 'config.yaml'), 'w'),
              sort_keys=False)
    print(f"Paper sweep  →  {agg_dir}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"n_train: {n_list}", flush=True)
    print(f"latent_dims: {d_list}\n", flush=True)

    all_metrics = []
    seed_dirs   = []
    t0 = time.time()

    for k, seed in enumerate(seeds):
        print(f"\n{'='*60}", flush=True)
        print(f"Seed {k+1}/{len(seeds)}  (seed={seed})", flush=True)
        print('='*60, flush=True)
        cfg = copy.deepcopy(cfg_base)
        cfg['seed'] = seed
        cfg['results_dir'] = os.path.join(agg_dir, f'seed_{seed}')
        os.makedirs(cfg['results_dir'], exist_ok=True)

        run_dir, metrics = run_iid(cfg, mode='single')
        all_metrics.append(metrics)
        seed_dirs.append(run_dir)

    # ---- aggregate ----
    print(f"\n{'='*60}", flush=True)
    print("Aggregating results ...", flush=True)

    # Build summary: mean ± std for every (det, n, tm)
    agg = {'n_train_list': n_list, 'latent_dim_list': d_list,
           'seeds': seeds, 'n_seeds': len(seeds)}

    for det in ['AMF', 'Reg-AMF', 'GMM-GLRT', 'Exact-GLRT',
                'AMF-rep', 'CEM']:
        for tm in ('additive', 'replacement'):
            curves = _collect_auc_curves(all_metrics, n_list, 'classical', tm, det)
            if curves is None: continue
            key = f'classical/{det}/{tm}'
            agg[key] = {'mean': curves.mean(0).tolist(),
                        'std':  curves.std(0).tolist()}

    for d in d_list:
        for tm in ('additive', 'replacement'):
            score_branch = (all_metrics[0].get('score', {})
                                          .get(f'd_{d}', {})
                                          .get(tm, {}))
            for det in score_branch.keys():
                out = []
                for m in all_metrics:
                    vals = (m.get('score', {})
                              .get(f'd_{d}', {})
                              .get(tm, {})
                              .get(det))
                    if vals is not None and len(vals) == len(n_list):
                        out.append(vals)
                if not out: continue
                curves = np.array(out)
                key = f'score/d{d}/{tm}/{det}'
                agg[key] = {'mean': curves.mean(0).tolist(),
                            'std':  curves.std(0).tolist()}

    json.dump(agg, open(os.path.join(agg_dir, 'metrics_agg.json'), 'w'), indent=2)

    # ---- print summary table ----
    for tm in ('additive', 'replacement'):
        print(f"\n=== {tm.upper()} (mean AUC @ n={n_list[-1]}) ===")
        n_idx = len(n_list) - 1
        for key, v in agg.items():
            if f'/{tm}/' not in key: continue
            det = key.split('/')[-1] if 'classical' in key else key.split('/')[-1]
            mu  = v['mean'][n_idx]; sd = v['std'][n_idx]
            print(f"  {key.split('/')[1]:>20}   {mu:.3f} ± {sd:.3f}")

    # ---- figures ----
    for d in d_list:
        for tm in ('additive', 'replacement'):
            cl = PAPER_DETS_ADD if tm == 'additive' else PAPER_DETS_REP
            sc = ['DSM-PCA'] if tm == 'additive' else ['DSM-PCA-rep']
            # re-use the structured data from all_metrics
            plot_sweep(all_metrics, n_list, d, tm,
                       dets_classical=[k for k in cl if 'DSM' not in k and 'LRao' not in k],
                       dets_score=[k for k in cl if 'DSM' in k or 'LRao' in k],
                       out_pdf=os.path.join(fig_dir, f'auc_vs_n_d{d}_{tm}.pdf'))

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal: {elapsed:.1f} min  →  {agg_dir}", flush=True)


if __name__ == '__main__':
    main()
