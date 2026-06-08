"""
study_lrao_sensitivity.py — Diagnostic: LRao is fast-converging but fragile.

Mission 4B. Demonstrates three properties of the LRao (linear-Fisher) detector:

  (a) CONVERGENCE — the LFI objective tr(J*) saturates after only a handful of
      epochs; long training buys nothing. Shown by the per-epoch trace.

  (b) NOT ADAPTED TO MULTICLASS — on a heterogeneous (multimodal) background the
      single global score/covariance cannot adapt, so LRao trails DSM/GMM-Levin.

  (c) DEGRADES WITH SAMPLES (multiclass) — adding background samples makes the
      global covariance/Jacobian estimate *worse*, so multiclass LRao AUC drops
      as n grows, unlike the stable single-class case.

(a) is computed here (the sweep does not record per-epoch traces). (b)+(c) are
read from the main n-sweep caches produced by make_pauc_figures.py so the numbers
match the paper exactly; if the caches are absent those panels are skipped.

Outputs (into --out):
    study_lrao_sensitivity.npz
    fig_lrao_convergence.pdf       tr(J*) vs epoch (single vs multi)
    fig_lrao_auc_vs_n.pdf          LRao AUC vs n, single vs multi (+DSM ref)
    fig_lrao_multiclass_gap.pdf    LRao vs DSM vs GMM-Levin vs AMF @ largest n

Usage:
    .venv/bin/python -u experiments/honest_pipeline/study_lrao_sensitivity.py \
        --single_cache experiments/honest_pipeline/results/pauc_figures/cache_single_n.pkl \
        --multi_cache  experiments/honest_pipeline/results/pauc_figures/cache_multi_n.pkl
    ... add --quick for a local smoke test of the convergence panel only.
"""

import argparse, os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP); sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import scipy.io
from pipeline import HonestDetectionPipeline
from dsm_model import ScoreNet, lfi_loss_mode2

_STYLE = {'font.family': 'serif', 'axes.spines.top': False,
          'axes.spines.right': False, 'figure.dpi': 200}
C_SINGLE = '#1f77b4'
C_MULTI  = '#d62728'


def load_bkg(dataset, target_cls, multi):
    mat  = scipy.io.loadmat(dataset)
    data = mat['data'].astype(np.float32); gt = mat['map'].astype(int)
    D = data.shape[-1]
    flat = data.reshape(-1, D); gt_flat = gt.reshape(-1)
    if multi:
        mask = (gt_flat != 0) & (gt_flat != target_cls)
    else:
        mask = gt_flat == 2          # meadows single-class background
    return flat[mask]


def train_lrao_history(d, tr_pca, hidden, epochs, delta, cutoff, lr, wd, bs, seed,
                       device='cpu'):
    """Train LRao, returning per-epoch tr(J*) (= -loss)."""
    torch.manual_seed(seed)
    model = ScoreNet(d, list(hidden), 'silu').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(tr_pca, dtype=torch.float32).to(device)
    N, bs = len(X), min(bs, len(tr_pca))
    trace = []
    for _ in range(epochs):
        perm = torch.randperm(N); last = np.nan
        try:
            for i in range(0, N, bs):
                b = X[perm[i:i + bs]]
                loss = lfi_loss_mode2(model, b, delta, cutoff, detach_sigma=False)
                if not torch.isfinite(loss):
                    raise FloatingPointError()
                opt.zero_grad(); loss.backward(); opt.step()
                last = float(-loss)
        except Exception:
            break
        trace.append(last)
    model.eval()
    return trace


def convergence_panel(args, device='cpu'):
    """Train LRao on single & multi backgrounds, record tr(J*) per epoch."""
    traces = {}
    for tag, multi, hidden in [('single-class', False, []),
                               ('multi-class', True, [64, 64])]:
        bkg = load_bkg(args.dataset, args.target_cls, multi)
        per_seed = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(bkg))
            bkg_tr = bkg[idx[:args.n]]
            pipe = HonestDetectionPipeline(latent_dim=args.d, norm='per_band_std')
            pipe.fit(bkg_tr)
            tr_pca = pipe.project(bkg_tr)
            tr = train_lrao_history(pipe.d, tr_pca, hidden, args.epochs,
                                    args.delta, args.cutoff, args.lr, args.wd,
                                    args.bs, seed, device=device)
            per_seed.append(tr)
            print(f"  {tag}: seed={seed} trJ[0]={tr[0]:.2f} -> "
                  f"trJ[-1]={tr[-1]:.2f}  ({len(tr)} ep)", flush=True)
        traces[tag] = per_seed
    return traces


def _mean_trace(per_seed):
    L = min(len(t) for t in per_seed)
    arr = np.array([t[:L] for t in per_seed])
    return arr.mean(0), arr.std(0)


def plot_convergence(traces, args, out_path):
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for tag, c in [('single-class', C_SINGLE), ('multi-class', C_MULTI)]:
            mu, sd = _mean_trace(traces[tag])
            x = np.arange(1, len(mu) + 1)
            # normalise each curve to its final value to compare convergence shape
            muN = mu / max(abs(mu[-1]), 1e-9)
            ax.plot(x, muN, color=c, lw=1.6, label=tag)
        ax.axhline(1.0, color='k', ls=':', lw=0.8, alpha=0.5)
        ax.set_xlabel('epoch', fontsize=9)
        ax.set_ylabel('tr$(\\hat J^*)$  /  final value', fontsize=9)
        ax.set_title('LRao objective converges in a few epochs', fontsize=9)
        ax.set_xlim(0, min(80, args.epochs))   # zoom on the early plateau
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def _agg_get(agg, det, n, metric='auc'):
    try:
        return agg['additive'][det][n][metric][0]
    except (KeyError, TypeError):
        return np.nan


def plot_auc_vs_n(cache_s, cache_m, out_path, metric='auc'):
    agg_s, meta_s = cache_s['agg'], cache_s['meta']
    agg_m, meta_m = cache_m['agg'], cache_m['meta']
    ns = meta_s['n_list']; nm = meta_m['n_list']
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.plot(ns, [_agg_get(agg_s, 'LRao', n, metric) for n in ns],
                'o-', color=C_SINGLE, lw=1.6, label='LRao (single-class)')
        ax.plot(nm, [_agg_get(agg_m, 'LRao', n, metric) for n in nm],
                's-', color=C_MULTI, lw=1.6, label='LRao (multi-class)')
        ax.plot(nm, [_agg_get(agg_m, 'DSM', n, metric) for n in nm],
                '^--', color='#2ca02c', lw=1.3, alpha=0.8, label='DSM (multi-class)')
        ax.set_xscale('log')
        ax.set_xlabel('$n_{\\mathrm{train}}$', fontsize=9)
        ax.set_ylabel(metric.upper(), fontsize=9)
        ax.set_title('LRao degrades with $n$ on multiclass background', fontsize=9)
        ax.legend(fontsize=8, loc='lower left')
        fig.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def plot_multiclass_gap(cache_m, out_path, metric='auc'):
    agg_m, meta_m = cache_m['agg'], cache_m['meta']
    n = meta_m['n_list'][-1]
    dets = [d for d in ['LRao', 'AMF', 'DSM', 'GMM-GLRT', 'GMM-GLRT-G']
            if d in agg_m.get('additive', {})]
    vals = [_agg_get(agg_m, d, n, metric) for d in dets]
    labels = {'LRao': 'LRao', 'AMF': 'AMF', 'DSM': 'DSM (ours)',
              'GMM-GLRT': 'GMM-Levin', 'GMM-GLRT-G': 'GMM-Levin\n(oracle)'}
    colors = {'LRao': '#2ca02c', 'AMF': '#1f77b4', 'DSM': '#ff7f0e',
              'GMM-GLRT': '#9467bd', 'GMM-GLRT-G': '#e377c2'}
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        x = np.arange(len(dets))
        ax.bar(x, vals, color=[colors[d] for d in dets], alpha=0.9)
        ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels([labels[d] for d in dets], fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=9); ax.set_ylim(0, 1.02)
        ax.set_title(f'Multiclass detectors @ n={n}', fontsize=9)
        fig.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/pavia-u.mat')
    p.add_argument('--out', default='experiments/honest_pipeline/results/study_lrao')
    p.add_argument('--target_cls', type=int, default=1)
    p.add_argument('--single_cache',
                   default='experiments/honest_pipeline/results/pauc_figures/cache_single_n.pkl')
    p.add_argument('--multi_cache',
                   default='experiments/honest_pipeline/results/pauc_figures/cache_multi_n.pkl')
    p.add_argument('--n',     type=int, default=1000)
    p.add_argument('--d',     type=int, default=20)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--delta', type=float, default=0.001)
    p.add_argument('--cutoff', type=float, default=1e-5)
    p.add_argument('--lr',    type=float, default=1e-3)
    p.add_argument('--wd',    type=float, default=1e-4)
    p.add_argument('--bs',    type=int, default=256)
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    p.add_argument('--metric', default='auc', choices=['auc', 'pauc'])
    p.add_argument('--device', default='auto', help="'auto'|'cuda'|'cpu'")
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()

    if args.quick:
        args.epochs = 60; args.seeds = [42]; args.n = 600

    dev = ('cuda' if (args.device in ('auto', 'cuda', 'gpu')
                      and torch.cuda.is_available()) else 'cpu')
    print(f"Device: {dev}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    print("=== (a) LRao convergence trace ===", flush=True)
    traces = convergence_panel(args, device=dev)
    plot_convergence(traces, args, os.path.join(args.out, 'fig_lrao_convergence.pdf'))

    saved = {'traces': {k: np.array([t[:min(len(x) for x in v)]
                                     for t in v]) for k, v in traces.items()},
             'meta': dict(n=args.n, d=args.d, epochs=args.epochs, seeds=args.seeds)}

    have_s = os.path.exists(args.single_cache)
    have_m = os.path.exists(args.multi_cache)
    if have_s and have_m:
        print("\n=== (b,c) AUC-vs-n + multiclass gap (from caches) ===", flush=True)
        cache_s = pickle.load(open(args.single_cache, 'rb'))
        cache_m = pickle.load(open(args.multi_cache, 'rb'))
        plot_auc_vs_n(cache_s, cache_m,
                      os.path.join(args.out, 'fig_lrao_auc_vs_n.pdf'), args.metric)
        plot_multiclass_gap(cache_m,
                            os.path.join(args.out, 'fig_lrao_multiclass_gap.pdf'),
                            args.metric)
    else:
        print(f"\n[skip b,c] caches not found "
              f"(single={have_s}, multi={have_m}); run make_pauc_figures.py first.",
              flush=True)

    np.savez_compressed(os.path.join(args.out, 'study_lrao_sensitivity.npz'),
                        **saved, allow_pickle=True)
    print(f"\nSaved study to {args.out}/", flush=True)


if __name__ == '__main__':
    main()
