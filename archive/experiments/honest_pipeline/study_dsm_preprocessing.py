"""
study_dsm_preprocessing.py — Diagnostic: DSM learning depends on preprocessing.

Mission 4A. Demonstrates that denoising score matching (DSM) only produces a
useful detection statistic when the background features are properly *scaled*.
In the raw pixel space, or in an un-normalised PCA space, the DSM either fails
to learn a meaningful score or yields a near-chance detector — whereas a simple
per-band standardisation recovers strong performance.

Single-class background (class 2, meadows), additive target (class 1, asphalt),
fixed n, d, rho. Five preprocessing variants compared:

    raw         skip_pca=True ,  norm=none          (full 103-D raw pixels)
    pca_only    skip_pca=False,  norm=none          (PCA-d, NO scaling)
    pca_std     skip_pca=False,  norm=per_band_std  (the working pipeline)
    pca_minmax  skip_pca=False,  norm=per_band_minmax
    pca_elm     skip_pca=False,  norm=elm            (robust 1/99 pct scaling)

AMF (full-D, global_max, scale-invariant) is shown as a reference.

Outputs (into --out dir):
    study_dsm_preprocessing.npz     per-variant AUC/pAUC + DSM loss curves
    fig_dsm_preproc_auc.pdf         DSM AUC & pAUC vs preprocessing (bars)
    fig_dsm_preproc_loss.pdf        DSM training-loss curves (log-y)

Usage:
    .venv/bin/python -u experiments/honest_pipeline/study_dsm_preprocessing.py
    .venv/bin/python -u experiments/honest_pipeline/study_dsm_preprocessing.py --quick
"""

import argparse, os, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP); sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import scipy.io
from pipeline import HonestDetectionPipeline, amf_score
from dsm_model import ScoreNet, dsm_loss, compute_scores
from final_paper_experiments.data_utils import compute_sigma_from_data
from make_pauc_figures import partial_auc_normalized, dr_at_fpr   # reuse metrics

# Variant table: (label, skip_pca, norm)
VARIANTS = [
    ('raw',        True,  'none'),
    ('pca_only',   False, 'none'),
    ('pca_std',    False, 'per_band_std'),
    ('pca_minmax', False, 'per_band_minmax'),
    ('pca_elm',    False, 'elm'),
]
VAR_LABELS = {'raw': 'raw 103-D', 'pca_only': 'PCA only\n(no scale)',
              'pca_std': 'PCA + std\n(ours)', 'pca_minmax': 'PCA + minmax',
              'pca_elm': 'PCA + elm'}


def plant_additive(bkg, s, amp, frac, seed):
    rng = np.random.RandomState(seed)
    n = len(bkg); k = int(frac * n)
    pos = rng.choice(n, k, replace=False)
    y = bkg.copy().astype(np.float32); lab = np.zeros(n, dtype=int); lab[pos] = 1
    y[pos] += amp * s
    return y, lab


def train_dsm_history(d, tr_pca, sigma, hidden, activation, epochs, lr, wd, bs, seed,
                      device='cpu'):
    """Train DSM, returning (model, per-epoch mean loss list)."""
    torch.manual_seed(seed)
    model = ScoreNet(d, list(hidden), activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(tr_pca, dtype=torch.float32).to(device)
    N, bs = len(X), min(bs, len(tr_pca))
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(N); tot = 0.0; nb = 0
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        hist.append(tot / max(nb, 1))
    model.eval()
    return model, hist


def score_dsm_additive(model, tr_pca, te_pca, s):
    z_tr = compute_scores(model, tr_pca); z_te = compute_scores(model, te_pca)
    z_bar = z_tr.mean(0)
    C = np.cov(z_tr, rowvar=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    norm = float(np.sqrt(max(float(s @ C @ s), 1e-12)))
    return -((z_te - z_bar) @ s) / norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/pavia-u.mat')
    p.add_argument('--out', default='experiments/honest_pipeline/results/study_dsm_preproc')
    p.add_argument('--target_cls', type=int, default=1)
    p.add_argument('--bkg_cls',    type=int, default=2)
    p.add_argument('--n',     type=int, default=1000)
    p.add_argument('--d',     type=int, default=20)
    p.add_argument('--rho',   type=float, default=0.01)
    p.add_argument('--amp',   type=float, default=0.15)
    p.add_argument('--frac',  type=float, default=0.10)
    p.add_argument('--test_n', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=1500)
    p.add_argument('--lr',    type=float, default=1e-3)
    p.add_argument('--wd',    type=float, default=1e-4)
    p.add_argument('--bs',    type=int, default=256)
    p.add_argument('--hidden', type=int, nargs='*', default=[64, 64])
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44, 45, 46])
    p.add_argument('--device', default='auto', help="'auto'|'cuda'|'cpu'")
    p.add_argument('--quick', action='store_true',
                   help='tiny settings for a local smoke test')
    args = p.parse_args()

    if args.quick:
        args.epochs = 60; args.seeds = [42]; args.n = 300; args.test_n = 500

    dev = ('cuda' if (args.device in ('auto', 'cuda', 'gpu')
                      and torch.cuda.is_available()) else 'cpu')
    print(f"Device: {dev}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    # ---- load single-class background + target ----
    mat  = scipy.io.loadmat(args.dataset)
    data = mat['data'].astype(np.float32); gt = mat['map'].astype(int)
    D_raw = data.shape[-1]
    flat = data.reshape(-1, D_raw); gt_flat = gt.reshape(-1)
    bkg_all = flat[gt_flat == args.bkg_cls]
    t_raw   = flat[gt_flat == args.target_cls].mean(0).astype(np.float32)
    print(f"bkg(cls {args.bkg_cls})={len(bkg_all)}px  "
          f"target(cls {args.target_cls})  hidden={args.hidden}", flush=True)

    # results[variant] = {'auc': [...per seed], 'pauc': [...], 'loss': [hist per seed]}
    results = {v[0]: {'auc': [], 'pauc': [], 'loss': []} for v in VARIANTS}
    amf_auc, amf_pauc = [], []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(bkg_all))
        bkg_tr = bkg_all[idx[:args.n]]
        bkg_te = bkg_all[idx[args.n:args.n + args.test_n]]

        # AMF reference (full-D global_max, scale-invariant)
        gm = float(bkg_tr.max() + 1e-12)
        t_gm = t_raw / gm
        te_gm, lab = plant_additive(bkg_te / gm, t_gm, args.amp, args.frac, seed)
        sc = amf_score(bkg_tr / gm, te_gm, t_gm)
        fpr, tpr, _ = roc_curve(lab, sc)
        amf_auc.append(float(sk_auc(fpr, tpr)))
        amf_pauc.append(partial_auc_normalized(fpr, tpr, 0.1))

        for name, skip_pca, norm in VARIANTS:
            pipe = HonestDetectionPipeline(latent_dim=args.d, norm=norm,
                                           skip_pca=skip_pca)
            pipe.fit(bkg_tr)
            d_eff  = pipe.d
            tr_pca = pipe.project(bkg_tr)
            te_pca = pipe.project(bkg_te)
            s_add  = pipe.signature_additive(t_raw)
            sigma  = compute_sigma_from_data(tr_pca, args.rho)

            model, hist = train_dsm_history(
                d_eff, tr_pca, sigma, args.hidden, 'silu',
                args.epochs, args.lr, args.wd, args.bs, seed, device=dev)

            te_pl, lab = plant_additive(te_pca, s_add, args.amp, args.frac, seed)
            sc = score_dsm_additive(model, tr_pca, te_pl, s_add)
            fpr, tpr, _ = roc_curve(lab, sc)
            a  = float(sk_auc(fpr, tpr))
            pa = partial_auc_normalized(fpr, tpr, 0.1)
            results[name]['auc'].append(a)
            results[name]['pauc'].append(pa)
            results[name]['loss'].append(hist)
            print(f"  seed={seed}  {name:11s} d={d_eff:3d}  AUC={a:.3f}  pAUC={pa:.3f}",
                  flush=True)

    # ---- aggregate ----
    def ms(x): x = np.asarray(x, float); return float(x.mean()), float(x.std())
    summary = {name: {'auc': ms(results[name]['auc']),
                      'pauc': ms(results[name]['pauc'])}
               for name, _, _ in VARIANTS}
    summary['AMF'] = {'auc': ms(amf_auc), 'pauc': ms(amf_pauc)}
    print("\n=== DSM AUC by preprocessing ===")
    for k in summary:
        print(f"  {k:11s}: AUC={summary[k]['auc'][0]:.3f}  pAUC={summary[k]['pauc'][0]:.3f}")

    np.savez_compressed(
        os.path.join(args.out, 'study_dsm_preprocessing.npz'),
        variants=[v[0] for v in VARIANTS],
        auc={k: results[k]['auc'] for k in results},
        pauc={k: results[k]['pauc'] for k in results},
        loss={k: np.array(results[k]['loss'], dtype=object) for k in results},
        amf_auc=amf_auc, amf_pauc=amf_pauc,
        meta=dict(n=args.n, d=args.d, rho=args.rho, amp=args.amp,
                  hidden=args.hidden, epochs=args.epochs, seeds=args.seeds),
        allow_pickle=True)

    _plot_auc(summary, args, os.path.join(args.out, 'fig_dsm_preproc_auc.pdf'))
    _plot_loss(results, args, os.path.join(args.out, 'fig_dsm_preproc_loss.pdf'))
    print(f"\nSaved study to {args.out}/", flush=True)


_STYLE = {'font.family': 'serif', 'axes.spines.top': False,
          'axes.spines.right': False, 'figure.dpi': 200}


def _plot_auc(summary, args, out_path):
    names = [v[0] for v in VARIANTS]
    labels = [VAR_LABELS[n] for n in names]
    auc_mu = [summary[n]['auc'][0] for n in names]
    auc_sd = [summary[n]['auc'][1] for n in names]
    pa_mu  = [summary[n]['pauc'][0] for n in names]
    pa_sd  = [summary[n]['pauc'][1] for n in names]
    colors = ['#b0b0b0', '#b0b0b0', '#d62728', '#7f7f7f', '#7f7f7f']
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        x = np.arange(len(names))
        for ax, mu, sd, ttl in [(axes[0], auc_mu, auc_sd, 'AUC'),
                                 (axes[1], pa_mu, pa_sd, 'pAUC (0→0.1)')]:
            ax.bar(x, mu, yerr=sd, color=colors, capsize=3, alpha=0.9)
            ax.axhline(summary['AMF'][ttl == 'pAUC (0→0.1)' and 'pauc' or 'auc'][0],
                       color='#1f77b4', ls='--', lw=1.6, label='AMF (ref)')
            ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.5)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_ylabel(ttl, fontsize=9); ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7.5, loc='upper left')
        fig.suptitle(f'DSM detection vs preprocessing  '
                     f'(single-class, n={args.n}, d={args.d}, '
                     f'$\\rho$={args.rho}, hidden={args.hidden})', fontsize=9, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


def _plot_loss(results, args, out_path):
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        cmap = {'raw': '#7f7f7f', 'pca_only': '#ff7f0e', 'pca_std': '#d62728',
                'pca_minmax': '#2ca02c', 'pca_elm': '#9467bd'}
        for name in results:
            curves = results[name]['loss']
            if not curves:
                continue
            L = min(len(c) for c in curves)
            arr = np.array([c[:L] for c in curves])
            mu = arr.mean(0)
            ax.plot(np.arange(1, L + 1), mu, color=cmap.get(name, None),
                    lw=1.5, label=VAR_LABELS[name].replace('\n', ' '))
        ax.set_yscale('log')
        ax.set_xlabel('epoch', fontsize=9)
        ax.set_ylabel('DSM loss (mean over seeds, log)', fontsize=9)
        ax.set_title('DSM training loss by preprocessing', fontsize=9)
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    print(f"  saved {out_path}", flush=True)


if __name__ == '__main__':
    main()
