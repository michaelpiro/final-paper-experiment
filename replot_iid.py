"""
replot_iid.py — Re-generate IID experiment figures from a saved aggregate directory.

Usage:
    python replot_iid.py <agg_dir> [--width W] [--height H] [--legend inside|outside]

Reads  <agg_dir>/metrics_aggregate.json  and rewrites
<agg_dir>/figures/*.pdf + *.png  with narrower dimensions.

Defaults: --width 4.0  --height 3.0  --legend inside
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# ---------------------------------------------------------------------------
DETECTOR_COLORS = {
    'AMF':       '#1f77b4',
    'GMM-Levin': '#9467bd',
    'DLTD':      '#e6550d',
    'SMGLRT':    '#8c564b',
    'DSM':       '#d62728',
    'DSM-lin':   '#9b2226',
    'DSM-MLP':   '#e07070',
    'LRao':      '#2ca02c',
}
OUR_DETS = {'DSM', 'DSM-lin', 'DSM-MLP', 'LRao'}

def _color(det):
    return DETECTOR_COLORS.get(det, '#444444')

def _savefig(fig, pdf_path, dpi=200):
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(pdf_path.replace('.pdf', '.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {os.path.basename(pdf_path)}")


def _apply_log_xticks(ax, x):
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')
        lbl.set_rotation_mode('anchor')
        lbl.set_fontsize(7)


def plot_vs(xvals, series, xlabel, ylabel, title, out_pdf,
            logx=False, series_std=None, figsize=(4.0, 3.0), legend='inside'):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.asarray(xvals, dtype=float)
    for det, ys in series.items():
        if ys is None or all(v != v for v in ys):
            continue
        ys = np.asarray(ys, dtype=float)
        style = dict(marker='D', lw=2.0, ms=4) if det in OUR_DETS \
            else dict(marker='o', lw=1.2, ms=3)
        c = _color(det)
        ax.plot(x, ys, color=c, label=det, **style)
        if series_std and det in series_std and series_std[det] is not None:
            sd = np.asarray(series_std[det], dtype=float)
            ax.fill_between(x, ys - sd, ys + sd, alpha=0.15, color=c)
    if logx:
        _apply_log_xticks(ax, x)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, which='both')
    if legend == 'outside':
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.)
    else:
        ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    _savefig(fig, out_pdf)


def plot_bar(dets, means, stds, title, out_pdf, n_seeds,
             figsize=(4.0, 2.8)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = [_color(d) for d in dets]
    ax.bar(dets, means, yerr=stds, color=colors, capsize=3, alpha=0.85,
           error_kw={'lw': 1.0})
    ax.set_ylabel('AUC', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7, axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _savefig(fig, out_pdf)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('agg_dir', help='Path to iid_*_agg_* directory')
    ap.add_argument('--width',  type=float, default=4.0,
                    help='Figure width in inches (default 4.0)')
    ap.add_argument('--height', type=float, default=3.0,
                    help='Figure height in inches (default 3.0)')
    ap.add_argument('--legend', choices=['inside', 'outside'], default='inside',
                    help='Legend placement (default inside)')
    args = ap.parse_args()

    agg_dir  = os.path.abspath(args.agg_dir)
    fig_dir  = os.path.join(agg_dir, 'figures')
    json_path = os.path.join(agg_dir, 'metrics_aggregate.json')

    if not os.path.exists(json_path):
        sys.exit(f"ERROR: {json_path} not found")
    os.makedirs(fig_dir, exist_ok=True)

    with open(json_path) as f:
        agg = json.load(f)

    fw = (args.width, args.height)
    leg = args.legend
    DETS     = list(agg['vs_n'].keys())
    n_list   = agg['n_list']
    rho_list = agg['rho_list']
    n_fixed  = agg['n_fixed']
    pfa      = agg['pfa']
    pauc_fpr = agg['pauc_fpr']
    n_seeds  = len(agg.get('seeds', []))
    suf = f'  ({n_seeds} seeds)' if n_seeds > 1 else ''

    def _m(sweep, metric):
        return {d: agg[sweep][d].get(metric) for d in DETS}

    def _s(sweep, metric):
        return {d: agg[sweep][d].get(f'{metric}_std') for d in DETS}

    print(f"Replotting  {agg_dir}")
    print(f"  figsize={fw}  legend={leg}  detectors={DETS}")

    # vs-n
    plot_vs(n_list, _m('vs_n','auc'), 'training samples  n', 'AUC',
            f'AUC vs n{suf}',
            os.path.join(fig_dir,'auc_vs_n.pdf'),
            logx=True, series_std=_s('vs_n','auc'), figsize=fw, legend=leg)

    plot_vs(n_list, _m('vs_n','pauc'),
            'training samples  n', f'partial AUC (Pfa<{pauc_fpr})',
            f'Partial AUC (Pfa<{pauc_fpr}) vs n{suf}',
            os.path.join(fig_dir,'pauc_vs_n.pdf'),
            logx=True, series_std=_s('vs_n','pauc'), figsize=fw, legend=leg)

    plot_vs(n_list, _m('vs_n','pd'),
            'training samples  n', f'Pd @ Pfa={pfa}',
            f'Pd @ Pfa={pfa} vs n{suf}',
            os.path.join(fig_dir,'pd_at_fa_vs_n.pdf'),
            logx=True, series_std=_s('vs_n','pd'), figsize=fw, legend=leg)

    # vs-rho
    plot_vs(rho_list, _m('vs_rho','auc'),
            r'DSM noise level  $\rho$', 'AUC',
            f'AUC vs ρ (n={n_fixed}){suf}',
            os.path.join(fig_dir,'auc_vs_rho.pdf'),
            logx=True, series_std=_s('vs_rho','auc'), figsize=fw, legend=leg)

    plot_vs(rho_list, _m('vs_rho','pd'),
            r'DSM noise level  $\rho$', f'Pd @ Pfa={pfa}',
            f'Pd @ Pfa={pfa} vs ρ (n={n_fixed}){suf}',
            os.path.join(fig_dir,'pdet_at_pfa_vs_rho.pdf'),
            logx=True, series_std=_s('vs_rho','pd'), figsize=fw, legend=leg)

    # AUC bar
    if 'roc_at_nmax' in agg:
        dets_r = list(agg['roc_at_nmax'].keys())
        means  = [agg['roc_at_nmax'][d]['auc_mean'] for d in dets_r]
        stds   = [agg['roc_at_nmax'][d]['auc_std']  for d in dets_r]
        plot_bar(dets_r, means, stds,
                 f'AUC at n={n_list[-1]}  (mean±std, {n_seeds} seeds)',
                 os.path.join(fig_dir,'roc_auc_bar_nmax.pdf'),
                 n_seeds, figsize=(max(fw[0], 3.0), fw[1]))

    print("Done.")


if __name__ == '__main__':
    main()
