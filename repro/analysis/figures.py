"""repro.analysis.figures — the two paper figures, from repro run outputs.

  iid_grid(iid_root)              -> figures/iid_grid.pdf
     reads  <iid_root>/iid_<mode>/iid_<mode>_agg_*/metrics_aggregate.json
     (written by repro.protocols.iid.run_iid_multi_seed)
  amp_sweep_spatial(spatial_root) -> figures/amp_sweep_spatial.pdf
     reads  <spatial_root>/<scene>/metrics.json
     (written by repro.protocols.spatial.run_scene)

One palette for every plot, shared legends, print-sized fonts. The published
size/style knobs are unchanged from the camera-ready figures.
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG = 'figures'


# ---- the ONE palette (IID palette + spatial + deep; grayscale-separable) ----
COLORS = {
    'AMF':        '#1f77b4', 'AMF-global': '#1f77b4', 'AMF-local': '#6baed6',
    'GMM-Levin':  '#9467bd',
    'DART':       '#d62728', 'L-DART':     '#ff7f0e',
    'DART-CFAR':  '#a50f15',
    'DARTS':      '#2ca02c', 'DARTS-CFAR': '#00441b',
    'L-LRao':     '#8c6d31', 'LRao':       '#e377c2',
    'THANTD':     '#8c564b', 'HTDNet':     '#7f7f7f',
    'TSTTD':      '#17becf', 'OSVAE':      '#bcbd22',
}
MARKERS = {
    'AMF': 'o', 'AMF-global': 'o', 'AMF-local': 'h', 'GMM-Levin': '^',
    'DART': 'D', 'L-DART': 's', 'DART-CFAR': 'd',
    'DARTS': 'v', 'DARTS-CFAR': '*',
    'L-LRao': '<', 'LRao': 'P',
    'THANTD': 'X', 'HTDNet': 'p', 'TSTTD': '>', 'OSVAE': '8',
}
# NOTE: L-LRao color changed from the old green (now DARTS's) to avoid a
# clash across IID and spatial panels; every detector keeps ONE color in
# every figure of the paper.

# =====================================================================
#  SIZE KNOBS — edit these and rerun; everything else adapts.
#  (width, height) in inches. On the page at width=\textwidth, the
#  printed height is  \textwidth * height/width  — so a WIDER figsize
#  gives a SHORTER, smaller-font figure, and vice versa.
# =====================================================================
IID_GRID_SIZE  = (8.8, 2.1)   # the 1x4 IID grid (v1 was 10.4x2.5, v2 8.8x2.5)
AMP_SWEEP_SIZE = (10.4, 2.9)  # the 1x3 spatial amplitude sweep
FS = dict(label=8, tick=7, legend=7.5, title=8)   # font sizes (pt)


def _style(ax, xlabel, ylabel, title=None, logx=True, xticks_at=None):
    if logx:
        ax.set_xscale('log')
    if xticks_at is not None:
        for v in xticks_at:
            ax.axvline(v, color='0.85', lw=0.7, ls='-', zorder=0)
    ax.grid(True, which='major', axis='y', alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=FS['label'])
    ax.set_ylabel(ylabel, fontsize=FS['label'])
    if title:
        ax.set_title(title, fontsize=FS['title'])
    ax.tick_params(labelsize=FS['tick'])


def _plot(ax, x, series_mu, series_sd, order):
    for det in order:
        if det not in series_mu:
            continue
        mu = np.asarray(series_mu[det], float)
        if np.all(np.isnan(mu)):
            continue
        c, m = COLORS.get(det, '#444444'), MARKERS.get(det, 'o')
        ax.plot(x, mu, color=c, marker=m, ms=3.2, lw=1.3, label=det)
        if series_sd and det in series_sd:
            sd = np.asarray(series_sd[det], float)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.13, color=c)


def _shared_legend(fig, handles, labels, ncol, y=1.02):
    fig.legend(handles, labels, loc='upper center', ncol=ncol,
               fontsize=FS['legend'], frameon=False,
               bbox_to_anchor=(0.5, y), handletextpad=0.4, columnspacing=1.1)


# ---------------------------------------------------------------------------
def iid_grid(iid_root='results', out_dir=None):
    global FIG
    FIG = out_dir or FIG
    os.makedirs(FIG, exist_ok=True)
    panels = []
    for mode in ('single', 'multi'):
        hits = sorted(glob.glob(os.path.join(
            iid_root, f'iid_{mode}', f'iid_{mode}_agg_*',
            'metrics_aggregate.json')))
        assert hits, f'no aggregate metrics for iid_{mode} under {iid_root}'
        m = json.load(open(hits[-1]))
        panels.append((mode, m))
    order = ['AMF', 'GMM-Levin', 'L-DART', 'DART', 'L-LRao', 'LRao']
    fig, axes = plt.subplots(1, 4, figsize=IID_GRID_SIZE)
    for i, (mode, m) in enumerate(panels):
        n = np.array(m['n_list'], float)
        rho = np.array(m['rho_list'], float)
        pfa = m['pfa']
        axn, axr = axes[2 * i], axes[2 * i + 1]
        _plot(axn, n, {d: m['vs_n'][d]['pd'] for d in m['vs_n']},
              {d: m['vs_n'][d].get('pd_std') for d in m['vs_n']}, order)
        _style(axn, 'training samples $n$', f'$P_d$ @ $P_{{fa}}$={pfa}' if i == 0 else '',
               title=f'Pavia {mode}: vs $n$', xticks_at=n)
        _plot(axr, rho, {d: m['vs_rho'][d]['pd'] for d in m['vs_rho']},
              {d: m['vs_rho'][d].get('pd_std') for d in m['vs_rho']}, order)
        _style(axr, r'DSM noise level $\rho$', '',
               title=f'Pavia {mode}: vs $\\rho$ (n={m["n_fixed"]})', xticks_at=rho)
    h, l = axes[2].get_legend_handles_labels()   # multi panel has all detectors
    _shared_legend(fig, h, l, ncol=6)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'iid_grid.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/iid_grid.pdf')


# ---------------------------------------------------------------------------
def amp_sweep_spatial(spatial_root='results/spatial', out_dir=None):
    global FIG
    FIG = out_dir or FIG
    os.makedirs(FIG, exist_ok=True)
    order = ['DART', 'DART-CFAR', 'DARTS', 'DARTS-CFAR',
             'AMF-global', 'AMF-local', 'GMM-Levin', 'LRao',
             'THANTD', 'HTDNet', 'TSTTD', 'OSVAE']
    titles = {'pavia4': 'Pavia University', 'sandiego': 'San Diego I',
              'sandiego2': 'San Diego II'}
    fig, axes = plt.subplots(1, 3, figsize=AMP_SWEEP_SIZE, sharey=True)
    for ax, scn in zip(axes, ('pavia4', 'sandiego', 'sandiego2')):
        m = json.load(open(os.path.join(spatial_root, scn, 'metrics.json')))
        ths = [float(t) for t in m['thetas']]
        dets = sorted({d for sr in m['rows'].values()
                       for r_ in sr.values() for d in r_})
        au = {d: {t: [m['rows'][str(t)][sd_][d]['auc']
                      for sd_ in m['rows'][str(t)]
                      if d in m['rows'][str(t)][sd_]]
                  for t in ths} for d in dets}
        mu = {d: [np.mean(au[d][t]) for t in ths] for d in dets}
        sd = {d: [np.std(au[d][t]) for t in ths] for d in dets}
        _plot(ax, np.array(ths), mu, sd, order)
        _style(ax, r'target amplitude $\theta$', 'AUC' if scn == 'pavia4' else '',
               title=titles[scn], xticks_at=ths)
        ax.set_ylim(0.28, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    _shared_legend(fig, h, l, ncol=6, y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'amp_sweep_spatial.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/amp_sweep_spatial.pdf')


if __name__ == '__main__':
    iid_grid()
    amp_sweep_spatial()
