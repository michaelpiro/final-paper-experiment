# TODO: adapt to the repro run output layout before use (copied from camera_ready)
"""camera_ready/scripts/make_figures.py — paper-ready combined figures.

One palette for every plot (extends the IID palette; deep baselines get their
own stable colors), shared legends, print-sized fonts, no per-panel legends.

Outputs (camera_ready/figures/):
    iid_grid.pdf         1x4: [single Pd@Pfa vs n | single vs rho |
                               multi Pd@Pfa vs n  | multi vs rho], one legend.
    amp_sweep_spatial.pdf 1x3: AUC vs theta (pavia4 | SD1 | SD2), 12 detectors,
                               one two-row legend.

Run: cd pythonProject && .venv/bin/python ../camera_ready/scripts/make_figures.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(CR, 'figures')
os.makedirs(FIG, exist_ok=True)

# ---- definitive LRao (lrao_val2): identical config on all scenes, 5 seeded
# models, validation early stopping (the LRao paper's prescribed usage).
# Overrides the sweep-run LRao (old registry recipe, single seed-42 model).
LRAO_RUN = os.path.join(CR, 'spatial', 'lrao_val2')


def _lrao_override(r, scn):
    p = os.path.join(LRAO_RUN, f'metrics__{scn}.json')
    if os.path.exists(p):
        rows = json.load(open(p))['rows']
        r['auc']['LRao'] = {t: [rows[t][s]['auc'] for s in sorted(rows[t])]
                            for t in rows}
    return r


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
def iid_grid():
    panels = []
    for mode in ('single', 'multi'):
        p = glob.glob(os.path.join(CR, 'iid', f'iid_{mode}',
                                   f'iid_{mode}_agg_*', 'metrics_aggregate.json'))[0]
        m = json.load(open(p))
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
def amp_sweep_spatial():
    order = ['DART', 'DART-CFAR', 'DARTS', 'DARTS-CFAR',
             'AMF-global', 'AMF-local', 'GMM-Levin', 'LRao',
             'THANTD', 'HTDNet', 'TSTTD', 'OSVAE']
    titles = {'pavia4': 'Pavia University', 'sandiego': 'San Diego I',
              'sandiego2': 'San Diego II'}
    fig, axes = plt.subplots(1, 3, figsize=AMP_SWEEP_SIZE, sharey=True)
    for ax, scn in zip(axes, ('pavia4', 'sandiego', 'sandiego2')):
        r = json.load(open(os.path.join(CR, 'spatial',
                                        f'spatial_sweep_{scn}',
                                        'theta_results.json')))
        r = _lrao_override(r, scn)
        ths = [float(t) for t in r['thetas']]
        mu = {d: [np.mean(r['auc'][d][str(t)]) for t in ths] for d in r['auc']}
        sd = {d: [np.std(r['auc'][d][str(t)]) for t in ths] for d in r['auc']}
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


def iid_grid_v2():
    """Variant: [vs-n single | vs-n multi | vs-rho single | vs-rho multi],
    shared y (Pd@Pfa) across all panels -> y numbers only on the leftmost;
    narrower figure."""
    ms = {}
    for mode in ('single', 'multi'):
        p = glob.glob(os.path.join(CR, 'iid', f'iid_{mode}',
                                   f'iid_{mode}_agg_*', 'metrics_aggregate.json'))[0]
        ms[mode] = json.load(open(p))
    order = ['AMF', 'GMM-Levin', 'L-DART', 'DART', 'L-LRao', 'LRao']
    fig, axes = plt.subplots(1, 4, figsize=IID_GRID_SIZE, sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    specs = [('single', 'vs_n', 'n_list', 'training samples $n$', 'single'),
             ('multi',  'vs_n', 'n_list', 'training samples $n$', 'multi'),
             ('single', 'vs_rho', 'rho_list', r'DSM noise level $\rho$', None),
             ('multi',  'vs_rho', 'rho_list', r'DSM noise level $\rho$', None)]
    for ax, (mode, blk, xkey, xlab, ttl) in zip(axes, specs):
        m = ms[mode]
        if ttl is None:                       # rho panels: show the fixed n
            ttl = f"{mode} ($n={m['n_fixed']}$)"
        x = np.array(m[xkey], float)
        _plot(ax, x, {d: m[blk][d]['pd'] for d in m[blk]},
              {d: m[blk][d].get('pd_std') for d in m[blk]}, order)
        _style(ax, xlab, '', title=ttl, xticks_at=x)
        if blk == 'vs_rho':                   # plain-number labels, no 10^k
            ticks = [1e-5, 1e-3, 0.1, 1, 10]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: '1e-5' if v == 1e-5 else f'{v:g}'))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel(f"$P_d$ @ $P_{{fa}}$={ms['single']['pfa']}",
                       fontsize=FS['label'])
    h, l = axes[1].get_legend_handles_labels()
    _shared_legend(fig, h, l, ncol=6, y=1.16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'iid_grid_v2.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/iid_grid_v2.pdf')


def _amp_sweep_variant(tag, xmode):
    """Axis variants for the spatial amplitude sweep.
    xmode: 'linear' | 'logticks' (log + labeled ticks at a subset of tested
    thetas) | 'sqrt' (power-0.5 scale, ticks at tested subset)."""
    order = ['DART', 'DART-CFAR', 'DARTS', 'DARTS-CFAR',
             'AMF-global', 'AMF-local', 'GMM-Levin', 'LRao',
             'THANTD', 'HTDNet', 'TSTTD', 'OSVAE']
    titles = {'pavia4': 'Pavia University', 'sandiego': 'San Diego I',
              'sandiego2': 'San Diego II'}
    LBL = [0.03, 0.075, 0.15, 0.3, 0.5, 0.7, 0.95]
    fig, axes = plt.subplots(1, 3, figsize=AMP_SWEEP_SIZE, sharey=True)
    for ax, scn in zip(axes, ('pavia4', 'sandiego', 'sandiego2')):
        r = json.load(open(os.path.join(CR, 'spatial',
                                        f'spatial_sweep_{scn}',
                                        'theta_results.json')))
        r = _lrao_override(r, scn)
        ths = [float(t) for t in r['thetas']]
        mu = {d: [np.mean(r['auc'][d][str(t)]) for t in ths] for d in r['auc']}
        sd = {d: [np.std(r['auc'][d][str(t)]) for t in ths] for d in r['auc']}
        _plot(ax, np.array(ths), mu, sd, order)
        for v in ths:
            ax.axvline(v, color='0.85', lw=0.7, ls='-', zorder=0)
        if xmode == 'linear':
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif xmode == 'logticks':
            ax.set_xscale('log')
            ax.set_xticks(LBL)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        elif xmode == 'sqrt':
            ax.set_xscale('function',
                          functions=(lambda x: np.sqrt(np.maximum(x, 0)),
                                     lambda x: x ** 2))
            ax.set_xticks(LBL)
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.grid(True, which='major', axis='y', alpha=0.3)
        ax.set_xlabel(r'target amplitude $\theta$', fontsize=FS['label'])
        ax.set_title(titles[scn], fontsize=FS['title'])
        ax.tick_params(labelsize=FS['tick'])
        ax.set_ylim(0.28, 1.02)
    axes[0].set_ylabel('AUC', fontsize=FS['label'])
    h, l = axes[0].get_legend_handles_labels()
    _shared_legend(fig, h, l, ncol=6, y=1.07)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'amp_sweep_spatial_{tag}.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote figures/amp_sweep_spatial_{tag}.pdf')


def iid_grid_v3():
    """Like v2 but the legend sits BELOW the panels (IEEE-ish)."""
    ms = {}
    for mode in ('single', 'multi'):
        p = glob.glob(os.path.join(CR, 'iid', f'iid_{mode}',
                                   f'iid_{mode}_agg_*', 'metrics_aggregate.json'))[0]
        ms[mode] = json.load(open(p))
    order = ['AMF', 'GMM-Levin', 'L-DART', 'DART', 'L-LRao', 'LRao']
    fig, axes = plt.subplots(1, 4, figsize=IID_GRID_SIZE, sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    specs = [('single', 'vs_n', 'n_list', 'training samples $n$', 'single'),
             ('multi',  'vs_n', 'n_list', 'training samples $n$', 'multi'),
             ('single', 'vs_rho', 'rho_list', r'DSM noise level $\rho$', None),
             ('multi',  'vs_rho', 'rho_list', r'DSM noise level $\rho$', None)]
    for ax, (mode, blk, xkey, xlab, ttl) in zip(axes, specs):
        m = ms[mode]
        if ttl is None:                       # rho panels: show the fixed n
            ttl = f"{mode} ($n={m['n_fixed']}$)"
        x = np.array(m[xkey], float)
        _plot(ax, x, {d: m[blk][d]['pd'] for d in m[blk]},
              {d: m[blk][d].get('pd_std') for d in m[blk]}, order)
        _style(ax, xlab, '', title=ttl, xticks_at=x)
        if blk == 'vs_rho':                   # plain-number labels, no 10^k
            ticks = [1e-5, 1e-3, 0.1, 1, 10]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: '1e-5' if v == 1e-5 else f'{v:g}'))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel(f"$P_d$ @ $P_{{fa}}$={ms['single']['pfa']}",
                       fontsize=FS['label'])
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, fontsize=FS['legend'],
               frameon=False, bbox_to_anchor=(0.5, -0.12),
               handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'iid_grid_v3.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/iid_grid_v3.pdf')


def iid_grid_v4():
    """2x2: rows = {single, multi}, cols = {vs n, vs rho}; shared axes."""
    ms = {}
    for mode in ('single', 'multi'):
        p = glob.glob(os.path.join(CR, 'iid', f'iid_{mode}',
                                   f'iid_{mode}_agg_*', 'metrics_aggregate.json'))[0]
        ms[mode] = json.load(open(p))
    order = ['AMF', 'GMM-Levin', 'L-DART', 'DART', 'L-LRao', 'LRao']
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.9), sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.42))
    for r, mode in enumerate(('single', 'multi')):
        m = ms[mode]
        for c, (blk, xkey, xlab) in enumerate(
                [('vs_n', 'n_list', 'training samples $n$'),
                 ('vs_rho', 'rho_list', r'DSM noise level $\rho$')]):
            ax = axes[r, c]
            x = np.array(m[xkey], float)
            _plot(ax, x, {d: m[blk][d]['pd'] for d in m[blk]},
                  {d: m[blk][d].get('pd_std') for d in m[blk]}, order)
            _style(ax, xlab if r == 1 else '', '', xticks_at=x)
            if r == 0:
                ax.set_title(['vs $n$', 'vs $\\rho$'][c], fontsize=FS['title'])
        axes[r, 0].set_ylabel(f"{mode}\n$P_d$ @ $P_{{fa}}$={m['pfa']}",
                              fontsize=FS['label'])
    h, l = axes[1, 0].get_legend_handles_labels()
    _shared_legend(fig, h, l, ncol=6, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'iid_grid_v4.{ext}'), dpi=200,
                    bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/iid_grid_v4.pdf')


if __name__ == '__main__':
    iid_grid()
    iid_grid_v2()
    iid_grid_v3()
    iid_grid_v4()
    amp_sweep_spatial()
    for tag, xm in (('linear', 'linear'), ('logticks', 'logticks'),
                    ('sqrt', 'sqrt')):
        _amp_sweep_variant(tag, xm)
