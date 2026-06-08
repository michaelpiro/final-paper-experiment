"""
plot_paper_2x2.py — 2×2 paper figure using normalised partial AUC (pAUC).
(Replaces full-AUC figure per reviewer feedback.)

pAUC = (1/P_fa_max) * integral_0^{P_fa_max} P_det(P_fa) dP_fa
     = average detection rate over the operating range P_fa ∈ [0, 0.1].

Layout (additive model only):
  [0,0]  Single-class  pAUC vs n      (rho=0.01, d=20)
  [0,1]  Multi-class   pAUC vs n      (rho=0.01, d=20)
  [1,0]  Single-class  P_det vs rho   (n=400,   d=20)   @ P_fa=0.1
  [1,1]  Multi-class   P_det vs rho   (n=1000,  d=20)   @ P_fa=0.1

Sources (pkl caches produced by make_pauc_figures.py):
  single n   : results/pauc_figures/cache_single_n.pkl
  single rho : results/pauc_figures/cache_single_rho.pkl
  multi  n   : results/pauc_figures/cache_multi_n.pkl
  multi  rho : results/pauc_figures/cache_multi_rho.pkl

Run:
    .venv/bin/python experiments/honest_pipeline/plot_paper_2x2.py
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE  = os.path.join(os.path.dirname(__file__), 'results')
# Cache dir (the 4 cache_*.pkl) and output dir are env-overridable so the same
# script works both locally and on Colab (where results live on Drive).
CACHE = os.environ.get('PAUC_CACHE_DIR', os.path.join(BASE, 'pauc_figures'))
OUTDIR = os.environ.get('PAUC_OUT_DIR', os.path.dirname(CACHE) or BASE)

# ── metrics ──────────────────────────────────────────────────────────────────
TARGET_FPR  = 0.1   # P_fa for ρ-sweep panels (P_det @ P_fa=0.1)
PAUC_MAX    = 0.1   # upper limit of pAUC integration (matches TARGET_FPR)

# ── colours & styles ─────────────────────────────────────────────────────────
STYLE = {
    'AMF':        dict(color='#1f77b4', ls='-',  marker='o', label='AMF'),
    'LRao':       dict(color='#2ca02c', ls='--', marker='s', label='LRao'),
    'DSM':        dict(color='#d62728', ls='-',  marker='^', label='DSM (ours)'),
    'GMM-GLRT':   dict(color='#9467bd', ls='-.', marker='D', label='GMM-Levin'),
    'GMM-GLRT-G': dict(color='#ff7f0e', ls=':',  marker='v', label='GMM-Levin (oracle)'),
}
ALPHA_FILL = 0.15
MS = 4
LW = 1.4

# ── helpers ──────────────────────────────────────────────────────────────────
def load_cache(name):
    path = os.path.join(CACHE, f'cache_{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            "Run make_pauc_figures.py first to generate the pkl caches.")
    with open(path, 'rb') as f:
        return pickle.load(f)   # keys: results, meta, agg


def get_pauc(agg, tm, det, key):
    """Return (mean, std) normalised pAUC for (tm, det, key).
    Used for n-sweep panels."""
    try:
        return tuple(agg[tm][det][key]['pauc'])
    except KeyError:
        return (float('nan'), float('nan'))


def get_pd(agg, tm, det, key):
    """Return (mean, std) P_det at TARGET_FPR for (tm, det, key).
    Used for ρ-sweep panels."""
    try:
        return tuple(agg[tm][det][key]['pd'][TARGET_FPR])
    except KeyError:
        return (float('nan'), float('nan'))


def plot_band(ax, xs, data, key, is_flat=False):
    """
    data : list of (mean, std) pairs matching xs.
    is_flat : if True, draw as dashed horizontal reference line (rho-independent).
    """
    if key not in STYLE:
        return
    st    = {**STYLE[key]}
    label = st.pop('label')
    color = st['color']
    means = np.array([d[0] for d in data])
    stds  = np.array([d[1] for d in data])

    if is_flat:
        ax.axhline(means[0], color=color, lw=LW, ls='--',
                   label=label, alpha=0.9)
        ax.axhspan(means[0] - stds[0], means[0] + stds[0],
                   color=color, alpha=0.06)
    else:
        ax.plot(xs, means, label=label, lw=LW, ms=MS, **st)
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=ALPHA_FILL)


# ── IEEE-style rcParams ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'lines.linewidth':   LW,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        300,
})

# ── load caches ───────────────────────────────────────────────────────────────
c_sn = load_cache('single_n')
c_sr = load_cache('single_rho')
c_mn = load_cache('multi_n')
c_mr = load_cache('multi_rho')

agg_sn, meta_sn = c_sn['agg'], c_sn['meta']
agg_sr, meta_sr = c_sr['agg'], c_sr['meta']
agg_mn, meta_mn = c_mn['agg'], c_mn['meta']
agg_mr, meta_mr = c_mr['agg'], c_mr['meta']

TM = 'additive'

# ── build panel data ──────────────────────────────────────────────────────────

# [0,0] Single-class: pAUC vs n
sn_xs = meta_sn['n_list']
sn_dets = ['AMF', 'LRao', 'DSM']   # no GMM in single-class
sn_data = {d: [get_pauc(agg_sn, TM, d, n) for n in sn_xs] for d in sn_dets}

# [0,1] Multi-class: pAUC vs n
mn_xs = meta_mn['n_list']
mn_dets = ['AMF', 'LRao', 'DSM', 'GMM-GLRT', 'GMM-GLRT-G']
mn_data = {}
for d in mn_dets:
    if d in agg_mn.get(TM, {}):
        mn_data[d] = [get_pauc(agg_mn, TM, d, n) for n in mn_xs]

# [1,0] Single-class: P_det vs rho
sr_xs = meta_sr['rho_list']
sr_dets = ['AMF', 'LRao', 'DSM']
sr_data = {d: [get_pd(agg_sr, TM, d, r) for r in sr_xs] for d in sr_dets}
# AMF and LRao are rho-independent → mark as flat
sr_flat = {'AMF', 'LRao'}

# [1,1] Multi-class: P_det vs rho
mr_xs  = meta_mr['rho_list']
mr_dets = ['AMF', 'LRao', 'DSM']
mr_data = {d: [get_pd(agg_mr, TM, d, r) for r in mr_xs] for d in mr_dets}
mr_flat = {'AMF', 'LRao'}
# Add GMM reference lines from n-sweep at ref_n=1000 (P_det @ P_fa=0.1)
ref_n = 1000
if ref_n in agg_mn.get(TM, {}).get('GMM-GLRT', {}):
    v = get_pd(agg_mn, TM, 'GMM-GLRT', ref_n)
    mr_data['GMM-GLRT'] = [v] * len(mr_xs)
    mr_flat.add('GMM-GLRT')
if ref_n in agg_mn.get(TM, {}).get('GMM-GLRT-G', {}):
    v = get_pd(agg_mn, TM, 'GMM-GLRT-G', ref_n)
    mr_data['GMM-GLRT-G'] = [v] * len(mr_xs)
    mr_flat.add('GMM-GLRT-G')
# AMF/LRao reference lines in rho-sweep also use P_det @ P_fa=0.1

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5))
fig.subplots_adjust(wspace=0.32, hspace=0.42)

YLABEL_PAUC = r'pAUC ($P_{\mathrm{fa}} \leq 0.1$)'
YLABEL_PD   = rf'$P_{{\mathrm{{det}}}}$ @ $P_{{\mathrm{{fa}}}}={TARGET_FPR}$'


def style_ax(ax, xlabel, ylabel, title, ylo=0.0):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.set_ylim(ylo, 1.02)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which='major', ls=':', lw=0.4, alpha=0.6)


# [0,0] Single-class pAUC vs n
ax = axes[0, 0]
for det, data in sn_data.items():
    plot_band(ax, sn_xs, data, det)
ax.set_xscale('log')
style_ax(ax, 'Training size $n$', YLABEL_PAUC, 'Single-class, pAUC vs. $n$')
ax.text(0.97, 0.05, r'$\rho{=}0.01,\;d{=}20$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color='gray')

# [0,1] Multi-class pAUC vs n
ax = axes[0, 1]
for det, data in mn_data.items():
    plot_band(ax, mn_xs, data, det)
ax.set_xscale('log')
style_ax(ax, 'Training size $n$', YLABEL_PAUC, 'Multi-class, pAUC vs. $n$', ylo=0.0)
ax.text(0.97, 0.05, r'$\rho{=}0.01,\;d{=}20$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color='gray')

# [1,0] Single-class P_det vs rho
ax = axes[1, 0]
for det, data in sr_data.items():
    plot_band(ax, sr_xs, data, det, is_flat=(det in sr_flat))
ax.set_xscale('log')
style_ax(ax, r'Noise level $\rho$', YLABEL_PD, r'Single-class, $P_{\rm det}$ vs. $\rho$')
ax.text(0.97, 0.05, r'$n{=}400,\;d{=}20$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color='gray')

# [1,1] Multi-class P_det vs rho
ax = axes[1, 1]
for det, data in mr_data.items():
    plot_band(ax, mr_xs, data, det, is_flat=(det in mr_flat))
ax.set_xscale('log')
style_ax(ax, r'Noise level $\rho$', YLABEL_PD, r'Multi-class, $P_{\rm det}$ vs. $\rho$', ylo=0.0)
ax.text(0.97, 0.05, r'$n{=}1000,\;d{=}20$',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color='gray')

# ── shared legend ─────────────────────────────────────────────────────────────
handles, labels = [], []
seen = set()
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            handles.append(h); labels.append(l); seen.add(l)

fig.legend(handles, labels,
           loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.04),
           frameon=False, fontsize=7)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTDIR, 'paper_2x2.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
