"""
plot_paper_2x2.py — Figure 2 (main IID figure): P_det @ P_fa=0.1 everywhere.

Layout (additive model only):
  [0,0]  Single-class  P_det vs n      (rho=0.01, d=20)
  [0,1]  Multi-class   P_det vs n      (rho=0.01, d=20)   + DSM PCA-dim variants
  [1,0]  Single-class  P_det vs rho    (n=400,   d=20)
  [1,1]  Multi-class   P_det vs rho    (n=1000,  d=20)

All panels report P_det at P_fa=0.1.  In the ρ-sweep panels (row 2) only DSM's
training noise σ depends on ρ; AMF / LRao / GMM-Levin are ρ-independent and are
drawn as flat dashed reference lines at their P_det@0.1 value.

Detectors: AMF, LRao, DSM (ours) + DSM variants, GMM-Levin (multi only).
(GMM-Levin oracle dropped per Ami's request — Levin + LRao are enough.)

Sources (pkl caches produced by make_pauc_figures.py):
  results/pauc_figures/cache_{single,multi}_{n,rho}.pkl

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
CACHE = os.environ.get('PAUC_CACHE_DIR', os.path.join(BASE, 'pauc_figures'))
OUTDIR = os.environ.get('PAUC_OUT_DIR', os.path.dirname(CACHE) or BASE)

TARGET_FPR = 0.1   # P_det @ P_fa=0.1 everywhere

# ── colours & styles ─────────────────────────────────────────────────────────
# DSM variants (different PCA dim) share the red family, distinct dashes/markers.
STYLE = {
    'AMF':       dict(color='#1f77b4', ls='-',  marker='o', label='AMF'),
    'LRao':      dict(color='#2ca02c', ls='--', marker='s', label='LRao'),
    'DSM':       dict(color='#d62728', ls='-',  marker='^', label='DSM (ours)'),
    'DSM-d40':   dict(color='#ff7f0e', ls='-',  marker='v', label='DSM, d=40'),
    'DSM-d64':   dict(color='#8c564b', ls='-',  marker='P', label='DSM, d=64'),
    'DSM-big':   dict(color='#e377c2', ls='-',  marker='X', label='DSM, [128,128]'),
    'GMM-GLRT':  dict(color='#9467bd', ls='-.', marker='D', label='GMM-Levin'),
}
# Legend / plot order
ORDER = ['AMF', 'LRao', 'GMM-GLRT', 'DSM', 'DSM-d40', 'DSM-d64', 'DSM-big']
ALPHA_FILL = 0.15
MS = 4
LW = 1.4


def load_cache(name):
    path = os.path.join(CACHE, f'cache_{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\nRun make_pauc_figures.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_pd(agg, tm, det, key):
    """(mean, std) P_det at TARGET_FPR for (tm, det, key)."""
    try:
        return tuple(agg[tm][det][key]['pd'][TARGET_FPR])
    except (KeyError, TypeError):
        return (float('nan'), float('nan'))


def dets_present(agg, tm):
    """Detector keys present for this panel, in canonical order."""
    have = agg.get(tm, {})
    return [d for d in ORDER if d in have] + \
           [d for d in have if d not in ORDER]


def plot_band(ax, xs, data, key, is_flat=False):
    if key not in STYLE:
        return
    st    = {**STYLE[key]}
    label = st.pop('label')
    color = st['color']
    means = np.array([d[0] for d in data])
    stds  = np.array([d[1] for d in data])
    if is_flat:
        ax.axhline(means[0], color=color, lw=LW, ls=st.get('ls', '--'),
                   label=label, alpha=0.9)
        ax.axhspan(means[0] - stds[0], means[0] + stds[0],
                   color=color, alpha=0.06)
    else:
        ax.plot(xs, means, label=label, lw=LW, ms=MS, **st)
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=ALPHA_FILL)


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
c_sn = load_cache('single_n');   agg_sn, meta_sn = c_sn['agg'], c_sn['meta']
c_sr = load_cache('single_rho'); agg_sr, meta_sr = c_sr['agg'], c_sr['meta']
c_mn = load_cache('multi_n');    agg_mn, meta_mn = c_mn['agg'], c_mn['meta']
c_mr = load_cache('multi_rho');  agg_mr, meta_mr = c_mr['agg'], c_mr['meta']

TM = 'additive'
FLAT = {'AMF', 'LRao', 'GMM-GLRT'}   # ρ-independent in the rho-sweep panels

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))
fig.subplots_adjust(wspace=0.30, hspace=0.46)
YLABEL = rf'$P_{{\mathrm{{det}}}}$ @ $P_{{\mathrm{{fa}}}}={TARGET_FPR}$'


def style_ax(ax, xlabel, title, ylo=0.0):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLABEL)
    ax.set_title(title, pad=4)
    ax.set_ylim(ylo, 1.02)
    ax.set_xscale('log')
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which='major', ls=':', lw=0.4, alpha=0.6)


def annot(ax, txt):
    ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6, color='gray')

# [0,0] Single-class P_det vs n
ax = axes[0, 0]
xs = meta_sn['n_list']
for det in dets_present(agg_sn, TM):
    plot_band(ax, xs, [get_pd(agg_sn, TM, det, n) for n in xs], det)
style_ax(ax, 'Training size $n$', 'Single-class,  $P_{\\rm det}$ vs. $n$')
annot(ax, r'$\rho{=}0.01,\;d{=}20$')

# [0,1] Multi-class P_det vs n  (+ DSM PCA-dim variants)
ax = axes[0, 1]
xs = meta_mn['n_list']
for det in dets_present(agg_mn, TM):
    plot_band(ax, xs, [get_pd(agg_mn, TM, det, n) for n in xs], det)
style_ax(ax, 'Training size $n$', 'Multi-class,  $P_{\\rm det}$ vs. $n$')
annot(ax, r'$\rho{=}0.01$')

# [1,0] Single-class P_det vs rho
ax = axes[1, 0]
xs = meta_sr['rho_list']
for det in dets_present(agg_sr, TM):
    plot_band(ax, xs, [get_pd(agg_sr, TM, det, r) for r in xs], det,
              is_flat=(det in FLAT))
style_ax(ax, r'Noise level $\rho$', r'Single-class,  $P_{\rm det}$ vs. $\rho$')
annot(ax, r'$n{=}400,\;d{=}20$')

# [1,1] Multi-class P_det vs rho  (GMM-Levin flat ref from n-sweep @ n=1000)
ax = axes[1, 1]
xs = meta_mr['rho_list']
mr_dets = dets_present(agg_mr, TM)
for det in mr_dets:
    plot_band(ax, xs, [get_pd(agg_mr, TM, det, r) for r in xs], det,
              is_flat=(det in FLAT))
# GMM-Levin isn't in the rho cache (multi rho sweep) → borrow its n-sweep value.
if 'GMM-GLRT' not in mr_dets and 'GMM-GLRT' in agg_mn.get(TM, {}):
    ref_n = meta_mr.get('n', 1000)
    ref_n = ref_n if ref_n in agg_mn[TM]['GMM-GLRT'] else meta_mn['n_list'][-1]
    v = get_pd(agg_mn, TM, 'GMM-GLRT', ref_n)
    plot_band(ax, xs, [v] * len(xs), 'GMM-GLRT', is_flat=True)
style_ax(ax, r'Noise level $\rho$', r'Multi-class,  $P_{\rm det}$ vs. $\rho$')
annot(ax, r'$n{=}1000,\;d{=}20$')

# ── shared legend (below, outside axes) ───────────────────────────────────────
handles, labels = [], []
seen = set()
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in seen:
            handles.append(h); labels.append(l); seen.add(l)
ncol = min(len(labels), 4)
fig.legend(handles, labels, loc='lower center', ncol=ncol,
           bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=7)

out_path = os.path.join(OUTDIR, 'paper_2x2.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
