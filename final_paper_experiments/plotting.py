"""
plotting.py — Paper-quality figure generation.

Style: clean, IEEE-style, 9pt font, outside legend, grid, PDF output.
"""

import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    # Fonts
    'font.family':       'serif',
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    # Lines & axes
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.6,
    'lines.markersize':   4,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3.5,
    'ytick.major.size':   3.5,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    # Grid
    'axes.grid':          True,
    'grid.color':         '0.88',
    'grid.linewidth':     0.6,
    'grid.linestyle':     '-',
    # Spines — keep only bottom + left (clean look)
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    # Legend
    'legend.framealpha':  0.95,
    'legend.edgecolor':   '0.75',
    'legend.borderpad':   0.4,
    'legend.labelspacing': 0.3,
    # Save
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':       42,   # TrueType → editable in Illustrator/Inkscape
    'ps.fonttype':        42,
})

# Colorblind-friendly, distinct palette
_COLORS = [
    '#1f77b4',  # blue
    '#e6550d',  # orange-red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#d62728',  # red
    '#e377c2',  # pink
    '#17becf',  # cyan
    '#bcbd22',  # olive
    '#7f7f7f',  # grey
]
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', 'p']


def _color(i):  return _COLORS[i % len(_COLORS)]
def _marker(i): return _MARKERS[i % len(_MARKERS)]


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(results: dict, save_path: str, title: str = ''):
    """
    Parameters
    ----------
    results   : {label: (fpr, tpr, auc)}
    save_path : path ending in .pdf or .png
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.2))

    for i, (label, (fpr, tpr, auc)) in enumerate(results.items()):
        ax.plot(fpr, tpr, color=_color(i), linewidth=1.6,
                label=f'{label}  ({auc:.3f})')

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color='0.6', linewidth=0.8,
            linestyle='--', zorder=0)

    ax.set_xlabel('False Alarm Rate')
    ax.set_ylabel('Detection Rate')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    if title:
        ax.set_title(title, pad=5)

    # Legend outside, right side
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0,
              handlelength=1.6)

    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detection rate at fixed false alarm rates
# ---------------------------------------------------------------------------

def plot_false_alarm_perf(results: dict, save_path: str,
                          fa_rates: list = None):
    """
    Grouped bar chart: detection rate at fixed FA operating points.

    Parameters
    ----------
    results  : {label: (fpr, tpr, auc)}
    fa_rates : FA thresholds to evaluate (default [0.001, 0.01, 0.05, 0.1])
    """
    if fa_rates is None:
        fa_rates = [0.001, 0.01, 0.05, 0.1]

    labels = list(results.keys())
    n_det  = len(labels)
    n_fa   = len(fa_rates)

    dr = np.zeros((n_det, n_fa))
    for i, label in enumerate(labels):
        fpr, tpr, _ = results[label]
        for j, fa in enumerate(fa_rates):
            dr[i, j] = float(np.interp(fa, fpr, tpr))

    x     = np.arange(n_fa)
    width = 0.75 / n_det

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for i, label in enumerate(labels):
        offset = (i - n_det / 2 + 0.5) * width
        ax.bar(x + offset, dr[i], width, label=label,
               color=_color(i), alpha=0.88, edgecolor='white', linewidth=0.4)

    fa_labels = ['0.1%', '1%', '5%', '10%']
    ax.set_xticks(x)
    ax.set_xticklabels(fa_labels)
    ax.set_xlabel('False Alarm Rate')
    ax.set_ylabel('Detection Rate')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0,
              handlelength=1.2)

    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# AUC vs number of training samples
# ---------------------------------------------------------------------------

def plot_auc_vs_n(results: dict, n_values: list, save_path: str):
    """
    Line plot: AUC as a function of number of training samples.

    Parameters
    ----------
    results  : {label: array-like (num_n,) or (num_n, num_seeds)}
               2-D input → mean ± 1 std shaded band.
    n_values : x-axis values (shown as explicit tick marks)
    save_path: path ending in .pdf or .png
    """
    # Wider figure to give room for outside legend
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    for i, (label, aucs) in enumerate(results.items()):
        aucs = np.array(aucs)
        c    = _color(i)
        m    = _marker(i)
        if aucs.ndim == 1:
            ax.plot(n_values, aucs, color=c, marker=m,
                    markersize=4, linewidth=1.6, label=label)
        else:
            mean = aucs.mean(axis=1)
            std  = aucs.std(axis=1)
            ax.plot(n_values, mean, color=c, marker=m,
                    markersize=4, linewidth=1.6, label=label)
            ax.fill_between(n_values, mean - std, mean + std,
                            color=c, alpha=0.15, linewidth=0)

    ax.set_xlabel('Number of training samples')
    ax.set_ylabel('AUC')

    # Log x-axis: natural spacing for training-sample grids like [100,200,500,1000,…]
    ax.set_xscale('log')
    ax.set_xticks(n_values)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x)//1000}k' if x >= 1000 else str(int(x))))
    ax.set_xlim(n_values[0] * 0.85, n_values[-1] * 1.15)

    # Y-axis: auto-range with 0.05 steps, floor at 0.4
    all_aucs = np.concatenate([np.array(v).flatten() for v in results.values()])
    y_min = max(0.40, np.floor(np.nanmin(all_aucs) * 20) / 20 - 0.05)
    y_max = min(1.00, np.ceil(np.nanmax(all_aucs)  * 20) / 20 + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    # Legend outside, right side — never overlaps data
    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0,
              handlelength=1.8,
              framealpha=0.95)

    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# AUC per target class (multiclass experiment)
# ---------------------------------------------------------------------------

def plot_auc_per_class(results: dict, save_path: str, title: str = ''):
    """
    Grouped bar chart: AUC per target class for each detector.

    Parameters
    ----------
    results : {detector_label: {class_id: auc}}
    """
    class_ids = sorted(next(iter(results.values())).keys())
    labels    = list(results.keys())
    n_det     = len(labels)
    n_cls     = len(class_ids)
    x         = np.arange(n_cls)
    width     = 0.75 / n_det

    fig, ax = plt.subplots(figsize=(max(5.0, n_cls * 0.65), 3.2))
    for i, label in enumerate(labels):
        aucs   = [results[label][c] for c in class_ids]
        offset = (i - n_det / 2 + 0.5) * width
        ax.bar(x + offset, aucs, width, label=label,
               color=_color(i), alpha=0.88, edgecolor='white', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f'cls {c}' for c in class_ids], rotation=45, ha='right')
    ax.set_ylabel('AUC')

    all_vals = [v for d in results.values() for v in d.values()]
    y_min = max(0.40, np.floor(min(all_vals) * 20) / 20 - 0.05)
    ax.set_ylim(y_min, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    if title:
        ax.set_title(title, pad=5)

    ax.legend(loc='upper left',
              bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0,
              handlelength=1.2)

    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utility: copy figure to comparison directory
# ---------------------------------------------------------------------------

def copy_to_comparison_dir(fig_path: str, comparisons_root: str,
                            comparison_tag: str, run_id: str):
    """
    Copy a figure into results/comparisons/{comparison_tag}/{run_id}_{fig_name}.
    """
    dest_dir = os.path.join(comparisons_root, comparison_tag)
    os.makedirs(dest_dir, exist_ok=True)
    fig_name = os.path.basename(fig_path)
    dest     = os.path.join(dest_dir, f'{run_id}_{fig_name}')
    shutil.copy2(fig_path, dest)
    return dest
