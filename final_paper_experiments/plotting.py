"""
plotting.py — Paper-quality figure generation.

Style: clean, minimal, 9pt font, thin axes, PDF output.
No excessive annotations, no grid by default.
"""

import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, roc_auc_score

matplotlib.rcParams.update({
    'font.size':        9,
    'axes.linewidth':   0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'lines.linewidth':   1.5,
    'legend.fontsize':   8,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.8',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype':      42,   # TrueType fonts in PDF (editable in Illustrator)
    'ps.fonttype':       42,
})

# Consistent color cycle across figures
_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _color(i):
    return _COLORS[i % len(_COLORS)]


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(results: dict, save_path: str, title: str = ''):
    """
    Plot ROC curves for multiple detectors.

    Parameters
    ----------
    results   : {label: (fpr, tpr, auc)}
    save_path : path ending in .pdf or .png
    title     : optional title (keep short for papers)
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for i, (label, (fpr, tpr, auc)) in enumerate(results.items()):
        ax.plot(fpr, tpr, color=_color(i), label=f'{label} ({auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.6, alpha=0.4)
    ax.set_xlabel('False Alarm Rate')
    ax.set_ylabel('Detection Rate')
    if title:
        ax.set_title(title, pad=4)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Detection rate at fixed false alarm rates
# ---------------------------------------------------------------------------

def plot_false_alarm_perf(results: dict, save_path: str,
                          fa_rates: list = None):
    """
    Bar chart: detection rate at several fixed false-alarm rates.

    Parameters
    ----------
    results  : {label: (fpr, tpr, auc)}
    fa_rates : list of FAR values to evaluate at (default [0.001, 0.01, 0.05, 0.1])
    """
    if fa_rates is None:
        fa_rates = [0.001, 0.01, 0.05, 0.1]

    labels   = list(results.keys())
    n_det    = len(labels)
    n_fa     = len(fa_rates)

    # Interpolate detection rate at each FA point
    dr = np.zeros((n_det, n_fa))
    for i, label in enumerate(labels):
        fpr, tpr, _ = results[label]
        for j, fa in enumerate(fa_rates):
            dr[i, j] = float(np.interp(fa, fpr, tpr))

    x     = np.arange(n_fa)
    width = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for i, label in enumerate(labels):
        offset = (i - n_det / 2 + 0.5) * width
        ax.bar(x + offset, dr[i], width, label=label,
               color=_color(i), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{fa:.3f}' for fa in fa_rates])
    ax.set_xlabel('False Alarm Rate')
    ax.set_ylabel('Detection Rate')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', ncol=max(1, n_det // 4))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    fig.tight_layout()
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
    results  : {label: array-like of shape (num_n,) or (num_n, num_seeds)}
               If shape is (num_n, num_seeds), mean ± std error bars are drawn.
    n_values : list of n_train values (x-axis, linear scale)
    save_path: path ending in .pdf or .png
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for i, (label, aucs) in enumerate(results.items()):
        aucs = np.array(aucs)
        if aucs.ndim == 1:
            ax.plot(n_values, aucs, color=_color(i), marker='o',
                    markersize=3, label=label)
        else:
            mean = aucs.mean(axis=1)
            std  = aucs.std(axis=1)
            ax.plot(n_values, mean, color=_color(i), marker='o',
                    markersize=3, label=label)
            ax.fill_between(n_values, mean - std, mean + std,
                            color=_color(i), alpha=0.15)

    ax.set_xlabel('Training samples')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc='lower right')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    fig.tight_layout()
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
    width     = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(max(4.0, n_cls * 0.6), 3.0))
    for i, label in enumerate(labels):
        aucs   = [results[label][c] for c in class_ids]
        offset = (i - n_det / 2 + 0.5) * width
        ax.bar(x + offset, aucs, width, label=label,
               color=_color(i), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'cls {c}' for c in class_ids], rotation=45, ha='right')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.05)
    if title:
        ax.set_title(title, pad=4)
    ax.legend(loc='lower right', ncol=max(1, n_det // 3))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Utility: copy figure to comparison directory
# ---------------------------------------------------------------------------

def copy_to_comparison_dir(fig_path: str, comparisons_root: str,
                            comparison_tag: str, run_id: str):
    """
    Copy a figure into results/comparisons/{comparison_tag}/{run_id}_{fig_name}.

    Parameters
    ----------
    fig_path         : source figure path
    comparisons_root : e.g. 'final_paper_experiments/results/comparisons'
    comparison_tag   : e.g. 'pca_dim_effect', 'n_samples_effect'
    run_id           : unique run identifier string
    """
    dest_dir = os.path.join(comparisons_root, comparison_tag)
    os.makedirs(dest_dir, exist_ok=True)
    fig_name = os.path.basename(fig_path)
    dest     = os.path.join(dest_dir, f'{run_id}_{fig_name}')
    shutil.copy2(fig_path, dest)
    return dest
