"""tsp_repro.figures — emit the figures/*.pdf files the TSP paper includes.

Implemented here (from data + score archives, no models needed):
  scenes_falsecolor.pdf   false-color composites with train/test boxes
  amp_sweep.pdf           AUC vs theta per scene (THETA_GRID)
  detection_maps.pdf      score maps (test box) per detector x scene

Phase-B stubs (source data exists elsewhere on the developer machine):
  null_calibration_bars.pdf  from final_paper_experiments/results/null_calibration_*
  sigma_rule.pdf             from SDSM sigma-sweep results
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tsp_repro import protocol as PR
from tsp_repro.tables import collect

DET_STYLE = {  # stable colors across all figures
    "DART": "#d62728", "DART-CFAR": "#ff9896",
    "DARTS": "#1f77b4", "DARTS-CFAR": "#17becf",
    "AMF": "#7f7f7f", "AMF-local": "#bcbd22", "GMM-Levin": "#9467bd",
    "LRao": "#2ca02c", "THANTD": "#8c564b", "HTDNet": "#e377c2",
    "OSVAE": "#c49c94", "TSTTD": "#f7b6d2",
}


def _false_color(data, bands=(60, 30, 10)):
    fc = data[..., list(bands)].astype(np.float64)
    for c in range(3):
        ch = fc[..., c]
        lo, hi = np.percentile(ch, [2, 98])
        fc[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-9), 0, 1)
    return fc


def _draw_box(ax, box, color, label):
    from matplotlib.patches import Rectangle
    r0, r1, c0, c1 = box
    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False,
                           edgecolor=color, linewidth=1.6))
    ax.text(c0, max(r0 - 4, 2), label, color=color, fontsize=8, weight="bold")


def scenes_falsecolor(out_pdf, scenes=("pavia4", "sandiego", "sandiego2")):
    fig, axes = plt.subplots(1, len(scenes), figsize=(4 * len(scenes), 5))
    for ax, name in zip(np.atleast_1d(axes), scenes):
        sc = PR.build_scene(name)
        ax.imshow(_false_color(sc["data"]))
        _draw_box(ax, sc["boxes"]["train"], "#00e5ff", "train")
        _draw_box(ax, sc["boxes"]["test"], "#ffd600", "test")
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out_pdf)


def amp_sweep(out_pdf, dirs, dets, scenes_sigs=(("pavia4", "bitumen"),
                                                ("sandiego", "aircraft"),
                                                ("sandiego2", "aircraft"))):
    """AUC vs theta per scene; dense (0,0.3] region emphasized (log x)."""
    from sklearn.metrics import roc_auc_score
    cells = collect(dirs)
    fig, axes = plt.subplots(1, len(scenes_sigs),
                             figsize=(4.2 * len(scenes_sigs), 3.4),
                             sharey=True)
    for ax, (scene, sig) in zip(np.atleast_1d(axes), scenes_sigs):
        curves = defaultdict(dict)
        for (sc, sg, det, model, th), paths in cells.items():
            if (sc, sg, model) != (scene, sig, "additive") or det not in dets:
                continue
            aucs = []
            for _, p in sorted(paths.items()):
                d = np.load(p)
                aucs.append(roc_auc_score(d["labels"], d["scores"]))
            curves[det][th] = (np.mean(aucs), np.std(aucs))
        for det in dets:
            if not curves[det]:
                continue
            ths = sorted(curves[det])
            mu = np.array([curves[det][t][0] for t in ths])
            sd = np.array([curves[det][t][1] for t in ths])
            c = DET_STYLE.get(det)
            ax.plot(ths, mu, "-o", ms=3, lw=1.4, color=c, label=det)
            ax.fill_between(ths, mu - sd, mu + sd, alpha=0.15, color=c)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\theta$")
        ax.set_title(f"{scene} ({sig})", fontsize=10)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("AUC")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 6), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out_pdf)


def detection_maps(out_pdf, dirs, dets, scene="pavia4", sig="bitumen",
                   theta=0.15, seed=42):
    """Score maps on the planted test box, one panel per detector."""
    cells = collect(dirs)
    sc = PR.build_scene(scene)
    H, W = sc["te_shape"]
    panels = []
    for det in dets:
        paths = cells.get((scene, sig, det, "additive", theta), {})
        if seed in paths:
            d = np.load(paths[seed])
            panels.append((det, d["scores"].reshape(H, W), d["labels"].reshape(H, W)))
    if not panels:
        raise FileNotFoundError(
            f"no score archives for {scene}/{sig} theta={theta} in {dirs}")
    n = len(panels) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    axes[0].imshow(_false_color(sc["data"][
        sc["boxes"]["test"][0]:sc["boxes"]["test"][1],
        sc["boxes"]["test"][2]:sc["boxes"]["test"][3]]))
    axes[0].set_title("test region", fontsize=9)
    for ax, (det, smap, lab) in zip(axes[1:], panels):
        ax.imshow(smap, cmap="inferno")
        ys, xs = np.where(lab == 1)
        ax.scatter(xs, ys, s=1.5, c="#00e5ff", alpha=0.5, linewidths=0)
        ax.set_title(det, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", out_pdf)


# --------------------------------------------------------------------------
# Phase-B stubs
# --------------------------------------------------------------------------
def null_calibration_bars(out_pdf, results_dir=None):
    raise NotImplementedError(
        "Phase B: build from final_paper_experiments/results/"
        "null_calibration_20260705 + _swap_20260705 (tables + raw npz).")


def sigma_rule(out_pdf, sweep_dir=None):
    raise NotImplementedError(
        "Phase B: build from the SDSM sigma-sweep results "
        "(hsi/experiments/sigma_sweep.py outputs).")
