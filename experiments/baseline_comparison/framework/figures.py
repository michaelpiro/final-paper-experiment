"""
figures.py — analysis/plotting that reads SAVED artifacts only (no retraining).

A results run dir is laid out as <run>/<provider>/<scenario>/<detector>/{metrics.json,
scores.npz, maps.npz}. Every function takes the run dir and returns a matplotlib
Figure, so notebook cells stay tiny and tweakable.
"""

from __future__ import annotations
import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    "CF-Attn-CFAR": "#1f77b4", "CF-Attn": "#aec7e8", "NeighborMLP": "#2ca02c",
    "DSM": "#ff7f0e", "MCLT": "#d62728", "AMF": "#9467bd", "Reg-AMF": "#c5b0d5",
    "CEM": "#7f7f7f", "GMM-Levin": "#e377c2", "Self-GMM": "#17becf",
    "Spatial-GMM": "#bcbd22",
}


# --------------------------------------------------------------------- loading
def _provider_root(run_dir):
    subs = [d for d in glob.glob(os.path.join(run_dir, "*")) if os.path.isdir(d)]
    return subs[0] if subs else run_dir


def scenarios(run_dir):
    pr = _provider_root(run_dir)
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(pr, "*"))
                  if os.path.isdir(d))


def detectors(run_dir, scenario):
    pr = _provider_root(run_dir)
    return sorted(os.path.basename(d) for d in
                  glob.glob(os.path.join(pr, scenario, "*")) if os.path.isdir(d))


def _det_dir(run_dir, scenario, det):
    return os.path.join(_provider_root(run_dir), scenario, det)


def load_metrics(run_dir, scenario, det):
    return json.load(open(os.path.join(_det_dir(run_dir, scenario, det),
                                        "metrics.json")))


def load_scores(run_dir, scenario, det):
    return dict(np.load(os.path.join(_det_dir(run_dir, scenario, det), "scores.npz"),
                        allow_pickle=True))


def load_maps(run_dir, scenario, det):
    p = os.path.join(_det_dir(run_dir, scenario, det), "maps.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


def _c(det):
    return COLORS.get(det, None)


# --------------------------------------------------------------------- figures
def auc_vs_amplitude(run_dir, scenario, model="additive", signature=None, dets=None):
    dets = dets or detectors(run_dir, scenario)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for det in dets:
        m = load_metrics(run_dir, scenario, det).get(model, {})
        sig = signature or (list(m)[0] if m else None)
        if sig not in m:
            continue
        amps = sorted(m[sig], key=float)
        ax.plot([float(a) for a in amps], [m[sig][a]["auc"] for a in amps],
                "-o", lw=1.8, ms=4, color=_c(det), label=det)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xlabel("amplitude θ"); ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{scenario} — {model} — sig={signature or 'first'}")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); return fig


def roc(run_dir, scenario, model="additive", signature=None, amplitude=None, dets=None):
    dets = dets or detectors(run_dir, scenario)
    fig, ax = plt.subplots(figsize=(6, 5))
    for det in dets:
        m = load_metrics(run_dir, scenario, det).get(model, {})
        sig = signature or (list(m)[0] if m else None)
        if sig not in m:
            continue
        amp = amplitude or sorted(m[sig], key=float)[-1]
        cell = m[sig].get(str(amp)) or m[sig].get(f"{float(amp):g}")
        if not cell:
            continue
        ax.plot(cell["roc"]["fpr"], cell["roc"]["tpr"], lw=1.6, color=_c(det),
                label=f"{det} ({cell['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{scenario} — {model} — sig={signature or 'first'} amp={amplitude or 'max'}")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); return fig


def replacement_sanity(run_dir, scenario, signature="orthogonal", amplitude=1.0,
                       dets=None):
    """Additive vs FULL-replacement AUC bars — 'did the model train' check."""
    dets = dets or detectors(run_dir, scenario)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dets)); w = 0.4
    for j, model in enumerate(["additive", "replacement"]):
        vals = []
        for det in dets:
            m = load_metrics(run_dir, scenario, det).get(model, {}).get(signature, {})
            cell = m.get(str(amplitude)) or m.get(f"{float(amplitude):g}")
            vals.append(cell["auc"] if cell else np.nan)
        ax.bar(x + j * w, vals, w, label=model)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(dets, rotation=30, ha="right")
    ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{scenario} — sig={signature} amp={amplitude}  (full-replacement = train sanity)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout(); return fig


def detection_maps(run_dir, scenario, model="additive", signature=None,
                   amplitude=None, dets=None, ncol=4):
    dets = dets or detectors(run_dir, scenario)
    key = None
    rows = []
    for det in dets:
        mp = load_maps(run_dir, scenario, det)
        if not mp:
            continue
        if key is None:
            m = load_metrics(run_dir, scenario, det).get(model, {})
            sig = signature or (list(m)[0] if m else None)
            amp = amplitude or sorted(m[sig], key=float)[-1]
            key = f"map|{model}|{sig}|{float(amp):g}"
            tgt_key = f"tgt|{model}|{sig}|{float(amp):g}"
        if key in mp:
            rows.append((det, mp[key], mp.get(tgt_key)))
    if not rows:
        return None
    n = len(rows) + 1; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(rows[0][2], cmap="gray"); axes[0].set_title("targets"); axes[0].axis("off")
    for ax, (det, smap, _) in zip(axes[1:], rows):
        ax.imshow(smap, cmap="inferno"); ax.set_title(det, fontsize=9); ax.axis("off")
    for ax in axes[len(rows) + 1:]:
        ax.axis("off")
    fig.suptitle(f"{scenario} — {model} — sig={signature or 'first'} amp={amplitude or 'max'}")
    fig.tight_layout(); return fig


def summary_table(run_dir, scenario, model="additive", signature=None):
    """Print an AUC table (detector x amplitude)."""
    dets = detectors(run_dir, scenario)
    rowsrc = load_metrics(run_dir, scenario, dets[0]).get(model, {})
    sig = signature or list(rowsrc)[0]
    amps = sorted(rowsrc.get(sig, {}), key=float)
    print(f"{scenario} | {model} | sig={sig}")
    print("  " + "detector".ljust(14) + "".join(f"{a:>8}" for a in amps))
    for det in dets:
        m = load_metrics(run_dir, scenario, det).get(model, {}).get(sig, {})
        print("  " + det.ljust(14) + "".join(
            f"{m.get(a, {}).get('auc', float('nan')):8.3f}" for a in amps))
