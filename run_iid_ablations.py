#!/usr/bin/env python3
"""Run + aggregate the IID LRao / DART ablation sweeps (local / Apple MPS).

Consumes an ablation PLAN (repro/configs/iid_{single,multi}_ablations.yaml) and
runs the published `run_iid` pipeline once per ablation entry, merging
{base < common < entry-overrides} into  results/iid_<mode>_ablations/<entry>/.
Afterwards it AGGREGATES every entry into one place:

    results/iid_<mode>_ablations/_aggregate/
        summary.csv         long-form: every detector, every ablation
        summary.md          focused tables (each ablation's ablated detector
                            + the baseline DART / LRao references)
        figures/*.png|pdf   overlay curves (AUC & Pd vs n and vs rho)

Usage:
    python run_iid_ablations.py                     # both modes: run + aggregate
    python run_iid_ablations.py single              # one mode
    python run_iid_ablations.py multi baseline lrao_paper_general
    python run_iid_ablations.py --aggregate-only    # re-aggregate existing runs
    python run_iid_ablations.py --dry-run           # print merged flags only
    python run_iid_ablations.py multi --scene=sandiego     # run on a SCENE
    python run_iid_ablations.py multi dart_wmw --scene=sandiego2

`--scene=<pavia4|sandiego|sandiego2>` sources the IID pools from the spatial
scene builder instead of the Pavia GT classes (the only way to run San Diego,
which has no per-pixel class GT) and writes to <results_root>__<scene>/.

Set env DEVICE=cpu to force CPU (MPS is used by default when available).
"""
import os
import sys
import csv
import glob
import json

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG_DIR = "repro/configs"
PLANS = {"single":         "iid_single_ablations.yaml",
         "multi":          "iid_multi_ablations.yaml",
         # second scene (Salinas, 204 bands) — same entries, different `base`
         "single_salinas": "iid_single_salinas_ablations.yaml",
         "multi_salinas":  "iid_multi_salinas_ablations.yaml"}
# detectors treated as references (drawn from the baseline run)
REF_DETS = ["DART", "LRao", "L-LRao"]
ABL_FLAG_KEYS = ("lrao_net", "lrao_preproc", "lrao_signal_aware", "dsm_preproc",
                 "dsm_variant", "whiten_mode", "lfi_sigma_reg")


def pick_device():
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_plan(mode):
    plan = yaml.safe_load(open(os.path.join(CFG_DIR, PLANS[mode])))
    base = yaml.safe_load(open(os.path.join(CFG_DIR, plan["base"])))
    return plan, base


def merged_cfg(base, plan, overrides, device):
    """base < common < entry-overrides, plus runtime fields."""
    cfg = dict(base)
    cfg.update(plan.get("common", {}) or {})
    cfg.update(overrides or {})
    # the base config owns `dataset` (Pavia, Salinas, ...); only fall back to
    # Pavia when it did not say. Overwriting it unconditionally silently ran
    # every plan on Pavia regardless of its `base`.
    cfg.setdefault("dataset", "repro/data/pavia-u.mat")
    cfg["device"] = device
    return cfg


# ===========================================================================
# aggregation
# ===========================================================================
def _find_run(ablation_dir):
    """Locate an ablation's metrics + config. Prefers a multi-seed aggregate,
    else the newest single-seed run. Returns (metrics, cfg) or None."""
    agg = glob.glob(os.path.join(ablation_dir, "**", "metrics_aggregate.json"),
                    recursive=True)
    single = glob.glob(os.path.join(ablation_dir, "**", "metrics.json"),
                       recursive=True)
    cands = agg + [p for p in single if p not in agg]
    cands = [p for p in cands if os.path.getsize(p) > 0]
    if not cands:
        return None
    # prefer aggregate; within a class, newest by mtime
    cands.sort(key=lambda p: (0 if p.endswith("metrics_aggregate.json") else 1,
                              -os.path.getmtime(p)))
    for path in cands:
        try:
            metrics = json.load(open(path))
        except Exception:
            continue
        if "vs_n" not in metrics:
            continue
        cfg_p = os.path.join(os.path.dirname(path), "config.yaml")
        cfg = yaml.safe_load(open(cfg_p)) if os.path.exists(cfg_p) else {}
        return metrics, cfg
    return None


def _ablated_detectors(cfg, available):
    """Which detector(s) an ablation changed (the 'ablated' detector(s) to plot).
    `baseline` is handled by name in collect(), so this is never called for it.
    A single entry may ablate BOTH DART and LRao (composite runs), so this is
    additive. Falls back to whatever LRao-family key is present (so runs made
    before the 'LRao-CNN' relabel still resolve)."""
    out = []
    # DART (DSM) ablation
    if str(cfg.get("dsm_preproc", "")) in ("mad", "wmw") or \
       str(cfg.get("dsm_variant", "")) in ("calibrated", "multi"):
        out += [d for d in ("DART",) if d in available]
    # LRao ablation (the ablated LRao, not the DART-matched default 'LRao')
    lrao_abl = (str(cfg.get("lrao_net", "mlp")) == "cnn"
                or str(cfg.get("lrao_preproc", "whiten")) == "mad"
                or bool(cfg.get("lrao_signal_aware", False)))
    if lrao_abl:
        for d in ("LRao-CNN", "LRao-abl", "L-LRao"):
            if d in available and d not in out:
                out.append(d)
                break
    if out:
        return out
    # no ablation flag matched → fall back to whatever LRao-family key is present
    for d in ("LRao", "L-LRao", "LRao-CNN"):
        if d in available:
            return [d]
    return []


def _curve(metrics, sweep, det, metric):
    """(x, y, y_std) for one detector/metric, or None."""
    node = metrics.get(sweep, {}).get(det, {})
    if metric not in node:
        return None
    x = metrics["n_list"] if sweep == "vs_n" else metrics["rho_list"]
    y = [float(v) if v is not None else np.nan for v in node[metric]]
    ystd = node.get(f"{metric}_std")
    ystd = [float(v) for v in ystd] if ystd else None
    n = min(len(x), len(y))
    return (list(x[:n]), y[:n], ystd[:n] if ystd else None)


def collect(mode, plan, root):
    """Gather every ablation's run. Returns dict keyed by ablation name."""
    runs = {}
    for name in plan["ablations"]:
        got = _find_run(os.path.join(root, name))
        if got is None:
            print(f"  [aggregate] no results for '{name}' — skipping")
            continue
        metrics, cfg = got
        available = list(metrics["vs_n"].keys())
        is_base = (name == "baseline")     # by name (all entries now carry flags)
        ablated = [] if is_base else _ablated_detectors(cfg, available)
        runs[name] = dict(metrics=metrics, cfg=cfg, available=available,
                          ablated=ablated, is_base=is_base)
    return runs


def write_csv(runs, out_path):
    """Long-form: every detector of every ablation, both sweeps, all metrics."""
    rows = []
    for name, r in runs.items():
        role_of = {d: "ablated" for d in r["ablated"]}
        for sweep, metrics_ in (("vs_n", ("auc", "pauc", "pd")),
                                ("vs_rho", ("auc", "pd"))):
            for det in r["available"]:
                for metric in metrics_:
                    c = _curve(r["metrics"], sweep, det, metric)
                    if c is None:
                        continue
                    x, y, ystd = c
                    for i, (xv, yv) in enumerate(zip(x, y)):
                        rows.append(dict(
                            ablation=name, detector=det,
                            role=role_of.get(det, "reference" if det in REF_DETS
                                             else "other"),
                            sweep=sweep, metric=metric, x=xv, y=yv,
                            y_std=(ystd[i] if ystd else "")))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ablation", "detector", "role",
                                          "sweep", "metric", "x", "y", "y_std"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _series(runs, sweep, metric):
    """Ordered list of (label, det, x, y, ystd, is_ref) to plot/tabulate:
    baseline references first, then each ablation's ablated detector."""
    out = []
    base = next((r for r in runs.values() if r["is_base"]), None)
    if base is not None:
        for det in REF_DETS:
            c = _curve(base["metrics"], sweep, det, metric)
            if c:
                out.append((f"baseline · {det}", det, *c, True))
    for name, r in runs.items():
        if r["is_base"]:
            continue
        for det in r["ablated"]:
            c = _curve(r["metrics"], sweep, det, metric)
            if c:
                # label ablated series by the ablation name alone — the run_iid
                # slot name ('L-LRao') is a confusing internal artifact (the CNN
                # ablations aren't linear), and the name already identifies it.
                out.append((name, det, *c, False))
    return out


def write_markdown(runs, mode, out_path):
    lines = [f"# IID {mode} — LRao / DART ablation summary\n",
             "Reference rows are the baseline detectors; each ablation row is "
             "the detector that ablation changed (its 'ablated' detector).\n"]
    titles = {("vs_n", "auc"): "AUC vs n", ("vs_n", "pd"): "Pd@Pfa vs n",
              ("vs_rho", "auc"): "AUC vs rho", ("vs_rho", "pd"): "Pd@Pfa vs rho"}
    xname = {"vs_n": "n", "vs_rho": "rho"}
    for sweep in ("vs_n", "vs_rho"):
        for metric in ("auc", "pd"):
            series = _series(runs, sweep, metric)
            if not series:
                continue
            # align on the union of x-values (robust to differing grids)
            xs = sorted({v for _l, _d, x, _y, _s, _r in series for v in x})
            lines.append(f"\n## {titles[(sweep, metric)]}\n")
            lines.append("| series | " +
                         " | ".join(f"{xname[sweep]}={v:g}" for v in xs) + " |")
            lines.append("|" + "---|" * (len(xs) + 1))
            for label, _det, x, y, _ystd, _isref in series:
                m = {xv: yv for xv, yv in zip(x, y)}
                cells = " | ".join(
                    (f"{m[v]:.3f}" if (v in m and m[v] == m[v]) else "")
                    for v in xs)
                lines.append(f"| {label} | {cells} |")
    open(out_path, "w").write("\n".join(lines) + "\n")


def plot_all(runs, mode, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    logx = True
    specs = [("vs_n", "auc", "n (training pixels)", "AUC"),
             ("vs_n", "pd", "n (training pixels)", "Pd @ Pfa"),
             ("vs_rho", "auc", r"DSM noise level $\rho$", "AUC"),
             ("vs_rho", "pd", r"DSM noise level $\rho$", "Pd @ Pfa")]
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for sweep, metric, xlabel, ylabel in specs:
        series = _series(runs, sweep, metric)
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ci = 0
        for label, _det, x, y, ystd, is_ref in series:
            x = np.asarray(x, float); y = np.asarray(y, float)
            if is_ref:
                ax.plot(x, y, "--", color="0.5" if "DART" in label else "0.15",
                        lw=1.4, alpha=0.9, label=label, zorder=1)
            else:
                c = cmap[ci % 10]; ci += 1
                ax.plot(x, y, "o-", color=c, lw=2.0, label=label, zorder=3)
                if ystd is not None:
                    ystd = np.asarray(ystd, float)
                    ax.fill_between(x, y - ystd, y + ystd, color=c, alpha=0.15)
        if logx:
            ax.set_xscale("log"); ax.set_xticks(series[0][2])
            ax.set_xticklabels(series[0][2]); ax.minorticks_off()
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"IID {mode}: {ylabel} {sweep.replace('vs_', 'vs ')} "
                     f"— ablations vs baseline")
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=2)
        fig.tight_layout()
        base = os.path.join(fig_dir, f"{metric}_{sweep}")
        fig.savefig(base + ".png", dpi=200); fig.savefig(base + ".pdf")
        plt.close(fig)


def aggregate(mode, plan, root):
    print(f"\n=== aggregating IID {mode} under {root} ===")
    runs = collect(mode, plan, root)
    if not runs:
        print("  [aggregate] nothing to aggregate"); return
    out = os.path.join(root, "_aggregate")
    os.makedirs(out, exist_ok=True)
    n = write_csv(runs, os.path.join(out, "summary.csv"))
    write_markdown(runs, mode, os.path.join(out, "summary.md"))
    plot_all(runs, mode, os.path.join(out, "figures"))
    abl = {k: v["ablated"] for k, v in runs.items() if not v["is_base"]}
    print(f"  {len(runs)} run(s), {n} csv rows -> {out}")
    print(f"  ablated detectors: {abl}")
    print(f"  wrote summary.csv, summary.md, figures/*.png")


# ===========================================================================
def main(argv):
    dry = "--dry-run" in argv
    agg_only = "--aggregate-only" in argv
    # --scene=<name> runs every ablation on a spatial SCENE (pavia4 / sandiego /
    # sandiego2) instead of the Pavia GT classes: the IID pools come from the
    # scene builder (see run_iid). Results go to <results_root>__<scene>/ so the
    # scenes do not overwrite each other (or the class-based runs).
    scene = next((a.split("=", 1)[1] for a in argv if a.startswith("--scene=")), None)
    argv = [a for a in argv if not a.startswith("--")]

    modes = [a for a in argv if a in PLANS] or ["single", "multi"]
    wanted = [a for a in argv if a not in PLANS]

    device = pick_device()
    print("device:", device, "| dry-run:", dry, "| aggregate-only:", agg_only,
          "| scene:", scene or "(class-based)")

    if not (dry or agg_only):
        from repro.protocols.iid import run_iid

    for mode in modes:
        plan, base = load_plan(mode)
        root = plan.get("results_root", f"results/iid_{mode}_ablations")
        if scene:
            root = f"{root}__{scene}"
        names = wanted or list(plan["ablations"])
        print(f"\n=== IID {mode}: {len(names)} ablation(s) -> {root} ===")
        if not agg_only:
            for name in names:
                if name not in plan["ablations"]:
                    print(f"  [skip] unknown ablation '{name}'"); continue
                cfg = merged_cfg(base, plan, plan["ablations"][name], device)
                if scene:
                    cfg["scene"] = scene
                cfg["results_dir"] = os.path.join(root, name)
                flags = {k: cfg.get(k) for k in
                         ("scene", "lrao_net", "lrao_preproc", "lrao_signal_aware",
                          "lfi_sigma_reg", "lfi_sigma_cutoff",
                          "dsm_preproc", "dsm_variant", "whiten_mode",
                          "dsm_sigma_rho", "run_lrao_mlp")}
                print(f"\n--- [{mode}:{name}] {flags} -> {cfg['results_dir']} ---")
                if dry:
                    continue
                run_iid(cfg, mode=plan.get("mode", mode))
        if not dry:
            aggregate(mode, plan, root)

    if dry:
        print("\n(dry run — nothing trained)")


if __name__ == "__main__":
    main(sys.argv[1:])
