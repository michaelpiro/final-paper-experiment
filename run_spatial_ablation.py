#!/usr/bin/env python3
"""Run the SPATIAL ablation plan (repro/configs/spatial_ablations.yaml).

The spatial counterpart of run_iid_ablations.py. Everything you would want to
change lives in the YAML plan, not here: edit `ablations:` to add or drop an
entry, `lean:` to control how long a sweep takes, `common:` for settings shared
by every entry.

Each entry is merged as  {base < lean (unless --full) < common < entry}  and run
into results_root/<entry-name>/, then summarized. An entry key that names a
config BLOCK (dart / darts / lrao / deep) is merged into that block; any other
key is set at the top level.

Usage:
    python run_spatial_ablation.py                       # every entry (lean)
    python run_spatial_ablation.py baseline dart_calibrated
    python run_spatial_ablation.py --full                # the published protocol
    python run_spatial_ablation.py --dry-run             # print merged flags only
    python run_spatial_ablation.py --list                # list entries and exit
    DEVICE=cpu python run_spatial_ablation.py
"""
import copy
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch

from repro.protocols import spatial as SP

PLAN = "repro/configs/spatial_ablations.yaml"
BLOCKS = ("dart", "darts", "lrao", "lrao2", "deep")


def pick_device():
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_plan(path=PLAN):
    plan = yaml.safe_load(open(path))
    base = SP.load_cfg(plan.get("base", "repro/configs/spatial.yaml"))
    # keys starting with '_' are YAML-anchor holders, not ablations
    plan["ablations"] = {k: v for k, v in (plan.get("ablations") or {}).items()
                         if not k.startswith("_")}
    return plan, base


def merged_cfg(base, plan, entry, lean=True):
    """base < lean < common < entry. A key naming a config BLOCK is merged into
    that block; anything else replaces the top-level key."""
    cfg = copy.deepcopy(base)
    if lean:
        cfg.update(plan.get("lean") or {})
    for src in (plan.get("common") or {}, entry or {}):
        for k, v in (src or {}).items():
            if k in BLOCKS and isinstance(v, dict):
                cfg[k] = {**cfg.get(k, {}), **v}
            else:
                cfg[k] = v
    return cfg


def describe(cfg):
    """The knobs worth seeing at a glance, per block."""
    out = {}
    for blk in ("dart", "darts"):
        b = cfg.get(blk, {})
        out[blk] = dict(variant=b.get("dsm_variant", "published"),
                        front=b.get("whiten_mode", "zca"),
                        rho=b.get("dsm_sigma_rho"))
        if b.get("whiten_mode") == "shrink":
            out[blk]["shrink"] = f"{b.get('shrink_space', 'cov')}/{b.get('shrink_lam')}"
    l = cfg.get("lrao", {})
    out["lrao"] = dict(variant=l.get("variant", "published"),
                       preproc=l.get("lrao_preproc", "robust"),
                       net=l.get("lrao_net", "mlp"),
                       sig_aware=bool(l.get("lrao_signal_aware", False)),
                       reg=l.get("lfi_sigma_reg", "none"))
    if cfg.get("lrao2"):                    # optional second LRao slot
        l2 = {**l, **cfg["lrao2"]}
        out[str(l2.get("label", "LRao-2"))] = dict(
            variant=l2.get("variant", "published"),
            preproc=l2.get("lrao_preproc", "robust"),
            net=l2.get("lrao_net", "mlp"),
            sig_aware=bool(l2.get("lrao_signal_aware", False)),
            detach=bool(l2.get("detach_sigma", True)),
            reg=l2.get("lfi_sigma_reg", "none"))
    return out


def main(argv):
    full = "--full" in argv
    dry = "--dry-run" in argv
    listing = "--list" in argv
    argv = [a for a in argv if not a.startswith("--")]

    plan, base = load_plan()
    root = plan.get("results_root", "results/spatial_ablation")
    names = [a for a in argv if a in plan["ablations"]] or list(plan["ablations"])
    unknown = [a for a in argv if a not in plan["ablations"]]
    for u in unknown:
        print(f"  [skip] unknown ablation {u!r}")

    if listing:
        print(f"{len(plan['ablations'])} entries in {PLAN} -> {root}")
        for k in plan["ablations"]:
            print("   ", k)
        return

    device = pick_device()
    print(f"device: {device} | {'FULL' if full else 'lean'} | "
          f"{len(names)} entr{'y' if len(names) == 1 else 'ies'} -> {root}")

    for name in names:
        cfg = merged_cfg(base, plan, plan["ablations"][name], lean=not full)
        out = os.path.join(root, name)
        print(f"\n--- [{name}] {describe(cfg)}")
        print(f"    scenes={cfg['scenes']} seeds={cfg['seeds']} "
              f"thetas={len(cfg['thetas'])} -> {out}", flush=True)
        if dry:
            continue
        for scene in cfg["scenes"]:
            SP.run_scene(scene, cfg, out_root=out, device=device)
        SP.summarize(out)

    if dry:
        print("\n(dry run — nothing trained)")
    else:
        print(f"\nDONE -> {root}/{{{','.join(names)}}}")


if __name__ == "__main__":
    main(sys.argv[1:])
