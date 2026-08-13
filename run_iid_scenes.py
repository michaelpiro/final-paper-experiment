#!/usr/bin/env python3
"""Run the IID experiment on the spatial SCENES — Pavia-4 and San Diego.

The IID sweeps (vs-n, vs-rho; AMF / GMM-Levin / DART / L-DART / LRao / L-LRao)
normally run on the Pavia GT classes (run_iid.py). San Diego has no per-pixel
class ground truth, only target REGIONS, so this runner sources the IID pools
from the spatial scene builder instead: background TRAIN pool = scene['tr'],
disjoint TEST pool = scene['te'], target signature = scene['sig'] (the same
pools the spatial protocol uses). Enabled by `cfg['scene']` in run_iid.

One run_iid per scene into results/iid_scenes/<scene>/. The detector/sweep
config is taken from iid_multi.yaml (multi-class background = a natural scene
mixture); per-scene sizing is applied below. n_train_list is auto-clamped to
each scene's available training pixels (~4000).

Usage:
    python run_iid_scenes.py                       # pavia4 + sandiego + sandiego2
    python run_iid_scenes.py sandiego              # one scene
    DEVICE=cpu python run_iid_scenes.py
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch

BASE_CFG = "repro/configs/iid_multi.yaml"
OUT_ROOT = "results/iid_scenes"

# Per-scene overrides. `amplitude` follows the spatial convention (Pavia 0.15,
# San Diego 0.075); n_train_list is clamped to the scene's train pixels anyway.
SCENES = {
    "pavia4":    dict(scene="pavia4",    amplitude=0.15,  test_size=2000,
                      n_fixed_for_rho=1000),
    "sandiego":  dict(scene="sandiego",  amplitude=0.075, test_size=1800,
                      n_fixed_for_rho=1000),
    "sandiego2": dict(scene="sandiego2", amplitude=0.075, test_size=2000,
                      n_fixed_for_rho=1000),
}
# IID sweep sizing shared across scenes (train pools are ~4000 px).
COMMON = dict(
    n_train_list=[20, 40, 60, 100, 200, 500, 1000, 2000, 4000],
    lrao_input_norm="robust",
)


def pick_device():
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(argv):
    device = pick_device()
    wanted = [a for a in argv if a in SCENES] or list(SCENES)
    print(f"device: {device} | scenes: {wanted}")

    from repro.protocols.iid import run_iid
    for name in wanted:
        cfg = yaml.safe_load(open(BASE_CFG))
        cfg.update(COMMON)
        cfg.update(SCENES[name])
        cfg.update(device=device,
                   results_dir=os.path.join(OUT_ROOT, name))
        print(f"\n=== IID on scene [{name}] -> {cfg['results_dir']} ===", flush=True)
        run_iid(cfg, mode="multi")

    print(f"\nDONE -> {OUT_ROOT}/{{{','.join(wanted)}}}")


if __name__ == "__main__":
    main(sys.argv[1:])
