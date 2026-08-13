#!/usr/bin/env python3
"""Spatial verification run — clean `repro` package (local / Apple MPS).

Runs the full published protocol: 3 scenes x 5 seeds x 12-theta sweep,
12 detectors (DART, DART-CFAR, DARTS, DARTS-CFAR, AMF-global, AMF-local,
GMM-Levin, LRao(val-ES) + THANTD, HTD-Net, TSTTD, OS-VAE). Saves raw scores
per (seed, theta), per-seed metrics (per-class Pfa on Pavia), checkpoints
(LRao every 10 epochs), then verifies against the published numbers and zips
everything.

Deep baselines load the bundled published checkpoints
(`repro/checkpoints/deep/`, 60 files) instead of retraining (hours).
Set `retrain_deep: true` in the config (or CFG['retrain_deep']=True below)
to retrain from scratch.

Local port of RunSpatial.ipynb: no Colab clone/download, runs from the repo
root, and uses Apple MPS when available (set env DEVICE=cpu to force CPU).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

from repro.protocols import spatial as SP
from repro.analysis.verify import verify_spatial
from repro.analysis.tables import make_tables


def pick_device():
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
print("device:", DEVICE)
for p in ("repro/data/pavia-u.mat", "repro/data/Sandiego.mat",
          "repro/data/Sandiego2.mat", "repro/data/sandiego_regions.json",
          "repro/data/sandiego2_regions.json"):
    assert os.path.exists(p), f"missing {p}"
print("all bundled data present")


def main():
    CFG = SP.load_cfg()
    print("scenes:", CFG["scenes"], " seeds:", CFG["seeds"])
    print("thetas:", CFG["thetas"])
    print("deep:", CFG["deep_detectors"], "| pretrained:",
          CFG["deep_pretrained"], "| retrain_deep:", CFG["retrain_deep"])
    # CFG['retrain_deep'] = True   # <- uncomment to retrain the deep baselines

    SP.run_scene("pavia4", CFG, out_root="results/spatial", device=DEVICE)
    SP.run_scene("sandiego", CFG, out_root="results/spatial", device=DEVICE)
    SP.run_scene("sandiego2", CFG, out_root="results/spatial", device=DEVICE)

    SP.summarize("results/spatial")

    verify_spatial("results/spatial")
    make_tables("results/spatial", dst="results/spatial/tables")

    # ---- package ----
    import zipfile
    with zipfile.ZipFile("spatial_verification.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk("results/spatial"):
            for fn in files:
                z.write(os.path.join(root, fn))
    print("zipped -> spatial_verification.zip")


if __name__ == "__main__":
    main()
