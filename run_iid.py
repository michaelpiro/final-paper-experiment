#!/usr/bin/env python3
"""IID verification run — clean `repro` package (local / Apple MPS).

Pavia single-class + multi-class IID experiments (the paper's `iid_grid`
figure): vs-n sweep and vs-rho sweep, 5 seeds, detectors
AMF / GMM-Levin(multi) / L-DART / DART / L-LRao / LRao.

The fixed LRao is NATIVE here (config `lrao_input_norm: robust` = per-band
median/IQR front-end; L-LRao forced linear) — no monkeypatching. The runner
`repro.protocols.iid.run_iid` is the verbatim published pipeline; it writes
metrics.json / scores.npz / loss curves / per-run figures, then everything is
zipped at the end.

Local port of RunIID.ipynb: no Colab clone/download, runs from the repo root,
and uses Apple MPS when available (set env DEVICE=cpu to force CPU).
"""
import os

# Unsupported MPS ops transparently fall back to CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Run from the repo root so the `repro/...` relative paths resolve.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch


def pick_device():
    """Prefer Apple MPS, then CUDA, then CPU. Override with env DEVICE."""
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
assert os.path.exists("repro/data/pavia-u.mat"), "missing repro/data/pavia-u.mat"

from repro.protocols.iid import run_iid


def main():
    # ---- single-class ----
    cfg_s = yaml.safe_load(open("repro/configs/iid_single.yaml"))
    cfg_s.update(dataset="repro/data/pavia-u.mat", device=DEVICE,
                 results_dir="results/iid_single", lrao_input_norm="robust")
    print({k: cfg_s[k] for k in ("n_train_list", "rho_list", "seed")
           if k in cfg_s})
    run_iid(cfg_s, mode="single")

    # ---- multi-class ----
    cfg_m = yaml.safe_load(open("repro/configs/iid_multi.yaml"))
    cfg_m.update(dataset="repro/data/pavia-u.mat", device=DEVICE,
                 results_dir="results/iid_multi", lrao_input_norm="robust")
    run_iid(cfg_m, mode="multi")

    # ---- package ----
    import zipfile
    with zipfile.ZipFile("iid_verification.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("results/iid_single", "results/iid_multi"):
            for root, _, files in os.walk(d):
                for fn in files:
                    z.write(os.path.join(root, fn))
    print("zipped -> iid_verification.zip")


if __name__ == "__main__":
    main()
