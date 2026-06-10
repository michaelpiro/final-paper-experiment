"""
validate_native.py — sanity-check the vendored deep/classical baselines in their
NATIVE regime (the real data + full-pixel targets the original papers used).

If our adapters reproduce the papers' high AUC on San Diego (real airplane
targets), the implementations are faithful — and their low AUC on our subpixel-
additive Pavia task is a genuine property of the method, NOT a bug.

Usage:
    python -u experiments/baseline_comparison/validate_native.py \
        [--dataset potential_spatial_baselines_code/code/TSTTD-main/Sandiego.mat] \
        [--detectors TSTTD,MCLT,E-CEM] [--epochs_scale 1.0]
"""

import argparse, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_ROOT, os.path.join(_ROOT, "experiments", "spatial"),
          os.path.join(_ROOT, "potential_spatial_baselines_code", "code", "TSTTD-main")):
    sys.path.insert(0, p)
os.chdir(_ROOT)

import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from ts_generation import ts_generation                       # from TSTTD repo

from experiments.baseline_comparison.framework import registry
from experiments.baseline_comparison.framework.detector_api import DetectorInput


def _standard(X):
    mn, mx = X.min(), X.max()
    return (X - mn) / (mx - mn) if mx != mn else X


def native_ctx(path, device, ts_type=7):
    m = sio.loadmat(path)
    data = m["data"].astype(np.float64)
    gt = m["map"] if "map" in m else m["gt"]
    data = _standard(data).astype(np.float32)
    H, W, B = data.shape
    prior = ts_generation(data, gt, ts_type).astype(np.float32).flatten()   # (B,)
    flat = data.reshape(-1, B)
    labels = (gt.reshape(-1) == 1).astype(int)
    ctx = DetectorInput(
        train_pix=flat, test_pix=flat, signature=prior,
        train_raw=flat, test_raw=flat, signature_raw=prior,
        sigma=0.1, device=device, seed=1,
        meta={"dataset": os.path.basename(path), "D": B, "D_raw": B, "spatial": False},
    )
    return ctx, labels, (H, W, B, int(labels.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default="potential_spatial_baselines_code/code/TSTTD-main/Sandiego.mat")
    ap.add_argument("--detectors", default="E-CEM,TSTTD,MCLT")
    ap.add_argument("--epochs_scale", type=float, default=1.0,
                    help="scale paper epochs (1.0 = faithful; <1 for a quick check)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    registry.ensure_loaded()
    ctx, labels, (H, W, B, ntgt) = native_ctx(args.dataset, device)
    print(f"Dataset {args.dataset}  {H}x{W}x{B}  targets={ntgt}/{H*W}  device={device}")
    print(f"{'detector':10s} {'AUC':>8s} {'fit_s':>7s}   (native regime: real full-pixel targets)")

    # paper epochs (scaled); other hyperparams stay at the adapters' paper defaults
    paper_epochs = {"TSTTD": {"epoch": 20}, "MCLT": {"mclt_epochs": 30}, "E-CEM": {}}
    for name in args.detectors.split(","):
        cfg = dict(paper_epochs.get(name, {}))
        for k in ("epoch", "mclt_epochs"):
            if k in cfg:
                cfg[k] = max(1, int(cfg[k] * args.epochs_scale))
        det = registry.build(name, cfg)
        t0 = time.time(); det.fit(ctx); fit_s = time.time() - t0
        scores = det.score(ctx)
        auc = roc_auc_score(labels, scores)
        print(f"{name:10s} {auc:8.4f} {fit_s:7.1f}", flush=True)


if __name__ == "__main__":
    main()
