"""
E-CEM — Ensemble-based Cascaded Constrained Energy Minimization.

Faithful self-contained port of:
  Zhao, Shi, Zou, Zhang, "Ensemble-Based Cascaded Constrained Energy Minimization
  for Hyperspectral Target Detection", Remote Sensing 2019.
  (upstream: potential_spatial_baselines_code/code/E_CEM-for-Hyperspectral-Target-Detection-master)

Differences from upstream: serial (no multiprocessing), no matplotlib/IO. The
algorithm is identical — multi-scale spectral-window CEM ensemble + cascaded
detection with sigmoid background suppression.

Target-agnostic (signature given at detect time), TRAINING-FREE / transductive
(uses the scored image's own autocorrelation), per-pixel spectral. Consumes raw
bands. needs_spatial=False -> runs on every dataset.
"""

from __future__ import annotations
import numpy as np

from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


def _cem(img, tgt, lam):
    """img (d, N), tgt (d,) -> (N,) CEM filter output."""
    d, N = img.shape
    R = (img @ img.T) / N
    w = np.linalg.solve(R + lam * np.eye(d), tgt)
    return w @ img


def _ms_scan(imgt, winsize, lam_lo, lam_hi, rng):
    """Multi-scale spectral scanning: slide a band-window, CEM each window."""
    d, M = imgt.shape
    winlen = max(2, int(d * winsize ** 2))
    rows = []
    for i in range(0, d - winlen + 1, 2):
        sub = imgt[i:i + winlen - 1, :]
        lam = rng.uniform(lam_lo, lam_hi)
        rows.append(_cem(sub, sub[:, -1], lam))
    if not rows:                                    # window too big for D
        rows = [_cem(imgt, imgt[:, -1], rng.uniform(lam_lo, lam_hi))]
    return np.asarray(rows)


def _ecem(img, tgt, windowsizes, num_layer, num_cem, Lambda, seed):
    """img (d, N), tgt (d,) -> (N,) detection scores."""
    rng = np.random.default_rng(seed)
    imgt = np.hstack([img, tgt[:, None]])           # append target column (d, N+1)
    lam_lo, lam_hi = Lambda / (1 + Lambda), Lambda
    mss = np.concatenate([_ms_scan(imgt, ws, lam_lo, lam_hi, rng)
                          for ws in windowsizes], 0)  # (R, N+1)
    forest = None
    for _ in range(num_layer):
        forest = np.stack([_cem(mss, mss[:, -1], rng.uniform(lam_lo, lam_hi))
                           for _ in range(num_cem)], 0)   # (num_cem, N+1)
        weights = 1.0 / (1.0 + np.exp(-forest.mean(0)))   # sigmoid suppression
        mss = mss * weights
    return forest[:, :-1].mean(0)                   # drop target col -> (N,)


@register("E-CEM")
class ECEM(Detector):
    needs_spatial = False
    space = "raw"

    def fit(self, ctx: DetectorInput) -> "Detector":
        return self                                  # transductive: nothing to fit

    def score(self, ctx: DetectorInput) -> np.ndarray:
        c = self.cfg
        img = ctx.test_raw.T.astype(np.float64)      # (d, N)
        tgt = ctx.signature_raw.astype(np.float64)   # (d,)
        return _ecem(img, tgt,
                     windowsizes=c.get("windowsize", [0.25, 0.5, 0.75, 1.0]),
                     num_layer=int(c.get("num_layer", 10)),
                     num_cem=int(c.get("num_cem", 6)),
                     Lambda=float(c.get("Lambda", 1e-6)),
                     seed=int(ctx.seed))
