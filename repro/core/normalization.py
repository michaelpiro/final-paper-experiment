"""Input-normalization front-ends for the score nets.

ZCA whitening lives on core.data.Whitening (verbatim port). This module adds
the two ROBUST variants used by the fixed LRao — both published:

  robust_whitening_iqr — mu = per-band median, W = diag(1/IQR)
                         (the camera-ready IID configuration)
  robust_whitening_mad — mu = per-band median, W = diag(1/(1.4826*MAD))
                         (the camera-ready spatial configuration, lrao_val2)
"""
import numpy as np

from .data import Whitening


def robust_whitening_iqr(train_raw, cfg=None):
    X = np.asarray(train_raw, dtype=np.float64)
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr = np.where(iqr > 1e-8, iqr, 1.0)
    return Whitening(med.astype(np.float32), np.diag(1.0 / iqr).astype(np.float32))


def robust_whitening_mad(train_raw, cfg=None):
    X = np.asarray(train_raw, dtype=np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = np.sqrt(np.maximum(mad ** 2, 1e-22))
    return Whitening(med.astype(np.float32), np.diag(1.0 / scale).astype(np.float32))
