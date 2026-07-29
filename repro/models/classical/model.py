"""Classical detectors as classes: AMF, AMFLocal, GMMLevin.

Training-free: fit() stores the background statistics; score() applies the
paper formula. The underlying math is the verbatim port in core.detectors
(the code that produced the published numbers).
"""
import numpy as np

from repro.core.detectors import GMMGLRTLevin, amf, amf_local


class AMF:
    """Global adaptive matched filter:
    T(y) = s^T Sigma^-1 (y - mu) / sqrt(s^T Sigma^-1 s), SCM from the
    training background; eig_floor=0 -> pure AMF (numerical guard only)."""

    def __init__(self, cfg):
        self.eig_floor = float(cfg.get('baseline_eig_floor', 0.0))
        self.tr = None

    def fit(self, tr_raw, *a, **k):
        self.tr = np.asarray(tr_raw)
        return self

    def score(self, test_pixels, sig):
        return amf(test_pixels, self.tr, sig, eig_floor=self.eig_floor)


class AMFLocal:
    """AMF on a per-pixel local SCM from each pixel's k x k window (diagonal
    loading `loading`; loading=0 -> numerical floor only)."""

    def __init__(self, cfg):
        self.loading = float(cfg.get('local_scm_loading', 0.0))
        self.window = cfg.get('amf_local_window')      # None -> dimension-aware

    @classmethod
    def dimension_aware_window(cls, D):
        """Smallest odd k with k^2 - 1 >= 2D (15 on 103 bands, 21 on 189)."""
        k = 3
        while k * k - 1 < 2 * D:
            k += 2
        return k

    def resolved_window(self, D):
        return int(self.window) if self.window else self.dimension_aware_window(D)

    def fit(self, *a, **k):
        return self

    def score(self, test_pixels, test_windows, sig, device='cpu'):
        return amf_local(test_pixels, test_windows, sig, device=device,
                         loading=self.loading)


class GMMLevin:
    """Levin (2019) product-of-GMMs GLRT, additive model, fill factor by grid
    search. fit() fits the background mixture once; score() runs the GLRT."""

    def __init__(self, cfg):
        self.p_steps = int(cfg.get('gmm_steps', 50))
        self.model = None

    def fit(self, tr_raw, *a, **k):
        self.model = GMMGLRTLevin().fit(np.asarray(tr_raw, np.float64))
        return self

    def score(self, test_pixels, sig):
        return self.model.score(test_pixels, sig, p_steps=self.p_steps)
