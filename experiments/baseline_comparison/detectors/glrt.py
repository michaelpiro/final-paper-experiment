"""
GLRT-family detectors: DLTD and SMGLRT.

  DLTD    — Distribution-Level Target Detection (Ma et al., IEEE GRSL 2026).
             Shared-covariance K-component GMM; requires K ≥ 3.
  SMGLRT  — Segmented-Mixing GLRT (Ma et al., IEEE JSTARS 2025).
             Per-segment MLE fill-factor α_k, shared-cov GMM; requires K ≥ 3.

Both operate in the feature (pix) space — for the raw-band no-PCA pipeline
pix == raw, so there is no distinction.
"""

from __future__ import annotations
import numpy as np

from final_paper_experiments.baselines.detectors import dltd, smglrt
from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


class _GLRTBase(Detector):
    """Caches train pixels (no learned model — GMM is fit at score time)."""
    def fit(self, ctx: DetectorInput) -> "Detector":
        self._train = ctx.train_pix
        return self

    def state(self):
        return {"cfg": self.cfg, "log": self._log, "train": self._train}

    def load_state(self, s):
        super().load_state(s)
        self._train = s["train"]


@register("DLTD")
class DLTDDetector(_GLRTBase):
    """Distribution-Level Target Detection (Ma et al. 2026). K ≥ 3."""
    def score(self, ctx: DetectorInput) -> np.ndarray:
        K = int(self.cfg.get("K", 3))
        K = max(K, 3)   # guard: K=1 is degenerate (constant score)
        return dltd(ctx.test_pix, self._train, ctx.signature, K=K)


@register("SMGLRT")
class SMGLRTDetector(_GLRTBase):
    """Segmented-Mixing GLRT (Ma et al. 2025). K ≥ 3."""
    def score(self, ctx: DetectorInput) -> np.ndarray:
        K          = int(self.cfg.get("K", 3))
        K          = max(K, 3)
        n_segments = self.cfg.get("n_segments", None)
        if n_segments is not None:
            n_segments = int(n_segments)
        return smglrt(ctx.test_pix, self._train, ctx.signature,
                      K=K, n_segments=n_segments)
