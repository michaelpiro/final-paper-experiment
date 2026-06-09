"""Classical / statistical baselines — thin wrappers over existing implementations."""

from __future__ import annotations
import numpy as np

from final_paper_experiments.baselines.detectors import amf, reg_amf, cem
from final_paper_experiments.baselines.gmm_glrt_levin import GMMGLRTLevin
from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


class _Cached(Detector):
    """Closed-form detector: fit just caches the background training pixels."""
    def fit(self, ctx: DetectorInput) -> "Detector":
        self._train = ctx.train_pix
        return self
    def state(self):
        return {"cfg": self.cfg, "log": self._log, "train": self._train}
    def load_state(self, s):
        super().load_state(s); self._train = s["train"]


@register("AMF")
class AMFDetector(_Cached):
    def score(self, ctx): return amf(ctx.test_pix, self._train, ctx.signature)


@register("Reg-AMF")
class RegAMFDetector(_Cached):
    def score(self, ctx):
        return reg_amf(ctx.test_pix, self._train, ctx.signature, ctx.sigma)


@register("CEM")
class CEMDetector(_Cached):
    def score(self, ctx): return cem(ctx.test_pix, self._train, ctx.signature)


@register("GMM-Levin")
class GMMLevinDetector(Detector):
    def fit(self, ctx: DetectorInput) -> "Detector":
        self._gmm = GMMGLRTLevin(seed=ctx.seed).fit(ctx.train_pix)
        return self
    def score(self, ctx):
        p_steps = int(self.cfg.get("p_steps", 50))
        return self._gmm.score(ctx.test_pix, ctx.signature,
                               model="additive", p_steps=p_steps)
    def state(self):
        return {"cfg": self.cfg, "log": self._log, "gmm": self._gmm}
    def load_state(self, s):
        super().load_state(s); self._gmm = s["gmm"]
