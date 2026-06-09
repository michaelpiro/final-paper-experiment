"""
Our score-based detectors, wrapping the EXACT trainers/scorers from the spatial
pipeline so results are directly comparable to run_colab.

  CF-Attn-CFAR / CF-Attn : closed-form attention Gaussian score  (needs_spatial)
  NeighborMLP            : top-K neighbour denoiser score          (needs_spatial)
  DSM                    : global per-pixel score (per-band standardised) — works
                           on non-spatial data too.
"""

from __future__ import annotations
import numpy as np
import torch

import numpy as np

from final_paper_experiments.baselines.detectors import dsm_additive
from dsm_model import ScoreNet, Whitening
from cfattn_model import (
    CFAttnGaussianScoreNet, score_cfattn_additive, score_cfattn_additive_cfar,
)
from neighbor_mlp_model import NeighborMLPDenoiser, score_nmlp_additive
from run_colab import (
    DEFAULT_CFG, _train_cfattn, _train_nmlp, _train_dsm, _make_whitening,
)

from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


def _full_cfg(cfg):
    return {**DEFAULT_CFG, **(cfg or {})}


def _placeholder_whitening(D):
    """Identity whitening with the right shape so load_state_dict can fill it."""
    return Whitening(np.zeros(D, dtype=np.float32), np.eye(D, dtype=np.float32))


class _Torch(Detector):
    """Common torch save/load: persist state_dict + arch metadata."""
    def _build(self):
        raise NotImplementedError
    def state(self):
        return {"cfg": self.cfg, "log": self._log, "meta": self._meta,
                "sd": {k: v.cpu() for k, v in self._model.state_dict().items()}}
    def load_state(self, s):
        super().load_state(s)
        self._meta = s["meta"]
        self._model = self._build()
        self._model.load_state_dict(s["sd"]); self._model.eval()


@register("CF-Attn-CFAR")
class CFAttnCFAR(_Torch):
    """RAW input + frozen ZCA whitening first layer; detection in whitened space."""
    needs_spatial = True
    space = "raw"

    def _build(self):
        c = _full_cfg(self.cfg)
        return CFAttnGaussianScoreNet(D=self._meta["D"], h=c["cfattn_h"],
                                      K=c["cfattn_K"], sigma=self._meta["sigma"],
                                      eps=c.get("cfattn_eps", 1e-4),
                                      whitening=_placeholder_whitening(self._meta["D"]))

    def fit(self, ctx):
        c = _full_cfg(self.cfg)
        self._model = _train_cfattn(ctx.train_raw, ctx.train_nbr_raw.astype(np.float32),
                                    c, ctx.device, ctx.seed)
        self._meta = {"D": ctx.train_raw.shape[1], "sigma": self._model.sigma}
        return self

    def score(self, ctx):
        # RAW signature: the model returns data-space scores; the CFAR head
        # whitens the signature internally (whitening-invariant statistic).
        return score_cfattn_additive_cfar(self._model, ctx.test_raw,
                                          ctx.test_nbr_raw.astype(np.float32),
                                          ctx.signature_raw)


@register("CF-Attn")
class CFAttn(CFAttnCFAR):
    def fit(self, ctx):
        super().fit(ctx)
        self._train_pix = ctx.train_raw.astype(np.float32)
        self._train_nbr = ctx.train_nbr_raw.astype(np.float32)
        return self
    def score(self, ctx):
        return score_cfattn_additive(self._model, ctx.test_raw,
                                     ctx.test_nbr_raw.astype(np.float32),
                                     self._train_pix, self._train_nbr,
                                     ctx.signature_raw)
    def state(self):
        s = super().state(); s["tr"] = (self._train_pix, self._train_nbr); return s
    def load_state(self, s):
        super().load_state(s); self._train_pix, self._train_nbr = s["tr"]


@register("NeighborMLP")
class NeighborMLP(_Torch):
    needs_spatial = True
    space = "raw"

    def _build(self):
        c = _full_cfg(self.cfg)
        return NeighborMLPDenoiser(D=self._meta["D"], d_lat=c["nmlp_d_lat"],
                                   K=c["nmlp_K"], hidden=c["nmlp_hidden"],
                                   n_layers=c["nmlp_n_layers"], sigma=self._meta["sigma"],
                                   activation=c["activation"],
                                   whitening=_placeholder_whitening(self._meta["D"]))

    def fit(self, ctx):
        c = _full_cfg(self.cfg)
        self._model = _train_nmlp(ctx.train_raw, ctx.train_nbr_raw.astype(np.float32),
                                  c, ctx.device)
        self._meta = {"D": ctx.train_raw.shape[1], "sigma": self._model.sigma}
        self._train_pix = ctx.train_raw.astype(np.float32)
        self._train_nbr = ctx.train_nbr_raw.astype(np.float32)
        return self

    def score(self, ctx):
        return score_nmlp_additive(self._model, ctx.test_raw,
                                   ctx.test_nbr_raw.astype(np.float32),
                                   self._train_pix, self._train_nbr,
                                   ctx.signature_raw)
    def state(self):
        s = super().state(); s["tr"] = (self._train_pix, self._train_nbr); return s
    def load_state(self, s):
        super().load_state(s); self._train_pix, self._train_nbr = s["tr"]


@register("DSM")
class DSM(_Torch):
    """Per-pixel DSM-LMP on RAW bands; frozen ZCA whitening replaces the old
    PCA + per-band z-score (whitening subsumes both: decorrelate + unit-variance)."""
    needs_spatial = False
    space = "raw"

    def _build(self):
        c = _full_cfg(self.cfg)
        return ScoreNet(self._meta["D"], list(c["dsm_hidden"]), c["activation"],
                        whitening=_placeholder_whitening(self._meta["D"]))

    def fit(self, ctx):
        c = _full_cfg(self.cfg)
        self._model = _train_dsm(ctx.train_raw, c, ctx.device)
        self._meta = {"D": ctx.train_raw.shape[1], "sigma": self._model.sigma}
        self._train_raw = ctx.train_raw.astype(np.float32)
        return self

    def score(self, ctx):
        # model returns DATA-SPACE scores; use the RAW signature directly
        return dsm_additive(ctx.test_raw.astype(np.float32), self._train_raw,
                            self._model, ctx.signature_raw.astype(np.float32))

    def state(self):
        s = super().state(); s["tr"] = self._train_raw; return s
    def load_state(self, s):
        super().load_state(s); self._train_raw = s["tr"]
