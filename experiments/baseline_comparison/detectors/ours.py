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

from final_paper_experiments.data_utils import compute_sigma_from_data
from final_paper_experiments.baselines.detectors import dsm_additive
from dsm_model import ScoreNet
from cfattn_model import (
    CFAttnGaussianScoreNet, score_cfattn_additive, score_cfattn_additive_cfar,
)
from neighbor_mlp_model import NeighborMLPDenoiser, score_nmlp_additive
from run_colab import DEFAULT_CFG, _train_cfattn, _train_nmlp, _train_dsm

from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


def _full_cfg(cfg):
    return {**DEFAULT_CFG, **(cfg or {})}


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
    needs_spatial = True

    def _build(self):
        c = _full_cfg(self.cfg)
        return CFAttnGaussianScoreNet(D=self._meta["D"], h=c["cfattn_h"],
                                      K=c["cfattn_K"], sigma=self._meta["sigma"],
                                      eps=c.get("cfattn_eps", 1e-4))

    def fit(self, ctx):
        c = _full_cfg(self.cfg)
        self._meta = {"D": ctx.train_pix.shape[1], "sigma": ctx.sigma}
        self._model = _train_cfattn(self._meta["D"], ctx.sigma, ctx.train_pix,
                                    ctx.train_nbr, c, ctx.device, ctx.seed)
        return self

    def score(self, ctx):
        return score_cfattn_additive_cfar(self._model, ctx.test_pix,
                                          ctx.test_nbr.astype(np.float32),
                                          ctx.signature)


@register("CF-Attn")
class CFAttn(CFAttnCFAR):
    def fit(self, ctx):
        super().fit(ctx)
        self._train_pix, self._train_nbr = ctx.train_pix, ctx.train_nbr.astype(np.float32)
        return self
    def score(self, ctx):
        return score_cfattn_additive(self._model, ctx.test_pix,
                                     ctx.test_nbr.astype(np.float32),
                                     self._train_pix, self._train_nbr, ctx.signature)
    def state(self):
        s = super().state(); s["tr"] = (self._train_pix, self._train_nbr); return s
    def load_state(self, s):
        super().load_state(s); self._train_pix, self._train_nbr = s["tr"]


@register("NeighborMLP")
class NeighborMLP(_Torch):
    needs_spatial = True

    def _build(self):
        c = _full_cfg(self.cfg)
        return NeighborMLPDenoiser(D=self._meta["D"], d_lat=c["nmlp_d_lat"],
                                   K=c["nmlp_K"], hidden=c["nmlp_hidden"],
                                   n_layers=c["nmlp_n_layers"], sigma=self._meta["sigma"],
                                   activation=c["activation"])

    def fit(self, ctx):
        c = _full_cfg(self.cfg)
        self._meta = {"D": ctx.train_pix.shape[1], "sigma": ctx.sigma}
        self._model = _train_nmlp(self._meta["D"], ctx.sigma, ctx.train_pix,
                                  ctx.train_nbr, c, ctx.device)
        self._train_pix, self._train_nbr = ctx.train_pix, ctx.train_nbr.astype(np.float32)
        return self

    def score(self, ctx):
        return score_nmlp_additive(self._model, ctx.test_pix,
                                   ctx.test_nbr.astype(np.float32),
                                   self._train_pix, self._train_nbr, ctx.signature)
    def state(self):
        s = super().state(); s["tr"] = (self._train_pix, self._train_nbr); return s
    def load_state(self, s):
        super().load_state(s); self._train_pix, self._train_nbr = s["tr"]


@register("DSM")
class DSM(_Torch):
    """Per-pixel DSM-LMP with per-band standardisation (matches the run_colab fix)."""
    needs_spatial = False

    def _build(self):
        c = _full_cfg(self.cfg)
        return ScoreNet(self._meta["D"], list(c["dsm_hidden"]), c["activation"])

    def fit(self, ctx):
        c = _full_cfg(self.cfg)
        D = ctx.train_pix.shape[1]
        mu = ctx.train_pix.mean(0).astype(np.float32)
        sd = (ctx.train_pix.std(0) + 1e-8).astype(np.float32)
        tr_z = ((ctx.train_pix - mu) / sd).astype(np.float32)
        sigma_dsm = compute_sigma_from_data(tr_z, c["dsm_sigma_rho"])
        self._meta = {"D": D, "sigma": sigma_dsm}
        self._mu, self._sd, self._tr_z = mu, sd, tr_z
        self._model = _train_dsm(D, sigma_dsm, tr_z, c, ctx.device)
        return self

    def score(self, ctx):
        z = ((ctx.test_pix - self._mu) / self._sd).astype(np.float32)
        s = (ctx.signature / self._sd).astype(np.float32)
        return dsm_additive(z, self._tr_z, self._model, s)

    def state(self):
        s = super().state()
        s["std"] = (self._mu, self._sd, self._tr_z); return s
    def load_state(self, s):
        super().load_state(s); self._mu, self._sd, self._tr_z = s["std"]
