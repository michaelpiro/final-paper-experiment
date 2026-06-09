"""
Two NEW GMM detectors that pick a Gaussian component, then run AMF with that
component's statistics.

  Self-GMM    : per test pixel, pick its OWN most-likely component (predict),
                run AMF with that component's (mu_k, Sigma_k).
  Spatial-GMM : per test pixel, majority-vote the component over its KxK spatial
                neighbourhood (excluding the centre), run AMF with the voted
                component's stats. (needs_spatial)

AMF with component stats:  T(y) = sᵀ Σ_k⁻¹ (y − μ_k) / sqrt(sᵀ Σ_k⁻¹ s).
"""

from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture

from ..framework.detector_api import Detector, DetectorInput
from ..framework.registry import register


def _amf_by_component(Y, s, comp, means, precis):
    """Vectorised AMF where pixel i uses component comp[i]'s stats."""
    out = np.empty(len(Y), np.float64)
    for k in range(len(means)):
        m = comp == k
        if not m.any():
            continue
        Pi = precis[k]                       # Σ_k⁻¹
        Pi_s = Pi @ s
        norm = np.sqrt(float(s @ Pi_s) + 1e-12)
        out[m] = (Y[m] - means[k]) @ Pi_s / norm
    return out


class _GMMBase(Detector):
    def fit(self, ctx: DetectorInput) -> "Detector":
        K = int(self.cfg.get("K", 5))
        Xtr = ctx.train_pix          # RAW bands (no PCA)
        # Scale-aware diagonal loading: raw hyperspectral bands have large, very
        # uneven variance, so an absolute reg_covar (1e-5) is negligible and a
        # full-cov K=9 GMM in ~103-dim goes singular. Default reg_covar to a
        # fraction of the mean band variance (overridable via cfg).
        reg = self.cfg.get("reg_covar", None)
        if reg is None:
            reg = float(self.cfg.get("reg_covar_rel", 1e-3)
                        * np.mean(np.var(Xtr, axis=0)))
        else:
            reg = float(reg)
        self._gm = GaussianMixture(n_components=K, covariance_type="full",
                                   n_init=int(self.cfg.get("n_init", 3)),
                                   reg_covar=reg,
                                   random_state=ctx.seed).fit(Xtr)
        self._means = self._gm.means_
        self._precis = np.linalg.inv(self._gm.covariances_)
        return self
    def state(self):
        return {"cfg": self.cfg, "log": self._log, "gm": self._gm,
                "means": self._means, "precis": self._precis}
    def load_state(self, s):
        super().load_state(s)
        self._gm, self._means, self._precis = s["gm"], s["means"], s["precis"]


@register("Self-GMM")
class SelfGMM(_GMMBase):
    def score(self, ctx):
        comp = self._gm.predict(ctx.test_pix)
        return _amf_by_component(ctx.test_pix, ctx.signature, comp,
                                 self._means, self._precis)


@register("Spatial-GMM")
class SpatialGMM(_GMMBase):
    needs_spatial = True

    def score(self, ctx):
        # component label of every neighbour, then per-pixel majority vote
        nbr = ctx.test_nbr                              # (n, k2-1, D)
        n, m, D = nbr.shape
        lab = self._gm.predict(nbr.reshape(n * m, D)).reshape(n, m)
        K = len(self._means)
        # majority vote (ties -> lowest index) via per-component counts
        counts = np.zeros((n, K), np.int32)
        for k in range(K):
            counts[:, k] = (lab == k).sum(1)
        comp = counts.argmax(1)
        return _amf_by_component(ctx.test_pix, ctx.signature, comp,
                                 self._means, self._precis)
