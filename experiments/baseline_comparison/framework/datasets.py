"""
datasets.py — providers that yield background "scenarios".

A provider knows how to build clean BACKGROUND data (train + test) plus the
spatial structure and a set of candidate target signatures. It does NOT plant
targets — the runner does that, sweeping (model x signature x amplitude).

Three providers:
  - PaviaScenarioProvider  (HARD: real scene, manual boxes, whole-image PCA)
  - IIDGMMProvider         (EASY: low-dim i.i.d. K-component GMM, non-spatial)
  - SpatialGMMProvider     (EASY: low-dim GMM with a smooth spatial label field)

Each yields `Scenario` objects. The runner turns a Scenario + a planting choice
into a `DetectorInput`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional
import json
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data,
)
from final_paper_experiments.evaluation import compute_signature
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


@dataclass
class Scenario:
    name: str
    spatial: bool
    # background pixels (CLEAN) — feature space
    train_pix: np.ndarray
    test_pix:  np.ndarray
    # raw band space (== feature space for synthetic providers)
    train_raw: np.ndarray
    test_raw:  np.ndarray
    sigma: float
    # candidate signatures: {sig_name: (s_feature (D,), s_raw (D_raw,))}
    signatures: dict
    # spatial structure (None when non-spatial)
    train_nbr: Optional[np.ndarray] = None
    test_nbr:  Optional[np.ndarray] = None
    train_nbr_raw: Optional[np.ndarray] = None
    test_nbr_raw:  Optional[np.ndarray] = None
    test_coords: Optional[np.ndarray] = None
    box_shape: Optional[tuple] = None
    test_gt_cls: Optional[np.ndarray] = None    # per-test-pixel class/component id
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _renorm_to(v: np.ndarray, target_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n * target_norm).astype(np.float32) if n > 1e-12 else v.astype(np.float32)


def _nbr(img: np.ndarray, k: int):
    """extract_neighborhoods wrapper: (H,W,D) np -> (centers, neighbors) np."""
    c, n = extract_neighborhoods(torch.tensor(img, dtype=torch.float32), k)
    return c.numpy(), n.numpy()


# ---------------------------------------------------------------------------
# HARD: Pavia real scenarios
# ---------------------------------------------------------------------------

class PaviaScenarioProvider:
    spatial = True

    def scenarios(self, cfg) -> Iterator[Scenario]:
        # NO PCA: every detector gets the RAW full-band data. Our score nets
        # whiten internally (frozen ZCA first layer); classical/deep baselines
        # consume raw bands directly. pix == raw, signature == signature_raw.
        seed = int(cfg.get("seed", 42))
        k = int(cfg.get("k", 5))
        rho = float(cfg.get("dsm_sigma_rho", 0.01))
        n_budget = int(cfg.get("n_budget", 2000))

        ds = cfg["dataset"]
        if not os.path.isabs(ds):
            ds = os.path.join(_ROOT, ds)
        # 'none' == original .mat sensor values (no scaling/normalization)
        data_norm, gt = load_and_normalize(ds, cfg.get("norm_mode", "none"))
        H, W, D_raw = data_norm.shape
        flat = data_norm.reshape(-1, D_raw)
        sigma = compute_sigma_from_data(flat, rho)

        boxes_path = cfg.get("manual_boxes_path",
                             "experiments/spatial/manual_boxes.json")
        if not os.path.isabs(boxes_path):
            boxes_path = os.path.join(_ROOT, boxes_path)
        boxes = json.load(open(boxes_path))
        which = cfg.get("scenarios", list(range(len(boxes))))

        for idx in which:
            sc = boxes[idx]
            tb, te = sc["train_box"], sc["test_box"]
            rng = np.random.default_rng(seed + idx * 100)

            tr_cr, tr_nr = _nbr(data_norm[tb[0]:tb[1], tb[2]:tb[3], :], k)
            if len(tr_cr) > n_budget:
                sub = rng.choice(len(tr_cr), n_budget, replace=False)
                tr_cr, tr_nr = tr_cr[sub], tr_nr[sub]

            te_cr, te_nr = _nbr(data_norm[te[0]:te[1], te[2]:te[3], :], k)
            te_gt = gt[te[0]:te[1], te[2]:te[3]].ravel()
            Hb, Wb = te[1] - te[0], te[3] - te[2]
            coords = np.stack(np.meshgrid(np.arange(Hb), np.arange(Wb), indexing="ij"),
                              -1).reshape(-1, 2)

            sig_raw, dom_cls, dom_name = compute_signature(
                gt[te[0]:te[1], te[2]:te[3]], data_norm[te[0]:te[1], te[2]:te[3]],
                float(cfg.get("sig_dom_weight", 0.8)),
                float(cfg.get("sig_mean_weight", 0.2)))
            sig_raw = sig_raw.astype(np.float32)
            # raw == feature: pix and signature live in the same raw band space
            signatures = {f"paper-{dom_name}": (sig_raw, sig_raw)}

            tr_cr = tr_cr.astype(np.float32); tr_nr = tr_nr.astype(np.float32)
            te_cr = te_cr.astype(np.float32); te_nr = te_nr.astype(np.float32)
            yield Scenario(
                name=f"pavia_s{idx}", spatial=True,
                train_pix=tr_cr, test_pix=te_cr,
                train_raw=tr_cr, test_raw=te_cr,
                sigma=sigma, signatures=signatures,
                train_nbr=tr_nr, test_nbr=te_nr,
                train_nbr_raw=tr_nr, test_nbr_raw=te_nr,
                test_coords=coords, box_shape=(Hb, Wb), test_gt_cls=te_gt,
                meta={"dataset": "pavia", "D": D_raw, "D_raw": D_raw,
                      "spatial": True, "dom_cls": int(dom_cls)},
            )


# ---------------------------------------------------------------------------
# synthetic GMM core
# ---------------------------------------------------------------------------

def _make_gmm(D: int, K: int, sep: float, rng, cov_scale: float = 1.0) -> tuple:
    """Return (means (K,D), covs (K,D,D), weights (K,)).

    `cov_scale` shrinks the WITHIN-component covariance. Tight components
    (small cov_scale) that are well separated (large sep) make the POOLED
    covariance dominated by between-mode scatter — exactly the regime where a
    single-Gaussian whitener (AMF) breaks while per-component detectors
    (Self-GMM, GMM-Levin) stay sharp.
    """
    means = rng.normal(0, sep, size=(K, D)).astype(np.float64)
    covs = np.empty((K, D, D))
    for k in range(K):
        A = rng.normal(0, 1, size=(D, D)) / np.sqrt(D)
        covs[k] = cov_scale * (A @ A.T + 0.1 * np.eye(D))     # SPD, tunable scale
    weights = rng.dirichlet(np.ones(K) * 2.0)
    return means, covs, weights


def _sample_gmm(means, covs, weights, n, rng):
    K = len(weights)
    comp = rng.choice(K, size=n, p=weights)
    X = np.empty((n, means.shape[1]))
    for k in range(K):
        m = comp == k
        if m.any():
            X[m] = rng.multivariate_normal(means[k], covs[k], size=int(m.sum()))
    return X.astype(np.float32), comp


def _gmm_signatures(means, weights, rng, ref_norm) -> dict:
    """comp-mean (entangled), between-components (separable), orthogonal (out-of-dist)."""
    K, D = means.shape
    big = int(np.argmax(weights))
    s_comp = means[big].astype(np.float32)
    a, b = rng.choice(K, size=2, replace=False)
    s_between = (means[a] - means[b]).astype(np.float32)
    # orthogonal to the span of all component means
    M = means - means.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=True)
    rank = int((np.linalg.svd(M, compute_uv=False) > 1e-8).sum())
    s_orth = (Vt[rank] if rank < D else rng.normal(size=D)).astype(np.float32)
    return {
        "comp_mean": _renorm_to(s_comp, ref_norm),
        "between":   _renorm_to(s_between, ref_norm),
        "orthogonal": _renorm_to(s_orth, ref_norm),
    }


class IIDGMMProvider:
    spatial = False

    def scenarios(self, cfg) -> Iterator[Scenario]:
        seed = int(cfg.get("seed", 0))
        rng = np.random.default_rng(seed)
        D = int(cfg.get("dim", 8)); K = int(cfg.get("K", 4))
        sep = float(cfg.get("separation", 2.0)); cov_scale = float(cfg.get("cov_scale", 1.0))
        n_tr = int(cfg.get("n_train", 4000)); n_te = int(cfg.get("n_test", 4000))
        rho = float(cfg.get("dsm_sigma_rho", 0.01))

        means, covs, weights = _make_gmm(D, K, sep, rng, cov_scale)
        train, _ = _sample_gmm(means, covs, weights, n_tr, rng)
        test, te_comp = _sample_gmm(means, covs, weights, n_te, rng)
        # Scale signatures to the WITHIN-component clutter (not the giant
        # comp-mean norm) so the amplitude sweep spans the weak->strong regime
        # where single-Gaussian AMF lags the per-component GMM detectors.
        within = float(np.sqrt(np.mean([np.mean(np.diag(c)) for c in covs])))
        ref = float(cfg.get("sig_norm_scale", 4.0)) * within
        sigs_f = _gmm_signatures(means, weights, rng, ref)
        signatures = {nm: (s, s) for nm, s in sigs_f.items()}   # raw == feature
        sigma = compute_sigma_from_data(train, rho)

        yield Scenario(
            name="iid_gmm", spatial=False,
            train_pix=train, test_pix=test, train_raw=train, test_raw=test,
            sigma=sigma, signatures=signatures, test_gt_cls=te_comp,
            meta={"dataset": "iid_gmm", "D": D, "D_raw": D, "spatial": False,
                  "K": K, "means": means.tolist()},
        )


class SpatialGMMProvider:
    spatial = True

    def _image(self, means, covs, weights, H, W, smooth, k, rng):
        """Smooth component-label field -> per-pixel GMM sample -> (img, labels)."""
        K, D = means.shape
        # smooth random scores per component -> argmax => spatially-coherent labels
        scores = np.stack([gaussian_filter(rng.normal(size=(H, W)), smooth)
                           for _ in range(K)], -1)
        scores += np.log(weights + 1e-9)[None, None, :]
        labels = scores.argmax(-1)
        img = np.empty((H, W, D), np.float32)
        for kk in range(K):
            m = labels == kk
            if m.any():
                img[m] = rng.multivariate_normal(means[kk], covs[kk],
                                                  size=int(m.sum())).astype(np.float32)
        return img, labels

    def scenarios(self, cfg) -> Iterator[Scenario]:
        seed = int(cfg.get("seed", 0))
        rng = np.random.default_rng(seed)
        D = int(cfg.get("dim", 8)); K = int(cfg.get("K", 4))
        sep = float(cfg.get("separation", 2.0)); cov_scale = float(cfg.get("cov_scale", 1.0))
        Htr, Wtr = cfg.get("train_shape", [60, 60])
        Hte, Wte = cfg.get("test_shape", [60, 60])
        smooth = float(cfg.get("spatial_smooth", 3.0))
        k = int(cfg.get("k", 5)); rho = float(cfg.get("dsm_sigma_rho", 0.01))

        means, covs, weights = _make_gmm(D, K, sep, rng, cov_scale)
        tr_img, _ = self._image(means, covs, weights, Htr, Wtr, smooth, k, rng)
        te_img, te_lab = self._image(means, covs, weights, Hte, Wte, smooth, k, rng)

        tr_c, tr_n = _nbr(tr_img, k)
        te_c, te_n = _nbr(te_img, k)
        coords = np.stack(np.meshgrid(np.arange(Hte), np.arange(Wte), indexing="ij"),
                          -1).reshape(-1, 2)
        # Scale signatures to the WITHIN-component clutter (not the giant
        # comp-mean norm) so the amplitude sweep spans the weak->strong regime
        # where single-Gaussian AMF lags the per-component GMM detectors.
        within = float(np.sqrt(np.mean([np.mean(np.diag(c)) for c in covs])))
        ref = float(cfg.get("sig_norm_scale", 4.0)) * within
        sigs_f = _gmm_signatures(means, weights, rng, ref)
        signatures = {nm: (s, s) for nm, s in sigs_f.items()}
        sigma = compute_sigma_from_data(tr_c, rho)

        yield Scenario(
            name="spatial_gmm", spatial=True,
            train_pix=tr_c, test_pix=te_c, train_raw=tr_c, test_raw=te_c,
            sigma=sigma, signatures=signatures,
            train_nbr=tr_n, test_nbr=te_n, train_nbr_raw=tr_n, test_nbr_raw=te_n,
            test_coords=coords, box_shape=(Hte, Wte), test_gt_cls=te_lab.ravel(),
            meta={"dataset": "spatial_gmm", "D": D, "D_raw": D, "spatial": True, "K": K},
        )


PROVIDERS = {
    "pavia": PaviaScenarioProvider,
    "iid_gmm": IIDGMMProvider,
    "spatial_gmm": SpatialGMMProvider,
}


def get_provider(name: str):
    if name not in PROVIDERS:
        raise KeyError(f"Unknown provider '{name}'. Available: {sorted(PROVIDERS)}")
    return PROVIDERS[name]()
