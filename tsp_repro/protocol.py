"""tsp_repro.protocol — scenes, amplitude grid, signatures, planting.

Every experiment in the TSP paper is a (scene, signature, theta, seed) cell.
This module owns the definitions; runner.py drives them.

Scenes
------
pavia4      Pavia University, the paper's Table-1 scenario 4: train box
            [85,193,207,306] side-cropped to ~4000 px, test box
            [419,508,250,334]. Signatures: bitumen (foreign class 7, the
            paper original), metal (class 5), and two mixtures.
sandiego    San Diego I  (Sandiego.mat,  boxes from sandiego_regions.json,
            signature = mean spectrum of the GT aircraft pixels).
sandiego2   San Diego II (Sandiego2.mat, boxes from sandiego2_regions.json).

NOTE (verify in Phase B): the San Diego signature normalization and the exact
mixture class pairs must be cross-checked against the archived generality run
(SDSM/results/generality_20260707) before the canonical numbers are quoted.
"""

import json
import os

import numpy as np
import scipy.io as sio

import tsp_repro  # noqa: F401  (path shim)
from colab_deep import paper_protocol as PP

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# --------------------------------------------------------------------------
# Canonical TSP settings
# --------------------------------------------------------------------------
# Amplitude grid: dense where detectors separate, sparse where they saturate.
THETA_GRID_DENSE = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
THETA_GRID_SPARSE = [0.40, 0.55, 0.70, 0.85, 0.95]
THETA_GRID = THETA_GRID_DENSE + THETA_GRID_SPARSE
THETA_REPLACEMENT = 0.95          # strong-target control cell
TABLE_THETAS = [0.075, 0.15, 0.225]  # headline table columns (MLSP-compatible)

EPOCHS = 2000                     # minimum training budget (TSP canonical)
SEEDS = [42, 43, 44, 45, 46]
TGT_FRACTION = 0.10
EDGE_GUARD = 3

PAVIA_TRAIN_BOX = [85, 193, 207, 306]
PAVIA_TEST_BOX = [419, 508, 250, 334]
PAVIA_N_BUDGET = 4000

# Pavia class ids: 1 asphalt, 2 meadows, 3 gravel, 4 trees, 5 metal sheets,
# 6 bare soil, 7 bitumen, 8 bricks, 9 shadows.
PAVIA_SIGNATURES = {
    "bitumen": (7,),              # paper original (foreign class)
    "metal": (5,),
    "mix_soil_bricks": (6, 8),    # off-library mixture 1
    "mix_metal_soil": (5, 6),     # off-library mixture 2
}


def sigma_mse_optimal(train_whitened: np.ndarray) -> float:
    """The paper's closed-form DSM noise level (Eq. sigma_rule):
    sigma = (D/n)^(1/6) * mean per-band std of the (whitened) training data."""
    n, D = train_whitened.shape
    scale = float(np.std(train_whitened, axis=0).mean())
    return float((D / n) ** (1.0 / 6.0) * scale)


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------
def _pavia_signature(data, gt, te_pix, classes):
    """Class-mean (or mean of class means) scaled to the mean test-pixel norm,
    exactly like the paper's foreign-signature construction."""
    D = data.shape[-1]
    flat, g = data.reshape(-1, D), gt.ravel()
    mus = [flat[g == c].mean(axis=0) for c in classes]
    mu = np.mean(mus, axis=0)
    scalar = float(np.linalg.norm(te_pix, axis=1).mean())
    return mu / (np.linalg.norm(mu) + 1e-12) * scalar


def _sandiego_signature(data, gtmap, te_pix, scale_to_test_norm=True):
    """Mean spectrum of the real aircraft (GT==1) pixels.
    TODO(verify): compare against the archived generality-run signature
    (median-normalized variant) before quoting canonical numbers."""
    D = data.shape[-1]
    mu = data.reshape(-1, D)[gtmap.ravel() == 1].mean(axis=0)
    if scale_to_test_norm:
        scalar = float(np.linalg.norm(te_pix, axis=1).mean())
        mu = mu / (np.linalg.norm(mu) + 1e-12) * scalar
    return mu


# --------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------
def _load_sandiego(path):
    m = sio.loadmat(path)
    return m["data"].astype(np.float64), (m["map"] > 0).astype(int)


def build_scene(name: str) -> dict:
    """Returns dict(tr, te, te_shape, sigs {label: vec}, data, gt, boxes)."""
    if name == "pavia4":
        data, gt = PP.get_pavia()
        tb = PP.side_crop(PAVIA_TRAIN_BOX, PAVIA_N_BUDGET)
        tr = PP.crop(data, tb)
        te = PP.crop(data, PAVIA_TEST_BOX)
        te_shape = (PAVIA_TEST_BOX[1] - PAVIA_TEST_BOX[0],
                    PAVIA_TEST_BOX[3] - PAVIA_TEST_BOX[2])
        sigs = {lbl: _pavia_signature(data, gt, te, cls)
                for lbl, cls in PAVIA_SIGNATURES.items()}
        boxes = dict(train=tb, test=list(PAVIA_TEST_BOX))
    elif name in ("sandiego", "sandiego2"):
        stem = "Sandiego.mat" if name == "sandiego" else "Sandiego2.mat"
        rj = ("sandiego_regions.json" if name == "sandiego"
              else "sandiego2_regions.json")
        data, gt = _load_sandiego(os.path.join(_HERE, "data", stem))
        regions = json.load(open(os.path.join(_HERE, "data", rj)))
        tb, xb = regions["train_box"], regions["test_box"]
        # region boxes are inclusive [r0, r1, c0, c1] -> half-open crops
        tb_h = [tb[0], tb[1] + 1, tb[2], tb[3] + 1]
        xb_h = [xb[0], xb[1] + 1, xb[2], xb[3] + 1]
        tr = PP.crop(data, tb_h)
        te = PP.crop(data, xb_h)
        te_shape = (xb_h[1] - xb_h[0], xb_h[3] - xb_h[2])
        sigs = {"aircraft": _sandiego_signature(data, gt, te)}
        boxes = dict(train=tb_h, test=xb_h)
    else:
        raise KeyError(f"unknown scene '{name}'")
    print(f"[{name}] train {len(tr)}px  test {len(te)}px ({te_shape[0]}x{te_shape[1]})"
          f"  signatures: {list(sigs)}", flush=True)
    return dict(name=name, tr=tr, te=te, te_shape=te_shape, sigs=sigs,
                data=data, gt=gt, boxes=boxes)


def plant(scene: dict, sig: np.ndarray, theta: float, model: str = "additive",
          seed: int = 0):
    """Plant targets into the clean test pixels; returns (planted, labels)."""
    return PP.plant_targets(scene["te"], sig, theta, TGT_FRACTION, model=model,
                            seed=seed, spatial_shape=scene["te_shape"],
                            edge_guard=EDGE_GUARD)


def default_cells(thetas=None):
    """The (model, theta) cells scored for every trained detector instance."""
    thetas = THETA_GRID if thetas is None else thetas
    return [("additive", th) for th in thetas] + [("replacement", THETA_REPLACEMENT)]
