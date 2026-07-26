"""sd_protocol.py — ARCHIVE-EXACT San Diego protocol for the camera-ready
deep-baseline runs (matches the MLSP generality archive to float precision).

Pinned against SDSM/results/generality_20260707/raw (2026-07-26):
  - crops: region-file boxes are HALF-OPEN as-is: data[r0:r1, c0:c1]
    (sandiego train 60x67=4020 / test 60x31=1860; sandiego2 4455 / 2226).
  - signature: mean spectrum of the GT aircraft pixels (full scene),
    rescaled to the MEDIAN test-pixel norm (verified: archived clean-pixel
    AMF scores match to 2e-5, GMM-Levin to 7e-5).
  - planting: additive y=w+theta*s (or replacement), edge_guard=3, and the
    target COUNT is round(0.10 * |eligible interior pool|) — NOT 10% of all
    test pixels (archive: 135/1860 and 169/2226).
Seeds 42-46, theta in {0.075, 0.15, 0.225} + replacement 0.95.
"""

import json
import os

import numpy as np
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

SEEDS = [42, 43, 44, 45, 46]
THETAS_ADD = [0.075, 0.15, 0.225]
THETA_REPL = 0.95
EDGE_GUARD = 3
TGT_FRACTION = 0.10

_EXPECT = {  # archive fingerprints; build() asserts against these
    "sandiego":  dict(tr=(4020, 189), te=(1860, 189), sig_norm=39598.9),
    "sandiego2": dict(tr=(4455, 189), te=(2226, 189), sig_norm=27929.4),
}


def _find(fname):
    for d in (os.path.join(_REPO, "tsp_repro", "data"),
              os.path.join(_HERE, "data"),
              os.path.join(_REPO, "SDSM", "data")):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fname)


def build(scene: str) -> dict:
    """scene in {'sandiego','sandiego2'} -> dict(tr, te, te_shape, sig, data, gt)."""
    stem = "Sandiego.mat" if scene == "sandiego" else "Sandiego2.mat"
    rj = json.load(open(_find(f"{scene}_regions.json")))
    m = sio.loadmat(_find(stem))
    data = m["data"].astype(np.float64)
    gt = (m["map"] > 0).astype(int)
    tb, xb = rj["train_box"], rj["test_box"]
    crop = lambda b: data[b[0]:b[1], b[2]:b[3]].reshape(-1, data.shape[-1])
    tr, te = crop(tb), crop(xb)
    te_shape = (xb[1] - xb[0], xb[3] - xb[2])
    mu = data.reshape(-1, data.shape[-1])[gt.ravel() == 1].mean(axis=0)
    sig = mu / np.linalg.norm(mu) * np.median(np.linalg.norm(te, axis=1))
    exp = _EXPECT[scene]
    assert tr.shape == exp["tr"] and te.shape == exp["te"], \
        f"{scene}: crop mismatch {tr.shape}/{te.shape} vs archive {exp}"
    assert abs(np.linalg.norm(sig) - exp["sig_norm"]) < 1.0, \
        f"{scene}: ||s||={np.linalg.norm(sig):.1f} != archive {exp['sig_norm']}"
    print(f"[{scene}] train {tr.shape} test {te.shape} ||s||="
          f"{np.linalg.norm(sig):.1f}  (archive-exact)", flush=True)
    return dict(name=scene, tr=tr, te=te, te_shape=te_shape, sig=sig,
                data=data, gt=gt)


def plant_pool(te, sig, theta, model="additive", seed=0, spatial_shape=None,
               edge_guard=EDGE_GUARD, frac=TGT_FRACTION):
    """Archive-semantics planting: n_targets = round(frac * |interior pool|)."""
    rng = np.random.RandomState(seed)
    N = len(te)
    H, W = spatial_shape
    rows, cols = np.unravel_index(np.arange(N), (H, W))
    ok = ((rows >= edge_guard) & (rows < H - edge_guard) &
          (cols >= edge_guard) & (cols < W - edge_guard))
    pool = np.where(ok)[0]
    n_t = int(round(frac * len(pool)))
    idx = rng.choice(pool, size=n_t, replace=False)
    planted = te.copy()
    if model == "additive":
        planted[idx] = planted[idx] + theta * sig[None, :]
    else:
        planted[idx] = (1 - theta) * planted[idx] + theta * sig[None, :]
    labels = np.zeros(N, dtype=np.int8)
    labels[idx] = 1
    return planted, labels


def cells():
    return [("additive", th) for th in THETAS_ADD] + [("replacement", THETA_REPL)]
