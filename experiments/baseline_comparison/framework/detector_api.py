"""
detector_api.py — the contract every baseline implements.

A `Detector` is given a `DetectorInput` bundle and knows NOTHING about whether
its data came from a real spatial scene, a synthetic GMM, or an i.i.d. sample.
It simply picks the fields it was designed to consume:

  - PCA / feature space  : train_pix, test_pix, signature            (always present)
  - raw band space       : train_raw, test_raw, signature_raw        (always present;
                           equals the feature arrays for synthetic data)
  - spatial structure    : train_nbr, test_nbr (+ _raw), test_coords, box_shape
                           (None when the dataset is non-spatial / i.i.d.)

Lifecycle:
  fit(ctx)   — train on the BACKGROUND only (ctx.test_pix is clean here).
  score(ctx) — return (n_test,) scores, higher = more target-like. ctx.test_pix
               may contain planted targets (the runner plants them).
  save/load  — persist/restore the fitted model so figures can be remade later
               without retraining.
  train_log  — optional {epoch: loss} dict for closed-form detectors return {}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace, field
from typing import Optional
import pickle

import numpy as np


@dataclass
class DetectorInput:
    """Everything a detector might need. Detectors pick what they use."""

    # --- feature (PCA) space ---
    train_pix: np.ndarray                      # (n_tr, D)
    test_pix:  np.ndarray                      # (n_te, D)  clean for fit, planted for score
    signature: np.ndarray                      # (D,) target signature (feature space)

    # --- raw band space (same rows; == feature arrays for synthetic data) ---
    train_raw: np.ndarray                      # (n_tr, D_raw)
    test_raw:  np.ndarray                      # (n_te, D_raw)
    signature_raw: np.ndarray                  # (D_raw,)

    # --- shared scalars ---
    sigma: float                               # DSM noise scale (feature space)
    device: str = "cpu"
    seed: int = 0

    # --- spatial structure (None for non-spatial datasets) ---
    train_nbr: Optional[np.ndarray] = None     # (n_tr, k2-1, D)
    test_nbr:  Optional[np.ndarray] = None     # (n_te, k2-1, D)
    train_nbr_raw: Optional[np.ndarray] = None # (n_tr, k2-1, D_raw)
    test_nbr_raw:  Optional[np.ndarray] = None # (n_te, k2-1, D_raw)
    test_coords: Optional[np.ndarray] = None   # (n_te, 2) row,col within test box
    box_shape: Optional[tuple] = None          # (H, W) of the test region for maps

    meta: dict = field(default_factory=dict)   # {'dataset','D','D_raw','spatial',...}

    # ------------------------------------------------------------------ helpers
    @property
    def spatial(self) -> bool:
        return bool(self.meta.get("spatial", self.test_nbr is not None))

    def test_image(self, raw: bool = True):
        """Reshape the (planted) test pixels back into an (H, W, C) image using
        box_shape. Used by transductive image-based detectors. None if no box."""
        if self.box_shape is None:
            return None
        H, W = self.box_shape
        arr = self.test_raw if raw else self.test_pix
        return arr.reshape(int(H), int(W), arr.shape[-1])

    def with_test(self, test_pix=None, test_raw=None) -> "DetectorInput":
        """Return a shallow copy with the test pixels swapped (planting)."""
        kw = {}
        if test_pix is not None:
            kw["test_pix"] = test_pix
        if test_raw is not None:
            kw["test_raw"] = test_raw
        return replace(self, **kw)


class Detector(ABC):
    """Base class for every baseline / method."""

    #: display name used in results + figures
    name: str = "detector"
    #: if True, the detector is skipped on non-spatial datasets
    needs_spatial: bool = False
    #: if True, the detector is TRANSDUCTIVE (trains on the test image each call);
    #: the runner skips fit + the train-pixel CFAR threshold for it, and calls
    #: score() once per planting cell. Implies needs a 2D image (box_shape).
    transductive: bool = False
    #: which input space the detector consumes ('pca' or 'raw') — informational
    space: str = "pca"

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = dict(cfg or {})
        self._log: dict = {}

    # -- lifecycle ----------------------------------------------------------
    @abstractmethod
    def fit(self, ctx: DetectorInput) -> "Detector":
        """Train on background pixels only. Return self."""

    @abstractmethod
    def score(self, ctx: DetectorInput) -> np.ndarray:
        """Return (n_test,) scores; higher = more target-like."""

    # -- persistence (override for nn.Module-based detectors) ---------------
    def state(self) -> dict:
        """Picklable state to persist. Override for torch models."""
        return {"cfg": self.cfg, "log": self._log}

    def load_state(self, state: dict) -> None:
        self.cfg = state.get("cfg", self.cfg)
        self._log = state.get("log", {})

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self.state(), fh)

    def load(self, path: str) -> "Detector":
        with open(path, "rb") as fh:
            self.load_state(pickle.load(fh))
        return self

    def train_log(self) -> dict:
        return self._log
