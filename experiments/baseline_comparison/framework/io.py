"""
io.py — per-(provider/scenario/detector) artifact persistence.

Layout:
  results_dir/<provider>/<scenario>/<detector>/
      model.pkl|model.pt     fitted detector state (via Detector.save)
      metrics.json           nested {model: {signature: {amp: {...}}}}
      scores.npz             cell scores/labels/thresholds + coords/gt/box
      maps.npz               detection maps + target/gt maps per cell
      train_log.json         {epoch: loss} (empty for closed-form)
      config.json            the run + detector config

Everything needed to remake figures WITHOUT retraining lives here.
"""

from __future__ import annotations

import json
import os

import numpy as np


def run_dir(base: str, provider: str, scenario: str, detector: str) -> str:
    d = os.path.join(base, provider, scenario, detector)
    os.makedirs(d, exist_ok=True)
    return d


def cell_key(model: str, sig: str, amp: float) -> str:
    return f"{model}|{sig}|{amp:g}"


def save_run(d: str, detector, metrics: dict, scores: dict, maps: dict,
             config: dict, train_log: dict) -> None:
    detector.save(os.path.join(d, "model.pkl"))
    json.dump(metrics, open(os.path.join(d, "metrics.json"), "w"), indent=2,
              default=_jsonify)
    json.dump(config, open(os.path.join(d, "config.json"), "w"), indent=2,
              default=_jsonify)
    json.dump(train_log, open(os.path.join(d, "train_log.json"), "w"), indent=2,
              default=_jsonify)
    if scores:
        np.savez_compressed(os.path.join(d, "scores.npz"), **scores)
    if maps:
        np.savez_compressed(os.path.join(d, "maps.npz"), **maps)


def load_metrics(d: str) -> dict:
    return json.load(open(os.path.join(d, "metrics.json")))


def load_scores(d: str) -> dict:
    return dict(np.load(os.path.join(d, "scores.npz"), allow_pickle=True))


def load_maps(d: str) -> dict:
    p = os.path.join(d, "maps.npz")
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else {}


def _jsonify(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
