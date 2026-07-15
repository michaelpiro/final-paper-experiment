"""tsp_repro.runner — scene x detector x seed x theta driver.

Design goals: resumable (per-(detector, scene, seed) checkpoints; per-cell
score archives are skipped if they already exist), so a Colab session that
disconnects loses at most one cell; and archival (raw scores + labels for
every cell, plus each detector's scores on the clean TRAIN background for
empirical CFAR thresholds), so any table/figure/metric is recomputable
without models.

Outputs under out_dir (double-underscore separated; parsed by tables.py):
    scores__{scene}__{sig}__{det}__seed{k}__{model}__{theta}.npz  (scores, labels)
    train__{scene}__{sig}__{det}__seed{k}.npz                     (scores)
    {scene}__{sig}__results.json                                  (AUC per cell)
"""

import json
import os
import zipfile

import numpy as np
import torch

from tsp_repro import protocol as PR
from tsp_repro.registry import CKPT_ALIAS, REGISTRY


def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def _cell_path(out_dir, scene, sig, det, seed, model, theta):
    return os.path.join(
        out_dir, f"scores__{scene}__{sig}__{det}__seed{seed}__{model}__{theta}.npz")


def run_scene(scene_name, detectors, sig_label=None, seeds=None, thetas=None,
              out_dir="results_tsp", ckpt_dir="ckpt_tsp", device=None,
              score_train=True):
    """Train (or resume) every trainable detector per seed, then score every
    (model, theta) cell of the protocol. Returns {det: {cell: [auc per seed]}}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or PR.SEEDS
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    scene = PR.build_scene(scene_name)
    sig_labels = [sig_label] if sig_label else list(scene["sigs"])
    all_results = {}

    for sl in sig_labels:
        results = {}
        all_results[sl] = results
        scene["_sig"] = scene["sigs"][sl]
        for seed in seeds:
            # ---- fit (with base-checkpoint sharing for CFAR variants) ----
            states = {}
            for det in detectors:
                spec = REGISTRY[det]
                if spec.artifact_only:
                    continue
                base = CKPT_ALIAS.get(det, det)
                if base in states:
                    states[det] = states[base]
                    continue
                if spec.fit is None:
                    states[det] = None
                    continue
                ck = os.path.join(ckpt_dir, f"{base}_{scene_name}_{sl}_seed{seed}.pt")
                states[det] = spec.fit(scene, seed, ck, device)
                states[base] = states[det]

            # ---- train-background scores (empirical CFAR thresholds) ----
            if score_train:
                for det in detectors:
                    spec = REGISTRY[det]
                    if spec.artifact_only or det in CKPT_ALIAS:
                        continue
                    p = os.path.join(
                        out_dir, f"train__{scene_name}__{sl}__{det}__seed{seed}.npz")
                    if os.path.exists(p):
                        continue
                    try:
                        tr_scores = spec.score(states[det], scene,
                                               np.asarray(scene["tr"], np.float32),
                                               device)
                        np.savez_compressed(p, scores=np.asarray(tr_scores, np.float32))
                    except Exception as e:  # spatial detectors need image-shaped input
                        print(f"    [train-scores] {det}: skipped ({e})", flush=True)

            # ---- planted cells ----
            for model, th in PR.default_cells(thetas):
                planted, labels = PR.plant(scene, scene["_sig"], th,
                                           model=model, seed=seed)
                for det in detectors:
                    spec = REGISTRY[det]
                    if spec.artifact_only:
                        continue
                    p = _cell_path(out_dir, scene_name, sl, det, seed, model, th)
                    key = f"{model}|{th}"
                    if os.path.exists(p):
                        d = np.load(p)
                        a = _auc(d["labels"], d["scores"])
                    else:
                        sc = np.asarray(spec.score(states[det], scene, planted,
                                                   device), np.float32)
                        a = _auc(labels, sc)
                        np.savez_compressed(p, scores=sc, labels=labels)
                    results.setdefault(det, {}).setdefault(key, []).append(a)
                    print(f"[{scene_name}/{sl}] {det} seed{seed} {model} "
                          f"th={th}: AUC={a:.4f}", flush=True)

        with open(os.path.join(out_dir, f"{scene_name}__{sl}__results.json"), "w") as f:
            json.dump(results, f, indent=1)

    print(f"\n=== {scene_name} summary (mean+/-std over {len(seeds)} seeds) ===")
    for sl, results in all_results.items():
        for det, cells in results.items():
            for k, v in cells.items():
                print(f"  [{sl}] {det:12s} {k}: {np.mean(v):.3f}+/-{np.std(v):.3f}")
    return all_results


def zip_results(dirs=("results_tsp", "ckpt_tsp"), zip_name="tsp_results.zip"):
    """Package results + checkpoints; auto-download on Colab."""
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    z.write(os.path.join(root, fn))
    print("zipped ->", zip_name, flush=True)
    try:
        from google.colab import files
        files.download(zip_name)
    except Exception:
        print("(not on Colab - zip left on disk)")
