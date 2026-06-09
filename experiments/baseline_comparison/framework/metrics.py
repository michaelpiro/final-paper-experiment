"""
metrics.py — statistics for one (detector, scenario, model, signature, amplitude)
cell, built on the existing evaluation utilities.
"""

from __future__ import annotations

import numpy as np

from final_paper_experiments.evaluation import (
    auc_safe, partial_auc, dr_at_fpr, roc_safe,
    cfar_threshold, per_class_fpr, scores_to_spatial_map,
)

DR_FPRS = (0.001, 0.01, 0.05, 0.10)


def cell_metrics(labels, scores, train_scores=None, gt_cls=None,
                 target_fpr=0.01, pauc_max=0.05) -> dict:
    """All scalar statistics for one planting cell.

    train_scores : clean training-pixel scores -> CFAR threshold (no test leakage).
    gt_cls       : per-test-pixel class/component id -> per-class FPR.
    """
    out = {
        "auc": float(auc_safe(labels, scores)),
        "pauc": float(partial_auc(labels, scores, fpr_max=pauc_max)),
        "dr": {str(f): v for f, v in dr_at_fpr(labels, scores, DR_FPRS).items()},
    }
    fpr, tpr, _ = roc_safe(labels, scores)
    out["roc"] = {"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr))}
    if train_scores is not None:
        thr = float(cfar_threshold(train_scores, target_fpr=target_fpr))
        out["threshold"] = thr
        if gt_cls is not None:
            out["per_class_fpr"] = per_class_fpr(scores, labels, gt_cls, thr)
    return out


def detection_map(scores, box_shape):
    """Scores -> (H,W) spatial map (NaN where no pixel)."""
    if box_shape is None:
        return None
    n = int(box_shape[0]) * int(box_shape[1])
    return scores_to_spatial_map(scores, np.arange(n), tuple(box_shape))
