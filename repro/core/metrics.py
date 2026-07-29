"""
metrics.py — ROC / detection metrics (numpy only, no torch).

partial_auc     — partial AUC over [0, fpr_max], normalised to [0, 1]
dr_at_fpr       — detection rate (TPR) at given FPR levels
auc_safe        — full AUC, nan on failure
roc_safe        — (fpr, tpr, auc)
cfar_threshold  — threshold from TRAINING background scores only
per_class_fpr   — per-class false-alarm rate on the test region
"""

import numpy as np
from sklearn.metrics import roc_curve, auc as sklearn_auc

# Pavia-University class id -> name (used for per-class false-alarm reporting).
CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks',    9: 'shadows',
}


def partial_auc(labels: np.ndarray, scores: np.ndarray, fpr_max: float = 0.05) -> float:
    """Partial AUC over [0, fpr_max], normalised to [0, 1] (perfect = 1)."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except Exception:
        return float('nan')
    tpr_at_max = float(np.interp(fpr_max, fpr, tpr))
    mask = fpr <= fpr_max
    fpr_cut = np.append(fpr[mask], fpr_max)
    tpr_cut = np.append(tpr[mask], tpr_at_max)
    return float(np.trapz(tpr_cut, fpr_cut)) / fpr_max


def dr_at_fpr(labels: np.ndarray, scores: np.ndarray,
              fpr_list=(0.001, 0.01, 0.05, 0.10)) -> dict:
    """Detection rate (TPR) at specific FPR values -> {fpr_str: dr}."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except Exception:
        return {str(f): float('nan') for f in fpr_list}
    return {str(f): float(np.interp(f, fpr, tpr)) for f in fpr_list}


def auc_safe(labels: np.ndarray, scores: np.ndarray) -> float:
    """Full AUC, nan on failure."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return float(sklearn_auc(fpr, tpr))
    except Exception:
        return float('nan')


def roc_safe(labels: np.ndarray, scores: np.ndarray):
    """(fpr_list, tpr_list, auc_float). Safe."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr.tolist(), tpr.tolist(), auc_safe(labels, scores)
    except Exception:
        return [0., 1.], [0., 1.], float('nan')


def cfar_threshold(bkg_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """CFAR threshold from TRAINING background scores at target_fpr.

    bkg_scores MUST be training-pixel scores only; using test pixels here would
    violate the CFAR guarantee. scores > threshold => declared target.
    """
    return float(np.quantile(bkg_scores, 1.0 - target_fpr))


def per_class_fpr(scores: np.ndarray, labels: np.ndarray,
                  cls_labels: np.ndarray, threshold: float) -> dict:
    """Per-class FPR on the test region (only background pixels contribute)."""
    result = {}
    bkg_mask = (labels == 0)
    for cid in np.unique(cls_labels[bkg_mask]):
        mask = bkg_mask & (cls_labels == cid)
        if mask.sum() == 0:
            continue
        result[CLS_NAMES.get(int(cid), f'cls{cid}')] = float((scores[mask] > threshold).mean())
    return result
