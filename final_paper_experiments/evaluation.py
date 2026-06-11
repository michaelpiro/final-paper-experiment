"""
evaluation.py — Shared evaluation helpers for the spatial experiment.

All functions accept numpy arrays and are backend-agnostic (no torch).

Functions
---------
partial_auc           — pAUC at [0, fpr_max], normalised to [0,1]
dr_at_fpr             — detection rate at several FPR levels
cfar_threshold        — threshold from TRAINING background scores only
per_class_fpr         — per-class FPR on the test region
compute_signature     — dominant-class offset signature (train-side only)
generate_random_boxes — 4 random (train, test) box pairs with seeded RNG
box_statistics        — class composition + dominant class of a box
scores_to_spatial_map — scatter test scores back into a 2D spatial map
"""

import numpy as np
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ---------------------------------------------------------------------------
# ROC-based metrics
# ---------------------------------------------------------------------------

def partial_auc(labels: np.ndarray, scores: np.ndarray,
                fpr_max: float = 0.05) -> float:
    """
    Partial AUC at [0, fpr_max], normalised to [0, 1].

    Normalisation: divide raw pAUC by fpr_max (area of a perfect detector
    in the [0, fpr_max] window is fpr_max × 1, so pAUC / fpr_max ∈ [0, 1]).
    Returns nan if fewer than 2 unique labels.
    """
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except Exception:
        return float('nan')
    # Interpolate TPR at exactly fpr_max
    tpr_at_max = float(np.interp(fpr_max, fpr, tpr))
    # Truncate to [0, fpr_max]
    mask = fpr <= fpr_max
    fpr_cut = np.append(fpr[mask], fpr_max)
    tpr_cut = np.append(tpr[mask], tpr_at_max)
    # np.trapz was renamed to np.trapezoid (NumPy 2.x) and removed in newer
    # versions -> resolve safely.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    raw_pauc = float(_trapz(tpr_cut, fpr_cut))
    return raw_pauc / fpr_max


def dr_at_fpr(labels: np.ndarray, scores: np.ndarray,
              fpr_list=(0.001, 0.01, 0.05, 0.10)) -> dict:
    """
    Detection rate (TPR) at specific FPR values.
    Returns {fpr_str: dr_float}.  Returns nan for each if not computable.
    """
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
    except Exception:
        return {str(f): float('nan') for f in fpr_list}
    return {str(f): float(np.interp(f, fpr, tpr)) for f in fpr_list}


def auc_safe(labels: np.ndarray, scores: np.ndarray) -> float:
    """Full AUC, returns nan on failure."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return float(sklearn_auc(fpr, tpr))
    except Exception:
        return float('nan')


def roc_safe(labels: np.ndarray, scores: np.ndarray):
    """Returns (fpr_list, tpr_list, auc_float). Safe version."""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr.tolist(), tpr.tolist(), auc_safe(labels, scores)
    except Exception:
        return [0., 1.], [0., 1.], float('nan')


# ---------------------------------------------------------------------------
# CFAR threshold (training pixels ONLY — no test labels used)
# ---------------------------------------------------------------------------

def cfar_threshold(bkg_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """
    CFAR threshold set from training background scores at target_fpr.

    IMPORTANT: bkg_scores must be scores of TRAINING pixels ONLY.
    Never pass test pixels here — that would violate the CFAR guarantee.

    Parameters
    ----------
    bkg_scores  : (n_train,) scores on training background pixels
    target_fpr  : desired false-alarm rate (default 1%)

    Returns
    -------
    threshold : float  — scores > threshold → declared target
    """
    return float(np.quantile(bkg_scores, 1.0 - target_fpr))


def per_class_fpr(scores: np.ndarray, labels: np.ndarray,
                  cls_labels: np.ndarray, threshold: float) -> dict:
    """
    Per-class FPR on the test region.

    For each unique class in cls_labels (excluding target pixels where
    labels==1), computes the fraction of that class's pixels that exceed
    the threshold (false alarms).

    Parameters
    ----------
    scores     : (n_test,) detection scores on test pixels
    labels     : (n_test,) binary — 1 = planted target, 0 = background
    cls_labels : (n_test,) ground-truth class id of each test pixel
    threshold  : detection threshold (e.g. from cfar_threshold on train)

    Returns
    -------
    per_cls_fpr : {class_name: fpr_float}
    """
    CLS_NAMES = {
        0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
        4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
        8: 'bricks',    9: 'shadows',
    }
    result = {}
    # Only background pixels (labels==0) contribute to FPR
    bkg_mask = (labels == 0)
    for cid in np.unique(cls_labels[bkg_mask]):
        mask = bkg_mask & (cls_labels == cid)
        if mask.sum() == 0:
            continue
        fpr_val = float((scores[mask] > threshold).mean())
        name = CLS_NAMES.get(int(cid), f'cls{cid}')
        result[name] = fpr_val
    return result


# ---------------------------------------------------------------------------
# Target signature construction (TRAINING side — no test labels)
# ---------------------------------------------------------------------------

def compute_signature(gt_patch: np.ndarray, raw_patch: np.ndarray,
                      w_dom: float = 0.8, w_mean: float = 0.2,
                      external_cls_pixels: np.ndarray = None) -> tuple:
    """
    Compute the target signature as a weighted combination of the dominant
    class mean and the patch mean.

    Formula: s = w_dom * mu_dominant + w_mean * mu_patch

    Parameters
    ----------
    gt_patch           : (H_box, W_box) or (N,) GT class labels for the box
    raw_patch          : (H_box, W_box, D) or (N, D) raw pixel values
    w_dom              : weight for dominant class mean (default 0.8)
    w_mean             : weight for overall patch mean  (default 0.2)
    external_cls_pixels: if not None, use these D-dim pixels for the dominant
                         class mean instead of pixels inside the box (cleaner)

    Returns
    -------
    s_raw     : (D,) raw-space signature
    dom_cls   : int   dominant class id
    dom_name  : str   dominant class name
    """
    CLS_NAMES = {
        0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
        4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
        8: 'bricks',    9: 'shadows',
    }
    gt_flat  = gt_patch.ravel()
    raw_flat = raw_patch.reshape(-1, raw_patch.shape[-1])

    # Dominant class (ignoring unlabeled=0)
    labeled  = gt_flat != 0
    if labeled.sum() == 0:
        labeled = np.ones(len(gt_flat), dtype=bool)   # fallback: use all
    cls_ids, cnts = np.unique(gt_flat[labeled], return_counts=True)
    dom_cls  = int(cls_ids[cnts.argmax()])
    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')

    if external_cls_pixels is not None:
        mu_dom = external_cls_pixels.mean(axis=0).astype(np.float32)
    else:
        dom_mask = (gt_flat == dom_cls)
        if dom_mask.sum() == 0:
            dom_mask = np.ones(len(gt_flat), dtype=bool)
        mu_dom = raw_flat[dom_mask].mean(axis=0).astype(np.float32)

    mu_patch = raw_flat.mean(axis=0).astype(np.float32)
    s_raw    = (w_dom * mu_dom + w_mean * mu_patch).astype(np.float32)
    return s_raw, dom_cls, dom_name


# ---------------------------------------------------------------------------
# Random box generation
# ---------------------------------------------------------------------------

def generate_random_boxes(gt: np.ndarray, n: int = 4,
                           min_pixels: int = 2000,
                           seeds=(42, 123, 456, 789)) -> list:
    """
    Generate n random (train_box, test_box) pairs on the Pavia-U image.

    Strategy:
    - Sample top-left corner (r0, c0) uniformly; fix a random width and height
      such that the box contains at least min_pixels pixels.
    - Train and test boxes are drawn from non-overlapping halves of the image
      (top half / bottom half, or left / right — determined by seed).
    - No class-composition constraints (diversity emerges from spatial spread).

    Parameters
    ----------
    gt          : (H, W) ground-truth label map
    n           : number of pairs to generate
    min_pixels  : minimum pixels inside each box
    seeds       : RNG seeds (one per pair)

    Returns
    -------
    pairs : list of dicts with keys 'train_box', 'test_box' ([r0,r1,c0,c1])
    """
    CLS_NAMES = {
        0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
        4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
        8: 'bricks',    9: 'shadows',
    }
    H, W = gt.shape
    pairs = []
    for i, seed in enumerate(seeds[:n]):
        rng = np.random.default_rng(seed)
        # Alternate between splitting image horizontally vs vertically
        if i % 2 == 0:
            # Split horizontally: train in top, test in bottom
            mid = H // 2
            train_region = (0, mid, 0, W)
            test_region  = (mid, H, 0, W)
        else:
            # Split vertically: train in left, test in right
            mid = W // 2
            train_region = (0, H, 0, mid)
            test_region  = (0, H, mid, W)

        def _random_box(region, rng, min_pix):
            r0r, r1r, c0r, c1r = region
            for _ in range(500):
                # Random size aiming for ~min_pix * 1.5 pixels
                target = int(min_pix * (1.5 + rng.uniform(0, 1)))
                side   = int(np.sqrt(target))
                h_box  = max(side, 40) + int(rng.integers(0, max(side//2, 20)))
                w_box  = max(side, 40) + int(rng.integers(0, max(side//2, 20)))
                h_box  = min(h_box, r1r - r0r)
                w_box  = min(w_box, c1r - c0r)
                if h_box < 10 or w_box < 10:
                    continue
                r0 = int(rng.integers(r0r, r1r - h_box + 1))
                c0 = int(rng.integers(c0r, c1r - w_box + 1))
                r1 = r0 + h_box
                c1 = c0 + w_box
                if (r1 - r0) * (c1 - c0) >= min_pix:
                    return [r0, r1, c0, c1]
            # Fallback: use the full region minus some padding
            pad = 10
            return [r0r + pad, r1r - pad, c0r + pad, c1r - pad]

        tr_box = _random_box(train_region, rng, min_pixels)
        te_box = _random_box(test_region,  rng, min_pixels)

        def _stats(box):
            r0, r1, c0, c1 = box
            patch = gt[r0:r1, c0:c1].ravel()
            cls_ids, cnts = np.unique(patch, return_counts=True)
            labeled = cls_ids != 0
            dom_cls  = int(cls_ids[cnts.argmax()]) if labeled.any() else int(cls_ids[cnts.argmax()])
            return {CLS_NAMES.get(int(c), f'cls{c}'): int(n_)
                    for c, n_ in zip(cls_ids, cnts)}, int(cnts.sum()), dom_cls

        tr_stats, tr_total, tr_dom = _stats(tr_box)
        te_stats, te_total, te_dom = _stats(te_box)

        pairs.append({
            'train_box':      tr_box,
            'train_stats':    tr_stats,
            'train_total':    tr_total,
            'train_dominant': CLS_NAMES.get(tr_dom, f'cls{tr_dom}'),
            'test_box':       te_box,
            'test_stats':     te_stats,
            'test_total':     te_total,
            'test_dominant':  CLS_NAMES.get(te_dom, f'cls{te_dom}'),
        })
    return pairs


# ---------------------------------------------------------------------------
# Box statistics
# ---------------------------------------------------------------------------

def box_statistics(gt: np.ndarray, data_patch: np.ndarray,
                   box: list) -> dict:
    """
    Compute class composition and spectral statistics for a box.

    Parameters
    ----------
    gt         : (H, W) full ground-truth label map
    data_patch : (H, W, D) full normalized data array
    box        : [r0, r1, c0, c1]

    Returns
    -------
    stats dict with keys:
        'box', 'n_pixels', 'class_counts', 'dominant_cls',
        'dominant_cls_id', 'class_fraction'
    """
    CLS_NAMES = {
        0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
        4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
        8: 'bricks',    9: 'shadows',
    }
    r0, r1, c0, c1 = box
    gt_crop   = gt[r0:r1, c0:c1].ravel()
    cls_ids, cnts = np.unique(gt_crop, return_counts=True)
    total = int(cnts.sum())
    cls_counts = {CLS_NAMES.get(int(c), f'cls{c}'): int(n)
                  for c, n in zip(cls_ids, cnts)}
    dom_idx = cnts.argmax()
    dom_id  = int(cls_ids[dom_idx])
    return {
        'box':              box,
        'n_pixels':         total,
        'class_counts':     cls_counts,
        'dominant_cls':     CLS_NAMES.get(dom_id, f'cls{dom_id}'),
        'dominant_cls_id':  dom_id,
        'class_fraction':   {k: v / total for k, v in cls_counts.items()},
    }


# ---------------------------------------------------------------------------
# Score → spatial map
# ---------------------------------------------------------------------------

def scores_to_spatial_map(scores: np.ndarray,
                           te_idx: np.ndarray,
                           box_shape: tuple,
                           fill: float = float('nan')) -> np.ndarray:
    """
    Scatter test-pixel scores back into a 2D grid matching the test box.

    Parameters
    ----------
    scores    : (n_test,) detection scores
    te_idx    : (n_test,) linear indices into the flattened box
    box_shape : (H_box, W_box)
    fill      : value for pixels not in te_idx (default nan)

    Returns
    -------
    smap : (H_box, W_box) spatial score map
    """
    smap = np.full(box_shape[0] * box_shape[1], fill, dtype=np.float32)
    smap[te_idx] = scores.astype(np.float32)
    return smap.reshape(box_shape)
