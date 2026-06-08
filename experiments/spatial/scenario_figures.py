"""
scenario_figures.py — All per-scenario visualizations for the spatial experiment.

Saves 5 PDF figures per (scenario, budget) pair:
  A. scenario_{sid}_n{budget}_boxes.pdf         — box context (2×3 grid)
  B. scenario_{sid}_n{budget}_targets.pdf       — target locations on test box
  C. scenario_{sid}_n{budget}_score_maps.pdf    — spatial score heatmaps
  D. scenario_{sid}_n{budget}_detection_on_gt.pdf — decision map on full GT
  E. scenario_{sid}_n{budget}_roc.pdf           — ROC curves (all detectors)

Also provides aggregated figure helpers:
  save_cfar_per_class_figure    — per-class FPR bar chart (all scenarios)
  save_auc_summary_figure       — AUC bar chart (all scenarios)
  save_dr_at_fpr_figure         — DR@FPR table/bar chart
  save_box_size_ablation_figure — AUC vs box size line plot
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, to_rgb
from sklearn.metrics import roc_curve, auc as sklearn_auc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks',    9: 'shadows',
}
CLS_COLORS_HEX = {
    0: '#000000', 1: '#808080', 2: '#00cc44', 3: '#d2691e',
    4: '#006400', 5: '#add8e6', 6: '#a52a2a', 7: '#9400d3',
    8: '#ff4500', 9: '#00008b',
}

METHOD_COLORS = {
    'CF-Attn-CFAR': '#1f77b4',
    'CF-Attn':      '#aec7e8',
    'NeighborMLP':  '#2ca02c',
    'DSM':          '#ff7f0e',
    'THANTD':       '#d62728',
    'AMF':          '#9467bd',
    'Reg-AMF':      '#c5b0d5',
    'GMM-GLRT':     '#8c564b',
    'GMM-Levin':    '#e377c2',
}


def _false_color(data_raw, box, bands=(60, 30, 10)):
    """Returns (H_box, W_box, 3) false-color RGB for the given box."""
    r0, r1, c0, c1 = box
    rgb = data_raw[r0:r1, c0:c1][..., list(bands)].astype(np.float32)
    lo  = np.percentile(rgb, 2,  axis=(0, 1), keepdims=True)
    hi  = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def _gt_colorimage(gt_crop):
    """Returns (H, W, 3) coloured GT map."""
    H, W = gt_crop.shape
    img  = np.zeros((H, W, 3), dtype=np.float32)
    for cid, hex_ in CLS_COLORS_HEX.items():
        img[gt_crop == cid] = to_rgb(hex_)
    return img


def _auc_safe(lab, sc):
    try:
        fpr, tpr, _ = roc_curve(lab, sc)
        return float(sklearn_auc(fpr, tpr))
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Fig A: Box context
# ---------------------------------------------------------------------------

def _fig_box_context(data_norm, gt, train_box, test_box):
    """2×3 grid: [false-color | gt-color | class histogram] for train & test."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, (box, title) in enumerate([(train_box, 'TRAIN'), (test_box, 'TEST')]):
        r0, r1, c0, c1 = box
        gt_crop  = gt[r0:r1, c0:c1]
        fc       = _false_color(data_norm, box)
        gt_img   = _gt_colorimage(gt_crop)

        axes[row, 0].imshow(fc)
        axes[row, 0].set_title(f'{title} — false color', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(gt_img)
        axes[row, 1].set_title(f'{title} — GT classes', fontsize=10)
        # legend patches
        unique_cls = np.unique(gt_crop)
        handles = [mpatches.Patch(color=CLS_COLORS_HEX.get(int(c), '#fff'),
                                  label=CLS_NAMES.get(int(c), f'cls{c}'))
                   for c in unique_cls]
        axes[row, 1].legend(handles=handles, loc='lower right', fontsize=7,
                            framealpha=0.8)
        axes[row, 1].axis('off')

        cls_ids, cnts = np.unique(gt_crop.ravel(), return_counts=True)
        names = [CLS_NAMES.get(int(c), f'cls{c}') for c in cls_ids]
        colors = [CLS_COLORS_HEX.get(int(c), '#999') for c in cls_ids]
        axes[row, 2].barh(names, cnts, color=colors)
        axes[row, 2].set_xlabel('pixel count')
        axes[row, 2].set_title(f'{title} — class composition', fontsize=10)
        total = cnts.sum()
        for xi, (name, cnt) in enumerate(zip(names, cnts)):
            axes[row, 2].text(cnt + total * 0.005, xi,
                              f'{cnt} ({100*cnt/total:.0f}%)',
                              va='center', fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig B: Target locations
# ---------------------------------------------------------------------------

def _fig_target_locations(data_norm, test_box, tgt_idx_dict, te_gt_flat):
    """False-color test box with planted target pixel markers."""
    n_models = len(tgt_idx_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    fc = _false_color(data_norm, test_box)

    for ax, (tm, tgt_idx) in zip(axes, tgt_idx_dict.items()):
        ax.imshow(fc)
        ys, xs = np.unravel_index(tgt_idx, (H_b, W_b))
        ax.scatter(xs, ys, s=12, c='white', marker='.', linewidths=0,
                   label=f'{len(tgt_idx)} targets')
        ax.set_title(f'{tm} — {len(tgt_idx)} planted targets', fontsize=10)
        ax.axis('off')

    fig.suptitle('Planted target locations (white dots)', fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig C: Score maps
# ---------------------------------------------------------------------------

def _fig_score_maps(scores_dict, labels_dict, tgt_idx_dict, thresholds_dict,
                    test_box):
    """
    One row per detector, single column: additive model.
    Spatial heatmap of scores over test box.
    White contour at 1% CFAR threshold (from training).
    White dots = planted targets.
    """
    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    n_pix = H_b * W_b

    target_models = [tm for tm in ('additive',)
                     if any(f'{name}_{tm}' in scores_dict
                            for name in METHOD_COLORS)]

    detectors = [nm for nm in METHOD_COLORS
                 if any(f'{nm}_{tm}' in scores_dict for tm in target_models)]
    if not detectors:
        return None
    n_det = len(detectors)
    n_col = len(target_models)

    fig, axes = plt.subplots(n_det, n_col, figsize=(5 * n_col, 3 * n_det),
                             squeeze=False)

    for row, nm in enumerate(detectors):
        for col, tm in enumerate(target_models):
            key = f'{nm}_{tm}'
            ax  = axes[row, col]
            if key not in scores_dict:
                ax.axis('off')
                continue
            sc    = scores_dict[key]
            lab   = labels_dict[key]
            thr   = thresholds_dict.get(key, None)
            tgt   = tgt_idx_dict.get(tm, np.array([], dtype=int))

            # Scatter scores into a spatial map (NaN for missing pixels)
            smap = np.full(n_pix, np.nan, dtype=np.float32)
            smap[:len(sc)] = sc   # assumes all test pixels, in order
            smap = smap.reshape(H_b, W_b)

            vmin, vmax = np.nanpercentile(sc, [1, 99])
            im = ax.imshow(smap, cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                           interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Threshold contour
            if thr is not None:
                ax.contour(smap, levels=[thr], colors='white', linewidths=1)

            # Target dots
            if len(tgt) > 0:
                ys, xs = np.unravel_index(tgt, (H_b, W_b))
                ax.scatter(xs, ys, s=8, c='yellow', marker='.', linewidths=0)

            auc_val = _auc_safe(lab, sc)
            ax.set_title(f'{nm} | {tm}  AUC={auc_val:.3f}', fontsize=9)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(nm, fontsize=9, rotation=0, labelpad=50, va='center')

    fig.suptitle('Score maps  (white contour = 1% threshold, yellow = targets)',
                 fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig D: Detection decision on full GT map
# ---------------------------------------------------------------------------

def _fig_detection_on_gt(scores_dict, labels_dict, thresholds_dict,
                          gt, test_box):
    """
    Full GT label map with test box highlighted.
    Inside test box: TP=green, FP=red, TN=dark, FN=blue.
    One subplot per (detector, target_model) pair.
    """
    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    n_pix = H_b * W_b

    target_models = [tm for tm in ('additive',)
                     if any(f'{nm}_{tm}' in scores_dict for nm in METHOD_COLORS)]
    detectors = [nm for nm in METHOD_COLORS
                 if any(f'{nm}_{tm}' in scores_dict for tm in target_models)]
    if not detectors:
        return None
    n_det = len(detectors)
    n_col = len(target_models)

    fig, axes = plt.subplots(n_det, n_col, figsize=(5 * n_col, 4 * n_det),
                             squeeze=False)

    gt_base = _gt_colorimage(gt) * 0.5   # dimmed background

    for row, nm in enumerate(detectors):
        for col, tm in enumerate(target_models):
            key = f'{nm}_{tm}'
            ax  = axes[row, col]
            if key not in scores_dict:
                ax.axis('off'); continue

            sc  = scores_dict[key]
            lab = labels_dict[key]
            thr = thresholds_dict.get(key, None)

            base_img = gt_base.copy()
            if thr is not None:
                detected = sc > thr
                # Inside test box overlay
                det_map  = np.zeros((H_b * W_b,), dtype=int)
                det_map[:len(sc)] = detected.astype(int)
                lab_map  = np.zeros((H_b * W_b,), dtype=int)
                lab_map[:len(lab)] = lab

                patch_img = base_img[r0:r1, c0:c1].copy()
                # TP=green, FP=red, FN=blue, TN=transparent (keep gt)
                tp_idx = np.where((det_map == 1) & (lab_map == 1))[0]
                fp_idx = np.where((det_map == 1) & (lab_map == 0))[0]
                fn_idx = np.where((det_map == 0) & (lab_map == 1))[0]
                flat   = patch_img.reshape(-1, 3)
                flat[tp_idx] = [0.0, 1.0, 0.0]   # green
                flat[fp_idx] = [1.0, 0.1, 0.1]   # red
                flat[fn_idx] = [0.1, 0.1, 1.0]   # blue
                base_img[r0:r1, c0:c1] = flat.reshape(H_b, W_b, 3)

            ax.imshow(base_img)
            # Box outline
            rect = mpatches.Rectangle(
                (c0, r0), c1-c0, r1-r0,
                lw=2, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f'{nm} | {tm}', fontsize=9)
            ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(fc='#00ff00', label='TP'),
                    Patch(fc='#ff1a1a', label='FP'),
                    Patch(fc='#1a1aff', label='FN')]
    axes[0, 0].legend(handles=legend_elems, loc='upper left', fontsize=8)

    fig.suptitle('Detection on full GT map  (threshold @ 1% FPR from train)',
                 fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Fig E: ROC curves
# ---------------------------------------------------------------------------

def _fig_roc(scores_dict, labels_dict):
    """ROC curves for all detectors (additive model)."""
    target_models = [tm for tm in ('additive',)
                     if any(f'{nm}_{tm}' in scores_dict for nm in METHOD_COLORS)]
    if not target_models:
        return None

    fig, axes = plt.subplots(1, len(target_models),
                             figsize=(6 * len(target_models), 5))
    if len(target_models) == 1:
        axes = [axes]

    for ax, tm in zip(axes, target_models):
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='random')
        for nm in METHOD_COLORS:
            key = f'{nm}_{tm}'
            if key not in scores_dict:
                continue
            sc, lab = scores_dict[key], labels_dict[key]
            try:
                fpr, tpr, _ = roc_curve(lab, sc)
                auc_val = float(sklearn_auc(fpr, tpr))
                ax.plot(fpr, tpr, lw=1.8, color=METHOD_COLORS.get(nm, 'k'),
                        label=f'{nm} ({auc_val:.3f})')
            except Exception:
                pass
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'ROC — {tm}', fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main entry point: save all 5 figures for one scenario/budget
# ---------------------------------------------------------------------------

def save_scenario_figures(sid: int, n_budget: int,
                           scores_dict: dict, labels_dict: dict,
                           thresholds_dict: dict,
                           te_gt_flat: np.ndarray,
                           tgt_idx_dict: dict,
                           data_norm: np.ndarray, gt: np.ndarray,
                           train_box: list, test_box: list,
                           fig_dir: str):
    """
    Save all 5 per-scenario figures.

    Parameters
    ----------
    sid            : scenario index (0-based)
    n_budget       : box-size budget (used in filename)
    scores_dict    : {'MethodName_additive': array, ...}
    labels_dict    : same keys, binary label arrays
    thresholds_dict: same keys, float thresholds from training pixels
    te_gt_flat     : (n_test,) GT class label for each test pixel
    tgt_idx_dict   : {'additive': array} target pixel indices
    data_norm      : (H, W, D_raw) normalized image
    gt             : (H, W) full GT label map
    train_box      : [r0, r1, c0, c1]
    test_box       : [r0, r1, c0, c1]
    fig_dir        : directory to write PDFs into
    """
    os.makedirs(fig_dir, exist_ok=True)
    prefix = f'scenario_{sid}_n{n_budget}'

    r0, r1, c0, c1 = test_box
    te_gt_crop = gt[r0:r1, c0:c1]

    # Fig A
    fig = _fig_box_context(data_norm, gt, train_box, test_box)
    path = os.path.join(fig_dir, f'{prefix}_boxes.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {os.path.basename(path)}", flush=True)

    # Fig B
    fig = _fig_target_locations(data_norm, test_box, tgt_idx_dict, te_gt_flat)
    path = os.path.join(fig_dir, f'{prefix}_targets.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {os.path.basename(path)}", flush=True)

    # Fig C
    fig = _fig_score_maps(scores_dict, labels_dict, tgt_idx_dict,
                           thresholds_dict, test_box)
    if fig is not None:
        path = os.path.join(fig_dir, f'{prefix}_score_maps.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [fig] {os.path.basename(path)}", flush=True)

    # Fig D
    fig = _fig_detection_on_gt(scores_dict, labels_dict, thresholds_dict,
                                gt, test_box)
    if fig is not None:
        path = os.path.join(fig_dir, f'{prefix}_detection_on_gt.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [fig] {os.path.basename(path)}", flush=True)

    # Fig E
    fig = _fig_roc(scores_dict, labels_dict)
    if fig is not None:
        path = os.path.join(fig_dir, f'{prefix}_roc.pdf')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [fig] {os.path.basename(path)}", flush=True)


# ---------------------------------------------------------------------------
# Aggregated figures (called once after all scenarios)
# ---------------------------------------------------------------------------

def save_cfar_per_class_figure(all_cfar: list, fig_dir: str,
                                detectors=None, target_model='additive'):
    """
    Per-class FPR bar chart aggregated across all scenarios.

    all_cfar : list of {detector_name: {class_name: fpr}} per scenario
    """
    if detectors is None:
        detectors = list(METHOD_COLORS.keys())
    os.makedirs(fig_dir, exist_ok=True)

    # Collect all class names across scenarios
    all_classes = set()
    for scenario_cfar in all_cfar:
        for det, cfar_d in scenario_cfar.items():
            all_classes.update(cfar_d.keys())
    all_classes = sorted(all_classes)
    if not all_classes:
        return

    # Average FPR per (detector, class) across scenarios
    n_det = len(detectors)
    n_cls = len(all_classes)
    fpr_mean = np.full((n_det, n_cls), np.nan)
    fpr_std  = np.full((n_det, n_cls), np.nan)

    for di, det in enumerate(detectors):
        for ci, cls in enumerate(all_classes):
            vals = [sc[det][cls]
                    for sc in all_cfar
                    if det in sc and cls in sc[det]]
            if vals:
                fpr_mean[di, ci] = np.mean(vals)
                fpr_std[di, ci]  = np.std(vals)

    x = np.arange(n_cls)
    width = 0.8 / max(n_det, 1)
    fig, ax = plt.subplots(figsize=(max(10, n_cls * 1.5), 5))
    for di, det in enumerate(detectors):
        offset = (di - n_det / 2 + 0.5) * width
        bars = ax.bar(x + offset, fpr_mean[di], width * 0.9,
                      yerr=fpr_std[di],
                      label=det,
                      color=METHOD_COLORS.get(det, 'grey'),
                      alpha=0.8, capsize=3)

    ax.axhline(0.01, color='black', linestyle='--', lw=1.5, label='1% target FPR')
    ax.set_xticks(x); ax.set_xticklabels(all_classes, rotation=30, ha='right')
    ax.set_ylabel('False Alarm Rate (FPR)'); ax.set_xlabel('Background Class')
    ax.set_title(f'Per-Class FPR — {target_model}  (threshold @ 1% from train)\n'
                 f'CF-Attn-CFAR should be flat near 1%', fontsize=11)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, min(ax.get_ylim()[1] * 1.2, 1.0))
    fig.tight_layout()
    path = os.path.join(fig_dir, 'cfar_per_class.pdf')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    print(f"  [fig] cfar_per_class.pdf", flush=True)


def save_auc_summary_figure(all_metrics: dict, fig_dir: str, n_budget: int):
    """AUC summary bar chart for the additive model."""
    os.makedirs(fig_dir, exist_ok=True)
    if not all_metrics:
        return

    # Gather AUC per detector across scenarios for this budget
    tm = 'additive'
    det_aucs = {}
    for sid_key, sid_data in all_metrics.items():
        bkey = f'n{n_budget}'
        if bkey not in sid_data:
            continue
        # sid_data[bkey][tm] = {'auc': {...}, 'pauc': {...}, ...}
        auc_dict = sid_data[bkey].get(tm, {})
        if isinstance(auc_dict, dict):
            auc_dict = auc_dict.get('auc', auc_dict)
        for nm, val in auc_dict.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                det_aucs.setdefault(nm, []).append(val)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    names = [n for n in det_aucs if det_aucs[n]]
    means = [np.nanmean(det_aucs[n]) for n in names]
    stds  = [np.nanstd(det_aucs[n])  for n in names]
    colors = [METHOD_COLORS.get(n, 'grey') for n in names]
    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel('AUC')
    ax.set_title(f'Additive model  (n={n_budget})', fontsize=11)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, min(v + 0.01, 0.99),
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'AUC summary (mean ± std over {len(all_metrics)} scenarios)',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(fig_dir, f'auc_summary_n{n_budget}.pdf')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    print(f"  [fig] auc_summary_n{n_budget}.pdf", flush=True)


def save_dr_at_fpr_figure(all_metrics: dict, fig_dir: str, n_budget: int):
    """DR@FPR bar chart for key FPR levels."""
    os.makedirs(fig_dir, exist_ok=True)
    fpr_keys = ['0.001', '0.01', '0.05', '0.1']
    det_dr   = {}

    for sid_key, sid_data in all_metrics.items():
        bkey = f'n{n_budget}'
        if bkey not in sid_data:
            continue
        # dr is stored under ['additive']['dr'] or under ['dr_additive']
        dr_src = (sid_data[bkey].get('additive', {}).get('dr', {})
                  or sid_data[bkey].get('dr_additive', {}))
        for nm, fpr_dict in dr_src.items():
            if not isinstance(fpr_dict, dict):
                continue
            for fk in fpr_keys:
                if fk in fpr_dict:
                    det_dr.setdefault(nm, {}).setdefault(fk, []).append(fpr_dict[fk])

    if not det_dr:
        return

    detectors = list(det_dr.keys())
    x = np.arange(len(fpr_keys))
    width = 0.8 / max(len(detectors), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for di, nm in enumerate(detectors):
        means = [np.nanmean(det_dr[nm].get(fk, [np.nan])) for fk in fpr_keys]
        offset = (di - len(detectors) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9,
               label=nm, color=METHOD_COLORS.get(nm, 'grey'), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'FPR={f}' for f in fpr_keys])
    ax.set_ylabel('Detection Rate (TPR)'); ax.set_ylim(0, 1)
    ax.set_title(f'DR@FPR — additive model  (n={n_budget})', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    path = os.path.join(fig_dir, f'dr_at_fpr_n{n_budget}.pdf')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    print(f"  [fig] dr_at_fpr_n{n_budget}.pdf", flush=True)


def save_box_size_ablation_figure(all_metrics: dict, fig_dir: str,
                                   budgets: list, target_model: str = 'additive'):
    """AUC vs box-size line plot per detector."""
    os.makedirs(fig_dir, exist_ok=True)
    det_auc_by_budget = {}

    for sid_key, sid_data in all_metrics.items():
        for n_budget in budgets:
            bkey = f'n{n_budget}'
            if bkey not in sid_data:
                continue
            auc_dict = sid_data[bkey].get(target_model, {})
            if isinstance(auc_dict, dict):
                auc_dict = auc_dict.get('auc', auc_dict)
            for nm, val in auc_dict.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    det_auc_by_budget.setdefault(nm, {}).setdefault(n_budget, []).append(val)

    if not det_auc_by_budget:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for nm, budget_dict in det_auc_by_budget.items():
        xs     = sorted(budget_dict.keys())
        means  = [np.nanmean(budget_dict[b]) for b in xs]
        stds   = [np.nanstd(budget_dict[b])  for b in xs]
        ax.errorbar(xs, means, yerr=stds, marker='o', lw=1.8, capsize=4,
                    color=METHOD_COLORS.get(nm, 'grey'), label=nm)

    ax.set_xlabel('Box size (min pixels)'); ax.set_ylabel('AUC')
    ax.set_title(f'AUC vs Box Size — {target_model}', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    path = os.path.join(fig_dir, f'box_size_ablation_{target_model}.pdf')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    print(f"  [fig] box_size_ablation_{target_model}.pdf", flush=True)
