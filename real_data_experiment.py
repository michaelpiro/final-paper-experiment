"""
Real-data experiment on the Pavia hyperspectral dataset.

Pipeline:
  1. Load dataset, show false-color image with class map overlay
  2. User picks background class (many pixels) and target class (few pixels)
  3. Normalize to [0,1], PCA to pca_dim dimensions
  4. Split background pixels: 4000 train / 1000 test (or 80/20 for <5000 pixels)
  5. Train DSM on train pixels with two sigma values
  6. Plant target signature on target_fraction of test pixels (additive model)
  7. Detect with: AMF, Reg.AMF (x2 sigmas), DSM (x2 sigmas)
  8. Plot ROC curves and report AUC for each detector

Usage:
    python real_data_experiment.py --dataset real_datasets/pavia-u.mat
"""

import argparse
import json
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from dsm_model import ScoreNet, train_dsm, compute_scores, train_lfi, compute_lfi_detector_scores
from visualize_dataset import load_dataset, false_color
from gaussian_iid_experiment import detector_amf, detector_reg_amf, detector_dsm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_class_map(rgb, gt_map):
    """Display false-color image alongside the ground-truth class map."""
    classes = np.unique(gt_map)
    cmap = plt.cm.get_cmap("tab10", len(classes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(rgb)
    ax1.set_title("False-color image")
    ax1.axis("off")

    ax2.imshow(gt_map, cmap=cmap, vmin=classes.min() - 0.5, vmax=classes.max() + 0.5)
    ax2.set_title("Ground-truth class map")
    ax2.axis("off")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap,
                              norm=mcolors.BoundaryNorm(
                                  np.arange(classes.min() - 0.5, classes.max() + 1.5), cmap.N)),
        ax=ax2, ticks=classes, shrink=0.8)
    cbar.set_label("Class ID")

    for cls in classes:
        count = (gt_map == cls).sum()
        # annotate centroid
        yx = np.argwhere(gt_map == cls)
        cy, cx = yx.mean(axis=0)
        ax2.text(cx, cy, str(int(cls)), ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold")

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    print("\nClass pixel counts:")
    for cls in classes:
        print(f"  class {int(cls):2d}: {(gt_map == cls).sum():6d} pixels")


def plot_roc_curves(results_dict, save_path):
    """
    Plot ROC curves for all detectors.
    results_dict: {label: (fpr, tpr, auc)}
    """
    n_det  = len(results_dict)
    colors = [plt.cm.get_cmap("tab10", n_det)(i) for i in range(n_det)]

    fig, ax = plt.subplots(figsize=(6, 5))
    for (label, (fpr, tpr, auc)), color in zip(results_dict.items(), colors):
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Alarm Rate", fontsize=12)
    ax.set_ylabel("Detection Rate", fontsize=12)
    ax.set_title("ROC curves — Real data (Pavia)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"ROC plot saved to {save_path}")
    plt.show(block=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg, dataset_path):
    results_dir   = cfg["results_dir"]
    pca_dim       = cfg["pca_dim"]
    sigma_values  = cfg["sigma_values"]
    amplitude     = cfg["amplitude"]
    tgt_fraction  = cfg["target_fraction"]
    total_samples = cfg.get("total_samples", 5000)
    test_fraction = cfg.get("test_fraction", 0.2)
    base_epochs   = cfg["base_epochs"]
    lr            = cfg["lr"]
    weight_decay  = cfg.get("weight_decay", 0.0)
    batch_size    = cfg["batch_size"]
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load and display
    # ------------------------------------------------------------------
    print(f"\n=== Loading {dataset_path} ===")
    import scipy.io
    mat  = scipy.io.loadmat(dataset_path)
    data = mat["data"].astype(np.float64)
    gt   = mat["map"].astype(int)

    # Normalize to [0,1]
    lo, hi = data.min(), data.max()
    data = (data - lo) / (hi - lo)

    rgb = false_color(data)

    classes     = sorted([int(c) for c in np.unique(gt)])
    class_sizes = {c: int((gt == c).sum()) for c in classes}

    no_display = cfg.get("_no_display", False)
    if not no_display:
        show_class_map(rgb, gt)

    # ------------------------------------------------------------------
    # Step 2 — Pick classes (from CLI args or interactively)
    # ------------------------------------------------------------------
    if "_bkg_cls" in cfg and "_target_cls" in cfg:
        bkg_cls = cfg["_bkg_cls"]
        tgt_cls = cfg["_target_cls"]
    else:
        print("\nPick classes (enter integer IDs shown in the map).")
        bkg_cls = int(input("  Background class ID (large class): ").strip())
        tgt_cls = int(input("  Target class ID    (small class): ").strip())

    assert bkg_cls in class_sizes, f"Class {bkg_cls} not found"
    assert tgt_cls in class_sizes, f"Class {tgt_cls} not found"
    print(f"  Background: class {bkg_cls} ({class_sizes[bkg_cls]} pixels)")
    print(f"  Target    : class {tgt_cls} ({class_sizes[tgt_cls]} pixels)")

    # ------------------------------------------------------------------
    # Step 3 — Extract pixels
    # ------------------------------------------------------------------
    H, W, B = data.shape
    all_flat = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    bkg_pixels = all_flat[gt_flat == bkg_cls]   # (N_bkg, B)
    tgt_pixels = all_flat[gt_flat == tgt_cls]   # (N_tgt, B)

    # Shuffle background
    rng = np.random.default_rng(seed=42)
    rng.shuffle(bkg_pixels)

    # ------------------------------------------------------------------
    # Step 4 — PCA
    # ------------------------------------------------------------------
    print(f"\n[Step 4] PCA to {pca_dim} dimensions ...")
    pca = PCA(n_components=pca_dim)
    pca.fit(all_flat)                           # fit on full image

    bkg_pca = pca.transform(bkg_pixels)        # (N_bkg, pca_dim)
    tgt_pca = pca.transform(tgt_pixels)        # (N_tgt, pca_dim)

    # Target signature: mean of target class pixels (PCA space), normalized
    s_raw = tgt_pca.mean(axis=0)
    s     = s_raw / (np.linalg.norm(s_raw) + 1e-12)
    print(f"  ||s||={np.linalg.norm(s):.4f}  (target: {len(tgt_pca)} pixels)")

    # ------------------------------------------------------------------
    # Step 5 — Split background
    # ------------------------------------------------------------------
    N       = len(bkg_pca)
    n_use   = min(total_samples, N)
    n_test  = max(1, int(round(n_use * test_fraction)))
    n_train = n_use - n_test

    bkg_pca = bkg_pca[:n_use]          # already shuffled above
    train_data = bkg_pca[:n_train]
    test_bkg   = bkg_pca[n_train:]
    print(f"\n[Step 5] Using {n_use}/{N} bkg pixels → "
          f"{n_train} train, {n_test} test  "
          f"(total_samples={total_samples}, test_fraction={test_fraction})")

    # ------------------------------------------------------------------
    # Step 6 — Plant target on test pixels
    # ------------------------------------------------------------------
    n_target = max(1, int(round(n_test * tgt_fraction)))
    labels   = np.zeros(n_test, dtype=int)
    rng2     = np.random.default_rng(seed=0)
    tgt_idx  = rng2.choice(n_test, size=n_target, replace=False)
    labels[tgt_idx] = 1

    test_data = test_bkg.copy()
    test_data[tgt_idx] += amplitude * s
    print(f"[Step 6] Planted target on {n_target}/{n_test} test pixels (amplitude={amplitude})")

    # ------------------------------------------------------------------
    # Step 7 — Train DSM and LFI models
    # ------------------------------------------------------------------
    hidden_dims  = cfg.get("hidden_dims", [])
    delta_theta  = cfg.get("delta_theta", 0.01)
    arch         = "linear" if not hidden_dims else f"MLP {hidden_dims}"
    print(f"\n[Step 7] Training models  architecture={arch}")

    models_dsm = {}
    for sigma in sigma_values:
        print(f"\n  [DSM] σ={sigma}, epochs={base_epochs} ...")
        model = ScoreNet(input_dim=pca_dim, hidden_dims=hidden_dims, activation="silu")
        model = train_dsm(model, train_data, sigma=sigma,
                          lr=lr, batch_size=batch_size, epochs=base_epochs,
                          weight_decay=weight_decay)
        torch.save(model.state_dict(), os.path.join(results_dir, f"model_dsm_sigma{sigma}.pt"))
        models_dsm[sigma] = model

    print(f"\n  [LFI/LRao] Δθ={delta_theta}, epochs={base_epochs} ...")
    model_lfi = ScoreNet(input_dim=pca_dim, hidden_dims=hidden_dims, activation="silu")
    model_lfi = train_lfi(model_lfi, train_data, s, delta_theta=delta_theta,
                          lr=lr, batch_size=batch_size, epochs=base_epochs,
                          weight_decay=weight_decay)
    torch.save(model_lfi.state_dict(), os.path.join(results_dir, "model_lfi.pt"))

    # ------------------------------------------------------------------
    # Step 8 — Evaluate detectors
    # ------------------------------------------------------------------
    print("\n[Step 8] Evaluating detectors ...")
    roc_results = {}
    auc_summary = {}

    def add_detector(label, scores):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        roc_results[label] = (fpr, tpr, auc)
        auc_summary[label] = auc
        print(f"  {label:<35} AUC={auc:.4f}")

    add_detector("AMF", detector_amf(test_data, train_data, s))

    for sigma in sigma_values:
        add_detector(f"Reg.AMF σ={sigma}",
                     detector_reg_amf(test_data, train_data, s, sigma))

    for sigma in sigma_values:
        add_detector(f"DSM σ={sigma}",
                     detector_dsm(test_data, train_data, models_dsm[sigma], s))

    add_detector("LRao (LFI)",
                 compute_lfi_detector_scores(model_lfi, train_data, test_data, s, delta_theta))

    # ------------------------------------------------------------------
    # Step 9 — Save and plot
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "auc_summary.json"), "w") as f:
        json.dump(auc_summary, f, indent=2)

    # Combined figure: ROC (left) + false-color with class overlay (right)
    n_detectors = len(roc_results)
    cmap   = plt.cm.get_cmap("tab10", n_detectors)
    colors = [cmap(i) for i in range(n_detectors)]

    fig, (ax_roc, ax_img) = plt.subplots(1, 2, figsize=(13, 5))

    for (label, (fpr, tpr, auc)), color in zip(roc_results.items(), colors):
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  AUC={auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax_roc.set_xlabel("False Alarm Rate", fontsize=12)
    ax_roc.set_ylabel("Detection Rate", fontsize=12)
    ax_roc.set_title(f"ROC — Pavia  bkg={bkg_cls}  target={tgt_cls}", fontsize=11)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # Right: false-color with bkg/target class outlines
    ax_img.imshow(rgb)
    bkg_mask = (gt == bkg_cls)
    tgt_mask = (gt == tgt_cls)
    ax_img.contour(bkg_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.5)
    ax_img.contour(tgt_mask.astype(float), levels=[0.5], colors=["red"],    linewidths=1.5)
    from matplotlib.patches import Patch
    ax_img.legend(handles=[Patch(facecolor="yellow", label=f"Background (class {bkg_cls})"),
                            Patch(facecolor="red",    label=f"Target (class {tgt_cls})")],
                  loc="upper right", fontsize=9)
    ax_img.set_title("False-color image", fontsize=12)
    ax_img.axis("off")

    # Info box
    info = (f"PCA d={pca_dim}  θ={amplitude}  "
            f"train={n_train}  test={n_test}\n"
            f"target pixels planted: {n_target} ({tgt_fraction*100:.0f}%)")
    ax_roc.text(0.02, 0.98, info, transform=ax_roc.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    save_path = os.path.join(results_dir, "result_figure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    if not cfg.get("_no_display", False):
        plt.show(block=True)
    plt.close(fig)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-data experiment on Pavia dataset")
    parser.add_argument("--dataset",    default="real_datasets/pavia-u.mat")
    parser.add_argument("--config",     default="real_data_config.yaml")
    parser.add_argument("--bkg",        type=int, default=None, help="Background class ID")
    parser.add_argument("--target",     type=int, default=None, help="Target class ID")
    parser.add_argument("--no-display", action="store_true", help="Skip interactive class map display")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Inject CLI class selection into config so run_experiment can use them
    if args.bkg    is not None: cfg["_bkg_cls"]    = args.bkg
    if args.target is not None: cfg["_target_cls"] = args.target
    if args.no_display:         cfg["_no_display"]  = True

    run_experiment(cfg, args.dataset)
