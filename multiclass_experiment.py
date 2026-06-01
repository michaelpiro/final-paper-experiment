"""
Multiclass real-data experiment on the Pavia hyperspectral dataset.

Background is composed of multiple classes (e.g. [2, 4, 5]).
Samples from each class are drawn proportionally to their dataset frequency.
Each class contributes 80% train / 20% test independently, then combined.

Pipeline:
  1. Load dataset, show false-color + class map
  2. User picks target class and background class list
  3. Normalize [0,1], PCA to pca_dim
  4. Per background class: sample proportionally (total_samples total),
     split 80/20 train/test, shuffle
  5. Train DSM on combined background train pixels
  6. Plant target on target_fraction of combined test pixels
  7. Evaluate: AMF, Reg.AMF (per sigma), DSM (per sigma)
  8. ROC curves + AUC figure

Usage:
    python multiclass_experiment.py --dataset real_datasets/pavia-u.mat
    python multiclass_experiment.py --dataset real_datasets/pavia-u.mat \\
        --target 9 --bkg 1 2 4 5
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

from dsm_model import ScoreNet, train_dsm, train_lfi, compute_lfi_detector_scores
from visualize_dataset import false_color
from gaussian_iid_experiment import detector_amf, detector_reg_amf, detector_dsm
from gmm_iid_experiment import detector_fitted_gmm, detector_gmm_glrt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show_class_map(rgb, gt_map):
    classes = np.unique(gt_map)
    cmap = plt.cm.get_cmap("tab10", len(classes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(rgb); ax1.set_title("False-color"); ax1.axis("off")

    im = ax2.imshow(gt_map, cmap=cmap,
                    vmin=classes.min() - 0.5, vmax=classes.max() + 0.5)
    ax2.set_title("Ground-truth class map"); ax2.axis("off")
    fig.colorbar(plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.BoundaryNorm(
            np.arange(classes.min()-0.5, classes.max()+1.5), cmap.N)),
        ax=ax2, ticks=classes, shrink=0.8).set_label("Class ID")

    for cls in classes:
        yx = np.argwhere(gt_map == cls)
        cy, cx = yx.mean(axis=0)
        ax2.text(cx, cy, str(int(cls)), ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold")

    plt.tight_layout(); plt.show(block=True); plt.close(fig)

    print("\nClass pixel counts:")
    for cls in classes:
        print(f"  class {int(cls):2d}: {(gt_map == cls).sum():6d} pixels")


def proportional_counts(class_sizes, total):
    """
    Given {class_id: n_pixels}, return {class_id: n_samples} summing to total,
    proportional to class sizes. Each class gets at least 1 sample.
    """
    total_pixels = sum(class_sizes.values())
    raw = {c: total * n / total_pixels for c, n in class_sizes.items()}
    counts = {c: max(1, int(round(v))) for c, v in raw.items()}
    # Adjust rounding to hit exactly `total`
    diff = total - sum(counts.values())
    # Add/remove from the largest classes
    keys_sorted = sorted(counts, key=lambda c: raw[c], reverse=True)
    for i in range(abs(diff)):
        c = keys_sorted[i % len(keys_sorted)]
        counts[c] += 1 if diff > 0 else -1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg, dataset_path, target_cls=None, bkg_classes=None):
    results_dir   = cfg["results_dir"]
    pca_dim       = cfg["pca_dim"]
    hidden_dims   = cfg.get("hidden_dims", [])
    sigma_values  = cfg["sigma_values"]
    amplitude     = cfg["amplitude"]
    tgt_fraction  = cfg["target_fraction"]
    total_samples = cfg["total_samples"]
    test_fraction = cfg["test_fraction"]
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

    lo, hi = data.min(), data.max()
    data = (data - lo) / (hi - lo)

    rgb = false_color(data)
    show_class_map(rgb, gt)

    # ------------------------------------------------------------------
    # Step 2 — Pick classes
    # ------------------------------------------------------------------
    all_classes  = sorted(int(c) for c in np.unique(gt))
    class_sizes  = {c: int((gt == c).sum()) for c in all_classes}

    if target_cls is None:
        print("\nEnter target class ID:")
        target_cls = int(input("  Target class: ").strip())
    if bkg_classes is None:
        print("Enter background class IDs (space-separated, e.g. 1 2 4 5):")
        bkg_classes = [int(x) for x in input("  Background classes: ").strip().split()]

    assert target_cls in class_sizes,  f"Target class {target_cls} not found"
    for c in bkg_classes:
        assert c in class_sizes, f"Background class {c} not found"
    assert target_cls not in bkg_classes, "Target class must differ from background classes"

    print(f"\n  Target    : class {target_cls} ({class_sizes[target_cls]} pixels)")
    for c in bkg_classes:
        print(f"  Background: class {c} ({class_sizes[c]} pixels)")

    # ------------------------------------------------------------------
    # Step 3 — PCA on full image
    # ------------------------------------------------------------------
    print(f"\n[Step 3] PCA to {pca_dim} dimensions ...")
    H, W, B = data.shape
    all_flat = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    pca = PCA(n_components=pca_dim)
    pca.fit(all_flat)

    # Target signature: mean of all target class pixels (PCA space), normalized
    tgt_pixels = pca.transform(all_flat[gt_flat == target_cls])
    s_raw = tgt_pixels.mean(axis=0)
    s     = s_raw / (np.linalg.norm(s_raw) + 1e-12)
    print(f"  Target: {len(tgt_pixels)} pixels, ||s||={np.linalg.norm(s):.4f}")

    # ------------------------------------------------------------------
    # Step 4 — Proportional sampling + per-class 80/20 split
    # ------------------------------------------------------------------
    bkg_sizes  = {c: class_sizes[c] for c in bkg_classes}
    counts     = proportional_counts(bkg_sizes, total_samples)
    rng        = np.random.default_rng(seed=42)

    train_list, test_list = [], []
    print(f"\n[Step 4] Sampling {total_samples} total background pixels "
          f"(test_fraction={test_fraction}):")

    for c in bkg_classes:
        pixels_c = pca.transform(all_flat[gt_flat == c])  # all pixels for class c
        n_c      = min(counts[c], len(pixels_c))
        idx      = rng.choice(len(pixels_c), size=n_c, replace=False)
        sampled  = pixels_c[idx]
        rng.shuffle(sampled)

        n_test_c  = max(1, int(round(n_c * test_fraction)))
        n_train_c = n_c - n_test_c

        train_list.append(sampled[:n_train_c])
        test_list.append(sampled[n_train_c:])
        print(f"  class {c}: {n_c} sampled → {n_train_c} train, {n_test_c} test")

    train_data = np.vstack(train_list)
    test_bkg   = np.vstack(test_list)
    rng.shuffle(train_data)
    rng.shuffle(test_bkg)

    n_train = len(train_data)
    n_test  = len(test_bkg)
    print(f"  Total: {n_train} train, {n_test} test")

    # ------------------------------------------------------------------
    # Step 5 — Plant target
    # ------------------------------------------------------------------
    n_target = max(1, int(round(n_test * tgt_fraction)))
    labels   = np.zeros(n_test, dtype=int)
    rng2     = np.random.default_rng(seed=0)
    tgt_idx  = rng2.choice(n_test, size=n_target, replace=False)
    labels[tgt_idx] = 1

    test_data = test_bkg.copy()
    test_data[tgt_idx] += amplitude * s
    print(f"\n[Step 5] Planted target: {n_target}/{n_test} pixels (θ={amplitude})")

    # ------------------------------------------------------------------
    # Step 6 — Train DSM models
    # ------------------------------------------------------------------
    delta_theta  = cfg.get("delta_theta", 0.01)
    arch = "linear" if not hidden_dims else f"MLP {hidden_dims}"
    print(f"\n[Step 6] Training models  arch={arch}")

    models = {}
    for sigma in sigma_values:
        print(f"\n  [DSM] σ={sigma}, epochs={base_epochs} ...")
        model = ScoreNet(input_dim=pca_dim, hidden_dims=hidden_dims, activation="silu")
        model = train_dsm(model, train_data, sigma=sigma,
                          lr=lr, batch_size=batch_size, epochs=base_epochs,
                          weight_decay=weight_decay)
        torch.save(model.state_dict(), os.path.join(results_dir, f"model_dsm_sigma{sigma}.pt"))
        models[sigma] = model

    print(f"\n  [LFI/LRao] Δθ={delta_theta}, epochs={base_epochs} ...")
    model_lfi = ScoreNet(input_dim=pca_dim, hidden_dims=hidden_dims, activation="silu")
    model_lfi = train_lfi(model_lfi, train_data, s, delta_theta=delta_theta,
                          lr=lr, batch_size=batch_size, epochs=base_epochs,
                          weight_decay=weight_decay)
    torch.save(model_lfi.state_dict(), os.path.join(results_dir, "model_lfi.pt"))

    # ------------------------------------------------------------------
    # Step 7 — Evaluate
    # ------------------------------------------------------------------
    print("\n[Step 7] Evaluating detectors ...")
    roc_results = {}
    auc_summary = {}

    def add_detector(label, scores):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        roc_results[label] = (fpr, tpr, auc)
        auc_summary[label] = float(auc)
        print(f"  {label:<35} AUC={auc:.4f}")

    add_detector("AMF", detector_amf(test_data, train_data, s))
    for sigma in sigma_values:
        add_detector(f"Reg.AMF σ={sigma}",
                     detector_reg_amf(test_data, train_data, s, sigma))
    K = len(bkg_classes)
    add_detector(f"Fitted GMM oracle (K={K})",
                 detector_fitted_gmm(test_data, train_data, s, amplitude, K=K))
    add_detector(f"GMM GLRT (K={K})",
                 detector_gmm_glrt(test_data, train_data, s, K=K))
    for sigma in sigma_values:
        add_detector(f"DSM σ={sigma}",
                     detector_dsm(test_data, train_data, models[sigma], s))
    add_detector("LRao (LFI)",
                 compute_lfi_detector_scores(model_lfi, train_data, test_data, s, delta_theta))

    with open(os.path.join(results_dir, "auc_summary.json"), "w") as f:
        json.dump(auc_summary, f, indent=2)

    # ------------------------------------------------------------------
    # Step 8 — Figure: ROC (left) + annotated image (right)
    # ------------------------------------------------------------------
    n_det  = len(roc_results)
    colors = [plt.cm.get_cmap("tab10", n_det)(i) for i in range(n_det)]

    fig, (ax_roc, ax_img) = plt.subplots(1, 2, figsize=(13, 5))

    for (label, (fpr, tpr, auc)), color in zip(roc_results.items(), colors):
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{label}  AUC={auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax_roc.set_xlabel("False Alarm Rate", fontsize=12)
    ax_roc.set_ylabel("Detection Rate", fontsize=12)
    ax_roc.set_title(
        f"ROC — Pavia multiclass\ntarget={target_cls}  bkg={bkg_classes}", fontsize=11)
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    info = (f"PCA d={pca_dim}  arch={arch}  θ={amplitude}\n"
            f"train={n_train}  test={n_test}  "
            f"targets planted={n_target} ({tgt_fraction*100:.0f}%)")
    ax_roc.text(0.02, 0.98, info, transform=ax_roc.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    # Image: outline background classes + target class
    ax_img.imshow(rgb)
    bkg_colors_map = ["yellow", "cyan", "lime", "orange", "magenta", "white"]
    from matplotlib.patches import Patch
    legend_handles = []
    for i, c in enumerate(bkg_classes):
        mask = (gt == c)
        col  = bkg_colors_map[i % len(bkg_colors_map)]
        ax_img.contour(mask.astype(float), levels=[0.5], colors=[col], linewidths=1.5)
        n_c = counts[c]
        legend_handles.append(Patch(facecolor=col, label=f"Bkg class {c} (n={n_c})"))
    tgt_mask = (gt == target_cls)
    ax_img.contour(tgt_mask.astype(float), levels=[0.5], colors=["red"], linewidths=2)
    legend_handles.append(Patch(facecolor="red", label=f"Target class {target_cls}"))
    ax_img.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax_img.set_title("False-color image", fontsize=12)
    ax_img.axis("off")

    fig.tight_layout()
    save_path = os.path.join(results_dir, "result_figure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.show(block=True)
    plt.close(fig)
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiclass real-data experiment")
    parser.add_argument("--dataset", default="real_datasets/pavia-u.mat")
    parser.add_argument("--config",  default="multiclass_config.yaml")
    parser.add_argument("--target",  type=int,  default=None,
                        help="Target class ID (skips interactive prompt)")
    parser.add_argument("--bkg",     type=int,  nargs="+", default=None,
                        help="Background class IDs (skips interactive prompt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg, args.dataset,
                   target_cls=args.target, bkg_classes=args.bkg)
