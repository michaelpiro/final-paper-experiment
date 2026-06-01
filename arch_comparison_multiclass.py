"""
Architecture comparison — multiclass background (Pavia dataset).

Same as arch_comparison.py but background is several classes sampled proportionally.
Default: target=1, bkg=[2,3,4,5].

Usage:
    python arch_comparison_multiclass.py
    python arch_comparison_multiclass.py --target 1 --bkg 2 3 4 5
"""

import argparse
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import scipy.io
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from dsm_model import ScoreNet, MixtureOfLinears, train_dsm
from visualize_dataset import false_color
from gaussian_iid_experiment import detector_dsm, detector_amf
from multiclass_experiment import proportional_counts
from real_data_experiment import show_class_map

ARCHITECTURES = [
    ("Linear",      lambda d: ScoreNet(d, hidden_dims=[])),
    ("MoL K=2",     lambda d: MixtureOfLinears(d, K=2, gate_hidden=5)),
    ("MoL K=3",     lambda d: MixtureOfLinears(d, K=3, gate_hidden=5)),
    ("MLP [64,64]", lambda d: ScoreNet(d, hidden_dims=[64, 64])),
]

DEFAULT_CFG = dict(
    sigma           = 0.05,
    amplitude       = 0.3,
    target_fraction = 0.1,
    total_samples   = 3000,
    test_fraction   = 0.2,
    base_epochs     = 1000,
    lr              = 0.001,
    weight_decay    = 0.0001,
    batch_size      = 256,
    pca_dims        = [5, 10, 20],
    results_dir     = "experiments/arch_comparison_multiclass",
    default_target  = 1,
    default_bkg     = [2, 3, 4, 5],
)


def run(cfg, dataset_path, target_cls, bkg_classes):
    os.makedirs(cfg["results_dir"], exist_ok=True)

    mat  = scipy.io.loadmat(dataset_path)
    data = mat["data"].astype(np.float64)
    gt   = mat["map"].astype(int)
    data = (data - data.min()) / (data.max() - data.min())
    rgb  = false_color(data)

    show_class_map(rgb, gt)

    H, W, B  = data.shape
    flat     = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    sigma           = cfg["sigma"]
    amplitude       = cfg["amplitude"]
    tgt_fraction    = cfg["target_fraction"]
    total_samples   = cfg["total_samples"]
    test_fraction   = cfg["test_fraction"]
    epochs          = cfg["base_epochs"]
    lr              = cfg["lr"]
    wd              = cfg["weight_decay"]
    bs              = cfg["batch_size"]
    pca_dims        = cfg["pca_dims"]

    print(f"\nTarget class: {target_cls}  |  Background classes: {bkg_classes}")
    print(f"Amplitude θ={amplitude}  |  σ={sigma}  |  total_samples={total_samples}")

    n_rows = len(pca_dims)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4.5 * n_rows), squeeze=False)

    for row, pca_dim in enumerate(pca_dims):
        ax = axes[row, 0]
        print(f"\n{'='*60}\nPCA dim = {pca_dim}\n{'='*60}")

        # --- PCA on full image ---
        pca = PCA(n_components=pca_dim)
        pca.fit(flat)

        # Target signature
        tgt_pca = pca.transform(flat[gt_flat == target_cls])
        s = tgt_pca.mean(axis=0); s /= np.linalg.norm(s)

        # --- Proportional background sampling ---
        bkg_sizes = {c: int((gt_flat == c).sum()) for c in bkg_classes}
        counts    = proportional_counts(bkg_sizes, total_samples)

        rng = np.random.default_rng(42)
        train_list, test_list = [], []
        for c in bkg_classes:
            px    = pca.transform(flat[gt_flat == c])
            n_c   = min(counts[c], len(px))
            idx   = rng.choice(len(px), size=n_c, replace=False)
            s2    = px[idx]; rng.shuffle(s2)
            n_t   = max(1, int(round(n_c * test_fraction)))
            train_list.append(s2[n_t:])
            test_list.append(s2[:n_t])

        train_data = np.vstack(train_list); rng.shuffle(train_data)
        test_bkg   = np.vstack(test_list);  rng.shuffle(test_bkg)

        print(f"  train={len(train_data)}  test={len(test_bkg)}")
        for c in bkg_classes:
            print(f"    class {c}: {counts[c]} sampled")

        # Plant target
        n_target = max(1, int(round(len(test_bkg) * tgt_fraction)))
        labels   = np.zeros(len(test_bkg), dtype=int)
        tgt_idx  = np.random.default_rng(0).choice(len(test_bkg), n_target, replace=False)
        labels[tgt_idx] = 1
        test_data = test_bkg.copy()
        test_data[tgt_idx] += amplitude * s

        # --- AMF baseline ---
        amf_auc     = roc_auc_score(labels, detector_amf(test_data, train_data, s))
        fpr, tpr, _ = roc_curve(labels, detector_amf(test_data, train_data, s))
        ax.plot(fpr, tpr, color="black", linewidth=2, linestyle="--",
                label=f"AMF  AUC={amf_auc:.4f}")
        print(f"  [AMF]  AUC={amf_auc:.4f}")

        # --- DSM architectures ---
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(ARCHITECTURES)))
        for (arch_name, arch_fn), color in zip(ARCHITECTURES, colors):
            model = arch_fn(pca_dim)
            n_p   = model.n_params()
            print(f"\n  [{arch_name}]  params={n_p}")
            model = train_dsm(model, train_data, sigma=sigma,
                              lr=lr, batch_size=bs, epochs=epochs,
                              weight_decay=wd, print_every=200)
            scores      = detector_dsm(test_data, train_data, model, s)
            auc         = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{arch_name} ({n_p}p)  AUC={auc:.4f}")
            print(f"  AUC={auc:.4f}")

        ax.plot([0,1],[0,1],"k--",linewidth=0.8,alpha=0.4)
        ax.set_title(
            f"PCA d={pca_dim}  |  train={len(train_data)}  "
            f"target={target_cls}  bkg={bkg_classes}", fontsize=10)
        ax.set_xlabel("False Alarm Rate"); ax.set_ylabel("Detection Rate")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Arch comparison (multiclass)  σ={sigma}  θ={amplitude}  epochs={epochs}",
        fontsize=12, y=1.01)
    fig.tight_layout()
    save_path = os.path.join(cfg["results_dir"], "arch_comparison_multiclass.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="real_datasets/pavia-u.mat")
    parser.add_argument("--config",  default="arch_comparison_multiclass_config.yaml")
    parser.add_argument("--target",  type=int, default=None)
    parser.add_argument("--bkg",     type=int, nargs="+", default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    with open(args.config) as f:
        cfg.update(yaml.safe_load(f))

    target_cls = args.target   if args.target is not None else cfg["default_target"]
    bkg_classes= args.bkg      if args.bkg    is not None else cfg["default_bkg"]
    print(f"Background: {bkg_classes}  |  Target: {target_cls}  |  amplitude: {cfg['amplitude']}")

    run(cfg, args.dataset, target_cls, bkg_classes)
