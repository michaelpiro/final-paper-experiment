"""
Architecture and PCA-dim comparison experiment.

For a single background class and target class:
  - Tries several score model architectures and PCA dimensions
  - Trains each on 3000 background samples
  - Evaluates detection ROC on held-out test pixels
  - Produces a single figure with all ROC curves

Architectures compared:
  Linear          : ScoreNet(hidden_dims=[])               ~400 params  (d=20)
  MoL K=2         : MixtureOfLinears(K=2, gate_hidden=5)  ~960 params
  MoL K=3         : MixtureOfLinears(K=3, gate_hidden=5)  ~1270 params
  MLP [64,64]     : ScoreNet(hidden_dims=[64,64])          ~4300 params

PCA dims: configurable list, e.g. [5, 10, 20]

Usage:
    python arch_comparison.py --dataset real_datasets/pavia-u.mat
    python arch_comparison.py --target 9 --bkg 2
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
from real_data_experiment import show_class_map

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CFG = dict(
    sigma        = 0.05,
    amplitude    = 0.3,       # target planting amplitude θ
    target_fraction = 0.1,
    n_train      = 3000,
    test_fraction= 0.2,
    base_epochs  = 1000,
    lr           = 0.001,
    weight_decay = 0.0001,
    batch_size   = 256,
    pca_dims     = [5, 10, 20],
    results_dir  = "experiments/arch_comparison",
    # defaults: bkg=2 (asphalt, large), target=1
    default_bkg    = 2,
    default_target = 1,
)

ARCHITECTURES = [
    ("Linear",       lambda d: ScoreNet(d, hidden_dims=[])),
    ("MoL K=2",      lambda d: MixtureOfLinears(d, K=2, gate_hidden=5)),
    ("MoL K=3",      lambda d: MixtureOfLinears(d, K=3, gate_hidden=5)),
    ("MLP [64,64]",  lambda d: ScoreNet(d, hidden_dims=[64, 64])),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg, dataset_path, bkg_cls, target_cls):
    os.makedirs(cfg["results_dir"], exist_ok=True)

    # --- Load & normalize ---
    mat  = scipy.io.loadmat(dataset_path)
    data = mat["data"].astype(np.float64)
    gt   = mat["map"].astype(int)
    data = (data - data.min()) / (data.max() - data.min())
    rgb  = false_color(data)

    show_class_map(rgb, gt)

    H, W, B = data.shape
    flat     = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    bkg_all = flat[gt_flat == bkg_cls]
    tgt_all = flat[gt_flat == target_cls]

    rng = np.random.default_rng(42)
    rng.shuffle(bkg_all)

    sigma        = cfg["sigma"]
    amplitude    = cfg["amplitude"]
    tgt_fraction = cfg["target_fraction"]
    n_train      = cfg["n_train"]
    test_fraction= cfg["test_fraction"]
    epochs       = cfg["base_epochs"]
    lr           = cfg["lr"]
    wd           = cfg["weight_decay"]
    bs           = cfg["batch_size"]
    pca_dims     = cfg["pca_dims"]

    # One figure: one row per pca_dim
    n_rows = len(pca_dims)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4.5 * n_rows), squeeze=False)

    for row, pca_dim in enumerate(pca_dims):
        ax = axes[row, 0]
        print(f"\n{'='*60}")
        print(f"PCA dim = {pca_dim}")
        print(f"{'='*60}")

        # --- PCA ---
        pca = PCA(n_components=pca_dim)
        pca.fit(flat)

        bkg_pca = pca.transform(bkg_all)
        tgt_pca = pca.transform(tgt_all)
        s = tgt_pca.mean(axis=0); s /= np.linalg.norm(s)

        # --- Split ---
        n_use   = min(n_train + max(1, int(n_train * test_fraction / (1 - test_fraction))),
                      len(bkg_pca))
        n_test  = max(1, int(round(n_use * test_fraction)))
        n_tr    = min(n_train, n_use - n_test)
        train_data = bkg_pca[:n_tr]
        test_bkg   = bkg_pca[n_tr:n_tr + n_test]

        # Plant target
        n_target = max(1, int(round(len(test_bkg) * tgt_fraction)))
        labels   = np.zeros(len(test_bkg), dtype=int)
        tgt_idx  = np.random.default_rng(0).choice(len(test_bkg), n_target, replace=False)
        labels[tgt_idx] = 1
        test_data = test_bkg.copy()
        test_data[tgt_idx] += amplitude * s

        print(f"  train={n_tr}  test={len(test_bkg)}  targets={n_target}  σ={sigma}")

        # --- AMF baseline ---
        amf_scores  = detector_amf(test_data, train_data, s)
        amf_auc     = roc_auc_score(labels, amf_scores)
        fpr, tpr, _ = roc_curve(labels, amf_scores)
        ax.plot(fpr, tpr, color="black", linewidth=2, linestyle="--",
                label=f"AMF  AUC={amf_auc:.4f}")
        print(f"  [AMF]  AUC={amf_auc:.4f}")

        # --- Train & evaluate each DSM architecture ---
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(ARCHITECTURES)))
        for (arch_name, arch_fn), color in zip(ARCHITECTURES, colors):
            model = arch_fn(pca_dim)
            n_p   = model.n_params()
            print(f"\n  [{arch_name}]  params={n_p}")
            model = train_dsm(model, train_data, sigma=sigma,
                              lr=lr, batch_size=bs, epochs=epochs,
                              weight_decay=wd, print_every=200)
            scores = detector_dsm(test_data, train_data, model, s)
            auc    = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{arch_name} ({n_p}p)  AUC={auc:.4f}")
            print(f"  AUC={auc:.4f}")

        ax.plot([0,1],[0,1],"k--",linewidth=0.8,alpha=0.4)
        ax.set_title(f"PCA d={pca_dim}  |  train={n_tr}  bkg={bkg_cls}  target={target_cls}",
                     fontsize=11)
        ax.set_xlabel("False Alarm Rate"); ax.set_ylabel("Detection Rate")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Architecture comparison — σ={sigma}  θ={amplitude}  epochs={epochs}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save_path = os.path.join(cfg["results_dir"], "arch_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {save_path}")
    plt.show(block=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="real_datasets/pavia-u.mat")
    parser.add_argument("--config",  default="arch_comparison_config.yaml")
    parser.add_argument("--target",  type=int, default=None)
    parser.add_argument("--bkg",     type=int, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    with open(args.config) as f:
        cfg.update(yaml.safe_load(f))

    import scipy.io as sio
    mat = sio.loadmat(args.dataset)
    gt  = mat["map"].astype(int)
    data_raw = mat["data"].astype(np.float64)
    data_raw = (data_raw - data_raw.min()) / (data_raw.max() - data_raw.min())
    rgb = false_color(data_raw)

    target_cls = args.target if args.target is not None else cfg["default_target"]
    bkg_cls    = args.bkg    if args.bkg    is not None else cfg["default_bkg"]
    print(f"Background class: {bkg_cls}  |  Target class: {target_cls}  |  amplitude: {cfg['amplitude']}")

    run(cfg, args.dataset, bkg_cls, target_cls)
