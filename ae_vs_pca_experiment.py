"""
Autoencoder vs PCA Dimensionality Reduction for DSM Detection.

Compares four dimensionality reduction strategies, each followed by a DSM detector:
  1. AE-clean  : Autoencoder trained on n target-free background samples
  2. AE-mixed  : Autoencoder trained on n samples with 10% target-contaminated
  3. PCA-clean : PCA fitted on the same n target-free background samples
  4. PCA-mixed : PCA fitted on the same n mixed samples

All reduce to latent_dim=5. A small DSM ScoreNet is then trained for 2000 epochs
on the encoded background training samples and evaluated on an unseen test set.

Usage:
    python ae_vs_pca_experiment.py --dataset real_datasets/pavia-u.mat \
        --bkg 2 --target 6 --no-display
"""

import argparse
import json
import os
import joblib
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from dsm_model import (Autoencoder, ScoreNet, train_autoencoder, train_dsm,
                       compute_scores)
from visualize_dataset import load_dataset, false_color
from gaussian_iid_experiment import detector_dsm, detector_amf
from real_data_experiment import show_class_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_ae(model: Autoencoder, data: np.ndarray) -> np.ndarray:
    """Return latent codes from the encoder half of the autoencoder."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32)
        return model.encoder(x).numpy()


def make_mixed_data(bkg: np.ndarray, s: np.ndarray, amplitude: float,
                    tgt_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of bkg where tgt_fraction of rows have amplitude*s added."""
    mixed = bkg.copy()
    n_tgt = max(1, int(round(len(bkg) * tgt_fraction)))
    idx   = rng.choice(len(bkg), size=n_tgt, replace=False)
    mixed[idx] += amplitude * s
    return mixed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict, dataset_path: str):
    results_dir   = cfg["results_dir"]
    latent_dim    = cfg.get("latent_dim", 5)
    ae_hidden     = cfg.get("ae_hidden_dims", [64])
    ae_epochs     = cfg.get("ae_epochs", 500)
    linear_ae_epochs = cfg.get("linear_ae_epochs", ae_epochs)
    dsm_epochs    = cfg.get("dsm_epochs", 2000)
    dsm_sigma     = cfg.get("dsm_sigma", 0.05)
    dsm_hidden    = cfg.get("dsm_hidden_dims", [32, 32])
    n_train_max   = cfg.get("n_train", 1600)
    n_test        = cfg.get("n_test", 400)
    amplitude     = cfg.get("amplitude", 0.15)
    tgt_fraction  = cfg.get("target_fraction", 0.10)
    lr            = cfg.get("lr", 1e-3)
    batch_size    = cfg.get("batch_size", 256)
    weight_decay  = cfg.get("weight_decay", 1e-4)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load and normalize
    # ------------------------------------------------------------------
    print(f"\n=== Loading {dataset_path} ===")
    import scipy.io
    mat  = scipy.io.loadmat(dataset_path)
    data = mat["data"].astype(np.float64)
    gt   = mat["map"].astype(int)

    lo, hi = data.min(), data.max()
    data = (data - lo) / (hi - lo)

    rgb     = false_color(data)
    classes = sorted([int(c) for c in np.unique(gt)])

    if not cfg.get("_no_display", False):
        show_class_map(rgb, gt)

    # ------------------------------------------------------------------
    # Step 2 — Pick classes
    # ------------------------------------------------------------------
    if "_bkg_cls" in cfg and "_target_cls" in cfg:
        bkg_cls = cfg["_bkg_cls"]
        tgt_cls = cfg["_target_cls"]
    else:
        print("\nPick classes (enter integer IDs shown in the map).")
        bkg_cls = int(input("  Background class ID: ").strip())
        tgt_cls = int(input("  Target class ID: ").strip())

    H, W, B = data.shape
    all_flat = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    bkg_pixels = all_flat[gt_flat == bkg_cls].copy()
    tgt_pixels = all_flat[gt_flat == tgt_cls].copy()
    print(f"  Background: class {bkg_cls}  ({len(bkg_pixels)} pixels)")
    print(f"  Target    : class {tgt_cls}  ({len(tgt_pixels)} pixels)")

    # ------------------------------------------------------------------
    # Step 3 — PCA on all pixels + split (mirrors real_data_experiment exactly)
    #          Used for AMF baseline and as the common test/train split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    rng.shuffle(bkg_pixels)

    n_avail = len(bkg_pixels)
    n_test  = min(n_test, n_avail // 5)
    n_train = min(n_train_max, n_avail - n_test)
    assert n_train > 0 and n_test > 0, "Not enough background pixels."

    train_bkg_raw = bkg_pixels[:n_train]
    test_bkg_raw  = bkg_pixels[n_train: n_train + n_test]
    print(f"\n  Train: {n_train}  Test: {n_test}")

    pca_all = PCA(n_components=latent_dim)
    pca_all.fit(all_flat)                                    # fit on full image

    bkg_pca_all = pca_all.transform(bkg_pixels)
    tgt_pca_all = pca_all.transform(tgt_pixels)

    s_all = tgt_pca_all.mean(axis=0)
    s_all = s_all / (np.linalg.norm(s_all) + 1e-12)

    # AMF train/test in PCA-all space — targets planted in PCA space, identical to real_data_experiment
    train_pca_all  = bkg_pca_all[:n_train]
    test_bkg_all   = bkg_pca_all[n_train: n_train + n_test]

    n_tgt_test = max(1, int(round(n_test * tgt_fraction)))
    labels     = np.zeros(n_test, dtype=int)
    tgt_idx    = np.random.default_rng(seed=0).choice(n_test, size=n_tgt_test, replace=False)
    labels[tgt_idx] = 1

    test_pca_all = test_bkg_all.copy()
    test_pca_all[tgt_idx] += amplitude * s_all
    print(f"  Planted {n_tgt_test}/{n_test} target pixels in test set (amplitude={amplitude})")

    # Raw-pixel target signature for AE/PCA-subset encoders
    s_raw = tgt_pixels.mean(axis=0)
    s     = s_raw / (np.linalg.norm(s_raw) + 1e-12)

    # Raw-pixel test set for AE/PCA-subset encoders (planted in raw space)
    test_data_raw = test_bkg_raw.copy()
    test_data_raw[tgt_idx] += amplitude * s

    # Build mixed training set for AE-mixed/PCA-mixed
    train_mixed_raw = make_mixed_data(train_bkg_raw, s, amplitude, tgt_fraction,
                                      rng=np.random.default_rng(seed=1))

    # ------------------------------------------------------------------
    # Step 6 — Four dimensionality reduction strategies → latent_dim=5
    # ------------------------------------------------------------------
    print(f"\n=== Dimensionality reduction to {latent_dim} dims ===")

    # --- 6a: AE-clean ---
    ae_clean_path = os.path.join(results_dir, "ae_clean.pt")
    ae_clean = Autoencoder(input_dim=B, hidden_dims=ae_hidden, latent_dim=latent_dim)
    if os.path.exists(ae_clean_path):
        ae_clean.load_state_dict(torch.load(ae_clean_path, weights_only=True))
        ae_clean.eval()
        print(f"\n[AE-clean] Loaded from {ae_clean_path}")
    else:
        print(f"\n[AE-clean] Training autoencoder on {n_train} background samples ...")
        ae_clean = train_autoencoder(ae_clean, train_bkg_raw, lr=lr, batch_size=batch_size,
                                     epochs=ae_epochs, weight_decay=weight_decay,
                                     print_every=ae_epochs // 5)
        torch.save(ae_clean.state_dict(), ae_clean_path)
    enc_ae_clean_train = encode_ae(ae_clean, train_bkg_raw)
    enc_ae_clean_test  = encode_ae(ae_clean, test_data_raw)

    # --- 6b: AE-mixed ---
    ae_mixed_path = os.path.join(results_dir, "ae_mixed.pt")
    ae_mixed = Autoencoder(input_dim=B, hidden_dims=ae_hidden, latent_dim=latent_dim)
    if os.path.exists(ae_mixed_path):
        ae_mixed.load_state_dict(torch.load(ae_mixed_path, weights_only=True))
        ae_mixed.eval()
        print(f"\n[AE-mixed] Loaded from {ae_mixed_path}")
    else:
        print(f"\n[AE-mixed] Training autoencoder on {n_train} mixed (10% target) samples ...")
        ae_mixed = train_autoencoder(ae_mixed, train_mixed_raw, lr=lr, batch_size=batch_size,
                                     epochs=ae_epochs, weight_decay=weight_decay,
                                     print_every=ae_epochs // 5)
        torch.save(ae_mixed.state_dict(), ae_mixed_path)
    enc_ae_mixed_train = encode_ae(ae_mixed, train_bkg_raw)
    enc_ae_mixed_test  = encode_ae(ae_mixed, test_data_raw)

    # --- 6c: PCA-clean ---
    pca_clean_path = os.path.join(results_dir, "pca_clean.joblib")
    if os.path.exists(pca_clean_path):
        pca_clean = joblib.load(pca_clean_path)
        print(f"\n[PCA-clean] Loaded from {pca_clean_path}")
    else:
        print(f"\n[PCA-clean] Fitting PCA on {n_train} background samples ...")
        pca_clean = PCA(n_components=latent_dim)
        pca_clean.fit(train_bkg_raw)
        joblib.dump(pca_clean, pca_clean_path)
        print(f"  Explained variance ratio: {pca_clean.explained_variance_ratio_.sum():.4f}")
    enc_pca_clean_train = pca_clean.transform(train_bkg_raw)
    enc_pca_clean_test  = pca_clean.transform(test_data_raw)

    # --- 6d: PCA-mixed ---
    pca_mixed_path = os.path.join(results_dir, "pca_mixed.joblib")
    if os.path.exists(pca_mixed_path):
        pca_mixed = joblib.load(pca_mixed_path)
        print(f"\n[PCA-mixed] Loaded from {pca_mixed_path}")
    else:
        print(f"\n[PCA-mixed] Fitting PCA on {n_train} mixed (10% target) samples ...")
        pca_mixed = PCA(n_components=latent_dim)
        pca_mixed.fit(train_mixed_raw)
        joblib.dump(pca_mixed, pca_mixed_path)
        print(f"  Explained variance ratio: {pca_mixed.explained_variance_ratio_.sum():.4f}")
    enc_pca_mixed_train = pca_mixed.transform(train_bkg_raw)
    enc_pca_mixed_test  = pca_mixed.transform(test_data_raw)

    # --- 6e: Linear AE (no hidden layers, no activation — PCA-style network) ---
    ae_linear_path = os.path.join(results_dir, "ae_linear.pt")
    ae_linear = Autoencoder(input_dim=B, hidden_dims=[], latent_dim=latent_dim,
                            latent_activation=None)
    if os.path.exists(ae_linear_path):
        ae_linear.load_state_dict(torch.load(ae_linear_path, weights_only=True))
        ae_linear.eval()
        print(f"\n[AE-linear] Loaded from {ae_linear_path}")
    else:
        print(f"\n[AE-linear] Training linear autoencoder (no hidden, no activation) ...")
        ae_linear = train_autoencoder(ae_linear, train_bkg_raw, lr=lr, batch_size=batch_size,
                                      epochs=linear_ae_epochs, weight_decay=weight_decay,
                                      print_every=linear_ae_epochs // 5)
        torch.save(ae_linear.state_dict(), ae_linear_path)
    enc_ae_linear_train = encode_ae(ae_linear, train_bkg_raw)
    enc_ae_linear_test  = encode_ae(ae_linear, test_data_raw)

    # --- 6f: Linear AE + ReLU bottleneck ---
    ae_relu_path = os.path.join(results_dir, "ae_relu.pt")
    ae_relu = Autoencoder(input_dim=B, hidden_dims=[], latent_dim=latent_dim,
                          latent_activation="relu")
    if os.path.exists(ae_relu_path):
        ae_relu.load_state_dict(torch.load(ae_relu_path, weights_only=True))
        ae_relu.eval()
        print(f"\n[AE-linear-relu] Loaded from {ae_relu_path}")
    else:
        print(f"\n[AE-linear-relu] Training linear autoencoder with ReLU bottleneck ...")
        ae_relu = train_autoencoder(ae_relu, train_bkg_raw, lr=lr, batch_size=batch_size,
                                    epochs=linear_ae_epochs, weight_decay=weight_decay,
                                    print_every=linear_ae_epochs // 5)
        torch.save(ae_relu.state_dict(), ae_relu_path)
    enc_ae_relu_train = encode_ae(ae_relu, train_bkg_raw)
    enc_ae_relu_test  = encode_ae(ae_relu, test_data_raw)

    # ------------------------------------------------------------------
    # Step 7 — Train a DSM on each encoded training set
    # ------------------------------------------------------------------
    print(f"\n=== Training DSM models (σ={dsm_sigma}, epochs={dsm_epochs}) ===")

    configs = [
        ("AE-clean",      enc_ae_clean_train,  enc_ae_clean_test),
        ("AE-mixed",      enc_ae_mixed_train,  enc_ae_mixed_test),
        ("PCA-clean",     enc_pca_clean_train, enc_pca_clean_test),
        ("PCA-mixed",     enc_pca_mixed_train, enc_pca_mixed_test),
        ("AE-linear",     enc_ae_linear_train, enc_ae_linear_test),
        ("AE-linear-relu",enc_ae_relu_train,   enc_ae_relu_test),
    ]

    trained_dsm = {}
    for name, enc_train, _ in configs:
        ckpt = os.path.join(results_dir, f"dsm_{name}.pt")
        model = ScoreNet(input_dim=latent_dim, hidden_dims=dsm_hidden, activation="silu")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            model.eval()
            print(f"\n  [{name}] Loaded from {ckpt}")
        else:
            print(f"\n  [{name}] Training ...")
            model = train_dsm(model, enc_train, sigma=dsm_sigma,
                              lr=lr, batch_size=batch_size, epochs=dsm_epochs,
                              weight_decay=weight_decay, print_every=dsm_epochs // 5)
            torch.save(model.state_dict(), ckpt)
        trained_dsm[name] = model

    # ------------------------------------------------------------------
    # Step 8 — Project target signature and evaluate detectors
    # ------------------------------------------------------------------
    print("\n=== Evaluating detectors ===")
    roc_results = {}
    auc_summary = {}

    def add(label, scores):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        roc_results[label] = (fpr, tpr, auc)
        auc_summary[label] = float(auc)
        print(f"  {label:<30}  AUC={auc:.4f}")

    ae_model_map = {
        "AE-clean":       ae_clean,
        "AE-mixed":       ae_mixed,
        "AE-linear":      ae_linear,
        "AE-linear-relu": ae_relu,
    }

    for name, enc_train, enc_test in configs:
        # Project target signature through the corresponding encoder
        if name in ae_model_map:
            with torch.no_grad():
                s_enc = ae_model_map[name].encoder(
                    torch.tensor(s[None, :], dtype=torch.float32)
                ).numpy()[0]
        else:
            # s @ V.T matches the planted direction (addition doesn't subtract PCA mean)
            pca_model = pca_clean if name == "PCA-clean" else pca_mixed
            s_enc = s @ pca_model.components_.T

        s_enc = s_enc / (np.linalg.norm(s_enc) + 1e-12)
        add(f"DSM [{name}]", detector_dsm(enc_test, enc_train, trained_dsm[name], s_enc))

    # AMF baseline: PCA-all space, planted in PCA space — identical pipeline to real_data_experiment
    add("AMF (PCA-all)", detector_amf(test_pca_all, train_pca_all, s_all))

    # ------------------------------------------------------------------
    # Step 9 — Save results and plot
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "auc_summary.json"), "w") as f:
        json.dump(auc_summary, f, indent=2)

    n_det  = len(roc_results)
    cmap   = plt.cm.get_cmap("tab10", n_det)
    colors = [cmap(i) for i in range(n_det)]

    fig, (ax_roc, ax_img) = plt.subplots(1, 2, figsize=(14, 5))

    for (label, (fpr, tpr, auc)), color in zip(roc_results.items(), colors):
        ax_roc.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  AUC={auc:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax_roc.set_xlabel("False Alarm Rate", fontsize=12)
    ax_roc.set_ylabel("Detection Rate", fontsize=12)
    ax_roc.set_title("AE vs PCA — DSM detector comparison", fontsize=11)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    info = (f"latent={latent_dim}  σ={dsm_sigma}  θ={amplitude}\n"
            f"train={n_train}  test={n_test}  tgt%={tgt_fraction*100:.0f}")
    ax_roc.text(0.02, 0.98, info, transform=ax_roc.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax_img.imshow(rgb)
    ax_img.contour((gt == bkg_cls).astype(float), levels=[0.5], colors=["yellow"], linewidths=1.5)
    ax_img.contour((gt == tgt_cls).astype(float), levels=[0.5], colors=["red"], linewidths=1.5)
    from matplotlib.patches import Patch
    ax_img.legend(handles=[Patch(facecolor="yellow", label=f"Background (class {bkg_cls})"),
                            Patch(facecolor="red",    label=f"Target (class {tgt_cls})")],
                  loc="upper right", fontsize=9)
    ax_img.set_title("False-color image", fontsize=12)
    ax_img.axis("off")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="real_datasets/pavia-u.mat")
    parser.add_argument("--config",     default="ae_vs_pca_config.yaml")
    parser.add_argument("--bkg",        type=int, default=None)
    parser.add_argument("--target",     type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.bkg    is not None: cfg["_bkg_cls"]    = args.bkg
    if args.target is not None: cfg["_target_cls"] = args.target
    if args.no_display:         cfg["_no_display"]  = True

    run_experiment(cfg, args.dataset)
