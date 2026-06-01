"""
Gaussian IID Experiment (Section 7.1 of the paper).

Pipeline:
  1. User marks background polygon → derive Gaussian mean
  2. User marks target polygon → compute normalized target signature
  3. PCA to d=pca_dim dimensions using all image pixels
  4. For each n: sample n Gaussian background samples, train 3 ScoreNets (sigma 0.5/1.0/2.0)
  5. Generate test data (background + 10% with additive target)
  6. Evaluate Oracle, AMF, DSM×3 detectors via AUC
  7. Save plots and results

Usage:
    python gaussian_iid_experiment.py --dataset real_datasets/Sandiego.mat
"""

import argparse
import json
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from dsm_model import ScoreNet, train_dsm, compute_scores, select_sigma_loo
from visualize_dataset import (load_dataset, false_color, select_polygon,
                                get_pixels_in_polygon, overlay_polygons)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detector_oracle(test_data, mu, Sigma, s, amplitude):
    """
    Two-simple-hypothesis LRT under additive model with known params.
    T(y) = log p(y|H1) - log p(y|H0)
         = amplitude * s^T Σ^{-1} (y - mu) - 0.5 * amplitude^2 * s^T Σ^{-1} s
    (the constant term doesn't affect ROC, but we keep it for correctness)
    """
    Sigma_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(len(s)))
    Si_s = Sigma_inv @ s
    scores = amplitude * (test_data - mu) @ Si_s - 0.5 * amplitude**2 * (s @ Si_s)
    return scores


def detector_amf(test_data, train_data, s):
    """
    Adaptive Matched Filter: T(y) = s^T Σ̂^{-1}(y - μ̂) / sqrt(s^T Σ̂^{-1} s)
    Covariance is regularized via eigenvalue clipping to handle ill-conditioned matrices.
    """
    d = len(s)
    mu_hat = train_data.mean(axis=0)
    Sigma_hat = np.cov(train_data, rowvar=False)
    if Sigma_hat.ndim == 0:
        Sigma_hat = np.array([[float(Sigma_hat)]])
    # Symmetrize and clip negative eigenvalues for a stable inverse
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
    eigvals = np.clip(eigvals, eigvals.max() * 1e-12, None)  # relative threshold
    Sigma_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    Si_s = Sigma_inv @ s
    norm = np.sqrt(s @ Si_s + 1e-12)
    scores = (test_data - mu_hat) @ Si_s / norm
    return scores


def detector_reg_amf(test_data, train_data, s, sigma):
    """
    Diagonal-loaded AMF — Theorem 1 of the paper.

        T_s(y) = s^T (Σ̂+σ²I)^{-1}(y-μ̂) / sqrt(s^T (Σ̂+σ²I)^{-1} Σ̂ (Σ̂+σ²I)^{-1} s)

    σ is a fixed loading parameter (same grid as DSM).
    """
    mu_hat    = train_data.mean(axis=0)
    Sigma_hat = np.cov(train_data, rowvar=False)
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2

    Sigma_reg = Sigma_hat + float(sigma) ** 2 * np.eye(len(s))
    Sigma_inv = np.linalg.inv(Sigma_reg)

    Si_s  = Sigma_inv @ s
    denom = np.sqrt(Si_s @ Sigma_hat @ Si_s + 1e-12)
    return (test_data - mu_hat) @ Si_s / denom


def detector_dsm(test_data, train_data, model, s):
    """
    DSM-LMP statistic: T_s(y) = -s^T(ψ̂(y) - z̄) / sqrt(s^T Ĉ_ψ s)
    """
    z_train = compute_scores(model, train_data)       # (n, d)
    z_bar = z_train.mean(axis=0)                       # (d,)
    C_psi = np.cov(z_train, rowvar=False)              # (d, d)
    if C_psi.ndim == 0:
        C_psi = np.array([[C_psi]])

    z_test = compute_scores(model, test_data)          # (n_test, d)
    centered = z_test - z_bar                          # (n_test, d)
    C_s = s @ C_psi @ s
    norm = np.sqrt(max(C_s, 1e-12))
    scores = -(centered @ s) / norm
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg, dataset_path):
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    results_dir = exp_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    pca_dim = exp_cfg["pca_dim"]
    n_values = exp_cfg["n_values"]
    amplitude = exp_cfg["amplitude"]
    n_test = exp_cfg["n_test"]
    target_fraction = exp_cfg["target_fraction"]
    sigma_values = train_cfg["sigma_values"]
    base_epochs = train_cfg["base_epochs"]
    n_seeds = exp_cfg.get("n_seeds", 1)

    # ------------------------------------------------------------------
    # Step 1 — Load dataset and get polygons
    # ------------------------------------------------------------------
    print(f"\n=== Loading dataset: {dataset_path} ===")
    data, _ = load_dataset(dataset_path)
    rgb = false_color(data)

    print("\n[Step 1a] Draw the BACKGROUND polygon.")
    bkg_mask = select_polygon(rgb, title="Background region — draw polygon, close when done")
    bkg_pixels = get_pixels_in_polygon(data, bkg_mask)
    if len(bkg_pixels) == 0:
        raise RuntimeError("No background pixels selected.")
    mu_raw = bkg_pixels.mean(axis=0)
    print(f"  Background pixels: {len(bkg_pixels)}, mean shape: {mu_raw.shape}")

    print("\n[Step 1b] Draw the TARGET polygon.")
    tgt_mask = select_polygon(rgb, title="Target region — draw polygon, close when done")
    tgt_pixels = get_pixels_in_polygon(data, tgt_mask)
    if len(tgt_pixels) == 0:
        raise RuntimeError("No target pixels selected.")
    s_raw = tgt_pixels.mean(axis=0)
    s_raw = s_raw / (np.linalg.norm(s_raw) + 1e-12)   # normalize to unit norm
    print(f"  Target pixels: {len(tgt_pixels)}, ||s||={np.linalg.norm(s_raw):.4f}")

    # Save false-color image with polygons
    overlay_polygons(rgb, [bkg_mask, tgt_mask],
                     labels=["Background", "Target"],
                     colors=["yellow", "red"],
                     save_path=os.path.join(results_dir, "false_color_polygons.png"))

    # ------------------------------------------------------------------
    # Step 2 — PCA on all image pixels
    # ------------------------------------------------------------------
    print(f"\n[Step 2] PCA to {pca_dim} dimensions ...")
    H, W, B = data.shape
    all_pixels = data.reshape(-1, B).astype(np.float64)

    pca = PCA(n_components=pca_dim)
    pca.fit(all_pixels)

    mu_pca = pca.transform(mu_raw.reshape(1, -1))[0]
    s_pca_raw = pca.transform(s_raw.reshape(1, -1))[0]
    s_pca = s_pca_raw / (np.linalg.norm(s_pca_raw) + 1e-12)   # re-normalize after PCA

    # PCA covariance from all image pixels in PCA space
    all_pca = pca.transform(all_pixels)
    Sigma_pca = np.cov(all_pca, rowvar=False)
    # Ensure strict PSD: symmetrize then clip negative eigenvalues
    Sigma_pca = (Sigma_pca + Sigma_pca.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Sigma_pca)
    eigvals = np.clip(eigvals, 1e-8, None)
    Sigma_pca = eigvecs @ np.diag(eigvals) @ eigvecs.T

    print(f"  μ_pca shape: {mu_pca.shape}, ||s_pca||={np.linalg.norm(s_pca):.4f} (should be ~1.0)")

    # Gaussian summary stats
    eigvals_sigma = np.linalg.eigvalsh(Sigma_pca)
    gauss_stats = {
        "pca_dim": int(pca_dim),
        "mu_norm": float(np.linalg.norm(mu_pca)),
        "sigma_eigenvalues_min": float(eigvals_sigma.min()),
        "sigma_eigenvalues_max": float(eigvals_sigma.max()),
        "sigma_condition_number": float(eigvals_sigma.max() / eigvals_sigma.min()),
        "sigma_trace": float(np.trace(Sigma_pca)),
        "s_dot_mu": float(s_pca @ mu_pca),
        "amplitude": float(amplitude),
        "snr_approx": float(amplitude * np.sqrt(s_pca @ np.linalg.solve(Sigma_pca, s_pca))),
    }
    stats_path = os.path.join(results_dir, "gaussian_stats.txt")
    os.makedirs(results_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        f.write("=== Gaussian background statistics (PCA space) ===\n")
        for k, v in gauss_stats.items():
            f.write(f"  {k:<35} {v:.6g}\n")
    print("\n  Gaussian stats:")
    for k, v in gauss_stats.items():
        print(f"    {k:<35} {v:.6g}")
    print(f"  Saved to {stats_path}")

    # ------------------------------------------------------------------
    # Generate fixed test data (shared across all n, so Oracle AUC is constant)
    # ------------------------------------------------------------------
    n_target = max(1, int(round(n_test * target_fraction)))
    rng_test = np.random.default_rng(seed=0)
    test_bkg = rng_test.multivariate_normal(mu_pca, Sigma_pca, size=n_test, method='eigh')
    labels = np.zeros(n_test, dtype=int)
    target_indices = rng_test.choice(n_test, size=n_target, replace=False)
    labels[target_indices] = 1
    test_data = test_bkg.copy()
    test_data[target_indices] += amplitude * s_pca

    # Oracle AUC is constant (true params, fixed test set) — compute once
    t_oracle = detector_oracle(test_data, mu_pca, Sigma_pca, s_pca, amplitude)
    auc_oracle_fixed = float(roc_auc_score(labels, t_oracle))
    print(f"\nOracle AUC (fixed): {auc_oracle_fixed:.4f}")

    # ------------------------------------------------------------------
    # Steps 3–5 — Loop over n, averaged over n_seeds
    # ------------------------------------------------------------------
    results = {}

    for n in n_values:
        print(f"\n=== n = {n} ===")

        seed_amf       = []
        seed_reg_amf   = {str(s): [] for s in sigma_values}
        seed_dsm       = {str(s): [] for s in sigma_values}
        seed_dsm_gcv   = []
        seed_gcv_sigma = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed=seed * 10000 + n)
            train_data = rng.multivariate_normal(mu_pca, Sigma_pca, size=n, method='eigh')

            seed_amf.append(roc_auc_score(labels, detector_amf(test_data, train_data, s_pca)))
            for sigma in sigma_values:
                seed_reg_amf[str(sigma)].append(
                    roc_auc_score(labels, detector_reg_amf(test_data, train_data, s_pca, sigma)))

            for sigma in sigma_values:
                sigma_tag = str(sigma)
                print(f"  seed={seed}  σ={sigma}  epochs={base_epochs} ...", end=" ", flush=True)
                model = ScoreNet(
                    input_dim=pca_dim,
                    hidden_dims=model_cfg.get("hidden_dims") or [],
                    activation=model_cfg.get("activation", "silu")
                )
                model = train_dsm(model, train_data, sigma=sigma,
                                  lr=train_cfg["lr"], batch_size=train_cfg["batch_size"],
                                  epochs=base_epochs)
                torch.save(model.state_dict(),
                           os.path.join(results_dir, f"model_n{n}_sigma{sigma}_seed{seed}.pt"))
                auc = float(roc_auc_score(labels, detector_dsm(test_data, train_data, model, s_pca)))
                seed_dsm[sigma_tag].append(auc)
                print(f"AUC={auc:.4f}")

            # GCV sigma selection + train one model with σ*
            sigma_star, loo_curve = select_sigma_loo(train_data, sigma_values)
            seed_gcv_sigma.append(sigma_star)
            print(f"  seed={seed}  GCV σ*={sigma_star}  LOO losses: "
                  f"{ {s: f'{v:.4f}' for s, v in loo_curve.items()} }")
            model_gcv = ScoreNet(input_dim=pca_dim,
                                 hidden_dims=model_cfg.get("hidden_dims") or [],
                                 activation=model_cfg.get("activation", "silu"))
            model_gcv = train_dsm(model_gcv, train_data, sigma=sigma_star,
                                  lr=train_cfg["lr"], batch_size=train_cfg["batch_size"],
                                  epochs=base_epochs)
            torch.save(model_gcv.state_dict(),
                       os.path.join(results_dir, f"model_n{n}_sigmaGCV_seed{seed}.pt"))
            auc_gcv = float(roc_auc_score(labels, detector_dsm(test_data, train_data, model_gcv, s_pca)))
            seed_dsm_gcv.append(auc_gcv)
            print(f"  DSM(GCV σ*={sigma_star}) AUC={auc_gcv:.4f}")

        auc_dsm_mean     = {t: float(np.mean(v)) for t, v in seed_dsm.items()}
        auc_dsm_std      = {t: float(np.std(v))  for t, v in seed_dsm.items()}
        auc_reg_amf_mean = {t: float(np.mean(v)) for t, v in seed_reg_amf.items()}
        auc_reg_amf_std  = {t: float(np.std(v))  for t, v in seed_reg_amf.items()}

        results[n] = {
            "auc_oracle"        : auc_oracle_fixed,
            "auc_amf_mean"      : float(np.mean(seed_amf)),
            "auc_amf_std"       : float(np.std(seed_amf)),
            "auc_reg_amf_mean"  : auc_reg_amf_mean,
            "auc_reg_amf_std"   : auc_reg_amf_std,
            "auc_dsm_mean"      : auc_dsm_mean,
            "auc_dsm_std"       : auc_dsm_std,
            "auc_dsm_gcv_mean"  : float(np.mean(seed_dsm_gcv)),
            "auc_dsm_gcv_std"   : float(np.std(seed_dsm_gcv)),
            "gcv_sigma_chosen"  : seed_gcv_sigma,
        }
        print(f"  Oracle={auc_oracle_fixed:.4f}  "
              f"AMF={results[n]['auc_amf_mean']:.4f}±{results[n]['auc_amf_std']:.4f}  "
              f"RegAMF={auc_reg_amf_mean}  "
              f"DSM(GCV)={results[n]['auc_dsm_gcv_mean']:.4f}±{results[n]['auc_dsm_gcv_std']:.4f}  "
              f"GCV σ*={seed_gcv_sigma}")

    # Save JSON
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ------------------------------------------------------------------
    # Step 6 — Plot AUC vs n
    # ------------------------------------------------------------------
    _plot_auc(results, n_values, sigma_values, results_dir, gauss_stats,
              rgb=rgb, masks=[bkg_mask, tgt_mask],
              mask_labels=["Background", "Target"],
              mask_colors=["yellow", "red"])
    print("Done.")


def _plot_auc(results, n_values, sigma_values, results_dir, gauss_stats=None,
              rgb=None, masks=None, mask_labels=None, mask_colors=None):
    ns = [n for n in n_values if n in results]

    auc_oracle = [results[n]["auc_oracle"] for n in ns]

    _palette   = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
    dsm_colors = {str(s): _palette[i % len(_palette)] for i, s in enumerate(sigma_values)}

    fig, (ax_auc, ax_img) = plt.subplots(1, 2, figsize=(13, 5),
                                          gridspec_kw={"width_ratios": [1.4, 1]})

    auc_amf_mean     = [results[n]["auc_amf_mean"]     for n in ns]
    auc_amf_std      = [results[n]["auc_amf_std"]      for n in ns]
    auc_gcv_mean = [results[n]["auc_dsm_gcv_mean"] for n in ns]
    auc_gcv_std  = [results[n]["auc_dsm_gcv_std"]  for n in ns]

    # --- Left: AUC vs n ---
    ax_auc.plot(ns, auc_oracle, color="black", linewidth=2, marker="o", label="Oracle")
    ax_auc.errorbar(ns, auc_amf_mean, yerr=auc_amf_std,
                    color="tab:blue", linewidth=2, marker="s", capsize=4, label="AMF")
    ax_auc.errorbar(ns, auc_gcv_mean, yerr=auc_gcv_std,
                    color="tab:olive", linewidth=2, marker="*", markersize=9, capsize=4,
                    linestyle="--", label="DSM (GCV σ*)")
    for sigma in sigma_values:
        tag = str(sigma)
        mean = [results[n]["auc_dsm_mean"].get(tag, float("nan")) for n in ns]
        std  = [results[n]["auc_dsm_std"].get(tag,  float("nan")) for n in ns]
        ax_auc.errorbar(ns, mean, yerr=std, color=dsm_colors[tag],
                        linewidth=2, marker="^", capsize=4, label=f"DSM σ={sigma}")
        reg_mean = [results[n]["auc_reg_amf_mean"].get(tag, float("nan")) for n in ns]
        reg_std  = [results[n]["auc_reg_amf_std"].get(tag,  float("nan")) for n in ns]
        ax_auc.errorbar(ns, reg_mean, yerr=reg_std, color=dsm_colors[tag],
                        linewidth=1.5, marker="P", capsize=4, linestyle="--",
                        label=f"Reg.AMF σ={sigma}")

    ax_auc.set_xlabel("n (secondary samples)", fontsize=12)
    ax_auc.set_ylabel("AUC", fontsize=12)
    ax_auc.set_title("Gaussian IID — AUC vs n", fontsize=12)
    ax_auc.legend(fontsize=9)
    ax_auc.grid(True, alpha=0.4)
    ax_auc.set_xticks(ns)
    ax_auc.tick_params(axis='x', rotation=45)

    if gauss_stats:
        info = (
            f"d={gauss_stats['pca_dim']}  θ={gauss_stats['amplitude']}  "
            f"||μ||={gauss_stats['mu_norm']:.2f}\n"
            f"Σ eigvals: [{gauss_stats['sigma_eigenvalues_min']:.2g}, "
            f"{gauss_stats['sigma_eigenvalues_max']:.2g}]  "
            f"cond={gauss_stats['sigma_condition_number']:.1f}\n"
            f"SNR≈{gauss_stats['snr_approx']:.3f}"
        )
        ax_auc.text(0.02, 0.04, info, transform=ax_auc.transAxes, fontsize=8,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    # --- Right: false-color image with polygon outlines ---
    if rgb is not None:
        ax_img.imshow(rgb)
        if masks and mask_colors:
            from matplotlib.patches import Patch
            handles = []
            for mask, label, color in zip(masks, mask_labels or [], mask_colors):
                ax_img.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2)
                handles.append(Patch(facecolor=color, edgecolor=color, label=label))
            ax_img.legend(handles=handles, loc="upper right", fontsize=8,
                          framealpha=0.7, edgecolor="white")
        ax_img.set_title("False-color image", fontsize=12)
        ax_img.axis("off")
    else:
        ax_img.axis("off")

    fig.tight_layout()
    save_path = os.path.join(results_dir, "result_figure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.show(block=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian IID Experiment")
    parser.add_argument("--dataset", default="real_datasets/pavia-u.mat",
                        help="Path to .mat hyperspectral dataset")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg, args.dataset)
