"""
GMM IID Experiment.

Background is a 3-component GMM with shared covariance (from PCA image stats).
Component means come from 3 user-drawn polygons. Weights are fixed: [0.6, 0.3, 0.1].

Detectors evaluated:
  - Oracle       : LRT with true GMM params and known amplitude θ
  - AMF          : Gaussian assumption, sample mean/cov from secondary data
  - Fitted GMM   : GMM fitted via EM on secondary data (K=3 known), LRT with fitted params
  - DSM σ=...    : nonlinear MLP trained with DSM objective, LMP statistic

Usage:
    python gmm_iid_experiment.py --dataset real_datasets/pavia-u.mat
"""

import argparse
import json
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp as _logsumexp

from dsm_model import ScoreNet, train_dsm, compute_scores
from visualize_dataset import (load_dataset, false_color, select_polygon,
                                get_pixels_in_polygon, overlay_polygons)
from gaussian_iid_experiment import detector_amf, detector_reg_amf, detector_dsm


# ---------------------------------------------------------------------------
# GMM helpers
# ---------------------------------------------------------------------------

def gmm_sample(means, cov, weights, n, rng):
    """Sample n points from a GMM with shared covariance."""
    K, d = means.shape
    counts = rng.multinomial(n, weights)
    parts = [rng.multivariate_normal(means[k], cov, size=counts[k], method='eigh')
             for k in range(K) if counts[k] > 0]
    samples = np.vstack(parts)
    rng.shuffle(samples)
    return samples


def gmm_log_prob(y, means, cov, weights):
    """
    Log-likelihood log p(y) under a GMM with shared covariance.
    y: (n, d),  returns (n,).
    """
    K = len(weights)
    log_pi = np.log(weights + 1e-300)
    cov_inv = np.linalg.inv(cov)
    log_det = np.linalg.slogdet(cov)[1]
    d = y.shape[1]
    log_comps = np.zeros((len(y), K))
    for k in range(K):
        diff = y - means[k]
        maha = (diff @ cov_inv * diff).sum(axis=1)
        log_comps[:, k] = log_pi[k] - 0.5 * (d * np.log(2 * np.pi) + log_det + maha)
    return logsumexp(log_comps, axis=1)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def detector_oracle_gmm(test_data, means, cov, weights, s, amplitude):
    """
    LRT oracle: knows true GMM params and amplitude θ.
    Under additive model H1: y = θs + w  →  w = y - θs
    T(y) = log p_GMM(y - θs) - log p_GMM(y)
    """
    log_p_H1 = gmm_log_prob(test_data - amplitude * s, means, cov, weights)
    log_p_H0 = gmm_log_prob(test_data,                 means, cov, weights)
    return log_p_H1 - log_p_H0


def gmm_log_prob_full(y, means, covs, weights):
    """
    Log-likelihood under a GMM with per-component covariances.
    covs: list of K (d×d) arrays — one per component.
    """
    K  = len(weights)
    d  = y.shape[1]
    log_pi    = np.log(weights + 1e-300)
    log_comps = np.zeros((len(y), K))
    for k in range(K):
        cov_inv_k = np.linalg.inv(covs[k])
        log_det_k = np.linalg.slogdet(covs[k])[1]
        diff      = y - means[k]
        maha      = (diff @ cov_inv_k * diff).sum(axis=1)
        log_comps[:, k] = log_pi[k] - 0.5 * (d * np.log(2 * np.pi) + log_det_k + maha)
    return logsumexp(log_comps, axis=1)


def _gmm_estimate_theta(y, s, means, cov_invs, s_prec_k, s_prec_mu_k, weights_log,
                        max_iter=20, tol=1e-6):
    """
    Fixed-point EM iteration to estimate amplitude θ for a single test point y.

    MLE condition (derivative of log p_GMM(y-θs) w.r.t. θ = 0):
        θ = [Σ_k r_k · s^T Σ_k^{-1}(y - μ_k)] / [Σ_k r_k · s^T Σ_k^{-1} s]

    where r_k are responsibilities at (y - θs).
    Each iteration is closed-form given r_k. Converges in ~5–10 steps.
    Clipped to θ ≥ 0 (one-sided H1).

    Pre-computed per call:
        s_prec_k[k]    = s^T Σ_k^{-1} s        (scalar)
        s_prec_mu_k[k] = s^T Σ_k^{-1} (y-μ_k)  (scalar, depends on y)
    """
    K = len(means)
    d = y.shape[0]
    theta = 0.0

    for _ in range(max_iter):
        # E-step: responsibilities at (y - θs)
        z = y - theta * s
        log_r = np.empty(K)
        for k in range(K):
            diff     = z - means[k]
            maha     = float(diff @ cov_invs[k] @ diff)
            log_det  = np.linalg.slogdet(np.linalg.inv(cov_invs[k]))[1]
            log_r[k] = weights_log[k] - 0.5 * (d * np.log(2*np.pi) + log_det + maha)
        log_r -= _logsumexp(log_r)
        r = np.exp(log_r)                                     # (K,)

        # M-step: closed-form θ given r_k
        numer = float(np.sum(r * (s_prec_mu_k - theta * s_prec_k)))
        denom = float(np.sum(r * s_prec_k)) + 1e-12
        theta_new = max(0.0, theta + numer / denom)           # one Newton step → converges fast

        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    return theta


def detector_gmm_glrt(test_data, train_data, s, K=3,
                      theta_min=0.0, theta_max=2.0, theta_steps=50):
    """
    GLRT with fitted GMM — amplitude θ estimated via grid search.

    Fits GMM via EM (sklearn, full per-component covariances).
    For each θ in the grid, evaluates log p(y - θs) for all test points at once
    (one vectorized score_samples call per grid point — very efficient).
    Picks θ̂(y) = argmax_θ log p(y - θs) per test point.

    T(y) = log p_fitted(y - θ̂s) - log p_fitted(y)
    """
    gm = GaussianMixture(n_components=K, covariance_type='full', n_init=5, random_state=0)
    gm.fit(train_data)

    theta_grid   = np.linspace(theta_min, theta_max, theta_steps)
    log_p_H0     = gm.score_samples(test_data)             # (n_test,)

    # Stack log p(y - θs) for each θ → shape (n_test, theta_steps)
    log_p_grid = np.column_stack([
        gm.score_samples(test_data - theta * s)
        for theta in theta_grid
    ])

    # For each test point pick the best θ
    log_p_H1 = log_p_grid.max(axis=1)                      # (n_test,)
    return log_p_H1 - log_p_H0


def detector_fitted_gmm(test_data, train_data, s, amplitude, K=3):
    """
    Fit GMM via EM on secondary data (full per-component covariances),
    then apply LRT with the fitted params and known θ.
    T(y) = log p_fitted(y - θs) - log p_fitted(y)
    """
    gm = GaussianMixture(n_components=K, covariance_type='full', n_init=5, random_state=0)
    gm.fit(train_data)

    weights_fit = gm.weights_
    means_fit   = gm.means_
    d           = train_data.shape[1]
    # Keep per-component covariances — do NOT average them
    covs_fit    = [gm.covariances_[k] + 1e-8 * np.eye(d) for k in range(K)]

    log_p_H1 = gmm_log_prob_full(test_data - amplitude * s, means_fit, covs_fit, weights_fit)
    log_p_H0 = gmm_log_prob_full(test_data,                 means_fit, covs_fit, weights_fit)
    return log_p_H1 - log_p_H0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg, dataset_path):
    exp_cfg   = cfg["experiment"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    results_dir = exp_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    pca_dim        = exp_cfg["pca_dim"]
    gmm_weights    = np.array(exp_cfg["gmm_weights"], dtype=float)
    gmm_weights   /= gmm_weights.sum()
    n_values       = exp_cfg["n_values"]
    amplitude      = exp_cfg["amplitude"]
    n_test         = exp_cfg["n_test"]
    target_fraction= exp_cfg["target_fraction"]
    sigma_values   = train_cfg["sigma_values"]
    base_epochs    = train_cfg["base_epochs"]
    n_seeds        = exp_cfg.get("n_seeds", 1)
    K = len(gmm_weights)

    # ------------------------------------------------------------------
    # Step 1 — Load dataset and select polygons
    # ------------------------------------------------------------------
    print(f"\n=== Loading dataset: {dataset_path} ===")
    data, _ = load_dataset(dataset_path)
    rgb = false_color(data)

    print(f"\n[Step 1a] Draw {K} BACKGROUND polygons (one per GMM component).")
    bkg_masks, mu_raw_list = [], []
    for k in range(K):
        mask = select_polygon(rgb,
            title=f"Background component {k+1}/{K}  weight={gmm_weights[k]:.2f} — draw polygon, close when done")
        pixels = get_pixels_in_polygon(data, mask)
        if len(pixels) == 0:
            raise RuntimeError(f"No pixels selected for background component {k+1}.")
        bkg_masks.append(mask)
        mu_raw_list.append(pixels.mean(axis=0))
        print(f"  Component {k+1}: {len(pixels)} pixels")

    print("\n[Step 1b] Draw the TARGET polygon.")
    tgt_mask  = select_polygon(rgb, title="Target region — draw polygon, close when done")
    tgt_pixels = get_pixels_in_polygon(data, tgt_mask)
    if len(tgt_pixels) == 0:
        raise RuntimeError("No target pixels selected.")
    s_raw  = tgt_pixels.mean(axis=0)
    s_raw /= np.linalg.norm(s_raw) + 1e-12
    print(f"  Target pixels: {len(tgt_pixels)}")

    all_masks  = bkg_masks + [tgt_mask]
    all_labels = [f"Bkg {k+1} (w={gmm_weights[k]:.2f})" for k in range(K)] + ["Target"]
    all_colors = ["yellow", "cyan", "lime", "red"]
    overlay_polygons(rgb, all_masks, labels=all_labels, colors=all_colors,
                     save_path=os.path.join(results_dir, "false_color_polygons.png"))

    # ------------------------------------------------------------------
    # Step 2 — PCA
    # ------------------------------------------------------------------
    print(f"\n[Step 2] PCA to {pca_dim} dimensions ...")
    H, W, B = data.shape
    all_pixels = data.reshape(-1, B).astype(np.float64)

    pca = PCA(n_components=pca_dim)
    pca.fit(all_pixels)

    means_pca  = np.array([pca.transform(m.reshape(1, -1))[0] for m in mu_raw_list])
    s_pca_raw  = pca.transform(s_raw.reshape(1, -1))[0]
    s_pca      = s_pca_raw / (np.linalg.norm(s_pca_raw) + 1e-12)

    all_pca   = pca.transform(all_pixels)
    Sigma_pca = np.cov(all_pca, rowvar=False)
    Sigma_pca = (Sigma_pca + Sigma_pca.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Sigma_pca)
    eigvals   = np.clip(eigvals, 1e-8, None)
    Sigma_pca = eigvecs @ np.diag(eigvals) @ eigvecs.T

    print(f"  GMM means shape: {means_pca.shape},  ||s_pca||={np.linalg.norm(s_pca):.4f}")

    # ------------------------------------------------------------------
    # Summary stats (no Fisher info — just geometry)
    # ------------------------------------------------------------------
    inter_mean_dist = np.mean([np.linalg.norm(means_pca[i] - means_pca[j])
                               for i in range(K) for j in range(i+1, K)])
    stats = {
        "pca_dim"             : int(pca_dim),
        "gmm_weights"         : list(np.round(gmm_weights, 4)),
        "amplitude"           : float(amplitude),
        "sigma_condition_num" : float(eigvals.max() / eigvals.min()),
        "sigma_trace"         : float(np.trace(Sigma_pca)),
        "mean_inter_dist"     : float(inter_mean_dist),
    }
    stats_path = os.path.join(results_dir, "gmm_stats.txt")
    with open(stats_path, "w") as f:
        f.write("=== GMM background statistics (PCA space) ===\n")
        for k, v in stats.items():
            f.write(f"  {k:<35} {v}\n")
    print("\n  GMM stats:")
    for k, v in stats.items():
        print(f"    {k:<35} {v}")

    # ------------------------------------------------------------------
    # Fixed test set (same for all n)
    # ------------------------------------------------------------------
    n_target  = max(1, int(round(n_test * target_fraction)))
    rng_test  = np.random.default_rng(seed=0)
    test_bkg  = gmm_sample(means_pca, Sigma_pca, gmm_weights, n_test, rng_test)
    labels    = np.zeros(n_test, dtype=int)
    tgt_idx   = rng_test.choice(n_test, size=n_target, replace=False)
    labels[tgt_idx] = 1
    test_data = test_bkg.copy()
    test_data[tgt_idx] += amplitude * s_pca

    # Oracle AUC — constant across n
    t_oracle       = detector_oracle_gmm(test_data, means_pca, Sigma_pca, gmm_weights, s_pca, amplitude)
    auc_oracle_fixed = float(roc_auc_score(labels, t_oracle))
    print(f"\nOracle AUC (fixed): {auc_oracle_fixed:.4f}")

    # ------------------------------------------------------------------
    # Loop over n, averaged over n_seeds
    # ------------------------------------------------------------------
    results = {}
    for n in n_values:
        print(f"\n=== n = {n} ===")

        seed_amf     = []
        seed_reg_amf = {str(s): [] for s in sigma_values}
        seed_fitted  = []
        seed_dsm     = {str(s): [] for s in sigma_values}

        for seed in range(n_seeds):
            rng        = np.random.default_rng(seed=seed * 10000 + n)
            train_data = gmm_sample(means_pca, Sigma_pca, gmm_weights, n, rng)

            seed_amf.append(roc_auc_score(labels, detector_amf(test_data, train_data, s_pca)))
            for sigma in sigma_values:
                seed_reg_amf[str(sigma)].append(
                    roc_auc_score(labels, detector_reg_amf(test_data, train_data, s_pca, sigma)))
            seed_fitted.append(roc_auc_score(labels,
                               detector_fitted_gmm(test_data, train_data, s_pca, amplitude, K=K)))

            for sigma in sigma_values:
                tag = str(sigma)
                print(f"  seed={seed}  σ={sigma}  epochs={base_epochs} ...", end=" ", flush=True)
                model = ScoreNet(
                    input_dim  = pca_dim,
                    hidden_dims= model_cfg.get("hidden_dims") or [],
                    activation = model_cfg.get("activation", "silu"),
                )
                model = train_dsm(model, train_data, sigma=sigma,
                                  lr=train_cfg["lr"], batch_size=train_cfg["batch_size"],
                                  epochs=base_epochs)
                torch.save(model.state_dict(),
                           os.path.join(results_dir, f"model_n{n}_sigma{sigma}_seed{seed}.pt"))
                auc = float(roc_auc_score(labels,
                            detector_dsm(test_data, train_data, model, s_pca)))
                seed_dsm[tag].append(auc)
                print(f"AUC={auc:.4f}")

        auc_dsm_mean     = {t: float(np.mean(v)) for t, v in seed_dsm.items()}
        auc_dsm_std      = {t: float(np.std(v))  for t, v in seed_dsm.items()}
        auc_reg_amf_mean = {t: float(np.mean(v)) for t, v in seed_reg_amf.items()}
        auc_reg_amf_std  = {t: float(np.std(v))  for t, v in seed_reg_amf.items()}

        results[n] = {
            "auc_oracle"         : auc_oracle_fixed,
            "auc_amf_mean"       : float(np.mean(seed_amf)),
            "auc_amf_std"        : float(np.std(seed_amf)),
            "auc_reg_amf_mean"   : auc_reg_amf_mean,
            "auc_reg_amf_std"    : auc_reg_amf_std,
            "auc_fitted_gmm_mean": float(np.mean(seed_fitted)),
            "auc_fitted_gmm_std" : float(np.std(seed_fitted)),
            "auc_dsm_mean"       : auc_dsm_mean,
            "auc_dsm_std"        : auc_dsm_std,
        }
        print(f"  Oracle={auc_oracle_fixed:.4f}  "
              f"AMF={results[n]['auc_amf_mean']:.4f}±{results[n]['auc_amf_std']:.4f}  "
              f"RegAMF={auc_reg_amf_mean}  "
              f"FittedGMM={results[n]['auc_fitted_gmm_mean']:.4f}±{results[n]['auc_fitted_gmm_std']:.4f}  "
              f"DSM={auc_dsm_mean}")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    _plot_auc(results, n_values, sigma_values, results_dir, stats,
              rgb=rgb, masks=all_masks, mask_labels=all_labels, mask_colors=all_colors)
    print("Done.")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_auc(results, n_values, sigma_values, results_dir, gmm_stats=None,
              rgb=None, masks=None, mask_labels=None, mask_colors=None):
    ns = [n for n in n_values if n in results]

    auc_oracle      = [results[n]["auc_oracle"]          for n in ns]
    auc_amf_mean    = [results[n]["auc_amf_mean"]        for n in ns]
    auc_amf_std     = [results[n]["auc_amf_std"]         for n in ns]
    auc_fitted_mean = [results[n]["auc_fitted_gmm_mean"] for n in ns]
    auc_fitted_std  = [results[n]["auc_fitted_gmm_std"]  for n in ns]

    _palette   = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    dsm_colors = {str(s): _palette[i % len(_palette)] for i, s in enumerate(sigma_values)}

    fig, (ax_auc, ax_img) = plt.subplots(1, 2, figsize=(13, 5),
                                          gridspec_kw={"width_ratios": [1.4, 1]})

    # --- Left: AUC vs n ---
    ax_auc.plot(ns, auc_oracle, color="black", linewidth=2, marker="o", label="Oracle (LRT, true GMM)")
    ax_auc.errorbar(ns, auc_amf_mean, yerr=auc_amf_std,
                    color="tab:blue", linewidth=2, marker="s", capsize=4, label="AMF")
    ax_auc.errorbar(ns, auc_fitted_mean, yerr=auc_fitted_std,
                    color="tab:gray", linewidth=2, marker="D", capsize=4, linestyle="--",
                    label="Fitted GMM (EM + LRT)")
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
    ax_auc.set_title("GMM IID — AUC vs n", fontsize=12)
    ax_auc.legend(fontsize=9)
    ax_auc.grid(True, alpha=0.4)
    ax_auc.set_xticks(ns)
    ax_auc.tick_params(axis='x', rotation=45)

    if gmm_stats:
        w = gmm_stats.get("gmm_weights", [])
        info = (
            f"d={gmm_stats.get('pca_dim')}  K=3  weights={[round(x,2) for x in w]}\n"
            f"θ={gmm_stats.get('amplitude')}  "
            f"cond(Σ)={gmm_stats.get('sigma_condition_num', 0):.1f}  "
            f"inter-dist={gmm_stats.get('mean_inter_dist', 0):.3f}"
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
    parser = argparse.ArgumentParser(description="GMM IID Experiment")
    parser.add_argument("--dataset", default="real_datasets/pavia-u.mat")
    parser.add_argument("--config",  default="gmm_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg, args.dataset)
