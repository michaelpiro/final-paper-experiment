"""
Sigma selection experiment for DSM.

Trains one ScoreNet per sigma in a grid, then evaluates several
data-driven sigma selection methods and compares them against the
oracle (best sigma by test AUC).

Selection methods tested:
  1. Held-out DSM loss      — minimize E[||ψ(x̃) - (x-x̃)/σ²||²] on val set
  2. Denoising L2           — minimize ||D(x̃)-x||²  (= σ⁴ × DSM loss, biased toward small σ)
  3. Normalized denoising   — minimize ||D(x̃)-x||² / σ⁴  (equivalent to DSM loss, unbiased)
  4. Median heuristic       — σ = median(pairwise distances) / √2  (snapped to grid)
  5. Silverman's rule       — σ = 1.06 · mean_std · n^{-1/5}  (trains own model)
  6. Scott's rule           — σ = mean_std · n^{-1/(d+4)}      (trains own model)

Silverman and Scott estimate σ directly from data and train a dedicated model,
so their result reflects how well the formula estimates the true optimal σ.

Pipeline mirrors real_data_experiment.py exactly (PCA on all pixels).

Usage:
    python sigma_selection_experiment.py --dataset real_datasets/pavia-u.mat \
        --bkg 2 --target 9 --no-display
"""

import argparse
import json
import os
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

from dsm_model import ScoreNet, train_dsm, dsm_loss, compute_scores
from visualize_dataset import false_color
from gaussian_iid_experiment import detector_dsm, detector_amf
from real_data_experiment import show_class_map


# ---------------------------------------------------------------------------
# Sigma selection methods
# ---------------------------------------------------------------------------

def select_by_dsm_loss(models: dict, val_data: np.ndarray) -> tuple:
    """Held-out DSM loss on val set for each trained model."""
    losses = {}
    for sigma, model in models.items():
        model.eval()
        X = torch.tensor(val_data, dtype=torch.float32)
        with torch.no_grad():
            loss = dsm_loss(model, X, sigma).item()
        losses[sigma] = loss
    best = min(losses, key=losses.get)
    return best, losses


def select_by_denoising_l2(models: dict, val_data: np.ndarray, normalize: bool) -> tuple:
    """
    For each model, add noise σ to val samples, denoise via Tweedie, measure L2.
        D(x̃) = x̃ + σ² · ψ_η(x̃)
        score = mean ||D(x̃) - x||²         (normalize=False)
        score = mean ||D(x̃) - x||² / σ⁴   (normalize=True  → equivalent to DSM loss)
    """
    scores = {}
    rng = np.random.default_rng(seed=99)
    for sigma, model in models.items():
        model.eval()
        eps   = rng.standard_normal(val_data.shape).astype(np.float32) * sigma
        x_noisy = val_data + eps
        X_noisy = torch.tensor(x_noisy, dtype=torch.float32)
        with torch.no_grad():
            psi   = model(X_noisy).numpy()
        denoised = x_noisy + sigma**2 * psi          # Tweedie
        l2 = np.mean(np.sum((denoised - val_data)**2, axis=1))
        scores[sigma] = l2 / (sigma**4) if normalize else l2
    best = min(scores, key=scores.get)
    return best, scores


def ism_loss_hutchinson(model: ScoreNet, val_data: np.ndarray,
                        n_hutchinson: int = 10) -> float:
    """
    Implicit Score Matching loss estimated via Hutchinson's trace estimator.

        ISM(ψ; x) = E[ ½||ψ(x)||² + tr(∂ψ/∂x) ]

    tr(∂ψ/∂x) ≈ (1/K) Σ_k  v_k^T (∂ψ/∂x · v_k)   where v_k ~ N(0, I)

    Each Hutchinson sample requires one forward + one backward pass.
    Evaluated on clean val_data (no noise added).
    """
    model.eval()
    X = torch.tensor(val_data, dtype=torch.float32)
    total_ism = 0.0

    for _ in range(n_hutchinson):
        x = X.clone().requires_grad_(True)
        psi = model(x)                                        # (n, d)
        norm_sq = 0.5 * (psi ** 2).sum(dim=-1).mean()        # ½ E[||ψ||²]

        # Hutchinson: tr(J) ≈ v^T J v  with v ~ N(0,I)
        v = torch.randn_like(psi)
        # scalar = v^T ψ, then grad w.r.t. x gives J^T v
        vjp = torch.autograd.grad((psi * v).sum(), x, create_graph=False)[0]  # (n, d)
        div_est = (vjp * v).sum(dim=-1).mean()                # E[v^T J v]

        total_ism += (norm_sq + div_est).item()

    return total_ism / n_hutchinson


def select_by_ism_loss(models: dict, val_data: np.ndarray,
                       n_hutchinson: int = 10) -> tuple:
    """Pick the model with lowest ISM loss on clean val data."""
    losses = {}
    for sigma, model in models.items():
        losses[sigma] = ism_loss_hutchinson(model, val_data, n_hutchinson)
    best = min(losses, key=losses.get)
    return best, losses


def select_by_median_heuristic(train_data: np.ndarray) -> float:
    """σ = median(pairwise L2 distances) / √2  (Gretton et al. 2012)."""
    dists = pdist(train_data)
    return float(np.median(dists) / np.sqrt(2))


def select_by_silverman(train_data: np.ndarray) -> float:
    """σ = 1.06 · mean_std · n^{-1/5}  (Silverman's rule of thumb)."""
    n, d = train_data.shape
    stds = train_data.std(axis=0)
    return float(1.06 * stds.mean() * n**(-1.0 / 5.0))


def select_by_scott(train_data: np.ndarray) -> float:
    """σ = mean_std · n^{-1/(d+4)}  (Scott's rule)."""
    n, d = train_data.shape
    stds = train_data.std(axis=0)
    return float(stds.mean() * n**(-1.0 / (d + 4.0)))


def snap_to_grid(value: float, grid: list) -> float:
    """Return the grid value closest to `value`."""
    return min(grid, key=lambda s: abs(s - value))


def train_or_load(sigma: float, pca_dim: int, hidden_dims: list,
                  train_data: np.ndarray, base_epochs: int, lr: float,
                  batch_size: int, weight_decay: float, ckpt: str) -> ScoreNet:
    """Train a ScoreNet for the given sigma, or load from checkpoint if it exists."""
    model = ScoreNet(input_dim=pca_dim, hidden_dims=hidden_dims, activation="silu")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
        print(f"  loaded from {ckpt}")
    else:
        model = train_dsm(model, train_data, sigma=sigma,
                          lr=lr, batch_size=batch_size, epochs=base_epochs,
                          weight_decay=weight_decay, print_every=base_epochs // 5)
        torch.save(model.state_dict(), ckpt)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict, dataset_path: str):
    results_dir   = cfg["results_dir"]
    pca_dim       = cfg["pca_dim"]
    sigma_grid    = cfg["sigma_grid"]
    amplitude     = cfg["amplitude"]
    tgt_fraction  = cfg["target_fraction"]
    total_samples = cfg.get("total_samples", 3000)
    test_fraction = cfg.get("test_fraction", 0.2)
    val_fraction  = cfg.get("val_fraction", 0.1)   # fraction of train used for val
    base_epochs   = cfg["base_epochs"]
    hidden_dims   = cfg.get("hidden_dims", [64, 64])
    lr            = cfg["lr"]
    weight_decay  = cfg.get("weight_decay", 1e-4)
    batch_size    = cfg["batch_size"]
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and normalize
    # ------------------------------------------------------------------
    print(f"\n=== Loading {dataset_path} ===")
    import scipy.io
    mat  = scipy.io.loadmat(dataset_path)
    data = mat["data"].astype(np.float64)
    gt   = mat["map"].astype(int)

    lo, hi = data.min(), data.max()
    data = (data - lo) / (hi - lo)

    rgb     = false_color(data)
    H, W, B = data.shape
    all_flat = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    if not cfg.get("_no_display", False):
        show_class_map(rgb, gt)

    if "_bkg_cls" in cfg and "_target_cls" in cfg:
        bkg_cls = cfg["_bkg_cls"]
        tgt_cls = cfg["_target_cls"]
    else:
        bkg_cls = int(input("  Background class ID: ").strip())
        tgt_cls = int(input("  Target class ID: ").strip())

    bkg_pixels = all_flat[gt_flat == bkg_cls].copy()
    tgt_pixels = all_flat[gt_flat == tgt_cls].copy()
    print(f"  Background: class {bkg_cls}  ({len(bkg_pixels)} pixels)")
    print(f"  Target    : class {tgt_cls}  ({len(tgt_pixels)} pixels)")

    # ------------------------------------------------------------------
    # PCA on all pixels (same as real_data_experiment)
    # ------------------------------------------------------------------
    pca = PCA(n_components=pca_dim)
    pca.fit(all_flat)

    bkg_pca = pca.transform(bkg_pixels)
    tgt_pca = pca.transform(tgt_pixels)

    s_raw = tgt_pca.mean(axis=0)
    s     = s_raw / (np.linalg.norm(s_raw) + 1e-12)

    # ------------------------------------------------------------------
    # Split: train / val / test
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    idx = np.arange(len(bkg_pca))
    rng.shuffle(idx)
    bkg_pca = bkg_pca[idx]

    N      = len(bkg_pca)
    n_use  = min(total_samples, N)
    n_test = max(1, int(round(n_use * test_fraction)))
    n_tv   = n_use - n_test                            # train + val pool
    n_val  = max(1, int(round(n_tv * val_fraction)))
    n_train = n_tv - n_val

    train_data = bkg_pca[:n_train]
    val_data   = bkg_pca[n_train: n_train + n_val]
    test_bkg   = bkg_pca[n_train + n_val: n_train + n_val + n_test]
    print(f"  Split → train={n_train}  val={n_val}  test={n_test}")

    # ------------------------------------------------------------------
    # Plant targets on test set
    # ------------------------------------------------------------------
    n_target = max(1, int(round(n_test * tgt_fraction)))
    labels   = np.zeros(n_test, dtype=int)
    tgt_idx  = np.random.default_rng(seed=0).choice(n_test, size=n_target, replace=False)
    labels[tgt_idx] = 1

    test_data = test_bkg.copy()
    test_data[tgt_idx] += amplitude * s
    print(f"  Planted {n_target}/{n_test} target pixels (amplitude={amplitude})")

    # ------------------------------------------------------------------
    # Train one DSM per sigma (load if checkpoint exists)
    # ------------------------------------------------------------------
    print(f"\n=== Training DSM models  σ_grid={sigma_grid} ===")
    models = {}
    for sigma in sigma_grid:
        ckpt = os.path.join(results_dir, f"dsm_sigma{sigma}.pt")
        model = ScoreNet(input_dim=pca_dim, hidden_dims=hidden_dims, activation="silu")
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            model.eval()
            print(f"  σ={sigma}  loaded from {ckpt}")
        else:
            print(f"\n  σ={sigma}  training for {base_epochs} epochs ...")
            model = train_dsm(model, train_data, sigma=sigma,
                              lr=lr, batch_size=batch_size, epochs=base_epochs,
                              weight_decay=weight_decay, print_every=base_epochs // 5)
            torch.save(model.state_dict(), ckpt)
        models[sigma] = model

    # ------------------------------------------------------------------
    # Oracle: AUC for every sigma
    # ------------------------------------------------------------------
    print("\n=== Oracle AUC per sigma ===")
    auc_per_sigma = {}
    for sigma, model in models.items():
        scores = detector_dsm(test_data, train_data, model, s)
        auc = roc_auc_score(labels, scores)
        auc_per_sigma[sigma] = float(auc)
        print(f"  σ={sigma:<6}  AUC={auc:.4f}")
    best_oracle_sigma = max(auc_per_sigma, key=auc_per_sigma.get)
    print(f"  → oracle best σ={best_oracle_sigma}  AUC={auc_per_sigma[best_oracle_sigma]:.4f}")

    # ------------------------------------------------------------------
    # Sigma selection methods
    # ------------------------------------------------------------------
    print("\n=== Sigma selection methods ===")
    selections = {}  # method → (selected_sigma, auc)

    # selections dict: method → (used_sigma, auc, estimated_sigma, model_or_None)
    # estimated_sigma == used_sigma for grid methods; differs for Silverman/Scott

    # 1. Held-out DSM loss
    s1, losses_dsm = select_by_dsm_loss(models, val_data)
    selections["DSM loss (val)"] = (s1, auc_per_sigma[s1], s1, None)
    print(f"  [DSM loss]          → σ={s1}  AUC={auc_per_sigma[s1]:.4f}   losses={losses_dsm}")

    # 2. Denoising L2 (raw, biased toward small σ)
    s2, scores_l2 = select_by_denoising_l2(models, val_data, normalize=False)
    selections["Denoising L2"] = (s2, auc_per_sigma[s2], s2, None)
    print(f"  [Denoising L2]      → σ={s2}  AUC={auc_per_sigma[s2]:.4f}   scores={scores_l2}")

    # 3. Normalized denoising L2 (÷σ⁴, equivalent to DSM loss)
    s3, scores_nl2 = select_by_denoising_l2(models, val_data, normalize=True)
    selections["Normalized L2 (÷σ⁴)"] = (s3, auc_per_sigma[s3], s3, None)
    print(f"  [Normalized L2]     → σ={s3}  AUC={auc_per_sigma[s3]:.4f}   scores={scores_nl2}")

    # 4. ISM loss (Hutchinson estimator on clean val data)
    n_hutch = cfg.get("n_hutchinson", 20)
    s4, losses_ism = select_by_ism_loss(models, val_data, n_hutchinson=n_hutch)
    selections["ISM loss (val)"] = (s4, auc_per_sigma[s4], s4, None)
    print(f"  [ISM loss]          → selected σ={s4}  AUC={auc_per_sigma[s4]:.4f}")
    for sig in sorted(losses_ism):
        marker = "  ← selected" if sig == s4 else ""
        print(f"      σ={sig:<7}  ISM={losses_ism[sig]:>10.4f}{marker}")

    # 5. Median heuristic (snapped to grid — no dedicated model needed)
    sigma_med = select_by_median_heuristic(train_data)
    s5 = snap_to_grid(sigma_med, sigma_grid)
    selections["Median heuristic"] = (s5, auc_per_sigma[s5], sigma_med, None)
    print(f"  [Median heuristic]  → estimated={sigma_med:.5f}  snapped={s5}  AUC={auc_per_sigma[s5]:.4f}")

    # 6. Silverman's rule — train dedicated model at the estimated σ
    sigma_sil = select_by_silverman(train_data)
    print(f"\n  [Silverman's rule]  → estimated σ={sigma_sil:.5f}  training dedicated model ...")
    model_sil = train_or_load(
        sigma_sil, pca_dim, hidden_dims, train_data, base_epochs, lr, batch_size, weight_decay,
        ckpt=os.path.join(results_dir, f"dsm_silverman_{sigma_sil:.6f}.pt"))
    auc_sil = float(roc_auc_score(labels, detector_dsm(test_data, train_data, model_sil, s)))
    selections["Silverman's rule"] = (sigma_sil, auc_sil, sigma_sil, model_sil)
    print(f"  [Silverman's rule]  → σ={sigma_sil:.5f}  AUC={auc_sil:.4f}")

    # 7. Scott's rule — train dedicated model at the estimated σ
    sigma_scott = select_by_scott(train_data)
    print(f"\n  [Scott's rule]      → estimated σ={sigma_scott:.5f}  training dedicated model ...")
    model_scott = train_or_load(
        sigma_scott, pca_dim, hidden_dims, train_data, base_epochs, lr, batch_size, weight_decay,
        ckpt=os.path.join(results_dir, f"dsm_scott_{sigma_scott:.6f}.pt"))
    auc_scott = float(roc_auc_score(labels, detector_dsm(test_data, train_data, model_scott, s)))
    selections["Scott's rule"] = (sigma_scott, auc_scott, sigma_scott, model_scott)
    print(f"  [Scott's rule]      → σ={sigma_scott:.5f}  AUC={auc_scott:.4f}")

    # AMF baseline
    auc_amf = float(roc_auc_score(labels, detector_amf(test_data, train_data, s)))
    print(f"\n  AMF baseline AUC={auc_amf:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "oracle_per_sigma": auc_per_sigma,
        "oracle_best":      {"sigma": best_oracle_sigma, "auc": auc_per_sigma[best_oracle_sigma]},
        "selections":       {k: {"estimated_sigma": v[2], "used_sigma": v[0], "auc": v[1]}
                             for k, v in selections.items()},

        "amf_auc":          auc_amf,
    }
    with open(os.path.join(results_dir, "sigma_selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, (ax_bar, ax_roc) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUC per sigma + method selections
    sigmas_sorted = sorted(auc_per_sigma)
    aucs_sorted   = [auc_per_sigma[s] for s in sigmas_sorted]
    x_pos = np.arange(len(sigmas_sorted))
    ax_bar.bar(x_pos, aucs_sorted, color="steelblue", alpha=0.7, label="Oracle AUC")
    ax_bar.axhline(auc_amf, color="black", linestyle="--", linewidth=1.5, label=f"AMF ({auc_amf:.3f})")

    colors_sel = plt.cm.get_cmap("tab10", len(selections))
    for i, (method, (sel_sigma, sel_auc, est_sigma, _)) in enumerate(selections.items()):
        if sel_sigma in sigmas_sorted:
            xi = sigmas_sorted.index(sel_sigma)
            ax_bar.scatter(xi, sel_auc + 0.005 * (i + 1), marker="v",
                           color=colors_sel(i), s=80, zorder=5,
                           label=f"{method} → σ={sel_sigma:.5g}")
        else:
            # Silverman / Scott: draw a vertical line at the estimated σ
            ax_bar.axvline(x=np.interp(sel_sigma, sigmas_sorted,
                                        np.arange(len(sigmas_sorted))),
                           color=colors_sel(i), linestyle=":", linewidth=1.5,
                           label=f"{method} → σ={sel_sigma:.5g}  AUC={sel_auc:.3f}")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([str(s) for s in sigmas_sorted])
    ax_bar.set_xlabel("σ", fontsize=12)
    ax_bar.set_ylabel("AUC", fontsize=12)
    ax_bar.set_title("Oracle AUC per σ + selection methods", fontsize=11)
    ax_bar.legend(fontsize=7, loc="lower right")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_ylim(max(0, min(aucs_sorted) - 0.05), min(1.0, max(aucs_sorted) + 0.08))

    # Right: ROC curves for all sigmas + best selection
    n_curves = len(models) + len(selections) + 1
    cmap = plt.cm.get_cmap("tab20", n_curves)
    ci = 0
    for sigma, model in sorted(models.items()):
        scores = detector_dsm(test_data, train_data, model, s)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax_roc.plot(fpr, tpr, color=cmap(ci), linewidth=1.5, linestyle="--",
                    alpha=0.6, label=f"DSM σ={sigma}  ({auc_per_sigma[sigma]:.3f})")
        ci += 1

    for method, (sel_sigma, sel_auc, _, dedicated_model) in selections.items():
        m = dedicated_model if dedicated_model is not None else models[sel_sigma]
        scores = detector_dsm(test_data, train_data, m, s)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax_roc.plot(fpr, tpr, color=cmap(ci), linewidth=2.5,
                    label=f"{method} (σ={sel_sigma:.5g}, AUC={sel_auc:.3f})")
        ci += 1

    fpr_amf, tpr_amf, _ = roc_curve(labels, detector_amf(test_data, train_data, s))
    ax_roc.plot(fpr_amf, tpr_amf, "k-", linewidth=2, label=f"AMF ({auc_amf:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax_roc.set_xlabel("False Alarm Rate", fontsize=12)
    ax_roc.set_ylabel("Detection Rate", fontsize=12)
    ax_roc.set_title("ROC curves", fontsize=11)
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(results_dir, "sigma_selection_figure.png")
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
    parser.add_argument("--config",     default="sigma_selection_config.yaml")
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
