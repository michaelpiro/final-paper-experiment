# Baseline Detectors — Notes & Formulas

All detectors are implemented in `baselines/detectors.py` with a unified API:
`scores = detector(test_data, train_data, s, ...)` → `np.ndarray (n_test,)`.

---

## 1. AMF — Adaptive Matched Filter

**Source:** Kelly (1986), classical Gaussian-background matched filter.

**Model:** Additive, Gaussian background `w ~ N(μ, Σ)`.

**Statistic:**
```
T_AMF(y) = s^T Σ̂⁻¹(y - μ̂) / sqrt(s^T Σ̂⁻¹ s)
```

**Notes:**
- Optimal under Gaussian additive model.
- Covariance estimated from training data with eigenvalue clipping for stability.
- File: `gaussian_iid_experiment.py → detector_amf`

---

## 2. Reg-AMF — Regularized (Diagonal-loaded) AMF

**Source:** Theorem 1 of the Rao-via-Score paper (our paper).

**Statistic:**
```
T_σ(y) = s^T (Σ̂+σ²I)⁻¹(y-μ̂) / sqrt(s^T (Σ̂+σ²I)⁻¹ Σ̂ (Σ̂+σ²I)⁻¹ s)
```

**Notes:**
- Shown to be the exact output of a linear DSM-based detector.
- σ is the DSM noise level, same as used for training.
- File: `gaussian_iid_experiment.py → detector_reg_amf`

---

## 3. LRao-IID — Linear Rao Detector for i.i.d. data

**Source:** Zschetzsche et al. 2026 ("Detection of weak signals under arbitrary noise distributions").

**Model:** Any background distribution. The test statistic is the **LLMP** (Locally Linearized Most Powerful):
```
T(y) = ĝ^T Ĉ_Ψ⁻¹ (Ψ(y) - μ̂_Ψ) / √Ĵ
```
where:
- `Ψ` is a learned transformation (ScoreNet) trained to maximize LFI: `J = ĝ^T Ĉ_Ψ⁻¹ ĝ`
- `ĝ = E[(Ψ(w+s·Δθ) - Ψ(w-s·Δθ)) / (2Δθ)]` (central-difference Jacobian)
- `Ĉ_Ψ = Cov(Ψ(w))` estimated from training data

**Training objective:** Maximize `J` (LFI) via gradient ascent.

**Files:**
- `dsm_model.py → train_lfi, compute_lfi_detector_scores`
- `baselines/detectors.py → lrao_iid`

---

## 4. LRao-CNN (Adapted to i.i.d.) — MLP variant

**Source:** Same paper as above (Zschetzsche et al. 2026). Original repo: `baselines/LRao-detector-main/`.

**Why adaptation?** The original CNN-LRao uses 1D Conv layers designed for **sequential time-series** data (harmonic signals in correlated noise). It exploits temporal correlations via PSD estimation (DFT or Yule-Walker). For **i.i.d. hyperspectral pixels**, there is no temporal structure, so:
- The 1D Conv → replaced by MLP (`Linear + Tanh` layers)
- PSD/DFT-based LFI estimation → replaced by covariance-based LFI (same formula as LRao-IID)
- Training objective: identical (maximize LFI)

**When to use which:**
| | LRao-IID (`ScoreNet + SiLU`) | LRao-MLP (`TrafoMLP + Tanh`) |
|---|---|---|
| Architecture | MLP with SiLU | MLP with Tanh (matching original repo) |
| Training | `train_lfi` | `train_lrao_mlp` (wraps `train_lfi`) |
| Objective | Same (maximize LFI) | Same (maximize LFI) |
| Use when | Standard setting | Comparing to original repo's style |

Both are mathematically equivalent — any difference in performance is due to activation function choice.

**Files:**
- `baselines/lrao_mlp.py → TrafoMLP, train_lrao_mlp, detect_lrao_mlp`

---

## 5. GMM-GLRT — Generalized Likelihood Ratio Test with Fitted GMM

**Source:** Our existing implementation.

**Model:** Additive, GMM background `p_w = Σ_k π_k N(μ_k, Σ_k)` (per-component covariances).

**Statistic:**
```
T(y) = log p̂_fitted(y - θ̂s) - log p̂_fitted(y)
θ̂(y) = argmax_θ log p̂(y - θs)   [grid search over θ ∈ [0, 2]]
```

**Notes:**
- K and θ_max are hyperparameters.
- Serves as an **oracle** in the GMM/multiclass setting (it knows the background model class).
- File: `gmm_iid_experiment.py → detector_gmm_glrt`

---

## 6. DLTD — Distribution-Level Target Detection

**Source:** Ma et al. (2026), "Distribution-Level Hyperspectral Target Detection Under Mixture of Gaussian", IEEE GRSL.

**Model:** GMM background with **shared covariance** Σ: `p_w = Σ_k π_k N(μ_k, Σ)`.

**Key idea:** Measure KL divergence between each background component N_k and the target distribution N_t = N(s, Σ). Score each pixel as a weighted similarity.

**Algorithm:**
1. Fit GMM (K components, shared Σ) via EM on training data.
2. For each pixel xᵢ and component k, define:
   - `u_ik = Σ^{-1/2}(μ_k - xᵢ)` (whitened deviation from component mean)
   - `v_i  = Σ^{-1/2}(xᵢ - s)` (whitened deviation from target)
3. Score (Eq. 6–8):
```
d_i = w_i · g_i

w_i = Σ_k π_k exp{-u_ik^T(u_ik + v_i)}
g_i = exp{-½ v_i^T v_i} / Σ_l π_l exp{-½ u_il^T u_il}
```
Note: `g_i` is the GLR statistic under MoG; `w_i` is the distribution-level weight.

**Notes:**
- K barely affects performance per ablation study in the paper.
- Works on raw hyperspectral bands (no PCA required), but we apply it in PCA space.
- Implemented in log-space for numerical stability.
- File: `baselines/detectors.py → dltd`

---

## 7. SMGLRT — Segmented-Mixing GLRT

**Source:** Ma et al. (2025), "Generalized Likelihood Ratio Test for Hyperspectral Subpixel Target Detection Based on Segmented Mixing Model", IEEE JSTARS.

**Key idea:** In the conventional replacement model `x = αt + (1-α)b`, all bands share one mixing coefficient α. The SMM assigns **per-segment** coefficients: adjacent bands within a segment share the same α, reducing overfitting vs. per-band coefficients.

**Model:** `H₀: x = b`,  `H₁: x = BlockDiag(α¹I,...,αᵐI)t + BlockDiag(β¹I,...,βᵐI)b`

**Background:** GMM with K components and shared Σ.

**Test statistic:**
```
T(y) = log Σ_k π_k N(y; α̂_k⊙t + (1-α̂_k)⊙μ_k, Σ) - log Σ_k π_k N(y; μ_k, Σ)

α̂^j_k = (t^j - μ_k^j)^T Σ_jj⁻¹(y^j - μ_k^j) / ||(t^j - μ_k^j)||²_{Σ_jj⁻¹}
```
clipped to [0, 1] per segment j.

**Note for PCA-reduced data:** With d=5 and 2–3 segments, SMGLRT degenerates toward GMM-GLRT (limited benefit of segmentation). The main advantage is for full-band (103-dim) data where the block-diagonal covariance reduces estimation variance.

**File:** `baselines/detectors.py → smglrt`

---

## 8. Replacement Model Detectors

### AMF-replacement
Gaussian-case closed-form LMP for the replacement model (from main2.tex Section 2.3):
```
T_rep(y) = [d - (y-μ̂)^T Σ̂⁻¹(y-s)] / sqrt(2d + (μ̂-s)^T Σ̂⁻¹(μ̂-s))
```

### DSM-replacement
Score-based LMP for the replacement model:
```
T_rep(y) = (ψ(y)^T(y-s) - r̄) / sqrt(Var{ψ(wᵢ)^T(wᵢ-s)})
r̄ = mean{ψ(wᵢ)^T(wᵢ-s)}  over training samples
```
Derived from `u_rep(y; 0) = ψ(y)^T(y-s) + d` (the +d cancels after centering).

**File:** `baselines/detectors.py → amf_replacement, dsm_replacement`
