# Rao via Score — Experiments

Score-based anomaly detection on hyperspectral imagery.
This repo contains all sweep scripts, baselines, and configs for the paper.

---

## Setup

```bash
git clone https://github.com/michaelpiro/final-paper-experiment.git
cd final-paper-experiment

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy scikit-learn torch matplotlib pyyaml tqdm
```

> **Important:** always use `.venv/bin/python` (not bare `python`). NumPy 2.x breaks PyTorch when using the system interpreter.

### Dataset

Download `pavia-u.mat` from the [IEEE GRSS benchmark](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes):

```
data/pavia-u.mat
```

It is a 610×340×103 hyperspectral image (Pavia University, ROSIS sensor) with 9 labeled land-cover classes:
`1=Asphalt  2=Meadows  3=Gravel  4=Trees  5=Metal sheets  6=Bare soil  7=Bitumen  8=Bricks  9=Shadows`

---

## Experiments

### IID sweeps (Section 8.1)

AUC vs. number of training samples, for all methods. Two settings:

| Config | Background | Target |
|--------|-----------|--------|
| Single-class | class 2 (meadows) | class 1 (asphalt) |
| Multi-class | all non-target classes | class 1 (asphalt) |

Three normalization orderings are compared:

| Mode | Description |
|------|-------------|
| `per_band_std` | per-band zero-mean / unit-std → PCA |
| `per_band_minmax` | per-band [0,1] → PCA |
| `pca_std` | raw PCA → divide each PC score by its training std |

**Run all 6 sweeps + spatial models overnight:**

```bash
bash run_tonight.sh
```

Or run a single sweep manually:

```bash
.venv/bin/python -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_n.yaml
```

Results land in `experiments/honest_pipeline/results/sweep_single_<timestamp>/`.

### Spatial models (Section 8.2)

Patch-based detectors that exploit spatial neighborhood structure.

```bash
.venv/bin/python -u experiments/spatial/run_thantd.py \
    --config experiments/spatial/thantd.yaml

# Skip THANTD (slow on CPU) and run only the spatial score models:
.venv/bin/python -u experiments/spatial/run_thantd.py \
    --config experiments/spatial/thantd.yaml --no-thantd
```

---

## Detectors

All detectors share the same pipeline:

1. **Affine normalization** of the background (per-band or PCA-based).
2. **PCA** reduction to `d` dimensions (typically 5).
3. **Train** a score network or fit a GMM on background PCA features.
4. **Score** each test pixel with the appropriate LMP statistic.

### IID detectors (run by `run_sweep.py`)

| Name | Model | Description |
|------|-------|-------------|
| **AMF** | Additive + Replacement | Adaptive Matched Filter — Gaussian, full 103-D (`global_max` norm). Classical reference. |
| **LRao-IID** | Additive + Replacement | Locally optimal Rao test via score matching (LFI objective). Theoretically optimal for weak signals θ→0. |
| **DSM** | Additive + Replacement | Denoising Score Matching — trains ψ(x)≈∇log p(x), uses LMP statistic. MLP with 2×64 hidden layers. |
| **DSM-linear** | Additive + Replacement | Same as DSM but with a single linear layer (no hidden dims). Should recover the Gaussian AMF solution analytically. Acts as a sanity baseline. |
| **GMM-GLRT (Levin)** | Additive + Replacement | Generalized LRT with a product-of-GMMs background model. ML estimation of fill factor p via 1-D grid search. Runs on multi-class background only (single Gaussian class doesn't benefit). |
| **GMM-GLRT (Gauss, oracle)** | Additive + Replacement | Simple K-Gaussian mixture, oracle amplitude θ. Upper bound for GMM-based methods. Multi-class only. |

**Scoring formulas:**

*Additive model* (y = background + θ·s):
- AMF: `sᵀΣ⁻¹(y−μ) / √(sᵀΣ⁻¹s)`
- DSM additive: `−(ψ(y)−ψ̄)ᵀs / √(sᵀCᵩs)` (standardized, CFAR)
- LRao-IID: LFI statistic J = ĝᵀĈᵩ⁻¹ĝ

*Replacement model* (y = (1−θ)·background + θ·s):
- AMF-rep: Gaussian replacement LMP: `[d − (y−μ)ᵀΣ⁻¹(y−s)] / √(2d + (μ−s)ᵀΣ⁻¹(μ−s))`
- DSM replacement: `(ψ(y)−ψ̄)ᵀ(y−s_rep)` normalized by training distribution

### Spatial detectors (run by `run_thantd.py`)

| Name | Description |
|------|-------------|
| **CF-Attn** | Covariance-Free Attention score model — attention over K-NN patch neighbors in PCA space |
| **NeighborMLP** | MLP score model that takes (center pixel, neighbor pixels) as joint input |
| **DSM (spatial)** | Standard IID DSM applied to the center pixel only (spatial neighbor context ignored) |
| **AMF** | Classical AMF (reference, no spatial context) |
| **THANTD** | Triplet Hybrid Attention Network (Liu et al. 2025). Operates on raw 103-D pixels. Slow on CPU — run on GPU. |

---

## Key files

```
experiments/
├── honest_pipeline/
│   ├── pipeline.py          ← HonestDetectionPipeline (norm → PCA → signatures)
│   ├── run_sweep.py         ← Main IID sweep runner
│   ├── sweep_n.yaml         ← Single-class, per_band_std
│   ├── sweep_multi_n.yaml   ← Multi-class, per_band_std
│   ├── sweep_n_perband.yaml ← Single-class, per_band_minmax
│   ├── sweep_multi_n_perband.yaml
│   ├── sweep_n_pcastd.yaml  ← Single-class, pca_std
│   └── sweep_multi_n_pcastd.yaml
└── spatial/
    ├── run_thantd.py        ← Spatial model runner (--no-thantd to skip THANTD)
    ├── thantd.yaml          ← Spatial experiment config
    ├── cfattn_model.py      ← CF-Attn model
    ├── neighbor_mlp_model.py
    └── thantd_model.py

dsm_model.py                 ← ScoreNet, DSM loss, LFI loss
final_paper_experiments/
├── data_utils.py            ← compute_sigma_from_data, loading helpers
└── baselines/
    ├── detectors.py         ← GMM-GLRT oracle scorers
    └── gmm_glrt_levin.py    ← GMMGLRTLevin (ML estimation, non-oracle)
```

---

## Config reference

Key fields shared across all sweep YAMLs:

```yaml
dataset:         data/pavia-u.mat
score_norm:      per_band_std     # or per_band_minmax / pca_std
target_cls:      1
bkg_cls:         2                # null for multi-class
n_train_list:    [50, 100, 200, 500, 750, 1000, 2000]
latent_dim_list: [5]
rho_list:        [0.001]          # DSM noise level: σ² = ρ · tr(Σ̂)/d
hidden_dims:     [64, 64]
dsm_epochs:      3000
seeds:           [42, 43, 44, 45, 46]
```

The `rho` parameter controls DSM noise: `σ² = ρ · (1/d) · tr(Σ̂_background)`.
A good default is `rho=0.001`.

---

## Output

Each sweep run creates a timestamped directory:

```
experiments/honest_pipeline/results/sweep_single_20260608_HHMMSS/
├── config.yaml
├── metrics.json          ← mean ± std AUC for all methods × (n, d, rho, amp)
├── progress.json         ← seeds completed so far (incremental)
├── figures/
│   ├── auc_vs_n_additive.pdf
│   ├── auc_vs_n_replacement.pdf
│   ├── auc_vs_d_*.pdf
│   ├── auc_vs_rho_*.pdf
│   └── auc_vs_amp_*.pdf
└── models/               ← saved DSM checkpoints per seed
```

---

## Notes on design

- **No target leakage**: target-class pixels are removed from background before any fitting. The `HonestDetectionPipeline` uses background pixels *only* to fit normalization, PCA, and the score network.
- **Affine invariance**: both additive and replacement statistics are invariant to consistent affine normalization. The signature is transformed with the *same* affine map as the data (direction rule for additive, point rule for replacement).
- **Deflection fraction ρ_d**: diagnostic reported per (d, seed) showing what fraction of the AMF signal energy is captured by the top-d PCA components. Should be ≥ 0.9 for meaningful comparison.
- **GMM-GLRT Jacobian** (replacement model): the Jacobian term `−r·log(1−p)` is mandatory. Without it the GLRT underweights large fill factors. Here `r` is the retained PCA rank (not full D=103 bands).
