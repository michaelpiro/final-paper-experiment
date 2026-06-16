# Target Detection via Denoising Score Matching

Reference implementation for the paper *"Target Detection via Denoising Score
Matching"*. We estimate the unknown background score with denoising score
matching (DSM) and plug it into the locally most powerful / one-sided Rao
detector. This gives **DART** (a denoising-score-assisted Rao test), its linear
special case **L-DART** (a diagonally loaded adaptive matched filter), and a
spatially adapted variant **DARTS** for correlated (hyperspectral) backgrounds,
with CFAR-normalised forms **DART-CFAR** / **DARTS-CFAR**.

Everything runs on the **Pavia University** hyperspectral scene (committed under
`data/`). One notebook reproduces every experiment, and every
performance-affecting hyperparameter is exposed in the config files.

---

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.9+. A GPU is optional (set `device: cuda` or pass
`device='cuda'` in the config); everything also runs on CPU.

## Run

**Notebook (recommended)** — open and run top-to-bottom:

```
notebooks/run_experiments.ipynb
```

It runs the IID single-class, IID multi-class, and spatial experiments and shows
all figures. Set `QUICK = True` in the first cell for a fast smoke test.

**Command line:**

```bash
# IID (single-class and multi-class)
python -c "import yaml, src.iid as m; m.run_iid(yaml.safe_load(open('configs/iid_single.yaml')), 'single')"
python -c "import yaml, src.iid as m; m.run_iid(yaml.safe_load(open('configs/iid_multi.yaml')),  'multi')"

# Spatial (single scenario; --dry-run for a fast check)
python src/spatial.py --config configs/spatial.yaml
python src/spatial.py --config configs/spatial.yaml --dry-run
```

Results (figures, metrics, scores) are written under `results/`.

---

## Detectors

| Name | What it is |
|---|---|
| **AMF** | Adaptive matched filter — global covariance in the IID experiment; a per-pixel local k×k sample covariance in the spatial experiment |
| **GMM-Levin** | Gaussian-mixture GLRT (Levin 2019), fill factor by grid search |
| **L-LRao** / **LRao** | Learned Rao detector (linear / one-hidden-layer MLP) |
| **L-DART** / **DART** | Our denoising-score Rao detector (linear / MLP) |
| **DARTS** | Spatially adapted DART (neighbour-conditioned score) |
| **DART-CFAR** / **DARTS-CFAR** | CFAR-normalised forms (local mean reduce + local-Fisher normalisation) |

IID figures show {AMF, GMM-Levin, L-DART, DART, L-LRao, LRao}; the spatial table
shows {AMF, GMM-Levin, DART, DART-CFAR, DARTS, DARTS-CFAR} (there, AMF uses a
local k×k sample covariance).

---

## Hyperparameters (all in `configs/`)

**Score model (DART / L-DART / DARTS).** Whitening is **ZCA only**
(`whiten_eig_floor`, `lrao_whiten_eig_floor`). The DSM noise level is set
manually as `dsm_sigma_rho` (numeric ρ → σ² = ρ·tr(Σ)/d in whitened space); its
sensitivity is reported as the ρ-sweep ablation — there is no automatic σ
selection. Architecture via `hidden_dims` (MLP → DART) vs `hidden_dims_2` (linear
→ L-DART), `activation`; optimisation via `dsm_epochs`, `lr`, `weight_decay`,
`batch_size`.

**Learned Rao (LRao / L-LRao).** `lfi_delta_theta`, `lfi_sigma_cutoff`,
`lfi_detach_sigma`, `lrao_grad_clip`, `lrao_epochs`, `run_lrao_mlp`; validation
early stopping `lrao_val_fraction`, `lrao_val_check_every`, `lrao_patience`,
`lrao_min_delta`.

**Classical.** AMF `baseline_eig_floor` (IID, global) / `amf_local_window`,
`local_scm_loading` (spatial, local k×k SCM); GMM-Levin `gmm_K`, `gmm_steps`.

**DARTS (spatial score net).** `k` (neighbourhood), `nmlp_K` (top-K neighbours),
`nmlp_d_lat`, `nmlp_enc_hidden`, `nmlp_score_hidden`, `nmlp_epochs`, `nmlp_lr`,
`nmlp_batch`.

**CFAR (DART-CFAR / DARTS-CFAR).** `cfar_lam` (local→global Fisher shrinkage),
`cfar_fisher_use_topk`, `sdsm_cfar_window` / `sdsm_cfar_guard` (DARTS-CFAR
window), `dsm_cfar_window` / `dsm_cfar_guard` (DART-CFAR window), `pfa_target`.

**IID experiment / data.** `bkg_cls`, `target_cls`, `exclude_classes`,
`amplitude`, `target_fraction`, `test_size`, `n_train_list`, `rho_list`,
`n_fixed_for_rho`, `pfa`, `pauc_fpr`, `seed`.

**Spatial experiment / data.** `scenario_index`, `random_scenario_seeds`,
`min_pixels`, `target_class`, `foreign_class`, `sig_dom_weight`,
`sig_mean_weight`, `amplitude`, `target_fraction`, `edge_guard`, `n_budget`,
`run_inpatch`, `run_foreign`, `seed`.

### Reproduce the paper

* **Fig. 2 (IID single-class):** `run_iid(configs/iid_single.yaml, 'single')` →
  `pauc_vs_n`, `pd_at_fa_vs_n`, `pdet_at_pfa_vs_rho`.
* **Fig. 3 (IID multi-class):** `run_iid(configs/iid_multi.yaml, 'multi')`.
* **Table 1 (spatial):** `src.spatial.run_multiseed(<spatial cfg>)`; sweep
  `cfar_lam` for the DART-CFAR / DARTS-CFAR ablation rows.

---

## Layout

```
data/                pavia-u.mat (committed)
src/
  data.py            loading, planting, ZCA whitening, neighbourhoods, boxes, signatures
  models.py          ScoreNet (DART/L-DART, learned Rao), NeighborMLPDenoiser (DARTS)
  detectors.py       AMF (global + local k×k SCM), GMM-Levin, DART/L-DART scoring
  metrics.py         AUC, pAUC, Pd@Pfa, CFAR threshold, per-class FPR
  iid.py             run_iid(cfg, mode)  — IID single/multi
  spatial.py         spatial comparison + CFAR + multi-seed
configs/             iid_single.yaml, iid_multi.yaml, spatial.yaml, manual_boxes.json
notebooks/           run_experiments.ipynb
```

## Data

`data/pavia-u.mat` (Pavia University, ROSIS; courtesy of Prof. Paolo Gamba,
University of Pavia) contains `data` (610×340×103) and `map` (610×340 labels,
0 = unlabeled). It is consumed raw — no normalization or PCA.
