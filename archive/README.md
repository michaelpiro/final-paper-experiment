# Archive

Retired code kept out of the active tree (git history preserves everything).
Moved during the 2026-06 "raw data + frozen whitening, no PCA" cleanup.

## experiments/
- `diag_dsm/` — old DSM iso-vs-diag preprocessing diagnostic (PCA-based). Superseded.
- `honest_pipeline/` — the dual-normalization "honest" IID pipeline + ELM /
  per-band / pca-std normalization sweeps and the two diagnostic studies
  (DSM preprocessing, LRao sensitivity). Superseded by the raw `iid_core` pipeline.

## root/
- `quick_elm.py`, `quick_pca_check.py`, `temp.py` — one-off scratch scripts.
- `run_elm_sweeps.sh`, `run_tonight.sh` — overnight runners for honest_pipeline sweeps.
- `0$.` — empty file from a botched shell redirect.

## spatial/
Exploratory spatial pipelines pruned from `experiments/spatial/` (kept: the
CF-Attn/NeighborMLP/DSM main pipeline `run_colab.py`, the entanglement-invariance
study `run_invariance.py`, and the THANTD baseline `run_thantd.py`):
- `run_cfattn.py`, `run_cfattn_jac.py` + `cfattn_jac_model.py` — single-model CF-Attn variants.
- `run_nas.py` — early neighbor-adapted-score (NAS) runner.
- `run_dom_offset.py`, `run_neighbor_mlp.py`, `run_thantd_ablation.py`,
  `eval_dominant_sig.py` — one-off ablations/eval scripts.
- `pick_boxes.py`, `pick_boxes_interactive.py` — manual box-picking tools
  (output `manual_boxes.json` is retained in the live tree).
- Orphaned configs: `cfattn.yaml`, `cfattn_jac.yaml`, `dom_offset.yaml`,
  `neighbor_mlp.yaml`, `thantd_ablation.yaml`, `spatial.yaml`.

## results/
- `colab_experiments/` — old honest_pipeline Colab result bundle (models/figures).
