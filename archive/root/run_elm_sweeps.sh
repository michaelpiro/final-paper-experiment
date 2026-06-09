#!/bin/bash
# ============================================================
# ELM post-PCA sweeps — run after run_tonight.sh completes
#   bash run_elm_sweeps.sh
# ============================================================
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=logs/tonight

mkdir -p $LOG

echo "=== ELM sweeps: $(date) ==="

# 1. single-class, pca_elm
echo "[1/2] sweep_n_pca_elm (single, pca_elm, 3000ep, 5 seeds) ..."
echo "  started: $(date)"
caffeinate -i $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_n_pca_elm.yaml \
    2>&1 | tee $LOG/sweep_single_pca_elm.log
echo "  finished: $(date)"

# 2. multi-class, pca_elm
echo "[2/2] sweep_multi_n_pca_elm (multi, pca_elm, 3000ep, 5 seeds) ..."
echo "  started: $(date)"
caffeinate -i $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_multi_n_pca_elm.yaml \
    2>&1 | tee $LOG/sweep_multi_pca_elm.log
echo "  finished: $(date)"

echo ""
echo "=== ELM sweeps done: $(date) ==="
