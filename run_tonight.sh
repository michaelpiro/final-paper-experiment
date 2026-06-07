#!/bin/bash
# ============================================================
# Overnight run script
# Run from project root:
#   bash run_tonight.sh
# ============================================================
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=logs/tonight

mkdir -p $LOG

echo "=== Starting overnight runs: $(date) ==="

# 1. IID sweep — single-class, per_band_std, 3000 epochs, 5 seeds
echo "[1/7] sweep_n (single, per_band_std, 3000ep, 5 seeds) ..."
nohup $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_n.yaml \
    > $LOG/sweep_single_std.log 2>&1 &
echo "  PID=$!"

# 2. IID sweep — multi-class, per_band_std, 3000 epochs, 5 seeds
echo "[2/7] sweep_multi_n (multi, per_band_std, 3000ep, 5 seeds) ..."
nohup $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_multi_n.yaml \
    > $LOG/sweep_multi_std.log 2>&1 &
echo "  PID=$!"

# 3. IID sweep — single-class, per_band_minmax, 3000 epochs, 5 seeds
echo "[3/7] sweep_n_perband (single, per_band_minmax, 3000ep, 5 seeds) ..."
nohup $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_n_perband.yaml \
    > $LOG/sweep_single_minmax.log 2>&1 &
echo "  PID=$!"

# 4. IID sweep — multi-class, per_band_minmax, 3000 epochs, 5 seeds
echo "[4/7] sweep_multi_n_perband (multi, per_band_minmax, 3000ep, 5 seeds) ..."
nohup $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_multi_n_perband.yaml \
    > $LOG/sweep_multi_minmax.log 2>&1 &
echo "  PID=$!"

# 5. IID sweep — single-class, pca_std, 3000 epochs, 5 seeds
echo "[5/7] sweep_n_pcastd (single, pca_std, 3000ep, 5 seeds) ..."
nohup $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_n_pcastd.yaml \
    > $LOG/sweep_single_pcastd.log 2>&1 &
echo "  PID=$!"

# 6. IID sweep — multi-class, pca_std, 3000 epochs, 5 seeds
echo "[6/7] sweep_multi_n_pcastd (multi, pca_std, 3000ep, 5 seeds) ..."
nohup $PY -u experiments/honest_pipeline/run_sweep.py \
    --config experiments/honest_pipeline/sweep_multi_n_pcastd.yaml \
    > $LOG/sweep_multi_pcastd.log 2>&1 &
echo "  PID=$!"

# 7. Spatial models WITHOUT THANTD (CF-Attn, NeighborMLP, DSM, AMF)
#    THANTD will be run separately on GPU tomorrow.
echo "[7/7] Spatial models (no THANTD) ..."
nohup $PY -u experiments/spatial/run_thantd.py \
    --config experiments/spatial/thantd.yaml \
    --no-thantd \
    > $LOG/spatial_no_thantd.log 2>&1 &
echo "  PID=$!"

echo ""
echo "All 7 jobs launched. Monitor with:"
echo "  tail -f $LOG/sweep_single_std.log"
echo "  tail -f $LOG/sweep_multi_std.log"
echo "  tail -f $LOG/sweep_single_minmax.log"
echo "  tail -f $LOG/sweep_multi_minmax.log"
echo "  tail -f $LOG/sweep_single_pcastd.log"
echo "  tail -f $LOG/sweep_multi_pcastd.log"
echo "  tail -f $LOG/spatial_no_thantd.log"
echo ""
echo "Check running jobs: ps aux | grep python | grep -v grep"
echo ""
echo "=== Started at: $(date) ==="
