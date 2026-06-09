#!/bin/bash
# ============================================================
# Overnight run script — sequential, no sleep mode
# Run from project root:
#   bash run_tonight.sh
#
# Uses caffeinate to prevent the Mac from sleeping.
# Jobs run one at a time — each waits for the previous to finish.
# ============================================================
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=logs/tonight

mkdir -p $LOG

echo "=== Starting overnight runs: $(date) ==="
echo "Preventing sleep with caffeinate ..."

run_job() {
    local num="$1"
    local total="$2"
    local label="$3"
    local config="$4"
    local logfile="$5"
    local extra="${6:-}"

    echo ""
    echo "[$num/$total] $label ..."
    echo "  log: $logfile"
    echo "  started: $(date)"
    caffeinate -i $PY -u experiments/honest_pipeline/run_sweep.py \
        --config "$config" $extra \
        2>&1 | tee "$logfile"
    echo "  finished: $(date)"
}

# 1. single-class, per_band_std
run_job 1 7 "sweep_n (single, per_band_std, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_n.yaml \
    $LOG/sweep_single_std.log

# 2. multi-class, per_band_std
run_job 2 7 "sweep_multi_n (multi, per_band_std, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_multi_n.yaml \
    $LOG/sweep_multi_std.log

# 3. single-class, per_band_minmax
run_job 3 7 "sweep_n_perband (single, per_band_minmax, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_n_perband.yaml \
    $LOG/sweep_single_minmax.log

# 4. multi-class, per_band_minmax
run_job 4 7 "sweep_multi_n_perband (multi, per_band_minmax, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_multi_n_perband.yaml \
    $LOG/sweep_multi_minmax.log

# 5. single-class, pca_std
run_job 5 7 "sweep_n_pcastd (single, pca_std, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_n_pcastd.yaml \
    $LOG/sweep_single_pcastd.log

# 6. multi-class, pca_std
run_job 6 7 "sweep_multi_n_pcastd (multi, pca_std, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_multi_n_pcastd.yaml \
    $LOG/sweep_multi_pcastd.log

# 7. Spatial models WITHOUT THANTD (CF-Attn, NeighborMLP, DSM, AMF)
echo ""
echo "[7/7] Spatial models (no THANTD) ..."
echo "  log: $LOG/spatial_no_thantd.log"
echo "  started: $(date)"
caffeinate -i $PY -u experiments/spatial/run_thantd.py \
    --config experiments/spatial/thantd.yaml \
    --no-thantd \
    2>&1 | tee $LOG/spatial_no_thantd.log
echo "  finished: $(date)"

# 8. single-class, pca_elm (post-PCA ELM scaling)
run_job 8 9 "sweep_n_pca_elm (single, pca_elm, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_n_pca_elm.yaml \
    $LOG/sweep_single_pca_elm.log

# 9. multi-class, pca_elm
run_job 9 9 "sweep_multi_n_pca_elm (multi, pca_elm, 3000ep, 5 seeds)" \
    experiments/honest_pipeline/sweep_multi_n_pca_elm.yaml \
    $LOG/sweep_multi_pca_elm.log

echo ""
echo "=== All 9 jobs done: $(date) ==="
