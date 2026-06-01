#!/bin/bash
# ============================================================
# Overnight sweep runner
# Runs all single-class configs in priority order, then multiclass.
# Results are saved independently per config — partial runs are never lost.
# ============================================================

set -euo pipefail
cd /Users/mac/Desktop/final_paper_experiment/pythonProject

PYTHON=".venv/bin/python"
SWEEPS="final_paper_experiments/experiments/sweeps"
LOGS="logs"
mkdir -p "$LOGS"

MASTER_LOG="$LOGS/sweep_master.log"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================" | tee -a "$MASTER_LOG"
echo "Sweep started: $START_TIME"                  | tee -a "$MASTER_LOG"
echo "PID: $$"                                     | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"

run_single() {
    local yaml="$1"
    local name
    name=$(basename "$yaml" .yaml)
    local log="$LOGS/${name}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo ">>> [$name] started at $(date '+%H:%M:%S')" | tee -a "$MASTER_LOG"

    if $PYTHON -m final_paper_experiments.experiments.single_class.run_experiment \
            --config "$yaml" --no-display \
            >> "$log" 2>&1; then
        echo "<<< [$name] DONE at $(date '+%H:%M:%S')" | tee -a "$MASTER_LOG"
    else
        echo "<<< [$name] FAILED at $(date '+%H:%M:%S') — check $log" | tee -a "$MASTER_LOG"
    fi
}

run_multi() {
    local yaml="$1"
    local name
    name=$(basename "$yaml" .yaml)
    local log="$LOGS/${name}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo ">>> [$name] started at $(date '+%H:%M:%S')" | tee -a "$MASTER_LOG"

    if $PYTHON -m final_paper_experiments.experiments.multiclass.run_experiment \
            --config "$yaml" --no-display \
            >> "$log" 2>&1; then
        echo "<<< [$name] DONE at $(date '+%H:%M:%S')" | tee -a "$MASTER_LOG"
    else
        echo "<<< [$name] FAILED at $(date '+%H:%M:%S') — check $log" | tee -a "$MASTER_LOG"
    fi
}

# ── Single-class configs (priority order) ─────────────────────────────────
run_single "$SWEEPS/01_cls2vs9_pca5_3seeds.yaml"     # 3 seeds — ~3h
run_single "$SWEEPS/02_cls2vs9_pca10.yaml"            # PCA 10   — ~1h
run_single "$SWEEPS/03_cls2vs9_pca15.yaml"            # PCA 15   — ~1h
run_single "$SWEEPS/04_cls2vs9_pca25.yaml"            # PCA 25   — ~1h
run_single "$SWEEPS/05_cls2vs9_arch128x128.yaml"      # arch     — ~1h
run_single "$SWEEPS/06_cls2vs9_arch64x3.yaml"         # deeper   — ~1h
run_single "$SWEEPS/07_cls2vs9_arch128x3.yaml"        # large    — ~1h
run_single "$SWEEPS/08_cls2vs9_amp010.yaml"           # amp 0.10 — ~1h
run_single "$SWEEPS/09_cls2vs9_amp020.yaml"           # amp 0.20 — ~1h
run_single "$SWEEPS/10_cls2vs9_amp030.yaml"           # amp 0.30 — ~1h
run_single "$SWEEPS/11_cls2vs1_pca5.yaml"             # cls pair — ~1h
run_single "$SWEEPS/12_cls2vs5_pca5.yaml"             # cls pair — ~1h
run_single "$SWEEPS/13_cls2vs7_pca5.yaml"             # cls pair — ~1h

# ── Multiclass (last — most expensive) ────────────────────────────────────
run_multi  "$SWEEPS/14_multiclass_pca5.yaml"          # 9 classes — ~4h

echo "" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
echo "All sweeps finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "============================================" | tee -a "$MASTER_LOG"
