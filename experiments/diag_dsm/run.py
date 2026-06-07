"""
Isotropic vs Diagonal DSM-noise comparison (Pavia-U), one entry point.

Trains two DSM models that differ ONLY in their training-noise covariance —
isotropic σ²I vs data-driven diagonal diag(σ_b²) — on the same global_max
data and PCA-d latent space, and compares additive + replacement AUC across
a sweep over latent_dim and n_train.  No other baselines are run.

Usage:
    .venv/bin/python -u run_diag_dsm.py --config diag_dsm.yaml

Knobs live in diag_dsm.yaml.  See diag_dsm_core.run_diag_dsm for the pipeline.
"""
import argparse
import os
import sys
import yaml

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP)   # for core.py (diag_dsm core)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from core import run_diag_dsm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_diag_dsm(cfg)


if __name__ == '__main__':
    main()
