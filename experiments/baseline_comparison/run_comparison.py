"""
run_comparison.py — CLI entry for the baseline-comparison framework.

Usage:
    python -u experiments/baseline_comparison/run_comparison.py \
        --config experiments/baseline_comparison/configs/synthetic_gmm.yaml \
        --results_dir /content/drive/.../baseline_results [--dry-run] \
        [--detectors AMF,CEM,DSM]
"""

import argparse
import os
import sys
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for p in (_ROOT, os.path.join(_ROOT, "experiments", "spatial")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_ROOT)

import yaml
import torch

from experiments.baseline_comparison.framework import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results_dir", default="experiments/baseline_comparison/results")
    ap.add_argument("--detectors", default=None,
                    help="comma-separated subset (default: config 'detectors')")
    ap.add_argument("--provider", default=None, help="override config provider")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.provider:
        cfg["provider"] = args.provider
    only = args.detectors.split(",") if args.detectors else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ("dryrun_" if args.dry_run else "") + cfg["provider"] + "_" + ts
    results_dir = os.path.join(args.results_dir, tag)
    os.makedirs(results_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(results_dir, "config.yaml"), "w"),
                   sort_keys=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Provider={cfg['provider']}  device={device}  dry_run={args.dry_run}")
    print(f"Results dir: {results_dir}", flush=True)

    runner.run(cfg, results_dir, only=only, dry_run=args.dry_run, device=device)
    print(f"\nDone. Results in {results_dir}", flush=True)


if __name__ == "__main__":
    main()
