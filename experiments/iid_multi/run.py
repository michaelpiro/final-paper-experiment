"""
IID multiclass experiment (Pavia-U): one target class, background = all
other classes (subject to `exclude_classes`).  Same pipeline + sweeps as
the single-class script; only the config differs.

Usage:
    .venv/bin/python -u run_iid_multi.py --config iid_multi.yaml
"""

import argparse
import os
import sys
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from iid_core import run_iid, replot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'))
    p.add_argument('--plot-only', metavar='RUN_DIR', default=None,
                   help='regenerate figures from a finished run dir (no training)')
    args = p.parse_args()
    if args.plot_only:
        replot(args.plot_only)
        return
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_iid(cfg, mode='multi')


if __name__ == '__main__':
    main()
