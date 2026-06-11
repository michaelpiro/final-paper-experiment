"""Diagnostic / hardened launcher for the IID-multi experiment.

Why this exists
---------------
On macOS, NumPy/SciPy/sklearn/torch are linked against Apple's Accelerate
(vecLib) BLAS/LAPACK. Accelerate has two failure modes that show up as a
hard SIGSEGV (exit code 139) instead of a catchable Python exception:

  1. Multi-threaded Accelerate routines crash under some workloads. Forcing
     a single BLAS thread very often eliminates the segfault outright.
  2. A LAPACK routine fed non-finite or singular input segfaults instead of
     raising (on Linux/OpenBLAS the same call raises and is caught).

This launcher:
  * sets the BLAS/OpenMP thread counts to 1 *before* numpy is imported
    (this MUST happen before the first numpy import to take effect), and
  * enables faulthandler, so if it still crashes the Python traceback —
    i.e. the exact file/line of the offending native call — is printed to
    stderr just before the process dies.

Usage
-----
    python experiments/iid_multi/run_debug.py            # uses config.yaml
    python experiments/iid_multi/run_debug.py --threads 4   # relax thread cap

If it now runs to completion, the bug was Accelerate threading: keep using
this launcher (or set those env vars in your run config). If it still dies,
copy the last ~15 lines of stderr (the faulthandler traceback) — that names
the precise crashing line so it can be guarded directly.
"""

import argparse
import os
import sys

# ---- must run BEFORE numpy/torch are imported anywhere ----
def _set_threads(n: int):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "ACCELERATE_NUM_THREADS"):
        os.environ[var] = str(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    p.add_argument("--threads", type=int, default=1,
                   help="BLAS/OMP thread cap (1 = safest on macOS Accelerate)")
    args = p.parse_args()

    _set_threads(args.threads)

    import faulthandler
    faulthandler.enable()                      # dump Python stack on SIGSEGV

    import yaml
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sys.path.insert(0, _ROOT)
    os.chdir(_ROOT)

    import torch
    try:
        torch.set_num_threads(args.threads)
    except Exception:
        pass

    from iid_core import run_iid
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_iid(cfg, mode="multi")


if __name__ == "__main__":
    main()
