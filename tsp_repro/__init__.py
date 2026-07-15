"""tsp_repro — reproduction pipeline for the TSP journal version.

Layout
------
vendor/src/     verbatim copy of SDSM/src (the code that produced the MLSP
                archive); imported as the plain package ``src`` via the path
                shim below, so the vendored modules' internal ``from src.x``
                imports work unchanged.
protocol.py     scenes, amplitude grid (THETA_GRID), planting, signatures.
registry.py     detector registry: one entry per detector (fit / score).
runner.py       per-scene x seed x theta driver with checkpoint resume,
                raw-score archiving, and zip packaging (Colab-friendly).
artifacts.py    resolver for checkpoints / raw scores (local dirs or a
                published release zip) so everything reproduces without
                retraining.
tables.py       emit the tables/*.tex files the paper \\input's.
figures.py      emit the figures/*.pdf files the paper includes.
ingest_zip.py   CLI: Colab results zip -> TSP repo tables/ + figures/.

Canonical TSP settings: EPOCHS >= 2000, sigma = (D/n)^(1/6) (mse_optimal,
whitened space), THETA_GRID dense in (0, 0.3] and sparse in (0.3, 1).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "vendor")
_REPO = os.path.dirname(_HERE)

# Make the vendored SDSM package importable as ``src`` (its own import name),
# and the deep-baseline modules importable as on the colab-deep-baselines
# branch (thantd_model lives in experiments/spatial, colab_deep at repo root).
for _p in (_VENDOR, _REPO, os.path.join(_REPO, "experiments", "spatial")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
