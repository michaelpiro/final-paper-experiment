"""tsp_repro.ingest_zip — Colab results zip -> TSP paper repo.

Usage (developer machine):
    .venv/bin/python -m tsp_repro.ingest_zip \
        --zip ~/Downloads/tsp_results.zip \
        --tsp ~/Desktop/final_paper_experiment/TSP

Extracts the zip, verifies the score archives it contains, regenerates the
tables and figures, and writes them straight into <tsp>/tables and
<tsp>/figures (Overleaf picks them up on the next push).
"""

import argparse
import os
import zipfile

from tsp_repro import figures as F
from tsp_repro import tables as T
from tsp_repro.registry import CLASSICAL, DEEP, OUR_DETECTORS

TABLE_DETS = OUR_DETECTORS + CLASSICAL + DEEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Colab tsp_results.zip")
    ap.add_argument("--tsp", required=True, help="path to the TSP paper repo")
    ap.add_argument("--workdir", default="tsp_artifacts")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    with zipfile.ZipFile(os.path.expanduser(args.zip)) as z:
        z.extractall(args.workdir)
    dirs = [os.path.join(args.workdir, d) for d in os.listdir(args.workdir)
            if os.path.isdir(os.path.join(args.workdir, d))]
    dirs = [d for d in dirs if any(f.startswith("scores__") or
                                   f.startswith("scores_")
                                   for f in os.listdir(d))] or [args.workdir]
    print("score dirs:", dirs)

    cells = T.collect(dirs)
    scenes = sorted({k[0] for k in cells})
    dets = sorted({k[2] for k in cells})
    print(f"found {len(cells)} cells | scenes: {scenes} | detectors: {dets}")

    tsp = os.path.expanduser(args.tsp)
    tab_dir = os.path.join(tsp, "tables")
    fig_dir = os.path.join(tsp, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    T.write_generality_tables(tab_dir, dirs,
                              [d for d in TABLE_DETS if d in dets])
    F.scenes_falsecolor(os.path.join(fig_dir, "scenes_falsecolor.pdf"))
    F.amp_sweep(os.path.join(fig_dir, "amp_sweep.pdf"), dirs,
                [d for d in TABLE_DETS if d in dets])
    try:
        F.detection_maps(os.path.join(fig_dir, "detection_maps.pdf"), dirs,
                         [d for d in TABLE_DETS if d in dets])
    except FileNotFoundError as e:
        print("detection_maps skipped:", e)
    print("\nDone. Review the diff in the TSP repo, then commit + push.")


if __name__ == "__main__":
    main()
