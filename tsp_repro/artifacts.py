"""tsp_repro.artifacts — locate checkpoints/raw scores without retraining.

Resolution order for an artifacts directory:
  1. $TSP_ARTIFACTS (explicit override)
  2. ./results_tsp + ./ckpt_tsp        (a fresh run in the working dir)
  3. ./tsp_artifacts/                  (an ingested/downloaded release zip)
  4. the local MLSP archive            (developer machine only; used for
     cross-checking the 2000-epoch canonical run against the MLSP numbers)

`fetch_release()` downloads the published results zip (GitHub release asset
on michaelpiro/final-paper-experiment) so reproduce-without-retraining works
on Colab too. Set the URL after the release is published.
"""

import os
import urllib.request
import zipfile

# TODO(Phase B): point at the published release asset of the canonical run.
RELEASE_URL = os.environ.get(
    "TSP_RELEASE_URL",
    "https://github.com/michaelpiro/final-paper-experiment/releases/download/"
    "tsp-v1/tsp_results.zip")

# Developer-machine locations of the MLSP-era archive (cross-check only).
MLSP_ARCHIVE = os.path.expanduser(
    "~/Desktop/final_paper_experiment/pythonProject/SDSM/results/generality_20260707")


def fetch_release(dest="tsp_artifacts", url=None):
    """Download + extract the published results zip; returns the directory."""
    os.makedirs(dest, exist_ok=True)
    zp = os.path.join(dest, "tsp_results.zip")
    if not os.path.exists(zp):
        u = url or RELEASE_URL
        print("downloading", u, flush=True)
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zp, "wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    print("artifacts ->", dest, flush=True)
    return dest


def scores_dirs():
    """Candidate directories containing scores_*.npz, in resolution order."""
    cands = []
    env = os.environ.get("TSP_ARTIFACTS")
    if env:
        cands.append(env)
    cands += ["results_tsp", os.path.join("tsp_artifacts", "results_tsp"),
              "tsp_artifacts"]
    if os.path.isdir(MLSP_ARCHIVE):
        cands += [os.path.join(MLSP_ARCHIVE, "raw"),
                  os.path.join(MLSP_ARCHIVE, "colab_deep_download",
                               "results_deep")]
    return [d for d in cands if os.path.isdir(d)]


def find_scores(pattern: str):
    """All scores files whose basename contains `pattern`, first hit per name."""
    seen, out = set(), []
    for d in scores_dirs():
        for fn in sorted(os.listdir(d)):
            if pattern in fn and fn.endswith(".npz") and fn not in seen:
                seen.add(fn)
                out.append(os.path.join(d, fn))
    return out
