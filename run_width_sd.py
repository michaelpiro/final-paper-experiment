#!/usr/bin/env python3
"""Width experiment — DART[200] + DARTS(enc 189-128-64-32, head 200) on
San Diego I--II.

Trains 5 seeds per scene, scores planted targets at theta=0.075 and 0.15 on
the same planted sets as the paper tables (labels bit-identical), DARTS
reported at lambda=0.1 like the table. Prints a comparison against the table's
DART[128], DARTS(0.1), and AMF values.

Local port of RunWidthSD.ipynb: no Colab clone/download, runs from the repo
root, and uses Apple MPS when available (set env DEVICE=cpu to force CPU).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from repro import scenes
from repro.protocols.spatial import load_cfg, _windows
from repro.core.data import plant_targets
from repro.core.metrics import partial_auc, dr_at_fpr
from repro.models.dart import DART
from repro.models.darts import DARTS


def pick_device():
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
print("device:", DEVICE)


def main():
    cfg = load_cfg()
    dcfg = dict(cfg["dart"]); dcfg["hidden"] = [200]
    scfg = dict(cfg["darts"])
    scfg["enc_hidden"] = [128, 64]    # encoder 189 -> 128 -> 64 -> 32
    scfg["d_lat"] = 32
    scfg["score_hidden"] = [200]      # head: [189 | 32 | 7x32] = 445 -> 200 -> 189

    os.makedirs("results/width_sd", exist_ok=True)
    res = {}
    for scn in ("sandiego", "sandiego2"):
        sc = scenes.build(scn, cfg)
        k = int(cfg["k"])
        _, tr_nbr = _windows(sc["tr"], sc["tr_shape"], k, DEVICE)
        _, te_nbr = _windows(sc["te"], sc["te_shape"], k, DEVICE)
        for seed in (42, 43, 44, 45, 46):
            d = DART(dcfg).fit(sc["tr"], seed, DEVICE,
                               ckpt=f"results/width_sd/dart200_{scn}_s{seed}.pt")
            m = DARTS(scfg).fit(sc["tr"], tr_nbr, seed, DEVICE,
                                ckpt=f"results/width_sd/darts_e128-64-32_h200_{scn}_s{seed}.pt")
            for th in (0.075, 0.15):
                pl, lab, _ = plant_targets(sc["te"], sc["sig"], th, 0.10,
                                           model="additive", seed=seed,
                                           spatial_shape=sc["te_shape"], edge_guard=3)
                pl = pl.astype(np.float32)
                sd_ = np.asarray(d.score(pl, sc["tr"], sc["sig"]), float)
                raw = np.asarray(m.score(pl, te_nbr, sc["tr"], tr_nbr, sc["sig"]), float)
                nrm = np.asarray(DARTS.local_moment_normalize(
                    raw, sc["te_shape"], win=k, guard=1,
                    cfar_lam=float(cfg["cfar_lam"])), float)
                for name, s_ in (("DART200", sd_), ("DARTS-new", nrm)):
                    res.setdefault(f"{scn}|{th}|{name}", []).append(
                        [roc_auc_score(lab, s_), partial_auc(lab, s_, 0.05),
                         dr_at_fpr(lab, s_, (0.05,))["0.05"]])
            print(f"{scn} seed {seed}: DART200@0.075="
                  f"{res[f'{scn}|0.075|DART200'][-1][0]:.3f}  DARTS-new@0.075="
                  f"{res[f'{scn}|0.075|DARTS-new'][-1][0]:.3f}", flush=True)
    json.dump(res, open("results/width_sd/results.json", "w"), indent=1)

    # ---- summary vs the published table ----
    REF = {"sandiego|0.075":  "table: DART128 0.924/0.462, DARTS(0.1) 0.998/0.961, AMF 0.994/0.937",
           "sandiego2|0.075": "table: DART128 0.710/0.091, DARTS(0.1) 0.944/0.569, AMF 0.909/0.350",
           "sandiego|0.15":   "DART128 0.997, DARTS(0.1) 1.000, AMF 0.997",
           "sandiego2|0.15":  "DART128 0.863, DARTS(0.1) 0.995, AMF 0.979"}
    print("=== width summary (AUC/pAUC/Pd@0.05, 5 seeds) ===")
    for key, v in res.items():
        scn, th, name = key.split("|")
        a = np.array(v)
        print(f"{scn:10s} th={th} {name:9s}: "
              f"{a[:,0].mean():.3f}±{a[:,0].std():.3f} / "
              f"{a[:,1].mean():.3f}±{a[:,1].std():.3f} / "
              f"{a[:,2].mean():.3f}±{a[:,2].std():.3f}   [{REF[f'{scn}|{th}']}]")

    # ---- package ----
    import shutil
    shutil.make_archive("width_sd", "zip", "results/width_sd")
    print("zipped -> width_sd.zip")


if __name__ == "__main__":
    main()
