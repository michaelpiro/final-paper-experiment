"""
runner.py — orchestrate the comparison.

For each scenario from the provider:
  build a clean-background DetectorInput
  for each detector (skip needs_spatial on non-spatial data):
     fit() once on background
     per signature: clean-train scores -> CFAR threshold (no test leakage)
     for each (target_model x signature x amplitude):
        plant targets (feature + raw, same seed) -> score() -> metrics + maps
     save model + metrics + scores + maps + config + train_log
"""

from __future__ import annotations

import os
import time
import numpy as np

from final_paper_experiments.data_utils import plant_targets
from .detector_api import DetectorInput
from .datasets import get_provider, Scenario
from . import registry, metrics as M, io


def _ctx(sc: Scenario, device, seed, test_pix=None, test_raw=None,
         test_nbr=None) -> DetectorInput:
    return DetectorInput(
        train_pix=sc.train_pix,
        test_pix=sc.test_pix if test_pix is None else test_pix,
        signature=np.zeros(sc.train_pix.shape[1], np.float32),  # set per cell
        train_raw=sc.train_raw,
        test_raw=sc.test_raw if test_raw is None else test_raw,
        signature_raw=np.zeros(sc.train_raw.shape[1], np.float32),
        sigma=sc.sigma, device=device, seed=seed,
        train_nbr=sc.train_nbr,
        test_nbr=sc.test_nbr if test_nbr is None else test_nbr,
        train_nbr_raw=sc.train_nbr_raw,
        # when the caller overrides test_nbr (e.g. the train-pixel CFAR pass),
        # the matching raw neighbors must follow it (raw == feature post-no-PCA)
        test_nbr_raw=sc.test_nbr_raw if test_nbr is None else test_nbr,
        test_coords=sc.test_coords, box_shape=sc.box_shape,
        meta=sc.meta,
    )


def _with_sig(ctx: DetectorInput, s_feat, s_raw) -> DetectorInput:
    from dataclasses import replace
    return replace(ctx, signature=s_feat.astype(np.float32),
                   signature_raw=s_raw.astype(np.float32))


def _should_load(dname, load):
    if not load:
        return False
    return load == "all" or dname in load


def run(cfg: dict, results_dir: str, only=None, dry_run=False, device="cpu",
        pretrained_dir=None, load=None):
    registry.ensure_loaded()
    provider = get_provider(cfg["provider"])
    pcfg = dict(cfg.get("provider_cfg", {})); pcfg.setdefault("seed", cfg.get("seed", 42))
    if dry_run:
        pcfg.update(cfg.get("dry_run_provider_cfg", {}))

    load = load if load is not None else cfg.get("load")   # 'all' or [names]
    if isinstance(load, str) and load not in ("all", "none"):
        load = [x for x in load.split(",") if x]
    if load == "none":
        load = None
    det_names = only or cfg["detectors"]
    det_cfgs = cfg.get("detector_cfg", {})
    models = cfg.get("target_models", ["additive"])
    amps = cfg.get("amplitudes", [0.15])
    frac = float(cfg.get("target_fraction", 0.10))
    sig_filter = cfg.get("signatures", "all")
    seed = int(cfg.get("seed", 42))
    if dry_run:
        amps = cfg.get("dry_run_amplitudes", amps[:2])

    summary = {}
    for sc in provider.scenarios(pcfg):
        print(f"\n=== scenario {sc.name} (spatial={sc.spatial}, "
              f"D={sc.meta.get('D')}) ===", flush=True)
        base = _ctx(sc, device, seed)
        sig_names = (list(sc.signatures) if sig_filter == "all"
                     else [s for s in sig_filter if s in sc.signatures])

        for dname in det_names:
            det = registry.build(dname, det_cfgs.get(dname))
            if det.needs_spatial and not sc.spatial:
                print(f"  [{dname}] skipped (needs spatial)", flush=True)
                continue
            if (det.transductive or det.image_based) and sc.box_shape is None:
                print(f"  [{dname}] skipped (needs a 2D image)", flush=True)
                continue
            dcfg = dict(det_cfgs.get(dname, {}))
            if dry_run:
                dcfg.update(cfg.get("dry_run_detector_cfg", {}))
                det = registry.build(dname, dcfg)

            if det.transductive:
                # trains on the test image each call -> no fit, no train-threshold
                fit_t = 0.0
                print(f"  [{dname}] transductive (per-cell train+detect)", flush=True)
            else:
                # LOAD a pretrained model instead of training (scenarios + GMM data
                # are deterministic across runs, so a saved model is exactly reusable).
                mpath = (os.path.join(pretrained_dir, cfg["provider"], sc.name, dname,
                                      "model.pkl") if pretrained_dir else None)
                if _should_load(dname, load) and mpath and os.path.exists(mpath):
                    det.load(mpath); fit_t = 0.0
                    print(f"  [{dname}] LOADED {mpath}", flush=True)
                else:
                    if _should_load(dname, load) and mpath:
                        print(f"  [{dname}] no pretrained at {mpath} -> training", flush=True)
                    t0 = time.time()
                    det.fit(_with_sig(base, sc.signatures[sig_names[0]][0],
                                      sc.signatures[sig_names[0]][1]))
                    fit_t = time.time() - t0
                    print(f"  [{dname}] fit {fit_t:.1f}s", flush=True)

            metr, scores_npz, maps_npz = {}, {}, {}
            scores_npz["gt_cls"] = (sc.test_gt_cls if sc.test_gt_cls is not None
                                    else np.array([]))
            if sc.box_shape is not None:
                scores_npz["box_shape"] = np.array(sc.box_shape)

            for sig in sig_names:
                s_f, s_r = sc.signatures[sig]
                ctx_sig = _with_sig(base, s_f, s_r)
                # CFAR threshold from CLEAN training pixels only. Skip for image-
                # based detectors (their input is the box image, not a pixel set).
                if det.transductive or det.image_based:
                    tr_scores = None
                else:
                    train_ctx = _with_sig(
                        _ctx(sc, device, seed, test_pix=sc.train_pix,
                             test_raw=sc.train_raw, test_nbr=sc.train_nbr), s_f, s_r)
                    tr_scores = det.score(train_ctx)
                    scores_npz[f"trainscore|{sig}"] = tr_scores

                for model in models:
                    for amp in amps:
                        pf, lab, tgt = plant_targets(sc.test_pix, s_f, amp, frac,
                                                     model=model, seed=seed)
                        pr, _, _ = plant_targets(sc.test_raw, s_r, amp, frac,
                                                 model=model, seed=seed)
                        ctx = _with_sig(base, s_f, s_r).with_test(pf, pr)
                        sc_vals = det.score(ctx)
                        cm = M.cell_metrics(lab, sc_vals, train_scores=tr_scores,
                                            gt_cls=sc.test_gt_cls)
                        metr.setdefault(model, {}).setdefault(sig, {})[f"{amp:g}"] = cm
                        key = io.cell_key(model, sig, amp)
                        scores_npz[f"score|{key}"] = sc_vals
                        scores_npz[f"label|{key}"] = lab
                        if sc.spatial:
                            maps_npz[f"map|{key}"] = M.detection_map(sc_vals, sc.box_shape)
                            maps_npz[f"tgt|{key}"] = M.detection_map(
                                lab.astype(np.float32), sc.box_shape)
                        print(f"    {dname} {model:11s} {sig:11s} amp={amp:<5g} "
                              f"AUC={cm['auc']:.3f}", flush=True)

            d = io.run_dir(results_dir, cfg["provider"], sc.name, dname)
            log = dict(det.train_log()); log["fit_seconds"] = fit_t
            io.save_run(d, det, metr, scores_npz, maps_npz,
                        config={"run": cfg, "detector": dcfg, "scenario": sc.name},
                        train_log=log)
            summary.setdefault(sc.name, {})[dname] = metr
            print(f"  [{dname}] saved -> {d}", flush=True)

    return summary
