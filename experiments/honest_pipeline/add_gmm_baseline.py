"""
add_gmm_baseline.py — compute ONLY the GMM-GLRT (oracle) baseline and merge
it into an existing sweep results dir, WITHOUT retraining any neural models.

Replicates the exact data split of run_sweep.run_one_seed (same seeds, same
n / target / background, same global_max scaling, same planted indices), so
the GMM-GLRT AUCs are directly comparable to the AMF/DSM/LRao already saved.

Usage:
    .venv/bin/python -u experiments/honest_pipeline/add_gmm_baseline.py \
        --run experiments/honest_pipeline/results/sweep_multi_n
"""

import argparse, os, sys, json, time
import numpy as np
import scipy.io

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP); sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import yaml
from sklearn.metrics import roc_auc_score
from final_paper_experiments.baselines.gmm_glrt_levin import GMMGLRTLevin

CLS_NAMES = {1:'asphalt',2:'meadows',3:'gravel',4:'trees',5:'metal_sheets',
             6:'bare_soil',7:'bitumen',8:'bricks',9:'shadows'}


def auc_safe(lab, sc):
    try:    return float(roc_auc_score(lab, sc))
    except: return float('nan')


def plant(bkg, s, amp, frac, model, seed):
    """Identical to run_sweep.plant — same seed → same indices/labels."""
    rng = np.random.RandomState(seed)
    n = len(bkg); k = int(frac * n)
    pos = rng.choice(n, k, replace=False)
    y = bkg.copy().astype(np.float32); lab = np.zeros(n, dtype=int); lab[pos] = 1
    if model == 'additive':   y[pos] += amp * s
    else:                     y[pos] = (1 - amp) * y[pos] + amp * s
    return y, lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='existing sweep results dir')
    args = p.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, 'config.yaml')))
    mfile = os.path.join(args.run, 'metrics.json')
    metrics = json.load(open(mfile))

    seeds    = cfg.get('seeds', [int(cfg.get('seed', 42))])
    n_list   = sorted(cfg['n_train_list'])
    amp_list = sorted(cfg.get('amp_list', [cfg['amplitude']]))
    print(f"Run dir : {args.run}")
    print(f"seeds={seeds}  n={n_list}  amp={amp_list}", flush=True)

    # ---- load RAW data ----
    mat  = scipy.io.loadmat(cfg['dataset'])
    data = mat['data'].astype(np.float32); gt = mat['map'].astype(int)
    flat = data.reshape(-1, data.shape[-1]); gt_flat = gt.reshape(-1)
    tcls = cfg['target_cls']
    if cfg.get('bkg_cls') is not None:
        bkg_all = flat[gt_flat == cfg['bkg_cls']]
    else:
        bkg_all = flat[(gt_flat != 0) & (gt_flat != tcls)]
    t_raw = flat[gt_flat == tcls].mean(0).astype(np.float32)
    print(f"bkg={len(bkg_all)}px  tgt cls {tcls}({CLS_NAMES.get(tcls,'?')})\n", flush=True)

    # ---- per-seed GMM-GLRT (oracle), same split as run_one_seed ----
    all_gmm = []
    for k, seed in enumerate(seeds):
        print(f"=== seed {k+1}/{len(seeds)} (seed={seed}) ===", flush=True)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(bkg_all))
        n_max = max(n_list); n_test = cfg['test_n']
        bkg_tr = bkg_all[idx[:n_max]]
        bkg_te = bkg_all[idx[n_max:n_max + n_test]]
        gm   = float(bkg_tr.max() + 1e-12)
        t_gm = t_raw / gm

        gmm_auc = {'additive': {}, 'replacement': {}}
        for n in n_list:
            t0 = time.time()
            det = GMMGLRTLevin(
                cond_tol=cfg.get('gmm_cond_tol', 1e3),
                max_dim=cfg.get('gmm_max_dim', 5),
                k_max=cfg.get('gmm_k_max', 5),
                seed=seed).fit(bkg_tr[:n] / gm)
            for tm in ('additive', 'replacement'):
                gmm_auc[tm][n] = {}
                for amp in amp_list:
                    te, lab = plant(bkg_te / gm, t_gm, amp,
                                    cfg['target_fraction'], tm, seed)
                    gmm_auc[tm][n][amp] = auc_safe(lab, det.score(te, t_gm, model=tm,
                                                                     p_steps=cfg.get('gmm_p_steps', 50),
                                                                     p_max=cfg.get('gmm_p_max', 1.0)))
            rA = amp_list[len(amp_list)//2]
            print(f"  n={n:>5}  add={gmm_auc['additive'][n][rA]:.3f} "
                  f"rep={gmm_auc['replacement'][n][rA]:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        all_gmm.append(gmm_auc)

    # ---- aggregate mean±std ----
    def agg(vals):
        a = np.array([v for v in vals if not np.isnan(v)])
        return [float(a.mean()), float(a.std())] if len(a) else [float('nan')]*2

    metrics['gmm'] = {
        tm: {str(n): {str(amp): agg([g[tm][n][amp] for g in all_gmm])
                       for amp in amp_list}
             for n in n_list}
        for tm in ('additive', 'replacement')}

    json.dump(metrics, open(mfile, 'w'), indent=2)
    print(f"\nMerged 'gmm' into {mfile}", flush=True)

    rA = amp_list[len(amp_list)//2]
    print(f"\nGMM-GLRT (mean over {len(seeds)} seeds, amp={rA}):")
    for tm in ('additive', 'replacement'):
        row = "  ".join(f"n={n}:{metrics['gmm'][tm][str(n)][str(rA)][0]:.3f}"
                        for n in n_list)
        print(f"  [{tm}] {row}")


if __name__ == '__main__':
    main()
