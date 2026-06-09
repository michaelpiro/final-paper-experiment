"""
run_contaminated.py — Contaminated-Gaussian experiment (Ami's suggestion).

Replicates the theory contaminated-Gaussian experiment using real HSI data
and learned scores. Background = majority_cls (80%) + contam_cls (20%).
No PCA, no data-adaptive normalization — raw spectral bands, optionally
divided by a fixed scale_factor for numerical stability.

Detectors:
    AMF            — classical adaptive matched filter
    DSM-linear     — linear score network (no hidden layers)
    DSM-small      — shallow MLP
    DSM-medium     — medium MLP

Results: AUC vs n_train for each detector × amplitude, mean ± std over seeds.

Usage:
    .venv/bin/python -u experiments/honest_pipeline/run_contaminated.py
    .venv/bin/python -u experiments/honest_pipeline/run_contaminated.py \\
        --config experiments/honest_pipeline/contaminated.yaml
"""

import argparse, json, os, sys, time
from datetime import datetime

import numpy as np
import scipy.io
import torch
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dsm_model import ScoreNet, dsm_loss, compute_scores

# ---------------------------------------------------------------------------
# Class names (Pavia University)
# ---------------------------------------------------------------------------
CLS_NAMES = {1: 'asphalt', 2: 'meadows', 3: 'gravel', 4: 'trees',
             5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
             8: 'bricks', 9: 'shadows'}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auc_safe(labels, scores):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return float(roc_auc_score(labels, scores))


def plant(bkg, s, amp, frac, seed):
    """Additive model: y = w + amp * s."""
    rng = np.random.RandomState(seed)
    n = len(bkg); k = int(frac * n)
    pos = rng.choice(n, k, replace=False)
    y = bkg.copy().astype(np.float32)
    lab = np.zeros(n, dtype=int); lab[pos] = 1
    y[pos] += amp * s
    return y, lab


def compute_sigma(data, rho):
    """sigma^2 = rho * tr(Sigma) / D — noise level from training data."""
    var = data.var(axis=0).mean()
    return float(np.sqrt(rho * var))


# ---------------------------------------------------------------------------
# AMF
# ---------------------------------------------------------------------------

def amf_score(train, test, s):
    """Adaptive Matched Filter: (y-mu)^T Si s / sqrt(s^T Si s)."""
    mu    = train.mean(0)
    Sigma = np.cov(train, rowvar=False)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])
    Si_s  = np.linalg.solve(Sigma + 1e-8 * np.eye(len(Sigma)), s)
    denom = float(np.sqrt(max(s @ Si_s, 1e-12)))
    return ((test - mu) @ Si_s) / denom


# ---------------------------------------------------------------------------
# DSM training
# ---------------------------------------------------------------------------

def train_dsm(D, tr, sigma, hidden_dims, activation, cfg, seed, label=''):
    torch.manual_seed(seed)
    model = ScoreNet(D, list(hidden_dims), activation)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    X  = torch.tensor(tr, dtype=torch.float32)
    N  = len(X); bs = min(cfg['batch_size'], N)
    pbar = tqdm(range(cfg['dsm_epochs']),
                desc=f'  DSM [{label}]', dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N); tot = 0.
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        pbar.set_postfix(loss=f"{tot / max(N // bs, 1):.3f}")
    model.eval()
    return model


def dsm_additive_score(model, tr, te, s):
    """DSM additive LMP: -(psi(y) - psi_bar)^T s / sqrt(s^T C_psi s)."""
    psi_tr = compute_scores(model, tr)
    psi_te = compute_scores(model, te)
    psi_bar = psi_tr.mean(0)
    C = np.cov(psi_tr, rowvar=False)
    if C.ndim == 0:
        C = np.array([[float(C)]])
    norm = float(np.sqrt(max(float(s @ C @ s), 1e-12)))
    return -((psi_te - psi_bar) @ s) / norm


# ---------------------------------------------------------------------------
# One-seed run
# ---------------------------------------------------------------------------

def run_one_seed(seed, bkg_all, t_raw, cfg, n_list, amp_list, D):
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(bkg_all))
    n_max = max(n_list); n_test = cfg['test_n']
    assert len(bkg_all) >= n_max + n_test, \
        f"Need {n_max + n_test} bkg pixels, have {len(bkg_all)}"

    bkg_tr_full = bkg_all[idx[:n_max]]
    bkg_te      = bkg_all[idx[n_max:n_max + n_test]]

    amf_auc  = {n: {} for n in n_list}
    dsm_auc  = {arch: {n: {} for n in n_list}
                for arch in cfg['architectures']}

    for n in n_list:
        bkg_tr = bkg_tr_full[:n]
        sigma  = compute_sigma(bkg_tr, cfg['rho'])

        # ---- AMF ----
        for amp in amp_list:
            te, lab = plant(bkg_te, t_raw, amp, cfg['target_fraction'], seed)
            sc = amf_score(bkg_tr, te, t_raw)
            amf_auc[n][amp] = auc_safe(lab, sc)

        # ---- DSM variants ----
        for arch_name, hidden_dims in cfg['architectures'].items():
            dsm_m = train_dsm(D, bkg_tr, sigma, hidden_dims,
                              cfg['activation'], cfg, seed,
                              label=f'{arch_name} n={n}')
            for amp in amp_list:
                te, lab = plant(bkg_te, t_raw, amp, cfg['target_fraction'], seed)
                sc = dsm_additive_score(dsm_m, bkg_tr, te, t_raw)
                dsm_auc[arch_name][n][amp] = auc_safe(lab, sc)

        ref = amp_list[len(amp_list) // 2]
        dsm_str = '  '.join(
            f'{a}={dsm_auc[a][n][ref]:.3f}' for a in cfg['architectures'])
        print(f"  n={n:5d}  AMF={amf_auc[n][ref]:.3f}  {dsm_str}", flush=True)

    return amf_auc, dsm_auc


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg(vals):
    return [float(np.mean(vals)), float(np.std(vals))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(_EXP, 'contaminated.yaml'))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- load data ----
    mat     = scipy.io.loadmat(cfg['dataset'])
    data    = mat['data'].astype(np.float32)
    gt      = mat['map'].astype(int)
    H, W, D = data.shape
    flat    = data.reshape(-1, D)
    gt_flat = gt.reshape(-1)
    print(f"Image {H}×{W}×{D}", flush=True)

    # ---- optional fixed scaling ----
    scale = cfg.get('scale_factor')
    if scale:
        flat = flat / float(scale)
        print(f"Scaled by 1/{scale}  (range now [{flat.min():.3f}, {flat.max():.3f}])",
              flush=True)

    # ---- build contaminated background ----
    maj_cls    = cfg['bkg_majority_cls']
    con_cls    = cfg['bkg_contam_cls']
    con_frac   = cfg['bkg_contam_frac']
    tcls       = cfg['target_cls']

    px_maj = flat[gt_flat == maj_cls]
    px_con = flat[gt_flat == con_cls]
    tgt_all = flat[gt_flat == tcls]
    t_raw   = tgt_all.mean(0).astype(np.float32)

    print(f"Majority  cls {maj_cls} ({CLS_NAMES[maj_cls]}): {len(px_maj)} px", flush=True)
    print(f"Contam    cls {con_cls} ({CLS_NAMES[con_cls]}): {len(px_con)} px  "
          f"({int(con_frac*100)}% of background)", flush=True)
    print(f"Target    cls {tcls}  ({CLS_NAMES[tcls]}): {len(tgt_all)} px  (REMOVED)",
          flush=True)

    # mix: use all contaminant pixels, scale majority to match desired ratio
    # n_con / (n_con + n_maj) = con_frac  →  n_maj = n_con * (1-frac)/frac
    n_con_take = len(px_con)
    n_maj_take = min(int(n_con_take * (1 - con_frac) / con_frac), len(px_maj))
    rng0       = np.random.default_rng(0)
    idx_maj    = rng0.choice(len(px_maj), n_maj_take, replace=False)
    idx_con    = rng0.choice(len(px_con), n_con_take, replace=False)
    bkg_all    = np.concatenate([px_maj[idx_maj], px_con[idx_con]], axis=0)
    print(f"Background pool: {n_maj_take} majority + {n_con_take} contaminant "
          f"= {len(bkg_all)} px  "
          f"(actual contam {100*n_con_take/len(bkg_all):.1f}%)\n", flush=True)

    n_list   = sorted(cfg['n_train_list'])
    amp_list = sorted(cfg['amp_list'])
    seeds    = cfg['seeds']

    # ---- per-seed loop ----
    all_amf = []; all_dsm = []
    for k, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Seed {k+1}/{len(seeds)}  (seed={seed})", flush=True)
        print('='*60, flush=True)
        amf_auc, dsm_auc = run_one_seed(
            seed, bkg_all, t_raw, cfg, n_list, amp_list, D)
        all_amf.append(amf_auc)
        all_dsm.append(dsm_auc)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate ----
    agg_amf = {n: {amp: _agg([a[n][amp] for a in all_amf])
                   for amp in amp_list} for n in n_list}
    agg_dsm = {arch: {n: {amp: _agg([a[arch][n][amp] for a in all_dsm])
                          for amp in amp_list} for n in n_list}
               for arch in cfg['architectures']}

    # ---- save ----
    stamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir  = os.path.join(cfg['results_dir'], f'contaminated_{stamp}')
    os.makedirs(run_dir, exist_ok=True)

    metrics = dict(
        n_list=n_list, amp_list=amp_list, seeds=seeds,
        target_cls=tcls, bkg_majority_cls=maj_cls, bkg_contam_cls=con_cls,
        bkg_contam_frac=con_frac, scale_factor=scale,
        architectures=cfg['architectures'],
        amf=agg_amf, dsm=agg_dsm,
    )
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    import shutil
    shutil.copy(args.config, os.path.join(run_dir, 'config.yaml'))

    # ---- print summary ----
    print(f"\n{'='*60}", flush=True)
    print(f"Results saved → {run_dir}", flush=True)
    ref_amp = amp_list[len(amp_list) // 2]
    print(f"\nSummary (amp={ref_amp}, mean ± std over {len(seeds)} seeds):", flush=True)
    print(f"{'n':>6}  {'AMF':>10}", '  '.join(f'{a:>14}' for a in cfg['architectures']))
    for n in n_list:
        amf_m, amf_s = agg_amf[n][ref_amp]
        row = f"{n:6d}  {amf_m:.3f}±{amf_s:.3f}"
        for arch in cfg['architectures']:
            m2, s2 = agg_dsm[arch][n][ref_amp]
            row += f"  {m2:.3f}±{s2:.3f}"
        print(row, flush=True)


if __name__ == '__main__':
    main()
