"""
diag_dsm_core.py — Isotropic vs Diagonal DSM noise comparison.

Compares TWO DSM noise models on the IID single-class problem:

  DSM-iso   Σ_n = σ²I,            σ²  = ρ·(1/d)·tr(Σ̂)        [current DSM]
  DSM-diag  Σ_n = diag(σ_b²),     σ_b = sqrt(ρ·Var_b)        [data-driven diagonal]

Both models share the SAME normalized data, the SAME PCA-d latent space, the
SAME architecture / lr / epochs and IDENTICAL weight init (same seed).  They
differ ONLY in the DSM training-noise covariance.  No other baselines are run.

Sweeps over latent_dim and n_train, exactly like iid_core.run_iid.  Saves
config.yaml, metrics.json, loss_curves.json and auc_vs_n figures to a
timestamped run dir, matching the other two experiments' output convention.
"""
import os
import json
import time
from datetime import datetime
from typing import List

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, compute_sigma_diag_from_data,
)
from final_paper_experiments.baselines.detectors import dsm_additive, dsm_replacement
from dsm_model import ScoreNet, dsm_loss

COLORS = {'DSM-iso': '#1f77b4', 'DSM-diag': '#d62728'}


def _ensure_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _train_dsm(train_lat, sigma, weighted, cfg, seed, label):
    """Train one DSM (sigma scalar=iso or (d,)=diag). Returns (model, loss_hist)."""
    torch.manual_seed(seed)                               # identical init both models
    d = train_lat.shape[1]
    model = ScoreNet(d, list(cfg['hidden_dims']), cfg['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    X = torch.tensor(train_lat, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch_size'], len(train_lat))
    hist = []
    pbar = tqdm(range(1, cfg['dsm_epochs'] + 1), desc=label,
                dynamic_ncols=True, leave=False)
    for _ in pbar:
        perm = torch.randperm(N); tot = 0.0; nb = 0
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma, weighted=weighted)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
        pbar.set_postfix(loss=f"{hist[-1]:.4f}")
    model.eval()
    return model, hist


def _plot_auc_vs_n(metrics, d, tm, n_list, out_pdf):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for det in ('DSM-iso', 'DSM-diag'):
        ys = metrics[f'd_{d}'][tm][det]
        ax.plot(n_list, ys, 'o-', label=det, color=COLORS[det], lw=2, ms=5)
    ax.set_xscale('log')
    ax.set_xlabel('n_train'); ax.set_ylabel('AUC')
    ax.set_title(f'{tm}  (d={d})')
    ax.grid(True, alpha=0.3); ax.legend()
    fig.savefig(out_pdf, bbox_inches='tight'); plt.close(fig)


def run_diag_dsm(cfg: dict):
    t0 = time.time()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'diag_dsm_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'))
    print(f"Run dir: {run_dir}", flush=True)

    n_list = sorted(set(_ensure_list(cfg['n_train_list'])))
    d_list = sorted(set(_ensure_list(cfg['latent_dim_list'])))
    seed = int(cfg['seed'])
    print(f"n_train_list = {n_list}\nlatent_dim_list = {d_list}", flush=True)

    data, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    flat = data.reshape(-1, data.shape[-1])
    gt_flat = gt.reshape(-1)
    bkg = flat[gt_flat == cfg['bkg_cls']]
    tgt = flat[gt_flat == cfg['target_cls']]
    s_raw = tgt.mean(axis=0)
    print(f"norm={cfg['norm_mode']}  bkg={len(bkg)}  tgt={len(tgt)}  "
          f"||s_raw||={np.linalg.norm(s_raw):.4f}\n", flush=True)

    metrics = {'n_train_list': n_list, 'latent_dim_list': d_list,
               'norm_mode': cfg['norm_mode']}
    loss_curves = {}

    for d in d_list:
        print(f"=== d={d} ===", flush=True)
        pca = PCA(n_components=d, random_state=seed).fit(flat)
        print(f"  PCA d={d}  explained var = {pca.explained_variance_ratio_.sum():.4f}",
              flush=True)
        s_pca_add = (pca.components_ @ s_raw).astype(np.float32)
        s_pca_rep = pca.transform(s_raw[None]).flatten().astype(np.float32)

        metrics[f'd_{d}'] = {
            'additive':    {'DSM-iso': [], 'DSM-diag': []},
            'replacement': {'DSM-iso': [], 'DSM-diag': []},
            'sigma_iso': [], 'sigma_diag': [],
        }

        for n in n_list:
            rng = np.random.RandomState(seed + n)
            idx = rng.permutation(len(bkg))
            tr_raw = bkg[idx[:n]]
            te_raw = bkg[idx[n:n + cfg['test_size']]].copy()

            n_pos = int(cfg['target_fraction'] * len(te_raw))
            pos = rng.choice(len(te_raw), n_pos, replace=False)
            labels = np.zeros(len(te_raw), dtype=int); labels[pos] = 1
            te_raw[pos] += cfg['amplitude'] * s_raw

            tr = pca.transform(tr_raw).astype(np.float32)
            te = pca.transform(te_raw).astype(np.float32)

            sig_iso  = compute_sigma_from_data(tr, cfg['dsm_sigma_rho'])
            sig_diag = compute_sigma_diag_from_data(tr, cfg['dsm_sigma_rho'])

            m_iso,  h_iso  = _train_dsm(tr, sig_iso,  False, cfg, seed,
                                        f'iso  d={d} n={n}')
            m_diag, h_diag = _train_dsm(tr, sig_diag, True,  cfg, seed,
                                        f'diag d={d} n={n}')

            au = lambda sc: float(roc_auc_score(labels, sc))
            res = {
                ('additive', 'DSM-iso'):  au(dsm_additive(te, tr, m_iso,  s_pca_add)),
                ('additive', 'DSM-diag'): au(dsm_additive(te, tr, m_diag, s_pca_add)),
                ('replacement', 'DSM-iso'):  au(dsm_replacement(te, tr, m_iso,  s_pca_rep)),
                ('replacement', 'DSM-diag'): au(dsm_replacement(te, tr, m_diag, s_pca_rep)),
            }
            for (tm, det), v in res.items():
                metrics[f'd_{d}'][tm][det].append(v)
            metrics[f'd_{d}']['sigma_iso'].append(float(sig_iso))
            metrics[f'd_{d}']['sigma_diag'].append(sig_diag.tolist())
            loss_curves[f'd{d}_n{n}_iso'] = h_iso
            loss_curves[f'd{d}_n{n}_diag'] = h_diag

            print(f"  n={n:<5}  "
                  f"ADD iso={res[('additive','DSM-iso')]:.3f} "
                  f"diag={res[('additive','DSM-diag')]:.3f}  |  "
                  f"REP iso={res[('replacement','DSM-iso')]:.3f} "
                  f"diag={res[('replacement','DSM-diag')]:.3f}", flush=True)

            json.dump(metrics, open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)
            json.dump(loss_curves, open(os.path.join(run_dir, 'loss_curves.json'), 'w'))

        _plot_auc_vs_n(metrics, d, 'additive', n_list,
                       os.path.join(fig_dir, f'auc_vs_n_additive_d{d}.pdf'))
        _plot_auc_vs_n(metrics, d, 'replacement', n_list,
                       os.path.join(fig_dir, f'auc_vs_n_replacement_d{d}.pdf'))

    print(f"\nDone in {time.time()-t0:.0f}s.  Results: {run_dir}", flush=True)
    return run_dir
