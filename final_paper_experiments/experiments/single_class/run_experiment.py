"""
Single-class IID experiment.

Pipeline (always in this order):
  1. Normalize full image  [or load pre-saved params if pretrained_dir is set]
  2. PCA on ALL pixels      [captures full image context]
  3. Compute target signature s = mean(tgt_pca) / ||mean(tgt_pca)||
  4. For each n_train in n_train_list:
       a. Sample n_train background pixels
       b. Load pre-trained DSM  OR  train DSM from scratch
       c. Optionally train LRao-IID from scratch
       d. Run all detectors for additive AND replacement test sets
       e. Save metrics
  5. Generate figures: ROC, AUC-vs-n, FA-perf

Pre-trained DSM mode (recommended after running run_pretrain.py):
  Set pretrained_dir in config — DSM is loaded from disk instead of trained.
  Only LRao-IID is trained from scratch (if run_lrao: true).

Usage:
    python -m final_paper_experiments.experiments.single_class.run_experiment \
        --config final_paper_experiments/experiments/single_class/config.yaml \
        --no-display
"""

import argparse
import json
import os
import sys
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, pca_reduce, compute_target_signature,
    compute_sigma_from_data, split_background, plant_targets,
    load_preprocessing,
)
from final_paper_experiments.experiments.pretrain.run_pretrain import (
    pretrain_subdir, dsm_checkpoint_dir,
)
from final_paper_experiments.checkpointing import Checkpointer
from final_paper_experiments.plotting import (
    plot_roc_curves, plot_false_alarm_perf, plot_auc_vs_n,
    copy_to_comparison_dir,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive, dsm_replacement,
    amf_replacement, lrao_iid, dltd, smglrt, gmm_glrt,
)
from final_paper_experiments.baselines.lrao_mlp import (
    TrafoMLP, train_lrao_mlp, detect_lrao_mlp,
)
from dsm_model import ScoreNet, train_dsm, train_lfi
from visualize_dataset import false_color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_id(cfg, norm_mode, seed):
    ds   = os.path.splitext(os.path.basename(cfg['dataset']))[0]
    arch = 'x'.join(str(h) for h in cfg['hidden_dims'])
    n_max = max(cfg['n_train_list'])
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    return (f"{ds}_cls{cfg['bkg_cls']}vs{cfg['target_cls']}"
            f"_{norm_mode}_{cfg['pca_dim']}d_{n_max}n_{arch}_s{seed}_{ts}")


def _auc(labels, scores):
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float('nan')


def _roc(labels, scores):
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr, tpr, _auc(labels, scores)
    except Exception:
        n = len(labels)
        return np.linspace(0, 1, n), np.linspace(0, 1, n), float('nan')


# ---------------------------------------------------------------------------
# Single seed, single normalization mode
# ---------------------------------------------------------------------------

def run_single(cfg: dict, norm_mode: str, seed: int, no_display: bool = True):
    run_id   = _run_id(cfg, norm_mode, seed)
    run_dir  = os.path.join(cfg['results_dir'], run_id)
    fig_dir  = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Save config copy
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump({**cfg, '_norm_mode': norm_mode, '_seed': seed}, f)

    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1+2. Normalize full image → PCA on all pixels
    # ------------------------------------------------------------------
    pretrained_dir = cfg.get('pretrained_dir', None)

    if pretrained_dir:
        # ── Load pre-saved preprocessing artifacts ─────────────────────
        sub = pretrain_subdir(pretrained_dir, norm_mode, cfg['pca_dim'])
        print(f"\n[1] Loading pre-trained preprocessing from {sub}")
        pca, _norm, _vmin, _ranges, gt_flat, class_pca = load_preprocessing(sub)

        bkg_pca = class_pca.get(cfg['bkg_cls'])
        tgt_pca = class_pca.get(cfg['target_cls'])
        if bkg_pca is None:
            raise ValueError(f"Class {cfg['bkg_cls']} not found in pretrained dir")
        if tgt_pca is None:
            raise ValueError(f"Class {cfg['target_cls']} not found in pretrained dir")

        print(f"  Background: class {cfg['bkg_cls']} ({len(bkg_pca)} pixels)")
        print(f"  Target:     class {cfg['target_cls']} ({len(tgt_pca)} pixels)")
        print(f"  [2] PCA already loaded ({cfg['pca_dim']}D)")
    else:
        # ── Compute fresh ───────────────────────────────────────────────
        print(f"\n[1] Loading {cfg['dataset']}  (normalization={norm_mode})")
        data, gt = load_and_normalize(cfg['dataset'], mode=norm_mode)
        H, W, B  = data.shape
        all_flat = data.reshape(-1, B)
        gt_flat  = gt.flatten()

        bkg_pixels = all_flat[gt_flat == cfg['bkg_cls']].copy()
        tgt_pixels = all_flat[gt_flat == cfg['target_cls']].copy()
        print(f"  Background: class {cfg['bkg_cls']} ({len(bkg_pixels)} pixels)")
        print(f"  Target:     class {cfg['target_cls']} ({len(tgt_pixels)} pixels)")

        print(f"\n[2] PCA → {cfg['pca_dim']} dims (fit on all pixels)")
        pca, bkg_pca, _, tgt_pca = pca_reduce(
            all_flat, bkg_pixels, bkg_pixels, tgt_pixels, cfg['pca_dim'])

    s = compute_target_signature(tgt_pca)
    print(f"  ||s|| = {np.linalg.norm(s):.4f}")

    # ------------------------------------------------------------------
    # 3. Metrics storage
    # ------------------------------------------------------------------
    n_train_list  = cfg['n_train_list']
    target_models = cfg['target_models']
    all_metrics   = {tm: {n: {} for n in n_train_list} for tm in target_models}
    auc_vs_n      = {tm: {} for tm in target_models}   # detector → [auc per n]

    # Fixed test split (same across all n_train)
    test_size = cfg['test_size']
    rng_global = np.random.default_rng(seed)
    idx = np.arange(len(bkg_pca))
    rng_global.shuffle(idx)
    bkg_pca_shuffled = bkg_pca[idx]

    # Reserve test pixels from the END of shuffled array to avoid overlap
    assert len(bkg_pca_shuffled) >= max(n_train_list) + test_size, \
        "Not enough background pixels for requested n_train_list + test_size"
    test_bkg = bkg_pca_shuffled[-(test_size):]

    # Plant targets on test (fixed for all n_train)
    test_sets = {}
    for tm in target_models:
        test_data, labels, _ = plant_targets(
            test_bkg, s,
            amplitude=cfg['amplitude'],
            tgt_fraction=cfg['target_fraction'],
            model=tm, seed=seed
        )
        test_sets[tm] = (test_data, labels)
        n_tgt = labels.sum()
        print(f"\n[3] Test set ({tm}): {test_size} pixels, {n_tgt} targets "
              f"(amplitude={cfg['amplitude']})")

    # ------------------------------------------------------------------
    # 4. Loop over n_train
    # ------------------------------------------------------------------
    for n_train in n_train_list:
        print(f"\n{'─'*50}")
        print(f"n_train = {n_train}")

        train_data = bkg_pca_shuffled[:n_train]

        # --- Compute σ ---
        if cfg['dsm_sigma'] == 'auto':
            sigma = compute_sigma_from_data(train_data, rho=cfg.get('dsm_sigma_rho', 0.01))
            print(f"  σ (auto) = {sigma:.5f}")
        else:
            sigma = float(cfg['dsm_sigma'])
            print(f"  σ (fixed) = {sigma:.5f}")

        # --- Train (or load) DSM ---
        dsm_model = ScoreNet(cfg['pca_dim'], cfg['hidden_dims'], cfg['activation'])
        dsm_loaded = False

        if pretrained_dir:
            sub = pretrain_subdir(pretrained_dir, norm_mode, cfg['pca_dim'])
            for fname in ('best_loss.pt', 'final.pt'):
                ckpt_path = os.path.join(
                    dsm_checkpoint_dir(sub, cfg['bkg_cls'], n_train),
                    'checkpoints', fname)
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, weights_only=True)
                    dsm_model.load_state_dict(ckpt['state_dict'])
                    print(f"\n  Loaded pre-trained DSM  (cls{cfg['bkg_cls']}, n={n_train})")
                    dsm_loaded = True
                    break
            if not dsm_loaded:
                print(f"\n  WARNING: pretrained DSM not found for "
                      f"cls{cfg['bkg_cls']}, n={n_train}. Training from scratch.")

        if not dsm_loaded:
            dsm_epochs  = cfg.get('dsm_epochs', cfg.get('epochs', 4000))
            dsm_run_dir = os.path.join(run_dir, f'n{n_train}', 'dsm')
            ckpt_dsm    = Checkpointer(dsm_run_dir, save_every=cfg['checkpoint_every'])
            print(f"\n  Training DSM (σ={sigma:.5f}, epochs={dsm_epochs}) ...")
            dsm_model = train_dsm(
                dsm_model, train_data, sigma,
                lr=cfg['lr'], batch_size=cfg['batch_size'],
                epochs=dsm_epochs, weight_decay=cfg['weight_decay'],
                print_every=max(1, dsm_epochs // 5),
                checkpointer=ckpt_dsm,
            )

        # --- Train LRao-IID ---
        lrao_model  = None
        lrao_epochs = cfg.get('lrao_epochs', cfg.get('epochs', 4000))
        if cfg.get('run_lrao', True):
            lrao_run_dir = os.path.join(run_dir, f'n{n_train}', 'lrao_iid')
            ckpt_lrao    = Checkpointer(lrao_run_dir, save_every=cfg['checkpoint_every'])
            print(f"\n  Training LRao-IID (epochs={lrao_epochs}) ...")
            lrao_model = ScoreNet(cfg['pca_dim'], cfg['hidden_dims'], cfg['activation'])
            lrao_model = train_lfi(
                lrao_model, train_data, s,
                delta_theta=cfg.get('lfi_delta_theta', 0.01),
                lr=cfg['lr'], batch_size=cfg['batch_size'],
                epochs=lrao_epochs, weight_decay=cfg['weight_decay'],
                print_every=max(1, lrao_epochs // 5),
                checkpointer=ckpt_lrao,
            )
        else:
            print(f"\n  Skipping LRao-IID  (run_lrao=false)")

        # --- Optionally train LRao-MLP ---
        lrao_mlp_model = None
        if cfg.get('run_lrao_mlp', False):
            print(f"\n  Training LRao-MLP ...")
            lrao_mlp_model = TrafoMLP(
                cfg['pca_dim'],
                hidden_dims=cfg.get('lrao_mlp_hidden', [64, 64]),
                activation='tanh',
            )
            mlp_run_dir  = os.path.join(run_dir, f'n{n_train}', 'lrao_mlp')
            ckpt_mlp     = Checkpointer(mlp_run_dir, save_every=cfg['checkpoint_every'])
            lrao_mlp_model = train_lrao_mlp(
                lrao_mlp_model, train_data, s,
                config={
                    'lr': cfg['lr'], 'batch_size': cfg['batch_size'],
                    'epochs': cfg.get('lrao_mlp_epochs', 2000),
                    'weight_decay': cfg['weight_decay'],
                    'delta_theta': cfg.get('lfi_delta_theta', 0.01),
                    'print_every': cfg.get('lrao_mlp_epochs', 2000) // 5,
                    'checkpointer': ckpt_mlp,
                }
            )

        # --- Evaluate all detectors ---
        for tm in target_models:
            test_data, labels = test_sets[tm]
            scores_dict = {}

            if tm == 'additive':
                scores_dict['AMF']      = amf(test_data, train_data, s)
                scores_dict['Reg-AMF']  = reg_amf(test_data, train_data, s, sigma)
                scores_dict['DSM']      = dsm_additive(test_data, train_data, dsm_model, s)
                scores_dict['GMM-GLRT'] = gmm_glrt(test_data, train_data, s, K=cfg['gmm_K'])
                scores_dict['DLTD']     = dltd(test_data, train_data, s, K=cfg['gmm_K'])
                scores_dict['SMGLRT']   = smglrt(test_data, train_data, s, K=cfg['gmm_K'])
                if lrao_model is not None:
                    scores_dict['LRao-IID'] = lrao_iid(test_data, train_data, lrao_model, s)
                if lrao_mlp_model is not None:
                    scores_dict['LRao-MLP'] = detect_lrao_mlp(
                        test_data, train_data, lrao_mlp_model, s)
            else:  # replacement
                scores_dict['AMF-rep']  = amf_replacement(test_data, train_data, s)
                scores_dict['DSM-rep']  = dsm_replacement(test_data, train_data, dsm_model, s)
                scores_dict['DLTD']     = dltd(test_data, train_data, s, K=cfg['gmm_K'])
                scores_dict['SMGLRT']   = smglrt(test_data, train_data, s, K=cfg['gmm_K'])
                if lrao_model is not None:
                    scores_dict['LRao-IID'] = lrao_iid(test_data, train_data, lrao_model, s)

            aucs = {det: _auc(labels, sc) for det, sc in scores_dict.items()}
            all_metrics[tm][n_train] = aucs
            print(f"\n  [{tm}] n={n_train}")
            for det, auc in aucs.items():
                print(f"    {det:<20}  AUC={auc:.4f}")

    # ------------------------------------------------------------------
    # 5. Build auc_vs_n structure
    # ------------------------------------------------------------------
    for tm in target_models:
        detectors = list(all_metrics[tm][n_train_list[0]].keys())
        auc_vs_n[tm] = {det: [all_metrics[tm][n][det] for n in n_train_list]
                        for det in detectors}

    # Save full metrics
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'n_train_list': n_train_list,
                   'auc_vs_n': auc_vs_n,
                   'all_metrics': {tm: {str(n): v
                                        for n, v in all_metrics[tm].items()}
                                   for tm in target_models},
                   'sigma': sigma,
                   'run_id': run_id}, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # ------------------------------------------------------------------
    # 6. Figures (at max n_train)
    # ------------------------------------------------------------------
    n_max     = max(n_train_list)
    train_max = bkg_pca_shuffled[:n_max]

    # Re-load best DSM model for ROC plot
    best_dsm_dir = os.path.join(run_dir, f'n{n_max}', 'dsm', 'checkpoints')
    best_dsm = ScoreNet(cfg['pca_dim'], cfg['hidden_dims'], cfg['activation'])
    if os.path.exists(os.path.join(best_dsm_dir, 'best_loss.pt')):
        ckpt = torch.load(os.path.join(best_dsm_dir, 'best_loss.pt'), weights_only=True)
        best_dsm.load_state_dict(ckpt['state_dict'])
    best_dsm.eval()

    best_lrao = None
    if cfg.get('run_lrao', True):
        best_lrao = ScoreNet(cfg['pca_dim'], cfg['hidden_dims'], cfg['activation'])
        best_lrao_dir = os.path.join(run_dir, f'n{n_max}', 'lrao_iid', 'checkpoints')
        if os.path.exists(os.path.join(best_lrao_dir, 'best_loss.pt')):
            ckpt = torch.load(os.path.join(best_lrao_dir, 'best_loss.pt'), weights_only=True)
            best_lrao.load_state_dict(ckpt['state_dict'])
        best_lrao.eval()

    for tm in target_models:
        test_data, labels = test_sets[tm]

        if tm == 'additive':
            roc_res = {
                'AMF':     _roc(labels, amf(test_data, train_max, s)),
                'Reg-AMF': _roc(labels, reg_amf(test_data, train_max, s, sigma)),
                'DSM':     _roc(labels, dsm_additive(test_data, train_max, best_dsm, s)),
                'DLTD':    _roc(labels, dltd(test_data, train_max, s, K=cfg['gmm_K'])),
                'SMGLRT':  _roc(labels, smglrt(test_data, train_max, s, K=cfg['gmm_K'])),
            }
            if best_lrao is not None:
                roc_res['LRao-IID'] = _roc(labels, lrao_iid(test_data, train_max, best_lrao, s))
        else:
            roc_res = {
                'AMF-rep': _roc(labels, amf_replacement(test_data, train_max, s)),
                'DSM-rep': _roc(labels, dsm_replacement(test_data, train_max, best_dsm, s)),
                'DLTD':    _roc(labels, dltd(test_data, train_max, s, K=cfg['gmm_K'])),
                'SMGLRT':  _roc(labels, smglrt(test_data, train_max, s, K=cfg['gmm_K'])),
            }
            if best_lrao is not None:
                roc_res['LRao-IID'] = _roc(labels, lrao_iid(test_data, train_max, best_lrao, s))

        roc_path = os.path.join(fig_dir, f'roc_{tm}.pdf')
        plot_roc_curves(roc_res, roc_path)
        print(f"  ROC figure saved → {roc_path}")

        fa_path = os.path.join(fig_dir, f'false_alarm_perf_{tm}.pdf')
        plot_false_alarm_perf(roc_res, fa_path)

        n_path = os.path.join(fig_dir, f'auc_vs_n_{tm}.pdf')
        plot_auc_vs_n(auc_vs_n[tm], n_train_list, n_path)
        print(f"  AUC-vs-n figure saved → {n_path}")

    # ------------------------------------------------------------------
    # 7. Copy to comparison dirs
    # ------------------------------------------------------------------
    comp_root = os.path.join(cfg['results_dir'], 'comparisons')
    for tm in target_models:
        for tag in ['pca_dim_effect', 'n_samples_effect', 'normalization_effect']:
            copy_to_comparison_dir(
                os.path.join(fig_dir, f'auc_vs_n_{tm}.pdf'),
                comp_root, tag, run_id)

    print(f"\nDone. Results in: {run_dir}")
    return run_dir, all_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='final_paper_experiments/experiments/single_class/config.yaml')
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    norm_modes = cfg['normalization']
    if isinstance(norm_modes, str):
        norm_modes = [norm_modes]

    num_seeds = cfg.get('num_seeds', 1)
    base_seed = cfg.get('base_seed', 42)

    for norm_mode in norm_modes:
        for seed in range(base_seed, base_seed + num_seeds):
            run_single(cfg, norm_mode, seed, no_display=args.no_display)


if __name__ == '__main__':
    main()
