"""
Pre-training runner.

Pipeline (once for all future experiments):
  1. Load + normalize full image  (per_band or global)
  2. PCA on ALL pixels             (captures full image context)
  3. Save preprocessing artifacts  (pca, norm params, per-class PCA pixels)
  4. For each class × each n_train:
       Sample n pixels from that class, train DSM, save checkpoints.

After this runs, any single_class or multiclass experiment can load the
pre-trained DSM directly instead of re-training.

Usage:
    python -m final_paper_experiments.experiments.pretrain.run_pretrain \
        --config final_paper_experiments/experiments/pretrain/config.yaml
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, pca_reduce,
    compute_sigma_from_data,
    save_preprocessing,
)
from final_paper_experiments.checkpointing import Checkpointer
from dsm_model import ScoreNet, train_dsm


def pretrain_subdir(pretrained_dir: str, norm_mode: str, pca_dim: int) -> str:
    return os.path.join(pretrained_dir, f'{norm_mode}_{pca_dim}d')


def rho_str(rho: float) -> str:
    """Canonical string for a rho value, e.g. 0.01 → 'rho0.01', 0.005 → 'rho0.005'."""
    # Strip trailing zeros after decimal so 0.010 → 'rho0.01', 0.100 → 'rho0.1'
    return f'rho{rho:.10f}'.rstrip('0').rstrip('.')


def dsm_checkpoint_dir(base: str, cls_id: int, n: int, rho: float) -> str:
    """
    Path structure:  base / cls{C} / n{n} / rho{rho} / dsm
    Preprocessing artifacts (PCA, norm params) live in base/ and are rho-independent.
    Only DSM checkpoints are rho-specific.
    """
    return os.path.join(base, f'cls{cls_id}', f'n{n}', rho_str(rho), 'dsm')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='final_paper_experiments/experiments/pretrain/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    norm_mode     = cfg['normalization']
    pca_dim       = cfg['pca_dim']
    pretrained_dir = cfg['pretrained_dir']
    save_dir      = pretrain_subdir(pretrained_dir, norm_mode, pca_dim)

    print('=' * 60)
    print(f'Pre-training  [{norm_mode}, PCA={pca_dim}d]')
    print(f'Output dir: {save_dir}')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Step 1+2: Normalize full image, fit PCA on all pixels
    # ------------------------------------------------------------------
    print(f'\n[1] Loading + normalizing  ({norm_mode}) ...')
    data, gt = load_and_normalize(cfg['dataset'], mode=norm_mode)
    H, W, B  = data.shape
    all_flat = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    print(f'[2] Fitting PCA on all {len(all_flat)} pixels → {pca_dim}D ...')
    pca, _, _, _ = pca_reduce(all_flat, all_flat[:1], all_flat[:1], all_flat[:1], pca_dim)

    # Compute per-band normalization parameters for saving
    if norm_mode == 'global':
        lo = data.min()
        hi = data.max()
        # Broadcast to (B,) for consistent storage
        vmin   = np.full(B, lo)
        ranges = np.full(B, hi - lo + 1e-12)
    else:  # per_band
        raw     = np.load(cfg['dataset']) if cfg['dataset'].endswith('.npy') else None
        # Re-derive from raw .mat
        import scipy.io
        mat_raw = scipy.io.loadmat(cfg['dataset'])
        raw_img = mat_raw['data'].astype(np.float64)
        vmin   = raw_img.min(axis=(0, 1))                   # (B,)
        ranges = raw_img.max(axis=(0, 1)) - vmin + 1e-12    # (B,)

    # Per-class PCA pixels
    exclude = set(cfg.get('exclude_classes', [0]))
    class_ids_all = sorted([int(c) for c in np.unique(gt_flat) if int(c) not in exclude])

    tc = cfg.get('classes', 'all')
    if tc == 'all':
        target_classes = class_ids_all
    else:
        target_classes = [int(c) for c in tc]

    print(f'[3] Saving preprocessing artifacts ...')
    class_pixels_pca = {}
    for cls_id in class_ids_all:
        raw_pixels = all_flat[gt_flat == cls_id]
        class_pixels_pca[cls_id] = pca.transform(raw_pixels)

    save_preprocessing(save_dir, pca, norm_mode, vmin, ranges,
                        gt_flat, class_pixels_pca)
    print(f'    Saved to {save_dir}')

    # ------------------------------------------------------------------
    # Step 4: Train DSM for each class × each n_train
    # rho is encoded in the checkpoint path so different rho values
    # coexist without overwriting each other.
    # ------------------------------------------------------------------
    n_train_list  = cfg['n_train_list']
    dsm_epochs    = cfg['dsm_epochs']
    base_seed     = cfg.get('base_seed', 42)
    rho           = cfg.get('dsm_sigma_rho', 0.01)

    print(f'\n[4] Training DSM  (rho={rho}, epochs={dsm_epochs})')

    total_runs = len(target_classes) * len(n_train_list)
    run_count  = 0
    t0_all     = time.time()

    for cls_id in target_classes:
        cls_pca = class_pixels_pca[cls_id]
        n_avail = len(cls_pca)
        print(f'\n{"─"*50}')
        print(f'Class {cls_id}  ({n_avail} pixels)  [rho={rho}]')

        # Fixed shuffle for this class (seed independent of rho)
        rng = np.random.default_rng(base_seed + cls_id)
        idx = np.arange(n_avail)
        rng.shuffle(idx)
        cls_pca_shuffled = cls_pca[idx]

        for n_train in n_train_list:
            run_count += 1

            if n_train > n_avail:
                print(f'  n={n_train:>5}  SKIP (only {n_avail} pixels available)')
                continue

            # rho is part of the checkpoint path
            ckpt_dir = dsm_checkpoint_dir(save_dir, cls_id, n_train, rho)

            # Skip if already trained
            final_path = os.path.join(ckpt_dir, 'checkpoints', 'final.pt')
            if os.path.exists(final_path):
                print(f'  n={n_train:>5}  [{rho_str(rho)}]  already trained — skipping')
                continue

            train_data = cls_pca_shuffled[:n_train]
            sigma = (compute_sigma_from_data(train_data, rho)
                     if cfg['dsm_sigma'] == 'auto' else float(cfg['dsm_sigma']))

            t0 = time.time()
            print(f'  n={n_train:>5}  σ={sigma:.5f}  [{run_count}/{total_runs}] ...',
                  end='', flush=True)

            ckpt = Checkpointer(ckpt_dir, save_every=cfg['checkpoint_every'])
            model = ScoreNet(pca_dim, cfg['hidden_dims'], cfg['activation'])
            model = train_dsm(
                model, train_data, sigma,
                lr=cfg['lr'], batch_size=cfg['batch_size'],
                epochs=dsm_epochs, weight_decay=cfg['weight_decay'],
                print_every=0,      # silent — progress shown per-n
                checkpointer=ckpt,
            )
            elapsed = time.time() - t0
            print(f'  done in {elapsed:.0f}s')

    elapsed_all = time.time() - t0_all
    print(f'\n{"="*60}')
    print(f'Pre-training complete in {elapsed_all/60:.1f} min')
    print(f'Artifacts saved to: {save_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
