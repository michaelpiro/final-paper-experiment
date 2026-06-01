"""
Multi-class IID experiment.

Background = proportional mix of all non-target classes.
Loops over each target class (or a specified subset).
Adds GMM-GLRT as an additional oracle baseline.

Usage:
    python -m final_paper_experiments.experiments.multiclass.run_experiment \
        --config final_paper_experiments/experiments/multiclass/config.yaml \
        --no-display
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from final_paper_experiments.data_utils import (
    load_and_normalize, pca_reduce, compute_target_signature,
    compute_sigma_from_data, split_background, plant_targets,
)
from final_paper_experiments.checkpointing import Checkpointer
from final_paper_experiments.plotting import (
    plot_roc_curves, plot_false_alarm_perf, plot_auc_vs_n,
    plot_auc_per_class, copy_to_comparison_dir,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive, dsm_replacement,
    amf_replacement, lrao_iid, dltd, smglrt, gmm_glrt,
)
from final_paper_experiments.baselines.lrao_mlp import (
    TrafoMLP, train_lrao_mlp, detect_lrao_mlp,
)
from dsm_model import ScoreNet, train_dsm, train_lfi
from multiclass_experiment import proportional_counts


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


def run_one_target(cfg, all_flat, gt_flat, pca,
                   target_cls, class_ids,
                   norm_mode, seed, results_dir):
    """Run all n_train experiments for a single target class."""

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ds = os.path.splitext(os.path.basename(cfg['dataset']))[0]
    arch = 'x'.join(str(h) for h in cfg['hidden_dims'])
    n_max = max(cfg['n_train_list'])
    run_id  = (f"{ds}_multi_tgt{target_cls}"
               f"_{norm_mode}_{cfg['pca_dim']}d_{n_max}n_{arch}_s{seed}_{ts}")
    run_dir = os.path.join(results_dir, run_id)
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump({**cfg, '_norm_mode': norm_mode,
                   '_seed': seed, '_target_cls': target_cls}, f)

    print(f"\n{'='*60}")
    print(f"Target class: {target_cls}  |  Run: {run_id}")

    # ---- PCA-transform pixels ----
    tgt_pixels  = all_flat[gt_flat == target_cls].copy()
    bkg_classes = [c for c in class_ids
                   if c != target_cls
                   and c not in cfg.get('exclude_classes', [0])]

    # Collect background pixels proportionally
    class_sizes = {c: int((gt_flat == c).sum()) for c in bkg_classes}
    total_bkg   = sum(class_sizes.values())
    bkg_pixels  = np.vstack([all_flat[gt_flat == c] for c in bkg_classes])

    _, bkg_pca, _, tgt_pca = pca_reduce(
        all_flat, bkg_pixels, bkg_pixels, tgt_pixels, cfg['pca_dim'])
    s = compute_target_signature(tgt_pca)

    # Shuffle background
    rng = np.random.default_rng(seed)
    idx = np.arange(len(bkg_pca))
    rng.shuffle(idx)
    bkg_pca = bkg_pca[idx]

    test_size = cfg['test_size']
    assert len(bkg_pca) >= max(cfg['n_train_list']) + test_size, \
        f"Not enough background pixels for class {target_cls}"

    test_bkg = bkg_pca[-test_size:]

    # Plant targets
    test_sets = {}
    for tm in cfg['target_models']:
        td, labels, _ = plant_targets(
            test_bkg, s, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        test_sets[tm] = (td, labels)

    n_train_list = cfg['n_train_list']
    all_metrics  = {tm: {n: {} for n in n_train_list} for tm in cfg['target_models']}
    auc_vs_n     = {tm: {} for tm in cfg['target_models']}

    for n_train in n_train_list:
        train_data = bkg_pca[:n_train]

        sigma = (compute_sigma_from_data(train_data, cfg.get('dsm_sigma_rho', 0.01))
                 if cfg['dsm_sigma'] == 'auto' else float(cfg['dsm_sigma']))

        # Train DSM
        ckpt_dsm = Checkpointer(
            os.path.join(run_dir, f'n{n_train}', 'dsm'), cfg['checkpoint_every'])
        dsm_model = ScoreNet(cfg['pca_dim'], cfg['hidden_dims'], cfg['activation'])
        dsm_model = train_dsm(dsm_model, train_data, sigma,
                              lr=cfg['lr'], batch_size=cfg['batch_size'],
                              epochs=cfg['epochs'], weight_decay=cfg['weight_decay'],
                              print_every=cfg['epochs'] // 5,
                              checkpointer=ckpt_dsm)

        # Train LRao-IID
        ckpt_lrao = Checkpointer(
            os.path.join(run_dir, f'n{n_train}', 'lrao_iid'), cfg['checkpoint_every'])
        lrao_model = ScoreNet(cfg['pca_dim'], cfg['hidden_dims'], cfg['activation'])
        lrao_model = train_lfi(lrao_model, train_data, s,
                               delta_theta=cfg.get('lfi_delta_theta', 0.01),
                               lr=cfg['lr'], batch_size=cfg['batch_size'],
                               epochs=cfg['epochs'], weight_decay=cfg['weight_decay'],
                               print_every=cfg['epochs'] // 5,
                               checkpointer=ckpt_lrao)

        for tm in cfg['target_models']:
            test_data, labels = test_sets[tm]
            scores = {}
            if tm == 'additive':
                scores['AMF']      = amf(test_data, train_data, s)
                scores['Reg-AMF']  = reg_amf(test_data, train_data, s, sigma)
                scores['DSM']      = dsm_additive(test_data, train_data, dsm_model, s)
                scores['LRao-IID'] = lrao_iid(test_data, train_data, lrao_model, s)
                scores['GMM-GLRT'] = gmm_glrt(test_data, train_data, s, K=cfg['gmm_K'])
                scores['DLTD']     = dltd(test_data, train_data, s, K=cfg['gmm_K'])
                scores['SMGLRT']   = smglrt(test_data, train_data, s, K=cfg['gmm_K'])
            else:
                scores['AMF-rep']  = amf_replacement(test_data, train_data, s)
                scores['DSM-rep']  = dsm_replacement(test_data, train_data, dsm_model, s)
                scores['LRao-IID'] = lrao_iid(test_data, train_data, lrao_model, s)
                scores['DLTD']     = dltd(test_data, train_data, s, K=cfg['gmm_K'])
                scores['SMGLRT']   = smglrt(test_data, train_data, s, K=cfg['gmm_K'])

            aucs = {det: _auc(labels, sc) for det, sc in scores.items()}
            all_metrics[tm][n_train] = aucs

    for tm in cfg['target_models']:
        dets = list(all_metrics[tm][n_train_list[0]].keys())
        auc_vs_n[tm] = {d: [all_metrics[tm][n][d] for n in n_train_list] for d in dets}

    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'target_cls': target_cls, 'n_train_list': n_train_list,
                   'auc_vs_n': auc_vs_n,
                   'all_metrics': {tm: {str(n): v for n, v in all_metrics[tm].items()}
                                   for tm in cfg['target_models']},
                   'run_id': run_id}, f, indent=2)

    # Figures
    n_max_train = bkg_pca[:max(n_train_list)]
    for tm in cfg['target_models']:
        test_data, labels = test_sets[tm]
        roc_res = {}
        for det, sc_list in zip(list(auc_vs_n[tm].keys()),
                                [auc_vs_n[tm][d] for d in auc_vs_n[tm]]):
            pass  # just use last n_train scores
        # Quick ROC from last n_train scores
        td, labels = test_sets[tm]
        td_max = n_max_train
        sigma_max = (compute_sigma_from_data(td_max, cfg.get('dsm_sigma_rho', 0.01))
                     if cfg['dsm_sigma'] == 'auto' else float(cfg['dsm_sigma']))
        if tm == 'additive':
            roc_res = {
                'AMF':  _roc(labels, amf(td, td_max, s)),
                'DSM':  _roc(labels, dsm_additive(td, td_max, dsm_model, s)),
                'DLTD': _roc(labels, dltd(td, td_max, s, K=cfg['gmm_K'])),
            }
        else:
            roc_res = {
                'AMF-rep':  _roc(labels, amf_replacement(td, td_max, s)),
                'DSM-rep':  _roc(labels, dsm_replacement(td, td_max, dsm_model, s)),
                'DLTD':     _roc(labels, dltd(td, td_max, s, K=cfg['gmm_K'])),
            }
        plot_roc_curves(roc_res, os.path.join(fig_dir, f'roc_{tm}.pdf'))
        plot_auc_vs_n(auc_vs_n[tm], n_train_list,
                      os.path.join(fig_dir, f'auc_vs_n_{tm}.pdf'))

    return run_id, all_metrics


def run_experiment(cfg, norm_mode, seed):
    ds = cfg['dataset']
    print(f"\nLoading {ds}  (norm={norm_mode})")
    data, gt = load_and_normalize(ds, mode=norm_mode)
    H, W, B  = data.shape
    all_flat = data.reshape(-1, B)
    gt_flat  = gt.flatten()

    class_ids = sorted([int(c) for c in np.unique(gt_flat)
                        if c not in cfg.get('exclude_classes', [0])])

    # Fit PCA once on all pixels
    from sklearn.decomposition import PCA
    pca = PCA(n_components=cfg['pca_dim']).fit(all_flat)

    tc = cfg.get('target_classes', 'all')
    if tc == 'all':
        target_classes = class_ids
    else:
        target_classes = list(tc)

    results_dir = cfg['results_dir']
    auc_per_class = {tm: {} for tm in cfg['target_models']}

    for target_cls in target_classes:
        run_id, metrics = run_one_target(
            cfg, all_flat, gt_flat, pca,
            target_cls, class_ids,
            norm_mode, seed, results_dir)
        for tm in cfg['target_models']:
            n_max = max(cfg['n_train_list'])
            auc_per_class[tm][target_cls] = {
                det: metrics[tm][n_max][det]
                for det in metrics[tm][n_max]}

    # Per-class AUC summary figure
    fig_dir = os.path.join(results_dir, f'multiclass_{norm_mode}_s{seed}')
    os.makedirs(fig_dir, exist_ok=True)
    for tm in cfg['target_models']:
        det_list = list(next(iter(auc_per_class[tm].values())).keys())
        res = {det: {cls: auc_per_class[tm][cls][det]
                     for cls in target_classes}
               for det in det_list}
        plot_auc_per_class(res, os.path.join(fig_dir, f'auc_per_class_{tm}.pdf'))

    summary = os.path.join(fig_dir, 'auc_per_class_summary.json')
    with open(summary, 'w') as f:
        json.dump(auc_per_class, f, indent=2)
    print(f"\nMulticlass summary → {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='final_paper_experiments/experiments/multiclass/config.yaml')
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
            run_experiment(cfg, norm_mode, seed)


if __name__ == '__main__':
    main()
