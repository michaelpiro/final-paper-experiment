"""
run_compare.py — Focused spatial detector comparison (single scenario).

Detectors
---------
  DSM            — our global per-pixel score (ScoreNet, no spatial context)
  CF-Attn        — spatial score net, global normalization      (Ours, spatial)
  CF-Attn-CFAR   — spatial score net, local-Fisher normalization (Ours, spatial)
  NeighborMLP    — spatial denoiser score net                    (Ours, spatial)
  AMF            — Adaptive Matched Filter (global SCM)
  AMF-local      — AMF on the per-pixel k×k window SCM (same window as spatial nets)
  CEM            — Constrained Energy Minimization (global autocorrelation)
  CEM-local      — CEM on the per-pixel k×k window autocorrelation
  GMM-Levin      — Levin product-GMM GLRT

Deep nets train on GPU (cuda) when available.

Metrics (per detector)
----------------------
  pAUC@0.05      — partial AUC over Pfa < 0.05
  AUC            — full ROC AUC
  Pd@Pfa=0.05    — detection rate at Pfa = 0.05
  Pfa per class  — false-alarm rate on each background class (CFAR thr @ 0.05)
  Pfa_avg/max    — macro-average / worst-class background false-alarm rate

Deliverables (saved in <results_dir>/compare_<ts>/)
---------------------------------------------------
  false_color.pdf        — RGB false color of the test box
  label_map.pdf          — ground-truth class map of the test box
  detection_maps.pdf     — per-detector spatial score maps
  roc.pdf                — all-detector ROC overlay
  pfa_per_class.pdf      — grouped per-class Pfa bars
  summary_table.csv/.md  — metric comparison table
  metrics.json, scores.npz

Usage
-----
  .venv/bin/python -u experiments/spatial/run_compare.py --dry-run
  python -u experiments/spatial/run_compare.py --config experiments/spatial/colab.yaml \
        --results_dir /content/drive/MyDrive/final_paper/compare_results
"""
import argparse, json, os, sys, time
from datetime import datetime

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import numpy as np
import torch
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from final_paper_experiments.data_utils import load_and_normalize, plant_targets
from final_paper_experiments.baselines.detectors import dsm_additive
from final_paper_experiments.baselines.gmm_glrt_levin import gmm_glrt_levin_additive
from final_paper_experiments.evaluation import (
    partial_auc, dr_at_fpr, auc_safe, roc_safe, cfar_threshold, per_class_fpr,
    compute_signature, generate_random_boxes, scores_to_spatial_map,
)
from cfattn_model import score_cfattn_additive, score_cfattn_additive_cfar
from neighbor_mlp_model import score_nmlp_additive
from local_detectors import amf_cem_local_scm, amf_global, cem_global

# Reuse the (whitening-aware, GPU) training helpers from the main runner.
from run_colab import (
    _crop_pca_box, _train_dsm, _train_cfattn, _train_nmlp,
)

CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks',    9: 'shadows',
}
CLS_COLORS_HEX = {
    0: '#000000', 1: '#808080', 2: '#00cc44', 3: '#d2691e',
    4: '#006400', 5: '#add8e6', 6: '#a52a2a', 7: '#9400d3',
    8: '#ff4500', 9: '#00008b',
}

# Fixed display order + colors for the 9 detectors.
DET_ORDER = ['DSM', 'CF-Attn', 'CF-Attn-CFAR', 'NeighborMLP',
             'AMF', 'AMF-local', 'CEM', 'CEM-local', 'GMM-Levin']
DET_COLORS = {
    'DSM':          '#ff7f0e',
    'CF-Attn':      '#aec7e8',
    'CF-Attn-CFAR': '#1f77b4',
    'NeighborMLP':  '#2ca02c',
    'AMF':          '#9467bd',
    'AMF-local':    '#c5b0d5',
    'CEM':          '#8c564b',
    'CEM-local':    '#e7969c',
    'GMM-Levin':    '#e377c2',
}

DEFAULT_CFG = dict(
    dataset='data/pavia-u.mat',
    norm_mode='none',
    manual_boxes_path='experiments/spatial/manual_boxes.json',
    scenario_index=0,            # which manual/random scenario to compare on
    min_pixels=2000,
    random_scenario_seeds=[42, 123, 456, 789],
    sig_dom_weight=0.8, sig_mean_weight=0.2,
    amplitude=0.15, target_fraction=0.10,
    n_budget=None,               # None = use the FULL train box (no subsampling).
                                 # int  = cut pixels from the SIDES only to a
                                 #        contiguous sub-box (spatial context kept).
    k=5,                         # spatial window (k×k) for nbr nets + local SCM
    local_scm_loading=1e-8,      # minimal diagonal loading for local SCMs (≈ no loading)
    baseline_eig_floor=1e-12,    # relative eigenvalue floor for AMF/CEM global (baselines)
    # CF-Attn
    cfattn_h=64, cfattn_K=9, cfattn_epochs=300, cfattn_lr=3e-4, cfattn_eps=1e-4,
    lam_ent=0.05, lam_div=0.05, lam_cov=1e-5,
    # NeighborMLP
    nmlp_d_lat=32, nmlp_K=8, nmlp_hidden=128, nmlp_n_layers=3,
    nmlp_epochs=300, nmlp_lr=3e-4, nmlp_batch=256,
    # DSM
    dsm_hidden=[64, 64], dsm_epochs=1000, dsm_lr=5e-4,
    # shared
    activation='silu', dsm_sigma_rho=0.01,
    whiten_mode='zca', whiten_eig_floor=1e-3,   # OUR nets' whitening floor (unchanged)
    batch_size=256, weight_decay=1e-4,
    gmm_steps=50, gmm_K=3,
    pfa_target=0.05,
    seed=42,
    results_dir='final_paper_experiments/results',
)

DRYRUN_OVERRIDES = dict(
    cfattn_epochs=8, nmlp_epochs=8, dsm_epochs=20,
    cfattn_K=4, nmlp_K=4,
    n_budget=400,   # dry-run only: contiguous side-crop for speed (still spatial)
)


# ---------------------------------------------------------------------------
def _side_crop_box(box, budget):
    """Cut pixels from the SIDES only → a contiguous, centered sub-box of about
    `budget` pixels. Preserves spatial context (no random subsampling).
    budget None/0/>=box ⇒ the full box is returned unchanged."""
    r0, r1, c0, c1 = box
    H, Wd = r1 - r0, c1 - c0
    if not budget or H * Wd <= int(budget):
        return [r0, r1, c0, c1]
    scale = (float(budget) / (H * Wd)) ** 0.5
    newH = max(int(round(H * scale)), 1)
    newW = max(int(round(Wd * scale)), 1)
    dr, dc = (H - newH) // 2, (Wd - newW) // 2
    return [r0 + dr, r0 + dr + newH, c0 + dc, c0 + dc + newW]


# ---------------------------------------------------------------------------
def _false_color(data_raw, box, bands=(60, 30, 10)):
    r0, r1, c0, c1 = box
    rgb = data_raw[r0:r1, c0:c1][..., list(bands)].astype(np.float32)
    lo  = np.percentile(rgb, 2,  axis=(0, 1), keepdims=True)
    hi  = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def _gt_colorimage(gt_crop):
    H, W = gt_crop.shape
    img  = np.zeros((H, W, 3), dtype=np.float32)
    for cid, hex_ in CLS_COLORS_HEX.items():
        img[gt_crop == cid] = to_rgb(hex_)
    return img


def _savefig(fig, path):
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(path.replace('.pdf', '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"  [fig] {os.path.basename(path)}", flush=True)


# ---------------------------------------------------------------------------
def score_all(pix, nbr, models, tr_raw, tr_nbr, sig, cfg, device):
    """Score every detector on (pix, nbr).  Returns {det_name: scores}."""
    pix = pix.astype(np.float32); nbr = nbr.astype(np.float32)
    cfattn, nmlp, dsm_net = models['cfattn'], models['nmlp'], models['dsm']
    out = {}
    floor = float(cfg.get('baseline_eig_floor', 1e-12))
    out['DSM']          = dsm_additive(pix, tr_raw, dsm_net, sig)
    out['CF-Attn']      = score_cfattn_additive(cfattn, pix, nbr, tr_raw, tr_nbr, sig)
    out['CF-Attn-CFAR'] = score_cfattn_additive_cfar(cfattn, pix, nbr, sig)
    out['NeighborMLP']  = score_nmlp_additive(nmlp, pix, nbr, tr_raw, tr_nbr, sig)
    out['AMF']          = amf_global(pix, tr_raw, sig, eig_floor=floor)
    out['CEM']          = cem_global(pix, tr_raw, sig, eig_floor=floor)
    out['GMM-Levin']    = gmm_glrt_levin_additive(pix, tr_raw, sig,
                                                  p_steps=cfg.get('gmm_steps', 50))
    amf_loc, cem_loc = amf_cem_local_scm(
        pix, nbr, sig, device=device,
        loading=float(cfg.get('local_scm_loading', 1e-8)))
    out['AMF-local'] = amf_loc
    out['CEM-local'] = cem_loc
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(_EXP, 'colab.yaml'))
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--scenario', type=int, default=None,
                    help='Override scenario_index from config')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if os.path.exists(args.config):
        with open(args.config) as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if k in DEFAULT_CFG})
    if args.dry_run:
        cfg.update(DRYRUN_OVERRIDES)
    if args.results_dir:
        cfg['results_dir'] = args.results_dir
    if args.scenario is not None:
        cfg['scenario_index'] = args.scenario

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    seed = int(cfg['seed'])
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'compare_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)

    # ---- Data + scenario ----
    print("Loading Pavia-U ...", flush=True)
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D = data_norm.shape
    print(f"Image {H}×{W}×{D}", flush=True)

    manual_path = cfg.get('manual_boxes_path')
    manual = json.load(open(manual_path)) if (manual_path and os.path.exists(manual_path)) else []
    random_sc = generate_random_boxes(
        gt, n=4, min_pixels=int(cfg['min_pixels']),
        seeds=tuple(cfg['random_scenario_seeds']))
    scenarios = manual + random_sc
    sidx = int(cfg['scenario_index']) % len(scenarios)
    scenario = scenarios[sidx]
    train_box, test_box = scenario['train_box'], scenario['test_box']
    print(f"Scenario {sidx}: train_box={train_box}  test_box={test_box}", flush=True)

    k = int(cfg['k'])

    # ---- Crop train / test (raw bands + k×k neighbors) ----
    # NO subsampling: use the full train box. If n_budget is set we cut pixels
    # from the SIDES only (contiguous sub-box) so spatial context is preserved.
    tr_box_eff = _side_crop_box(train_box, cfg.get('n_budget'))
    tr_raw, tr_nbr = _crop_pca_box(data_norm, tr_box_eff, k)
    tr_raw = tr_raw.astype(np.float32); tr_nbr = tr_nbr.astype(np.float32)
    print(f"train={len(tr_raw)} px  (box {tr_box_eff}, full={cfg.get('n_budget') is None})",
          flush=True)

    r0, r1, c0, c1 = test_box
    H_b, W_b = r1 - r0, c1 - c0
    te_raw, te_nbr = _crop_pca_box(data_norm, test_box, k)
    te_raw = te_raw.astype(np.float32); te_nbr = te_nbr.astype(np.float32)
    te_gt  = gt[r0:r1, c0:c1].ravel()
    te_idx = np.arange(len(te_raw))
    print(f"test={len(te_raw)} px  ({H_b}×{W_b})", flush=True)

    # ---- Signature ----
    sig, dom_cls, dom_name = compute_signature(
        gt[r0:r1, c0:c1], data_norm[r0:r1, c0:c1],
        w_dom=float(cfg['sig_dom_weight']), w_mean=float(cfg['sig_mean_weight']))
    sig = sig.astype(np.float32)
    print(f"signature: dominant={dom_name}  ||s||={np.linalg.norm(sig):.4f}", flush=True)

    # ---- Train deep nets (GPU) ----
    print("Training deep nets ...", flush=True)
    t0 = time.time()
    dsm_net = _train_dsm(tr_raw, cfg, device);                 print(f"  DSM done ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    # cfattn  = _train_cfattn(tr_raw, tr_nbr, cfg, device, seed); print(f"  CF-Attn done ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    nmlp    = _train_nmlp(tr_raw, tr_nbr, cfg, device);        print(f"  NeighborMLP done ({time.time()-t0:.0f}s)", flush=True)
    models = {
        'dsm': dsm_net,
        # 'cfattn': cfattn,
        'nmlp': nmlp}

    # ---- Plant targets + score ----
    planted, labels, tgt_idx = plant_targets(
        te_raw, sig, cfg['amplitude'], cfg['target_fraction'],
        model='additive', seed=seed)
    planted = planted.astype(np.float32)
    print(f"planted {int(labels.sum())} targets", flush=True)

    print("Scoring detectors (test) ...", flush=True)
    test_scores  = score_all(planted, te_nbr, models, tr_raw, tr_nbr, sig, cfg, device)
    print("Scoring detectors (train, for CFAR threshold) ...", flush=True)
    train_scores = score_all(tr_raw, tr_nbr, models, tr_raw, tr_nbr, sig, cfg, device)

    # ---- Metrics ----
    pfa_t = float(cfg.get('pfa_target', 0.05))
    rows = []
    pfa_per_class = {}
    roc_curves = {}
    for det in DET_ORDER:
        sc = np.asarray(test_scores[det], dtype=np.float64)
        thr = cfar_threshold(np.asarray(train_scores[det], dtype=np.float64),
                             target_fpr=pfa_t)
        pcf = per_class_fpr(sc, labels, te_gt, thr)     # {clsname: fpr}
        pcf = {kk: vv for kk, vv in pcf.items() if kk != 'unlabeled'} or pcf
        pfa_per_class[det] = pcf
        pfa_vals = list(pcf.values()) if pcf else [float('nan')]
        fpr, tpr, auc_v = roc_safe(labels, sc)
        roc_curves[det] = (fpr, tpr, auc_v)
        rows.append({
            'Detector':     det,
            'pAUC@0.05':    partial_auc(labels, sc, fpr_max=0.05),
            'AUC':          auc_v,
            'Pd@Pfa=0.05':  dr_at_fpr(labels, sc, fpr_list=(pfa_t,))[str(pfa_t)],
            'Pfa_avg':      float(np.nanmean(pfa_vals)),
            'Pfa_max':      float(np.nanmax(pfa_vals)),
        })

    # ---- Summary table (CSV + Markdown) ----
    cols = ['Detector', 'pAUC@0.05', 'AUC', 'Pd@Pfa=0.05', 'Pfa_avg', 'Pfa_max']
    csv_path = os.path.join(run_dir, 'summary_table.csv')
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) if c == 'Detector' else f'{r[c]:.4f}'
                             for c in cols) + '\n')
    md_path = os.path.join(run_dir, 'summary_table.md')
    with open(md_path, 'w') as f:
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r['Detector'] if c == 'Detector'
                                      else f'{r[c]:.3f}' for c in cols) + ' |\n')
    print("\n=== Summary ===", flush=True)
    print(open(md_path).read(), flush=True)

    # ---- metrics.json + scores.npz ----
    json.dump({'scenario_index': sidx, 'train_box': train_box, 'test_box': test_box,
               'dom_cls': dom_cls, 'dom_name': dom_name, 'pfa_target': pfa_t,
               'rows': rows, 'pfa_per_class': pfa_per_class},
              open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2, default=str)
    npz = {f'score_{d}': test_scores[d] for d in DET_ORDER}
    npz['labels'] = labels; npz['te_gt'] = te_gt; npz['tgt_idx'] = tgt_idx
    np.savez(os.path.join(run_dir, 'scores.npz'), **npz)

    # ---- Figures ----
    print("\nSaving figures ...", flush=True)
    # false color
    fig, ax = plt.subplots(figsize=(5, 5 * H_b / max(W_b, 1)))
    ax.imshow(_false_color(data_norm, test_box)); ax.axis('off')
    ax.set_title(f'False color — test box (scenario {sidx})', fontsize=9)
    _savefig(fig, os.path.join(run_dir, 'false_color.pdf'))

    # label map
    fig, ax = plt.subplots(figsize=(5, 5 * H_b / max(W_b, 1)))
    ax.imshow(_gt_colorimage(gt[r0:r1, c0:c1])); ax.axis('off')
    ax.set_title(f'Label map — test box (dominant={dom_name})', fontsize=9)
    present = sorted(np.unique(te_gt))
    handles = [plt.matplotlib.patches.Patch(color=CLS_COLORS_HEX.get(int(c), '#777'),
               label=CLS_NAMES.get(int(c), f'cls{c}')) for c in present]
    ax.legend(handles=handles, fontsize=6, loc='center left',
              bbox_to_anchor=(1.0, 0.5))
    _savefig(fig, os.path.join(run_dir, 'label_map.pdf'))

    # detection maps grid
    ncol = 3
    nrow = int(np.ceil(len(DET_ORDER) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for j, det in enumerate(DET_ORDER):
        smap = scores_to_spatial_map(test_scores[det], te_idx, (H_b, W_b))
        ax = axes[j]
        im = ax.imshow(smap, cmap='inferno')
        ax.set_title(f'{det}  (AUC={roc_curves[det][2]:.3f})', fontsize=8)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(len(DET_ORDER), len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'Detection score maps — scenario {sidx}', fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(run_dir, 'detection_maps.pdf'))

    # ROC overlay
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
    for det in DET_ORDER:
        fpr, tpr, auc_v = roc_curves[det]
        ax.plot(fpr, tpr, color=DET_COLORS[det], lw=1.6,
                label=f'{det} (AUC={auc_v:.3f})')
    ax.set_xlabel('False Alarm Rate'); ax.set_ylabel('Detection Rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
    ax.set_title(f'ROC — scenario {sidx}', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    _savefig(fig, os.path.join(run_dir, 'roc.pdf'))

    # per-class Pfa grouped bars
    classes = sorted({c for d in DET_ORDER for c in pfa_per_class[d]})
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(classes)), 4))
    bw = 0.8 / len(DET_ORDER)
    xpos = np.arange(len(classes))
    for di, det in enumerate(DET_ORDER):
        vals = [pfa_per_class[det].get(c, 0.0) for c in classes]
        ax.bar(xpos + di * bw, vals, bw, label=det, color=DET_COLORS[det])
    ax.axhline(pfa_t, color='k', ls=':', lw=1, label=f'target Pfa={pfa_t}')
    ax.set_xticks(xpos + 0.4 - bw / 2); ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Per-class Pfa'); ax.set_title(f'Per-class false-alarm rate (CFAR thr @ {pfa_t})', fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _savefig(fig, os.path.join(run_dir, 'pfa_per_class.pdf'))

    print(f"\nDone.  Results: {run_dir}", flush=True)
    if args.dry_run:
        expect = ['summary_table.csv', 'summary_table.md', 'metrics.json',
                  'scores.npz', 'false_color.pdf', 'label_map.pdf',
                  'detection_maps.pdf', 'roc.pdf', 'pfa_per_class.pdf']
        ok = all(os.path.exists(os.path.join(run_dir, e)) for e in expect)
        print("DRY-RUN:", "ALL OUTPUTS PRESENT ✓" if ok else "MISSING OUTPUTS ✗")
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
