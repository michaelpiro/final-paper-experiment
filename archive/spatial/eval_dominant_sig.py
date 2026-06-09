"""
eval_dominant_sig.py — Dominant-component target signature evaluation.

Loads pretrained CF-Attn + DSM models from a saved cfattn run, then
re-evaluates them on the SAME train/test pixels using the PC1 direction
(dominant background variance axis) as the planted target signature.

Why: a class-specific signature (e.g. metal sheets) is spectrally far from
every background class — even AMF trivially succeeds.  PC1 is the hardest
possible direction: background pixels naturally vary along it, so a planted
bump looks like ordinary background fluctuation.  The spatial model's local
adaptation is what separates signal from background variation here.

Creates a full self-contained run directory with:
  - config.yaml, metrics.json
  - figures/patches.png        — train/test crops + GT, boxes on full scene
  - figures/spectra.pdf        — raw-space mean spectra of all labeled classes
                                  in the boxes + the planted PC1 signature
  - figures/roc_additive.pdf   — ROC curves
  - figures/roc_replacement.pdf

Usage:
    .venv/bin/python eval_dominant_sig.py
    .venv/bin/python eval_dominant_sig.py --source results/cfattn_20260606_202332
"""

import argparse, os, sys, json, pickle, time
from datetime import datetime

import numpy as np
import torch
import yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))
sys.path.insert(0, _EXP)   # for cfattn_model
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from sklearn.metrics import roc_auc_score, roc_curve

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive, dsm_replacement, amf_replacement,
    gmm_glrt, gmm_glrt_replacement, exact_glrt_replacement,
)
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods
from dsm_model import ScoreNet
from cfattn_model import (
    CFAttnGaussianScoreNet,
    score_cfattn_additive, score_cfattn_replacement,
)

# ---- Pavia-U class labels ----
CLS_NAMES = {
    0: 'unlabeled',  1: 'asphalt',   2: 'meadows',
    3: 'gravel',     4: 'trees',     5: 'metal_sheets',
    6: 'bare_soil',  7: 'bitumen',   8: 'bricks',
    9: 'shadows',
}
# Visually distinct palette for spectra plots
CLS_COLORS = {
    1: '#555555', 2: '#4daf4a', 3: '#a65628', 4: '#1a7a1a',
    5: '#ff7f00', 6: '#984ea3', 7: '#000000', 8: '#e41a1c',
    9: '#377eb8',
}


def auc_safe(labels, scores):
    try:    return float(roc_auc_score(labels, scores))
    except: return float('nan')


def roc_safe(labels, scores):
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr.tolist(), tpr.tolist(), auc_safe(labels, scores)
    except:
        return [0., 1.], [0., 1.], float('nan')


# ---------------------------------------------------------------------------
# Figure 1: patches + GT
# ---------------------------------------------------------------------------
def save_patches_figure(data_norm, gt, cfg, s_raw, fig_dir):
    """False-color crops, GT overlays, full scene with boxes."""
    H, W, _ = data_norm.shape
    bands = (60, 30, 10)

    def _rgb(crop):
        rgb = crop[..., list(bands)].astype(np.float32)
        lo  = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
        hi  = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
        return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)

    full_rgb = _rgb(data_norm)
    n_cls = int(gt.max()) + 1

    fig = plt.figure(figsize=(12, 7))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], height_ratios=[1, 1])

    def _plot_crop(ax_rgb, ax_gt, box, title):
        r0, r1, c0, c1 = box
        ax_rgb.imshow(_rgb(data_norm[r0:r1, c0:c1]))
        ax_rgb.set_title(f"{title} — false color ({r1-r0}×{c1-c0})")
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        gt_sub = gt[r0:r1, c0:c1]
        ax_gt.imshow(gt_sub, cmap='tab10', vmin=0, vmax=max(n_cls - 1, 9),
                     interpolation='nearest')
        ax_gt.set_title(f"{title} — GT labels")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])
        # Annotate classes present
        cls_ids, cnts = np.unique(gt_sub, return_counts=True)
        note = "\n".join(f"cls{int(c)}: {CLS_NAMES.get(int(c),'?')} ({n}px)"
                         for c, n in zip(cls_ids, cnts) if int(c) != 0)
        ax_gt.set_xlabel(note, fontsize=7)

    _plot_crop(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
               cfg['train_box'], 'TRAIN')
    _plot_crop(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
               cfg['test_box'],  'TEST')

    ax_full = fig.add_subplot(gs[:, 2])
    ax_full.imshow(full_rgb)
    for box, col, lab in [(cfg['train_box'], 'lime', 'train'),
                          (cfg['test_box'],  'red',  'test')]:
        r0, r1, c0, c1 = box
        ax_full.add_patch(plt.Rectangle((c0, r0), c1-c0, r1-r0,
                          linewidth=2, edgecolor=col, facecolor='none',
                          label=lab))
    ax_full.set_title(f"Pavia-U ({H}×{W})"); ax_full.legend(loc='lower right')
    ax_full.set_xticks([]); ax_full.set_yticks([])

    fig.tight_layout()
    out = os.path.join(fig_dir, 'patches.png')
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  saved {out}", flush=True)


# ---------------------------------------------------------------------------
# Figure 2: raw spectra — class means + PC1 signature
# ---------------------------------------------------------------------------
def save_spectra_figure(data_norm, gt, cfg, s_raw, pc1_raw, fig_dir, sig_label='PC1'):
    """
    For every labeled class found in EITHER box, plot its mean raw spectrum.
    Also plot the PC1 direction (scaled to amplitude) as the target signature.
    All in raw 103-D global_max-normalized space.
    """
    flat = data_norm.reshape(-1, data_norm.shape[-1])
    gt_flat = gt.reshape(-1)
    bands = np.arange(data_norm.shape[-1])

    # Collect classes present in train OR test box
    def cls_in_box(box):
        r0, r1, c0, c1 = box
        return set(int(c) for c in np.unique(gt[r0:r1, c0:c1]) if c != 0)

    present = cls_in_box(cfg['train_box']) | cls_in_box(cfg['test_box'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    for ax, box_key, title in [
            (axes[0], 'train_box', 'TRAIN box — class spectra'),
            (axes[1], 'test_box',  'TEST box — class spectra')]:

        r0, r1, c0, c1 = cfg[box_key]
        box_flat = data_norm[r0:r1, c0:c1].reshape(-1, data_norm.shape[-1])
        box_gt   = gt[r0:r1, c0:c1].reshape(-1)

        # Background mean ± 1 std (all labeled pixels)
        bkg_mask = box_gt != 0
        if bkg_mask.any():
            bkg = box_flat[bkg_mask]
            bkg_mean = bkg.mean(0)
            bkg_std  = bkg.std(0)
            ax.fill_between(bands, bkg_mean - bkg_std, bkg_mean + bkg_std,
                            alpha=0.12, color='gray', label='bkg ±1σ')
            ax.plot(bands, bkg_mean, color='gray', lw=1.2,
                    linestyle='--', label='bkg mean')

        # Per-class mean spectra
        for cls_id in sorted(present):
            mask = box_gt == cls_id
            if not mask.any():
                continue
            mean_spec = box_flat[mask].mean(0)
            color = CLS_COLORS.get(cls_id, 'black')
            ax.plot(bands, mean_spec, lw=1.5, color=color,
                    label=f"cls{cls_id}: {CLS_NAMES.get(cls_id,'?')} (n={mask.sum()})")

        # PC1 signature: show the full unit-norm PC1 direction.
        # We plot 1*PC1 (not scaled by amplitude) to show that the
        # signature shape overlaps heavily with the class spectra —
        # demonstrating it is NOT a distinctive spectral target.
        ax.plot(bands, pc1_raw, color='magenta', lw=2.2,
                linestyle='-', label=f'{sig_label} (target signature)')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Band index')
        ax.set_ylabel('Reflectance (global_max norm.)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Raw spectra in train/test boxes — target signature = {sig_label}\n"
        "Signature overlaps with natural class variation (not spectrally distinctive)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(fig_dir, 'spectra.pdf')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}", flush=True)


# ---------------------------------------------------------------------------
# Figure 3: ROC curves
# ---------------------------------------------------------------------------
def save_roc_figure(rocs, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (fpr, tpr, a) in rocs.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} ({a:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=None,
                   help='Source cfattn run dir. Default: latest results/cfattn_*/')
    p.add_argument('--sig', default='pc1',
                   help='Target signature in PCA space. Examples: '
                        'pc1  pc1+pc2  pc2  pc1-pc2  (components summed/subtracted)')
    args = p.parse_args()

    # ---- locate source run ----
    if args.source:
        src = args.source
    else:
        _spatial_results = os.path.join('experiments', 'spatial', 'results')
        dirs = sorted([d for d in os.listdir(_spatial_results)
                       if d.startswith('cfattn_')], reverse=True)
        assert dirs, f"No cfattn_* runs found in {_spatial_results}"
        src = os.path.join(_spatial_results, dirs[0])
    print(f"Source run: {src}", flush=True)
    src_cfg = yaml.safe_load(open(os.path.join(src, 'config.yaml')))

    # ---- create output run dir ----
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sig_tag = args.sig.replace('+', 'plus').replace('-', 'minus')
    run_dir = os.path.join(src_cfg['results_dir'], f'dominant_sig_{sig_tag}_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Build config for THIS run (copy src, override signature info)
    cfg = dict(src_cfg)
    cfg['signature'] = args.sig
    cfg['source_run'] = src
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'),
              sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    # ---- load image + PCA ----
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw = data_norm.shape
    all_flat  = data_norm.reshape(-1, D_raw)
    gt_flat   = gt.reshape(-1)
    pca = pickle.load(open(os.path.join(src, 'models', 'pca.pkl'), 'rb'))
    D   = pca.n_components_
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)
    print(f"Image {H}x{W}x{D_raw}  PCA→{D}  "
          f"explained={pca.explained_variance_ratio_.sum():.4f}", flush=True)

    # Verify boxes are free of target class
    CLS_NAMES_LOCAL = CLS_NAMES
    tgt_cls = cfg['target_cls']
    for nm, bx in [('train_box', cfg['train_box']), ('test_box', cfg['test_box'])]:
        r0, r1, c0, c1 = bx
        gt_sub = gt[r0:r1, c0:c1]
        n_tgt  = int(np.sum(gt_sub == tgt_cls))
        cls_ids, cnts = np.unique(gt_sub, return_counts=True)
        comp = ", ".join(f"{CLS_NAMES_LOCAL.get(int(c), f'cls{c}')}={int(n)}"
                        for c, n in zip(cls_ids, cnts))
        print(f"  {nm}: {comp}", flush=True)
        assert n_tgt == 0, \
            f"{nm} contains {n_tgt} target-class pixels — re-pick boxes!"

    # ---- reconstruct exact train/test pixel sets ----
    seed = int(cfg['seed'])
    rng  = np.random.default_rng(seed)

    def crop_box(box):
        r0, r1, c0, c1 = box
        sub = torch.tensor(pca_img[r0:r1, c0:c1, :], dtype=torch.float32)
        centers, nbrs = extract_neighborhoods(sub, cfg['k'])
        return centers.numpy(), nbrs.numpy()

    tr_pix, tr_nbr = crop_box(cfg['train_box'])
    te_pix, te_nbr = crop_box(cfg['test_box'])

    def subsample(pix, nbr, n):
        if len(pix) <= n: return pix, nbr
        idx = rng.choice(len(pix), n, replace=False)
        return pix[idx], nbr[idx]

    tr_pix, tr_nbr = subsample(tr_pix, tr_nbr, cfg['train_n'])
    te_pix, te_nbr = subsample(te_pix, te_nbr, cfg['test_n'])
    print(f"train={len(tr_pix)} px  test={len(te_pix)} px", flush=True)

    # ---- build signature from --sig spec ----
    import re

    def _parse_sig(spec, D, D_raw):
        """Return (s_pca [D,], s_raw [D_raw,]) for a spec like 'pc1+pc2'."""
        tokens = re.findall(r'[+-]?pc\d+', spec.replace(' ', ''))
        s_pca = np.zeros(D, dtype=np.float32)
        s_raw = np.zeros(D_raw, dtype=np.float32)
        for tok in tokens:
            sign = -1.0 if tok.startswith('-') else 1.0
            idx  = int(re.search(r'\d+', tok).group()) - 1
            assert 0 <= idx < D, f"Component pc{idx+1} out of range (D={D})"
            s_pca[idx] += sign
            s_raw       += sign * pca.components_[idx]
        return s_pca, s_raw

    if args.sig == 'dom_offset':
        # Dominant-class offset:
        #   s = 0.9 * mean(dominant_cls in test box) + 0.1 * mean(all test box pixels)
        # Computed in RAW space then projected to PCA.
        r0, r1, c0, c1 = cfg['test_box']
        test_raw_box = all_flat.reshape(H, W, D_raw)[r0:r1, c0:c1].reshape(-1, D_raw)
        test_gt_box  = gt[r0:r1, c0:c1].reshape(-1)

        # Find dominant labeled class in test box (by pixel count, excluding cls0)
        labeled_mask = test_gt_box != 0
        if labeled_mask.any():
            cls_ids, cnts = np.unique(test_gt_box[labeled_mask], return_counts=True)
            dom_cls = int(cls_ids[cnts.argmax()])
        else:
            dom_cls = 0
        print(f"  Dominant labeled class in test box: cls{dom_cls} "
              f"({CLS_NAMES.get(dom_cls,'?')}) — {int(cnts.max())} px", flush=True)

        dom_mean  = test_raw_box[test_gt_box == dom_cls].mean(axis=0).astype(np.float32)
        test_mean = test_raw_box.mean(axis=0).astype(np.float32)
        sig_raw   = 0.9 * dom_mean + 0.1 * test_mean                 # (D_raw,)
        s_pca     = pca.transform(sig_raw[None]).flatten().astype(np.float32)  # (D,)
    else:
        s_pca, sig_raw = _parse_sig(args.sig, D, D_raw)
    sig_norm_raw = np.linalg.norm(sig_raw)

    # SNR: project train pixels onto sig direction, measure std
    proj_train = tr_pix @ s_pca  # dot with PCA-space signature
    tr_std_sig  = float(proj_train.std())
    snr = cfg['amplitude'] * np.linalg.norm(s_pca) / (tr_std_sig + 1e-12)

    print(f"\nTarget signature: {args.sig}  (in PCA-{D} space)", flush=True)
    print(f"  s_pca = {s_pca}", flush=True)
    print(f"  ||s_raw|| = {sig_norm_raw:.4f}", flush=True)
    print(f"  bkg std along sig = {tr_std_sig:.4f}", flush=True)
    print(f"  SNR (amp·||s||/std) = {snr:.3f}  ({'very hard' if snr < 0.5 else 'moderate'})\n",
          flush=True)

    # Use sig_raw (unit: raw space) for the spectra figure
    pc1_raw = sig_raw  # reuse the existing figure variable name

    # ---- noise level ----
    sigma = compute_sigma_from_data(pca.transform(all_flat), cfg['dsm_sigma_rho'])
    print(f"sigma = {sigma:.5f}", flush=True)

    # ---- load pretrained models ----
    cfattn = CFAttnGaussianScoreNet(
        D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
        sigma=sigma, eps=cfg.get('cfattn_eps', 1e-4))
    cfattn.load_state_dict(
        torch.load(os.path.join(src, 'models', 'cfattn.pt'),
                   map_location='cpu')['state_dict'])
    cfattn.eval()

    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'])
    dsm_net.load_state_dict(
        torch.load(os.path.join(src, 'models', 'dsm.pt'),
                   map_location='cpu')['state_dict'])
    dsm_net.eval()
    print("Pretrained CF-Attn + DSM loaded.\n", flush=True)

    # ---- figures 1 & 2 ----
    print("[Figures] saving patches + spectra ...", flush=True)
    save_patches_figure(data_norm, gt, cfg, pc1_raw, fig_dir)
    save_spectra_figure(data_norm, gt, cfg, pc1_raw, pc1_raw, fig_dir,
                        sig_label=args.sig)

    # ---- plant + score ----
    metrics = {}; rocs = {}
    for tm in ('additive', 'replacement'):
        planted, labels, _ = plant_targets(
            te_pix, s_pca, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        n_pos = int(labels.sum())
        print(f"[{tm}]  planted {n_pos}/{len(labels)} targets", flush=True)

        te_nbr_f = te_nbr.astype(np.float32)

        if tm == 'additive':
            sc_cf  = score_cfattn_additive(
                cfattn, planted, te_nbr_f, tr_pix, tr_nbr.astype(np.float32), s_pca)
            sc_dsm = dsm_additive(planted, tr_pix, dsm_net, s_pca)
            sc_gmm = gmm_glrt(planted, tr_pix, s_pca, K=cfg['gmm_K'],
                              theta_max=cfg['gmm_theta_max'],
                              theta_steps=cfg['gmm_theta_steps'])
        else:
            sc_cf  = score_cfattn_replacement(
                cfattn, planted, te_nbr_f, tr_pix, tr_nbr.astype(np.float32), s_pca)
            sc_dsm = dsm_replacement(planted, tr_pix, dsm_net, s_pca)
            sc_gmm = gmm_glrt_replacement(planted, tr_pix, s_pca, K=cfg['gmm_K'],
                                          theta_max=cfg['gmm_theta_max'],
                                          theta_steps=cfg['gmm_theta_steps'])

        det_scores = {
            'CF-Attn':  sc_cf,
            'DSM':      sc_dsm,
            'AMF':      amf(planted, tr_pix, s_pca),
            'Reg-AMF':  reg_amf(planted, tr_pix, s_pca, sigma),
            'GMM-GLRT': sc_gmm,
        }
        if tm == 'replacement':
            det_scores['AMF-rep']    = amf_replacement(planted, tr_pix, s_pca)
            det_scores['Exact-GLRT'] = exact_glrt_replacement(planted, tr_pix, s_pca)

        metrics[tm] = {k: auc_safe(labels, v) for k, v in det_scores.items()}
        rocs[tm]    = {k: roc_safe(labels, v) for k, v in det_scores.items()}
        line = "  ".join(f"{k}={v:.3f}" for k, v in metrics[tm].items())
        print(f"  {line}", flush=True)

    # ---- ROC figures ----
    save_roc_figure(rocs['additive'],
                    f'Additive — PC1 target (SNR={snr:.2f})',
                    os.path.join(fig_dir, 'roc_additive.pdf'))
    save_roc_figure(rocs['replacement'],
                    f'Replacement — PC1 target (SNR={snr:.2f})',
                    os.path.join(fig_dir, 'roc_replacement.pdf'))

    # ---- bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    det_colors = ['#d62728','#1f77b4','#aec7e8','#6baed6','#9467bd',
                  '#08306b','#ff7f0e']
    for ax, tm in zip(axes, ('additive', 'replacement')):
        names = list(metrics[tm].keys())
        vals  = list(metrics[tm].values())
        bars  = ax.bar(names, vals,
                       color=det_colors[:len(names)])
        ax.set_ylim(0.4, 1.0); ax.set_title(f'{tm.capitalize()} — PC1 target')
        ax.set_ylabel('AUC')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, min(v+0.01, 0.99),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'PC1 target signature  (SNR={snr:.2f}, amp={cfg["amplitude"]})',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'auc_bar.pdf')); plt.close(fig)

    # ---- save metrics ----
    full_metrics = {
        'signature': args.sig,
        'snr': snr, 'amplitude': cfg['amplitude'],
        'source_run': src,
        **metrics}
    json.dump(full_metrics,
              open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)
    json.dump(rocs,
              open(os.path.join(run_dir, 'rocs.json'), 'w'), indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
