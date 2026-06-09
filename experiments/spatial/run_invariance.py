"""
run_invariance.py — Spectral-entanglement invariance experiment.

Paper point (Section 5, invariance argument):

    Our spatial score detector (CF-Attn) is INVARIANT to the spectral
    correlation between target and background.  Pure spectral-angle detectors
    (THANTD) and matched filters (AMF) only work when the target is spectrally
    SEPARABLE from the background; they collapse to near-random when the target
    is spectrally ENTANGLED with the background (a subpixel additive target
    whose signature is a linear combination of the local background materials).

We use the SAME heterogeneous test boxes as the main experiment (the real,
challenging setup — a spatially mixed background is the whole point), hold the
box and the additive target model FIXED, and vary ONLY the target's spectral
DIRECTION across three regimes (all signatures renormalized to the SAME L2 norm,
so amplitude is matched and only direction changes):

    A. ENTANGLED   — signature = 0.8·(dominant background class) + 0.2·patch_mean.
                     The original target: a linear combination of the local materials.
    B. DISTINCT    — signature = mean of a real class that is ABSENT from the
                     test box (trees/metal_sheets excluded as "too easy").  In Pavia
                     the materials are mutually low-rank, so a real absent class is
                     usually only mildly distinct — an honest intermediate case.
    C. SYNTH-⟂     — synthetic signature orthogonal to the present scene materials
                     in the FULL raw band space.  Rigorously "not a linear
                     combination of the scene materials".

NO PCA: every detector consumes the raw full-band data; our score nets whiten
internally (a frozen ZCA first layer fit per scenario). The orthogonal regime-C
signature is built in the full raw band space (orthogonal complement of the
scene-material span), which is more rigorous than the old PCA-subspace version.

Expected pattern (the story):
    regime          CF-Attn(ours)   THANTD   AMF
    A entangled        high          low     low      ← only spatial works
    B distinct         high          high    high     ← everyone works
    C synth-perp          high          high    high     ← everyone works
    → CF-Attn is high everywhere (invariant); THANTD/AMF flip with separability.

Usage:
    # local smoke test (tiny, CPU):
    .venv/bin/python -u experiments/spatial/run_invariance.py --dry-run

    # full run (Colab T4):
    python -u experiments/spatial/run_invariance.py \
        --config experiments/spatial/colab.yaml \
        --results_dir /content/drive/MyDrive/final_paper/invariance_results
"""

import argparse, json, os, sys, time, pickle
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

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import amf, reg_amf, dsm_additive
from final_paper_experiments.evaluation import auc_safe, roc_safe, compute_signature
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods

from cfattn_model import (
    CFAttnGaussianScoreNet,
    score_cfattn_additive, score_cfattn_additive_cfar,
)
from neighbor_mlp_model import NeighborMLPDenoiser, score_nmlp_additive
from dsm_model import ScoreNet
from thantd_model import THANTD, score_thantd

# Reuse the EXACT trainers from the main pipeline so results are comparable.
from run_colab import (
    _subsample, _crop_pca_box, _crop_raw_box,
    _train_cfattn, _train_nmlp, _train_dsm, _train_thantd,
    _placeholder_whitening, _whitened_sigma,
    DEFAULT_CFG, CLS_NAMES,
)

REGIME_COLORS = {'A: entangled': '#d62728', 'B: distinct': '#1f77b4',
                 'C: synth-perp': '#2ca02c'}
DET_COLORS = {
    'CF-Attn-CFAR': '#1f77b4', 'CF-Attn': '#aec7e8', 'NeighborMLP': '#2ca02c',
    'DSM': '#ff7f0e', 'THANTD': '#d62728', 'AMF': '#9467bd', 'Reg-AMF': '#c5b0d5',
}


def _build_cfattn(cfg, D, sigma):
    return CFAttnGaussianScoreNet(D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
                                  sigma=sigma, eps=cfg.get('cfattn_eps', 1e-4),
                                  whitening=_placeholder_whitening(D))


def _build_nmlp(cfg, D, sigma):
    return NeighborMLPDenoiser(D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
                               hidden=cfg['nmlp_hidden'], n_layers=cfg['nmlp_n_layers'],
                               sigma=sigma, activation=cfg['activation'],
                               whitening=_placeholder_whitening(D))


def _build_dsm(cfg, D):
    return ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'],
                    whitening=_placeholder_whitening(D))


def _find_ckpt(pretrained_dir, sid_idx, sid_label, n_budget, name):
    """Locate a model checkpoint in either the run_colab (spatial) layout or the
    invariance layout. Returns the path, or None if not found.

    The two trainings share the SAME background-only objective, the SAME boxes,
    the SAME PCA (random_state=seed) and the SAME per-scenario seed, so a
    signature-agnostic model (cfattn/nmlp/dsm) trained by run_colab is identical
    to one trained here and can be reused directly.
    """
    cands = [
        # run_colab (spatial) layout
        os.path.join(pretrained_dir, f'scenario_{sid_idx}', f'n{n_budget}',
                     'models', f'{name}_s{sid_idx}_n{n_budget}.pt'),
        # invariance layout (saved by this script)
        os.path.join(pretrained_dir, f'scenario_{sid_label}', 'models', f'{name}.pt'),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def find_homogeneous_boxes(gt, bg_class, box_h=26, box_w=26,
                           stride=4, min_purity=0.85):
    """Find two DISJOINT windows where `bg_class` dominates the labeled pixels.

    Returns (train_box, test_box) as [r0,r1,c0,c1], or None if not found.
    Purity = fraction of labeled pixels in the window that belong to bg_class.
    A homogeneous single-material background makes a different real material a
    genuinely DISTINCT target (high residual outside the 1-D material span),
    mirroring THANTD's airplane-on-uniform-tarmac regime.
    """
    H, W = gt.shape
    is_c = (gt == bg_class).astype(np.int64)
    is_lab = (gt != 0).astype(np.int64)
    cands = []
    for r in range(0, H - box_h, stride):
        for c in range(0, W - box_w, stride):
            nc = int(is_c[r:r+box_h, c:c+box_w].sum())
            nl = int(is_lab[r:r+box_h, c:c+box_w].sum())
            if nl >= 0.5 * box_h * box_w and nc >= min_purity * nl:
                cands.append((nc, r, c))
    if len(cands) < 2:
        return None
    cands.sort(reverse=True)                       # most class-c pixels first
    _, r0, c0 = cands[0]
    train = [r0, r0+box_h, c0, c0+box_w]
    # pick the best NON-overlapping candidate for the test box
    def overlap(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])
    for _, r, c in cands[1:]:
        test = [r, r+box_h, c, c+box_w]
        if not overlap(train, test):
            return train, test
    return None


def _renorm(v, target_norm):
    n = np.linalg.norm(v)
    return (v / n * target_norm).astype(np.float32) if n > 0 else v.astype(np.float32)


def _residual_fraction(s, basis):
    """Fraction of ||s|| lying OUTSIDE the row-span of `basis` (orthonormal rows)."""
    proj = basis.T @ (basis @ s)
    return float(np.linalg.norm(s - proj) / (np.linalg.norm(s) + 1e-12))


def _orthonormal_basis(M):
    """Orthonormal rows spanning the row-space of M (p, D)."""
    U, S, Vt = np.linalg.svd(np.atleast_2d(M), full_matrices=False)
    if S.size == 0 or S[0] == 0:
        return np.zeros((0, M.shape[-1]), dtype=M.dtype)
    r = int((S > 1e-8 * S[0]).sum())
    return Vt[:r]


def build_signatures(gt, data_norm, class_means, test_box, dom_cls, rng):
    """Return dict regime -> (s_raw, info).  All raw signatures share one L2 norm.

    NO PCA: every detector sees the full raw bands, so the orthogonal regime-C
    signature is built in the FULL raw band space (orthogonal complement of the
    scene-material span) — more rigorous than the old PCA-subspace version.
    """
    r0, r1, c0, c1 = test_box
    te_gt   = gt[r0:r1, c0:c1].ravel()
    te_raw  = data_norm[r0:r1, c0:c1].reshape(-1, data_norm.shape[-1])
    box_mean = te_raw.mean(0).astype(np.float32)

    # "Scene materials" = the means of the labeled classes PRESENT in the test box.
    # The distinctness of a target is how much of it lies OUTSIDE the span of these
    # materials — i.e. how far it is from being a linear combination of them
    # (exactly the user's criterion).
    present_lab = [int(c) for c in np.unique(te_gt) if c != 0]
    M_scene  = np.stack([class_means[c] for c in present_lab]).astype(np.float32)
    U_scene  = _orthonormal_basis(M_scene)                # raw-space material span

    # ---- A: entangled (dominant background class) — a linear combo of scene mats ----
    sig_A = (0.8 * class_means[dom_cls] + 0.2 * box_mean).astype(np.float32)
    ref_norm = float(np.linalg.norm(sig_A))

    # ---- B: distinct real class ABSENT from the box (excl trees/metal as "too easy") ----
    # Choose the absent class LEAST expressible as a linear combination of the
    # present scene materials (max residual outside U_scene).
    absent = [c for c in class_means
              if c not in present_lab and c not in (4, 5)]   # avoid trees(4), metal(5)
    if not absent:                                            # fallback: allow trees/metal
        absent = [c for c in class_means if c not in present_lab]
    distinct_cls = max(absent,
                       key=lambda c: _residual_fraction(class_means[c], U_scene))
    sig_B = class_means[distinct_cls].astype(np.float32)

    # ---- C: synthetic, ORTHOGONAL to the scene-material span in the FULL raw band
    #         space — rigorously "not any linear combination of the scene materials". ----
    z = rng.standard_normal(data_norm.shape[-1])
    if U_scene.shape[0]:
        z = z - U_scene.T @ (U_scene @ z)                  # remove material directions
    z = z / (np.linalg.norm(z) + 1e-12)
    sig_C = z.astype(np.float32)                            # raw-space orthogonal dir

    # Renormalize all to the SAME L2 norm (matched amplitude; only DIRECTION varies)
    sig_A = _renorm(sig_A, ref_norm)
    sig_B = _renorm(sig_B, ref_norm)
    sig_C = _renorm(sig_C, ref_norm)

    info = {
        'distinct_cls': int(distinct_cls),
        'distinct_name': CLS_NAMES.get(distinct_cls, str(distinct_cls)),
        'present_classes': sorted(int(c) for c in np.unique(te_gt)),
        'present_materials': [CLS_NAMES.get(c) for c in present_lab],
        'resid_A': _residual_fraction(sig_A, U_scene),
        'resid_B': _residual_fraction(sig_B, U_scene),
        'resid_C': _residual_fraction(sig_C, U_scene),
        'ref_norm': ref_norm,
    }
    return {'A: entangled': sig_A, 'B: distinct': sig_B, 'C: synth-perp': sig_C}, info


def run_scenario(sid, sid_idx, scenario, cfg, data_norm, gt,
                 class_means, results_dir, run_thantd, device, dry_run,
                 pretrained_dir=None):
    train_box = scenario['train_box']
    test_box  = scenario['test_box']
    scen_dir  = os.path.join(results_dir, f'scenario_{sid}')
    mdl_dir   = os.path.join(scen_dir, 'models')
    os.makedirs(mdl_dir, exist_ok=True)

    print(f"\n{'='*60}\nScenario {sid}  device={device}", flush=True)
    # Same per-scenario seed convention as run_colab (cfg.seed + idx*100) so the
    # training-pixel subsample (and thus regime A) matches the main experiment.
    seed = int(cfg['seed']) + sid_idx * 100
    rng  = np.random.default_rng(seed)
    torch.manual_seed(seed)

    D_raw = data_norm.shape[-1]
    D = D_raw                       # NO PCA: raw band space everywhere
    k = int(cfg['k'])
    n_budget = cfg['box_size_ablation'][0]

    # ---- crop + subsample training pixels (RAW bands) ----
    tr_raw_full, tr_nbr_full = _crop_pca_box(data_norm, train_box, k)
    tr_raw, tr_nbr, tr_idx   = _subsample(tr_raw_full, tr_nbr_full, n_budget, rng)

    # ---- full test box (RAW bands) ----
    r0, r1, c0, c1 = test_box
    te_raw, te_nbr = _crop_pca_box(data_norm, test_box, k)
    te_gt          = gt[r0:r1, c0:c1].ravel()
    te_nbr_f = te_nbr.astype(np.float32)
    tr_nbr_f = tr_nbr.astype(np.float32)
    print(f"  train={len(tr_raw)}px  test={len(te_raw)}px", flush=True)

    # sigma for classical detectors (our nets compute whitened σ=√ρ internally)
    sigma = compute_sigma_from_data(tr_raw, cfg['dsm_sigma_rho'])

    dom_cls = int(compute_signature(gt[r0:r1, c0:c1],
                                    data_norm[r0:r1, c0:c1], 0.8, 0.2)[1])

    # ---- signatures (A/B/C), matched amplitude ----
    sigs, info = build_signatures(gt, data_norm, class_means,
                                  test_box, dom_cls, rng)
    print(f"  dominant(bkg)={CLS_NAMES.get(dom_cls)}  "
          f"distinct(absent)={info['distinct_name']}", flush=True)
    print(f"  residual-outside-local-bkg-span:  "
          f"A={info['resid_A']:.3f}  B={info['resid_B']:.3f}  "
          f"C={info['resid_C']:.3f}", flush=True)

    n_budget = cfg['box_size_ablation'][0]

    # ---- signature-AGNOSTIC models (background only): LOAD if available, else TRAIN ----
    # These do not depend on the planting model/amplitude, so they are reused as-is
    # for the strong/full-pixel verification runs.
    def _load(name, build):
        if pretrained_dir is None:
            return None
        ck = _find_ckpt(pretrained_dir, sid_idx, sid, n_budget, name)
        if ck is None:
            return None
        m = build().to(device)
        m.load_state_dict(torch.load(ck, map_location=device)['state_dict'])
        m.eval()
        print(f"  [{name}] loaded {ck}", flush=True)
        return m

    sig0 = _whitened_sigma(cfg)
    cfattn = _load('cfattn', lambda: _build_cfattn(cfg, D, sig0))
    if cfattn is None:
        print("  [CF-Attn] training ...", flush=True); t0 = time.time()
        cfattn = _train_cfattn(tr_raw, tr_nbr, cfg, device, seed)
        torch.save({'state_dict': cfattn.state_dict(), 'cfg': cfg},
                   os.path.join(mdl_dir, 'cfattn.pt'))
        print(f"    done {time.time()-t0:.0f}s", flush=True)

    nmlp = _load('nmlp', lambda: _build_nmlp(cfg, D, sig0))
    if nmlp is None:
        print("  [NeighborMLP] training ...", flush=True); t0 = time.time()
        nmlp = _train_nmlp(tr_raw, tr_nbr, cfg, device)
        torch.save({'state_dict': nmlp.state_dict(), 'cfg': cfg},
                   os.path.join(mdl_dir, 'nmlp.pt'))
        print(f"    done {time.time()-t0:.0f}s", flush=True)

    # DSM: RAW input + internal frozen whitening (replaces PCA + per-band z-score).
    dsm_net = _load('dsm', lambda: _build_dsm(cfg, D))
    if dsm_net is None:
        print("  [DSM] training ...", flush=True); t0 = time.time()
        dsm_net = _train_dsm(tr_raw, cfg, device)
        torch.save({'state_dict': dsm_net.state_dict(), 'cfg': cfg},
                   os.path.join(mdl_dir, 'dsm.pt'))
        print(f"    done {time.time()-t0:.0f}s", flush=True)

    # ---- evaluate each regime ----
    results = {'info': info, 'auc': {}, 'roc': {}}
    plant_model = cfg.get('plant_model', 'additive')
    for regime, sig_raw in sigs.items():
        pl_raw, lab, _ = plant_targets(te_raw, sig_raw, cfg['amplitude'],
                                       cfg['target_fraction'], model=plant_model,
                                       seed=seed)

        # all detectors operate on RAW bands with the RAW signature; our score
        # nets whiten internally and return data-space scores.
        det = {
            'CF-Attn-CFAR': score_cfattn_additive_cfar(cfattn, pl_raw, te_nbr_f, sig_raw),
            'CF-Attn':      score_cfattn_additive(cfattn, pl_raw, te_nbr_f,
                                                  tr_raw, tr_nbr_f, sig_raw),
            'NeighborMLP':  score_nmlp_additive(nmlp, pl_raw, te_nbr_f,
                                                tr_raw, tr_nbr_f, sig_raw),
            'DSM':          dsm_additive(pl_raw, tr_raw, dsm_net, sig_raw),
            'AMF':          amf(pl_raw, tr_raw, sig_raw),
            'Reg-AMF':      reg_amf(pl_raw, tr_raw, sig_raw, sigma),
        }
        if run_thantd:
            print(f"  [THANTD] training for regime {regime} ...", flush=True)
            th = _train_thantd(D_raw, tr_raw, sig_raw, cfg, device,
                               np.random.default_rng(seed))
            det['THANTD'] = score_thantd(th, sig_raw, pl_raw)

        results['auc'][regime] = {nm: float(auc_safe(lab, sc))
                                  for nm, sc in det.items()}
        results['roc'][regime] = {nm: roc_safe(lab, sc) for nm, sc in det.items()}
        line = "  ".join(f"{nm}={results['auc'][regime][nm]:.3f}"
                         for nm in det)
        print(f"  [{regime}] {line}", flush=True)

    json.dump({'scenario_id': sid, 'train_box': train_box, 'test_box': test_box,
               'dom_name': CLS_NAMES.get(dom_cls), **{k2: v for k2, v in info.items()},
               'auc': results['auc']},
              open(os.path.join(scen_dir, 'invariance_metrics.json'), 'w'),
              indent=2, default=str)
    _save_figures(sid, results, scen_dir)
    return results


def _save_figures(sid, results, scen_dir):
    fig_dir = os.path.join(scen_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    regimes   = list(results['auc'].keys())
    detectors = list(next(iter(results['auc'].values())).keys())

    # --- grouped bar: detector (x) × regime (color) ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(detectors)); w = 0.8 / len(regimes)
    for j, rg in enumerate(regimes):
        vals = [results['auc'][rg][d] for d in detectors]
        ax.bar(x + j * w, vals, w, label=rg,
               color=REGIME_COLORS.get(rg, None))
    ax.axhline(0.5, ls='--', c='grey', lw=1)
    ax.set_xticks(x + w * (len(regimes) - 1) / 2)
    ax.set_xticklabels(detectors, rotation=20, ha='right')
    ax.set_ylabel('AUC'); ax.set_ylim(0.4, 1.0)
    ax.set_title(f'Scenario {sid} — invariance to target–background spectral correlation')
    ax.legend(title='target regime'); ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'invariance_auc_s{sid}.pdf'),
                bbox_inches='tight'); plt.close(fig)

    # --- ROC panels, one per regime ---
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4.5))
    if len(regimes) == 1:
        axes = [axes]
    for ax, rg in zip(axes, regimes):
        for d in detectors:
            fpr, tpr = results['roc'][rg][d][0], results['roc'][rg][d][1]
            ax.plot(fpr, tpr, lw=1.6, color=DET_COLORS.get(d, None),
                    label=f'{d} ({results["auc"][rg][d]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(rg); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.legend(fontsize=7, loc='lower right'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Scenario {sid} — ROC by target regime')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'invariance_roc_s{sid}.pdf'),
                bbox_inches='tight'); plt.close(fig)
    print(f"  [fig] invariance_auc_s{sid}.pdf, invariance_roc_s{sid}.pdf", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--bg_classes', default='manual',
                    help='"manual" (default) = the heterogeneous manual_boxes.json '
                         'scenarios (the real, challenging setup). Or a comma-separated '
                         'list of class ids to instead use homogeneous single-material '
                         'boxes (diagnostic only).')
    ap.add_argument('--plant_model', default='additive',
                    choices=['additive', 'replacement'],
                    help='target model. "replacement" (y=(1-θ)w+θs) with high amplitude '
                         '≈ full-pixel targets — THANTD\'s native regime.')
    ap.add_argument('--amplitude', type=float, default=None,
                    help='override target amplitude θ (default = config). Use a strong '
                         'value (e.g. 1.0) to verify THANTD on strong/full-pixel targets.')
    ap.add_argument('--pretrained_dir', default=None,
                    help='path to a prior run (this invariance script OR run_colab) '
                         'whose signature-agnostic models (cfattn/nmlp/dsm) + pca.pkl '
                         'are LOADED instead of retrained. THANTD is still retrained '
                         '(it depends on the signature). Ideal for the strong/full-pixel '
                         'verification: reuse the already-trained models, only re-score.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-thantd', action='store_true')
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.config:
        cfg.update(yaml.safe_load(open(args.config)))
    cfg['plant_model'] = args.plant_model
    if args.amplitude is not None:
        cfg['amplitude'] = args.amplitude
    print(f"Planting: model={cfg['plant_model']}  amplitude={cfg['amplitude']}  "
          f"fraction={cfg['target_fraction']}", flush=True)
    if args.dry_run:
        cfg.update(dict(box_size_ablation=[400], cfattn_epochs=8, nmlp_epochs=8,
                        dsm_epochs=15, thantd_epochs=4, cfattn_K=4, nmlp_K=4))
    rd = args.results_dir or 'final_paper_experiments/results'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(rd, f'invariance_{ts}')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results dir: {results_dir}", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    # NO PCA: raw sensor values; our nets whiten internally (per-scenario frozen ZCA).
    data_norm, gt = load_and_normalize(cfg['dataset'], cfg.get('norm_mode', 'none'))
    H, W, _ = data_norm.shape
    flat = data_norm.reshape(-1, data_norm.shape[-1])
    gtf  = gt.ravel()
    print(f"RAW band space: D={data_norm.shape[-1]} (no PCA)", flush=True)
    class_means = {int(c): flat[gtf == c].mean(0).astype(np.float32)
                   for c in np.unique(gtf) if c != 0}

    # ---- assemble scenarios ----
    scenarios = []   # list of (label, {'train_box','test_box'})
    if args.bg_classes == 'manual':
        boxes = json.load(open(cfg['manual_boxes_path']))
        for i, b in enumerate(boxes):
            scenarios.append((f'manual{i}', b))
    else:
        for c in [int(x) for x in args.bg_classes.split(',')]:
            hb = find_homogeneous_boxes(gt, c)
            if hb is None:
                print(f"  [skip] no homogeneous box pair for class "
                      f"{CLS_NAMES.get(c)} ({c})", flush=True)
                continue
            tr_b, te_b = hb
            scenarios.append((f'{CLS_NAMES.get(c, c)}',
                              {'train_box': tr_b, 'test_box': te_b}))
    if args.dry_run:
        scenarios = scenarios[:1]
    print(f"Scenarios: {[s[0] for s in scenarios]}", flush=True)

    all_auc = {}
    for sid_idx, (label, box) in enumerate(scenarios):
        print(f"\n### {label}  train={box['train_box']} test={box['test_box']}",
              flush=True)
        res = run_scenario(label, sid_idx, box, cfg, data_norm, gt,
                           class_means, results_dir,
                           run_thantd=(not args.no_thantd), device=device,
                           dry_run=args.dry_run, pretrained_dir=args.pretrained_dir)
        all_auc[label] = res['auc']
        json.dump(all_auc, open(os.path.join(results_dir, 'all_invariance_auc.json'), 'w'),
                  indent=2)

    # ---- console summary table ----
    print(f"\n{'='*60}\nINVARIANCE SUMMARY (AUC)\n{'='*60}", flush=True)
    for sk, auc in all_auc.items():
        print(f"\n{sk}", flush=True)
        regimes = list(auc.keys())
        dets = list(next(iter(auc.values())).keys())
        print(f"  {'detector':14s}" + "".join(f"{rg:>16s}" for rg in regimes), flush=True)
        for d in dets:
            print(f"  {d:14s}" + "".join(f"{auc[rg][d]:16.3f}" for rg in regimes),
                  flush=True)
    print(f"\nAll results: {results_dir}", flush=True)


if __name__ == '__main__':
    main()
