"""
run_colab.py — Colab-ready comprehensive spatial experiment runner.

Runs 8 scenarios (4 manual + 4 random) × box-size ablation with all detectors:
    CF-Attn-CFAR (Ours — local Fisher norm)
    CF-Attn      (Ours — global norm)
    NeighborMLP  (Ours — spatial MLP)
    THANTD       (Baseline — triplet hybrid attention)
    DSM          (Baseline — global score, no spatial context)
    AMF          (Classical baseline)
    Reg-AMF      (Classical baseline)

Usage:
    # Dry-run first (verify outputs locally):
    .venv/bin/python -u experiments/spatial/run_colab.py --dry-run

    # Full run (on Colab T4):
    python -u run_colab.py --config colab.yaml --results_dir /drive/MyDrive/spatial_results

    # Resume interrupted run (skips existing checkpoints):
    python -u run_colab.py --config colab.yaml --results_dir /drive/MyDrive/spatial_results

    # Skip THANTD (no GPU):
    python -u run_colab.py --no-thantd

Honest evaluation:
    - Threshold set from TRAINING pixels only (cfar_threshold in evaluation.py)
    - THANTD uses thantd_use_secondary=true (clean secondary data mode)
    - Signature computed from dominant class in test box (GT is public, no labels)
    - No hyperparameters tuned on test labels
"""

import argparse, json, os, sys, time, pickle
from datetime import datetime
from pathlib import Path

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
from sklearn.cluster import KMeans
from tqdm import tqdm

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive,
    gmm_glrt,
)
from final_paper_experiments.baselines.gmm_glrt_levin import (
    gmm_glrt_levin_additive,
)
from final_paper_experiments.evaluation import (
    partial_auc, dr_at_fpr, auc_safe, roc_safe,
    cfar_threshold, per_class_fpr,
    compute_signature, generate_random_boxes, box_statistics,
    scores_to_spatial_map,
)
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods
from dsm_model import ScoreNet, dsm_loss, Whitening
from cfattn_model import (
    CFAttnGaussianScoreNet, cfattn_dsm_loss,
    score_cfattn_additive,
    score_cfattn_additive_cfar,
)
from neighbor_mlp_model import (
    NeighborMLPDenoiser, neighbor_mlp_dsm_loss,
    score_nmlp_additive,
)
from thantd_model import (
    THANTD, build_thantd_samples, train_thantd, score_thantd,
)
from scenario_figures import (
    save_scenario_figures,
    save_cfar_per_class_figure, save_auc_summary_figure,
    save_dr_at_fpr_figure, save_box_size_ablation_figure,
)

CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks',    9: 'shadows',
}

# ---------------------------------------------------------------------------
# Default config (overridden by colab.yaml / --dry-run)
# ---------------------------------------------------------------------------
DEFAULT_CFG = dict(
    dataset='data/pavia-u.mat',
    norm_mode='none',           # RAW sensor values — no scaling (no PCA anywhere)
    manual_boxes_path='experiments/spatial/manual_boxes.json',
    sig_dom_weight=0.8,
    sig_mean_weight=0.2,
    random_scenario_seeds=[42, 123, 456, 789],
    min_pixels=2000,
    box_size_ablation=[2000, 4000, 8000],
    amplitude=0.15,
    target_fraction=0.10,
    latent_dim=20,
    k=5,
    # CF-Attn
    cfattn_h=64, cfattn_K=9, cfattn_epochs=300,
    cfattn_lr=3e-4, cfattn_eps=1e-4,
    lam_ent=0.05, lam_div=0.05, lam_cov=1e-5,
    # NeighborMLP
    nmlp_d_lat=32, nmlp_K=8, nmlp_hidden=128, nmlp_n_layers=3,
    nmlp_epochs=300, nmlp_lr=3e-4, nmlp_batch=256,
    # THANTD
    thantd_m=7, thantd_d=64, thantd_heads=4,
    thantd_epochs=300, thantd_batch=64, thantd_lr=1e-4,
    thantd_margin=0.3, thantd_lambda=0.5, thantd_alpha=0.5,
    thantd_n_pairs=2048, thantd_use_secondary=True,
    # DSM
    dsm_hidden=[64, 64], dsm_epochs=1000, dsm_lr=5e-4,
    # shared
    activation='silu',
    dsm_sigma_rho=0.01,
    # frozen whitening front-end (replaces PCA): raw data in, nets whiten internally
    whiten_mode='zca',          # 'zca' | 'pca' | 'cholesky'  (easy one-line swap)
    whiten_eig_floor=1e-5,      # RELATIVE eigenvalue floor (× λ_max)
    batch_size=256,
    weight_decay=1e-4,
    seed=42,
    results_dir='final_paper_experiments/results',
)

# ---------------------------------------------------------------------------
# DRY-RUN overrides
# ---------------------------------------------------------------------------
DRYRUN_OVERRIDES = dict(
    box_size_ablation=[200],
    cfattn_epochs=10, nmlp_epochs=10, dsm_epochs=20, thantd_epochs=5,
    cfattn_K=4, nmlp_K=4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subsample(pix, nbr, n, rng):
    if len(pix) <= n:
        return pix, nbr, np.arange(len(pix))
    idx = rng.choice(len(pix), n, replace=False)
    return pix[idx], nbr[idx], idx


def _crop_pca_box(pca_img, box, k):
    r0, r1, c0, c1 = box
    sub = torch.tensor(pca_img[r0:r1, c0:c1, :], dtype=torch.float32)
    centers, nbrs = extract_neighborhoods(sub, k)
    return centers.numpy(), nbrs.numpy()


def _crop_raw_box(data_norm, box):
    r0, r1, c0, c1 = box
    return data_norm[r0:r1, c0:c1].reshape(-1, data_norm.shape[-1])


def _make_whitening(tr_raw, cfg, device):
    """Frozen whitening front-end fit on the RAW training background."""
    W = Whitening.from_data(np.asarray(tr_raw, dtype=np.float32),
                            mode=cfg.get('whiten_mode', 'zca'),
                            eig_floor=float(cfg.get('whiten_eig_floor', 1e-5)))
    return W.to(device)


def _whiten_np(W, X, device):
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        return W(t).cpu().numpy()


def _whitened_sigma(cfg):
    """In whitened space cov ≈ I, so σ² = ρ·1 ⇒ σ = √ρ (matches the reference)."""
    return float(np.sqrt(cfg['dsm_sigma_rho']))


def _placeholder_whitening(D):
    """Identity whitening of the right shape so load_state_dict can fill its buffers."""
    return Whitening(np.zeros(D, dtype=np.float32), np.eye(D, dtype=np.float32))


def _train_cfattn(tr_raw, tr_nbr_raw, cfg, device, seed):
    """Train CF-Attn on RAW data; whiten internally, kmeans init in whitened space."""
    D = tr_raw.shape[1]
    W = _make_whitening(tr_raw, cfg, device)
    tr_w = _whiten_np(W, tr_raw, device)       # whitened atoms for kmeans init
    sigma = _whitened_sigma(cfg)
    cfattn = CFAttnGaussianScoreNet(
        D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
        sigma=sigma, eps=cfg.get('cfattn_eps', 1e-4), whitening=W).to(device)
    km = KMeans(n_clusters=cfg['cfattn_K'], init='k-means++',
                n_init=5, random_state=seed, max_iter=100).fit(tr_w)
    cfattn.comp_mu.data.copy_(
        torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(cfattn.parameters(), lr=cfg['cfattn_lr'],
                            weight_decay=cfg['weight_decay'])
    Xtr = torch.tensor(tr_raw, dtype=torch.float32).to(device)
    Ntr = torch.tensor(tr_nbr_raw, dtype=torch.float32).to(device)
    P   = len(Xtr)
    pbar = tqdm(range(cfg['cfattn_epochs']), desc='CF-Attn', dynamic_ncols=True, leave=False)
    last = float('nan')
    for ep in pbar:
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, P, cfg['batch_size']):
            sel = perm[i:i+cfg['batch_size']]
            loss, di = cfattn_dsm_loss(cfattn, Xtr[sel], Ntr[sel],
                                       lam_ent=cfg.get('lam_ent', 0.05),
                                       lam_div=cfg.get('lam_div', 0.05),
                                       lam_cov=cfg.get('lam_cov', 1e-5))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(di); nb += 1
        last = ep_loss / max(nb, 1)
        pbar.set_postfix(loss=f'{last:.4f}')
    cfattn._final_loss = last
    cfattn.eval()
    return cfattn


def _train_nmlp(tr_raw, tr_nbr_raw, cfg, device):
    """Train NeighborMLP on RAW data; whiten internally."""
    D = tr_raw.shape[1]
    W = _make_whitening(tr_raw, cfg, device)
    sigma = _whitened_sigma(cfg)
    nmlp = NeighborMLPDenoiser(
        D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
        hidden=cfg['nmlp_hidden'], n_layers=cfg['nmlp_n_layers'],
        sigma=sigma, activation=cfg['activation'], whitening=W).to(device)
    opt  = torch.optim.AdamW(nmlp.parameters(), lr=cfg['nmlp_lr'],
                              weight_decay=cfg['weight_decay'])
    Xtr = torch.tensor(tr_raw, dtype=torch.float32).to(device)
    Ntr = torch.tensor(tr_nbr_raw, dtype=torch.float32).to(device)
    P   = len(Xtr)
    pbar = tqdm(range(cfg['nmlp_epochs']), desc='NeighborMLP', dynamic_ncols=True, leave=False)
    last = float('nan')
    for ep in pbar:
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, P, cfg['nmlp_batch']):
            sel = perm[i:i+cfg['nmlp_batch']]
            loss = neighbor_mlp_dsm_loss(nmlp, Xtr[sel], Ntr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        pbar.set_postfix(loss=f'{last:.4f}')
    nmlp._final_loss = last
    nmlp.eval()
    return nmlp


def _train_dsm(tr_raw, cfg, device):
    """Train per-pixel DSM on RAW data; whiten internally (replaces PCA + z-score)."""
    D = tr_raw.shape[1]
    W = _make_whitening(tr_raw, cfg, device)
    sigma = _whitened_sigma(cfg)
    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'],
                       whitening=W).to(device)
    dsm_net.sigma = sigma          # expose for downstream metadata
    opt     = torch.optim.Adam(dsm_net.parameters(), lr=cfg['dsm_lr'],
                               weight_decay=cfg['weight_decay'])
    Xtr = torch.tensor(tr_raw, dtype=torch.float32).to(device)
    P   = len(Xtr)
    pbar = tqdm(range(cfg['dsm_epochs']), desc='DSM', dynamic_ncols=True, leave=False)
    last = float('nan')
    for ep in pbar:
        perm = torch.randperm(P, device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, P, cfg['batch_size']):
            b = Xtr[perm[i:i+cfg['batch_size']]]
            loss = dsm_loss(dsm_net, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        pbar.set_postfix(loss=f'{last:.4f}')
    dsm_net._final_loss = last
    dsm_net.eval()
    return dsm_net


def _train_thantd(D_raw, tr_raw, sig_raw, cfg, device, rng):
    bkg_for_samples = tr_raw if cfg.get('thantd_use_secondary', True) else None
    a_smp, p_smp, n_smp = build_thantd_samples(
        tr_raw, sig_raw,
        alpha=cfg.get('thantd_alpha', 0.5),
        n_samples=cfg.get('thantd_n_pairs', 2048),
        rng=rng, bkg_pool=bkg_for_samples)
    thantd = THANTD(b=D_raw, m=cfg.get('thantd_m', 7),
                    d=cfg.get('thantd_d', 64),
                    n_heads=cfg.get('thantd_heads', 4))
    train_thantd(thantd, a_smp, p_smp, n_smp,
                 epochs=cfg.get('thantd_epochs', 300),
                 batch_size=cfg.get('thantd_batch', 64),
                 lr=cfg.get('thantd_lr', 1e-4),
                 margin=cfg.get('thantd_margin', 0.3),
                 lam=cfg.get('thantd_lambda', 0.5),
                 device=device)
    return thantd


# ---------------------------------------------------------------------------
# Per-scenario runner
# ---------------------------------------------------------------------------

def run_scenario(sid, scenario, n_budget, cfg,
                 data_norm, gt, results_dir,
                 run_thantd=True, device='cpu', dry_run=False):
    """
    Train models and evaluate all detectors for one (scenario, budget) pair.
    Checkpoints are saved/loaded for crash recovery.
    """
    train_box = scenario['train_box']
    test_box  = scenario['test_box']

    scen_dir = os.path.join(results_dir, f'scenario_{sid}', f'n{n_budget}')
    mdl_dir  = os.path.join(scen_dir, 'models')
    fig_dir  = os.path.join(scen_dir, 'figures')
    os.makedirs(mdl_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    metrics_path = os.path.join(scen_dir, 'metrics.json')
    scores_path  = os.path.join(scen_dir, 'scores.npz')

    print(f"\n{'='*60}", flush=True)
    print(f"Scenario {sid}  n_budget={n_budget}  device={device}", flush=True)
    print(f"  train_box={train_box}  test_box={test_box}", flush=True)

    seed = int(cfg['seed']) + sid * 100
    rng  = np.random.default_rng(seed)
    torch.manual_seed(seed)

    H, W, D_raw = data_norm.shape
    D = D_raw                       # NO PCA: nets/detectors work in raw band space
    k = int(cfg['k'])

    # ---- Crop + subsample training pixels (RAW bands) ----
    tr_raw_full, tr_nbr_full = _crop_pca_box(data_norm, train_box, k)

    tr_raw, tr_nbr, tr_idx = _subsample(tr_raw_full, tr_nbr_full, n_budget, rng)
    print(f"  train={len(tr_raw)} px", flush=True)

    # ---- Full test box (ALL pixels, no subsampling), RAW bands ----
    r0t, r1t, c0t, c1t = test_box
    H_b, W_b = r1t - r0t, c1t - c0t
    te_raw_full, te_nbr_full = _crop_pca_box(data_norm, test_box, k)
    te_gt_full               = gt[r0t:r1t, c0t:c1t].ravel()
    te_idx_full              = np.arange(len(te_raw_full))  # all test pixels, in order
    print(f"  test={len(te_raw_full)} px", flush=True)

    # ---- Target signature (RAW band space) ----
    w_dom  = float(cfg.get('sig_dom_weight', 0.8))
    w_mean = float(cfg.get('sig_mean_weight', 0.2))
    te_raw_box = data_norm[r0t:r1t, c0t:c1t]
    sig_raw, dom_cls, dom_name = compute_signature(
        gt[r0t:r1t, c0t:c1t], te_raw_box,
        w_dom=w_dom, w_mean=w_mean)
    sig_raw = sig_raw.astype(np.float32)
    print(f"  signature: {w_dom}·{dom_name} + {w_mean}·patch_mean  "
          f"||s_raw||={np.linalg.norm(sig_raw):.4f}", flush=True)

    # ---- sigma for the classical detectors (Reg-AMF). Our score nets compute
    #      their own whitened-space sigma (=sqrt(rho)) internally.
    sigma = compute_sigma_from_data(tr_raw, cfg['dsm_sigma_rho'])

    # ---- Train models (with checkpoint save/load) ----
    def ckpt(name):
        return os.path.join(mdl_dir, f'{name}_s{sid}_n{n_budget}.pt')

    # CF-Attn
    cf_ckpt = ckpt('cfattn')
    if os.path.exists(cf_ckpt):
        print("  [CF-Attn] loading checkpoint", flush=True)
        cfattn = CFAttnGaussianScoreNet(
            D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
            sigma=_whitened_sigma(cfg), eps=cfg.get('cfattn_eps', 1e-4),
            whitening=_placeholder_whitening(D))
        cfattn.load_state_dict(torch.load(cf_ckpt, map_location='cpu')['state_dict'])
        cfattn.eval()
    else:
        print("  [CF-Attn] training ...", flush=True)
        t0 = time.time()
        cfattn = _train_cfattn(tr_raw, tr_nbr, cfg, device, seed)
        torch.save({'state_dict': cfattn.state_dict(), 'cfg': cfg}, cf_ckpt)
        print(f"  [CF-Attn] done in {time.time()-t0:.0f}s", flush=True)

    # NeighborMLP
    nmlp_ckpt = ckpt('nmlp')
    if os.path.exists(nmlp_ckpt):
        print("  [NeighborMLP] loading checkpoint", flush=True)
        nmlp = NeighborMLPDenoiser(
            D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
            hidden=cfg['nmlp_hidden'], n_layers=cfg['nmlp_n_layers'],
            sigma=_whitened_sigma(cfg), activation=cfg['activation'],
            whitening=_placeholder_whitening(D))
        nmlp.load_state_dict(torch.load(nmlp_ckpt, map_location='cpu')['state_dict'])
        nmlp.eval()
    else:
        print("  [NeighborMLP] training ...", flush=True)
        t0 = time.time()
        nmlp = _train_nmlp(tr_raw, tr_nbr, cfg, device)
        torch.save({'state_dict': nmlp.state_dict(), 'cfg': cfg}, nmlp_ckpt)
        print(f"  [NeighborMLP] done in {time.time()-t0:.0f}s", flush=True)

    # DSM
    dsm_ckpt = ckpt('dsm')
    if os.path.exists(dsm_ckpt):
        print("  [DSM] loading checkpoint", flush=True)
        dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'],
                           whitening=_placeholder_whitening(D))
        dsm_net.load_state_dict(torch.load(dsm_ckpt, map_location='cpu')['state_dict'])
        dsm_net.eval()
    else:
        print("  [DSM] training ...", flush=True)
        t0 = time.time()
        dsm_net = _train_dsm(tr_raw, cfg, device)
        torch.save({'state_dict': dsm_net.state_dict(), 'cfg': cfg}, dsm_ckpt)
        print(f"  [DSM] done in {time.time()-t0:.0f}s", flush=True)

    # THANTD
    thantd_model = None
    if run_thantd:
        th_ckpt = ckpt('thantd')
        if os.path.exists(th_ckpt):
            print("  [THANTD] loading checkpoint", flush=True)
            thantd_model = THANTD(b=D_raw, m=cfg.get('thantd_m', 7),
                                   d=cfg.get('thantd_d', 64),
                                   n_heads=cfg.get('thantd_heads', 4))
            thantd_model.load_state_dict(
                torch.load(th_ckpt, map_location='cpu')['state_dict'])
            thantd_model.eval()
        else:
            print("  [THANTD] training ...", flush=True)
            t0 = time.time()
            thantd_model = _train_thantd(D_raw, tr_raw, sig_raw, cfg, device, rng)
            torch.save({'state_dict': thantd_model.state_dict(), 'cfg': cfg}, th_ckpt)
            print(f"  [THANTD] done in {time.time()-t0:.0f}s", flush=True)

    # ---- Evaluate all detectors ----
    print("  [Eval] scoring all detectors ...", flush=True)
    scores_out = {}
    labels_out = {}
    thresholds_out = {}
    tgt_idx_dict = {}
    metrics = {'scenario_id': sid, 'n_budget': n_budget,
               'train_box': train_box, 'test_box': test_box,
               'signature': f'{w_dom}·{dom_name}+{w_mean}·patch_mean',
               'dom_cls': dom_cls, 'dom_name': dom_name}

    te_nbr_f = te_nbr_full.astype(np.float32)
    tr_nbr_f = tr_nbr.astype(np.float32)

    for tm in ('additive',):
        planted_raw, labels, tgt_idx = plant_targets(
            te_raw_full, sig_raw, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        tgt_idx_dict[tm] = tgt_idx

        # our score nets: RAW input (whiten internally) + RAW signature (data-space scores)
        sc_cfa_cfar = score_cfattn_additive_cfar(
            cfattn, planted_raw, te_nbr_f, sig_raw)
        sc_cfa      = score_cfattn_additive(
            cfattn, planted_raw, te_nbr_f, tr_raw, tr_nbr_f, sig_raw)
        sc_nmlp     = score_nmlp_additive(
            nmlp, planted_raw, te_nbr_f, tr_raw, tr_nbr_f, sig_raw)
        sc_dsm      = dsm_additive(planted_raw, tr_raw, dsm_net, sig_raw)

        sc_amf    = amf(planted_raw, tr_raw, sig_raw)
        sc_regamf = reg_amf(planted_raw, tr_raw, sig_raw, sigma)

        sc_gmmglrt = gmm_glrt(planted_raw, tr_raw, sig_raw,
                              K=cfg.get('gmm_K', 3),
                              theta_steps=cfg.get('gmm_steps', 50))
        sc_levin   = gmm_glrt_levin_additive(planted_raw, tr_raw, sig_raw,
                                              p_steps=cfg.get('gmm_steps', 50))

        det_scores = {
            'CF-Attn-CFAR': sc_cfa_cfar,
            'CF-Attn':      sc_cfa,
            'NeighborMLP':  sc_nmlp,
            'DSM':          sc_dsm,
            'AMF':          sc_amf,
            'Reg-AMF':      sc_regamf,
            'GMM-GLRT':     sc_gmmglrt,
            'GMM-Levin':    sc_levin,
        }
        if thantd_model is not None:
            det_scores['THANTD'] = score_thantd(thantd_model, sig_raw, planted_raw)

        for nm, sc in det_scores.items():
            key = f'{nm}_{tm}'
            scores_out[key] = sc
            labels_out[key] = labels

        # ---- Compute training-pixel scores for CFAR threshold ----
        # Use CLEAN training pixels (no planted targets) for threshold setting
        with torch.no_grad():
            tr_sc_cfar = score_cfattn_additive_cfar(
                cfattn, tr_raw, tr_nbr_f, sig_raw)
            tr_sc_cfa  = score_cfattn_additive(
                cfattn, tr_raw, tr_nbr_f, tr_raw, tr_nbr_f, sig_raw)
            tr_sc_nmlp = score_nmlp_additive(
                nmlp, tr_raw, tr_nbr_f, tr_raw, tr_nbr_f, sig_raw)
            tr_sc_dsm  = dsm_additive(tr_raw, tr_raw, dsm_net, sig_raw)

        tr_sc_amf    = amf(tr_raw, tr_raw, sig_raw)
        tr_sc_regamf = reg_amf(tr_raw, tr_raw, sig_raw, sigma)
        tr_sc_gmmglrt = gmm_glrt(tr_raw, tr_raw, sig_raw,
                                 K=cfg.get('gmm_K', 3),
                                 theta_steps=cfg.get('gmm_steps', 50))
        tr_sc_levin   = gmm_glrt_levin_additive(tr_raw, tr_raw, sig_raw,
                                                 p_steps=cfg.get('gmm_steps', 50))
        train_scores = {
            'CF-Attn-CFAR': tr_sc_cfar,
            'CF-Attn':      tr_sc_cfa,
            'NeighborMLP':  tr_sc_nmlp,
            'DSM':          tr_sc_dsm,
            'AMF':          tr_sc_amf,
            'Reg-AMF':      tr_sc_regamf,
            'GMM-GLRT':     tr_sc_gmmglrt,
            'GMM-Levin':    tr_sc_levin,
        }
        if thantd_model is not None:
            train_scores['THANTD'] = score_thantd(thantd_model, sig_raw, tr_raw)

        # Set CFAR thresholds from training pixels only
        for nm, tr_sc in train_scores.items():
            thr = cfar_threshold(tr_sc, target_fpr=0.01)
            thresholds_out[f'{nm}_{tm}'] = thr

        # ---- Per-class FPR ----
        cfar_dict = {}
        for nm, sc in det_scores.items():
            thr = thresholds_out[f'{nm}_{tm}']
            cfar_dict[nm] = per_class_fpr(sc, labels, te_gt_full, thr)

        # ---- Metrics ----
        auc_dict   = {nm: auc_safe(labels, sc)
                      for nm, sc in det_scores.items()}
        pauc_dict  = {nm: partial_auc(labels, sc, fpr_max=0.05)
                      for nm, sc in det_scores.items()}
        dr_dict    = {nm: dr_at_fpr(labels, sc)
                      for nm, sc in det_scores.items()}
        roc_dict   = {nm: roc_safe(labels, sc)
                      for nm, sc in det_scores.items()}

        metrics[tm] = {'auc': auc_dict, 'pauc': pauc_dict,
                       'dr': dr_dict, 'cfar': cfar_dict}

        line = "  ".join(f"{k}={v:.3f}" for k, v in auc_dict.items()
                         if not np.isnan(v))
        print(f"  [{tm}] AUC: {line}", flush=True)

    # ---- Save scores.npz ----
    np_dict = {}
    for k, v in scores_out.items():
        np_dict[f'score_{k}'] = v
    for k, v in labels_out.items():
        np_dict[f'label_{k}'] = v
    for k, v in thresholds_out.items():
        np_dict[f'thr_{k}'] = np.array([v])
    np_dict['te_gt_cls']   = te_gt_full
    np_dict['te_idx_full'] = te_idx_full
    for tm, idx in tgt_idx_dict.items():
        np_dict[f'tgt_idx_{tm}'] = idx
    np.savez(scores_path, **np_dict)

    # ---- Save metrics.json ----
    json.dump(metrics, open(metrics_path, 'w'), indent=2,
              default=lambda x: float(x) if hasattr(x, '__float__') else x)

    # ---- Save all figures ----
    save_scenario_figures(
        sid=sid, n_budget=n_budget,
        scores_dict=scores_out,
        labels_dict=labels_out,
        thresholds_dict=thresholds_out,
        te_gt_flat=te_gt_full,
        tgt_idx_dict=tgt_idx_dict,
        data_norm=data_norm,
        gt=gt,
        train_box=train_box,
        test_box=test_box,
        fig_dir=fig_dir,
    )

    print(f"  → saved: {scen_dir}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Dry-run verification
# ---------------------------------------------------------------------------

def run_dry_run_checks(sid, n_budget, results_dir):
    """Print PASS/FAIL for all expected output files."""
    scen_dir = os.path.join(results_dir, f'scenario_{sid}', f'n{n_budget}')
    fig_dir  = os.path.join(scen_dir, 'figures')
    mdl_dir  = os.path.join(scen_dir, 'models')
    prefix   = f'scenario_{sid}_n{n_budget}'

    checks = [
        (os.path.join(scen_dir, 'scores.npz'),              'scores.npz'),
        (os.path.join(scen_dir, 'metrics.json'),             'metrics.json'),
        (os.path.join(fig_dir, f'{prefix}_boxes.pdf'),       f'{prefix}_boxes.pdf'),
        (os.path.join(fig_dir, f'{prefix}_targets.pdf'),     f'{prefix}_targets.pdf'),
        (os.path.join(fig_dir, f'{prefix}_score_maps.pdf'),  f'{prefix}_score_maps.pdf'),
        (os.path.join(fig_dir, f'{prefix}_detection_on_gt.pdf'),
                                                              f'{prefix}_detection_on_gt.pdf'),
        (os.path.join(fig_dir, f'{prefix}_roc.pdf'),         f'{prefix}_roc.pdf'),
    ]
    # Also check at least one model checkpoint
    for nm in ('cfattn', 'nmlp', 'dsm'):
        path = os.path.join(mdl_dir, f'{nm}_s{sid}_n{n_budget}.pt')
        checks.append((path, f'models/{nm}_s{sid}_n{n_budget}.pt'))

    all_pass = True
    print("\n" + "="*50)
    print("DRY-RUN VERIFICATION:")
    for path, name in checks:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else 0
        ok     = exists and (size > 1024)   # at least 1KB
        status = '[PASS]' if ok else '[FAIL]'
        if not ok:
            all_pass = False
        print(f"  {status} {name}  ({size} bytes)")

    # Check scores.npz keys
    npz_path = os.path.join(scen_dir, 'scores.npz')
    if os.path.exists(npz_path):
        npz = np.load(npz_path)
        expected_keys = ['score_CF-Attn-CFAR_additive', 'score_NeighborMLP_additive',
                         'score_CF-Attn_additive', 'score_DSM_additive',
                         'score_AMF_additive', 'te_gt_cls']
        for k in expected_keys:
            ok = k in npz
            print(f"  {'[PASS]' if ok else '[FAIL]'} scores.npz key: {k}")
            if not ok:
                all_pass = False

    # Check metrics.json structure
    mj_path = os.path.join(scen_dir, 'metrics.json')
    if os.path.exists(mj_path):
        mj = json.load(open(mj_path))
        for field in ('additive',):
            ok = field in mj and 'auc' in mj[field] and 'cfar' in mj[field]
            print(f"  {'[PASS]' if ok else '[FAIL]'} metrics.json[{field}] has auc+cfar")
            if not ok:
                all_pass = False

    print("="*50)
    print(f"\n{'ALL CHECKS PASSED ✓' if all_pass else 'SOME CHECKS FAILED ✗ — fix before Colab run'}")
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      default=os.path.join(_EXP, 'colab.yaml'))
    p.add_argument('--results_dir', default=None)
    p.add_argument('--dry-run',     action='store_true',
                   help='Run 1 scenario with tiny settings; verify all outputs')
    p.add_argument('--no-thantd',   action='store_true',
                   help='Skip THANTD (no GPU required)')
    args = p.parse_args()

    # ---- Load config ----
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    else:
        print(f"Config not found at {args.config}, using defaults.", flush=True)

    if args.dry_run:
        cfg.update(DRYRUN_OVERRIDES)
        print("DRY-RUN MODE: small settings to verify pipeline.", flush=True)

    if args.results_dir:
        cfg['results_dir'] = args.results_dir

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'colab_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)
    print(f"Results dir: {run_dir}", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    seed = int(cfg['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Load data ----
    print("Loading Pavia-U ...", flush=True)
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw   = data_norm.shape
    all_flat      = data_norm.reshape(-1, D_raw)
    print(f"Image {H}×{W}×{D_raw}  norm={cfg['norm_mode']}", flush=True)

    # ---- NO PCA: every detector consumes raw bands; our nets whiten internally
    #      (frozen ZCA first layer, fit per-scenario on the training box). ----
    print(f"RAW band space: D={D_raw} (no PCA, whiten_mode={cfg.get('whiten_mode','zca')})",
          flush=True)

    # ---- Load scenarios ----
    manual_path = cfg.get('manual_boxes_path',
                          os.path.join(_EXP, 'manual_boxes.json'))
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            manual_scenarios = json.load(f)
        print(f"Loaded {len(manual_scenarios)} manual scenarios from {manual_path}")
    else:
        print(f"WARNING: {manual_path} not found. "
              f"Run pick_boxes_interactive.py first, or only random scenarios will be used.")
        manual_scenarios = []

    random_scenarios = generate_random_boxes(
        gt, n=4,
        min_pixels=int(cfg.get('min_pixels', 2000)),
        seeds=tuple(cfg.get('random_scenario_seeds', [42, 123, 456, 789])))

    all_scenarios = manual_scenarios + random_scenarios
    if args.dry_run:
        all_scenarios = all_scenarios[:1]   # only one scenario for dry-run

    print(f"Total scenarios: {len(all_scenarios)} "
          f"({len(manual_scenarios)} manual + {len(random_scenarios)} random)", flush=True)

    # ---- Save scenario info ----
    json.dump(all_scenarios, open(os.path.join(run_dir, 'all_scenarios.json'), 'w'), indent=2)

    # ---- Main loop ----
    t_start = time.time()
    all_metrics = {}

    for n_budget in cfg['box_size_ablation']:
        print(f"\n{'#'*60}")
        print(f"BOX SIZE BUDGET: n={n_budget} pixels")

        for sid, scenario in enumerate(all_scenarios):
            key = f'scenario_{sid}'
            if key not in all_metrics:
                all_metrics[key] = {}

            m = run_scenario(
                sid=sid,
                scenario=scenario,
                n_budget=n_budget,
                cfg=cfg,
                data_norm=data_norm,
                gt=gt,
                results_dir=run_dir,
                run_thantd=(not args.no_thantd),
                device=device,
                dry_run=args.dry_run,
            )
            all_metrics[key][f'n{n_budget}'] = {
                tm: m.get(tm, {}) for tm in ('additive',)
            }
            # Also store cfar + dr separately for aggregated figures
            for tm in ('additive',):
                if tm in m:
                    all_metrics[key][f'n{n_budget}'][f'dr_{tm}'] = m[tm].get('dr', {})
                    all_metrics[key][f'n{n_budget}'][f'cfar_{tm}'] = m[tm].get('cfar', {})

        # Save running results
        json.dump(all_metrics,
                  open(os.path.join(run_dir, 'all_metrics.json'), 'w'),
                  indent=2, default=str)

    # ---- Aggregated figures ----
    print(f"\n{'='*60}")
    print("Saving aggregated figures ...", flush=True)
    agg_fig_dir = os.path.join(run_dir, 'figures')
    os.makedirs(agg_fig_dir, exist_ok=True)

    for n_budget in cfg['box_size_ablation']:
        # Collect per-class CFAR across scenarios (additive model)
        all_cfar_additive = []
        for key, sid_data in all_metrics.items():
            bkey = f'n{n_budget}'
            if bkey in sid_data and 'cfar_additive' in sid_data[bkey]:
                all_cfar_additive.append(sid_data[bkey]['cfar_additive'])

        if all_cfar_additive:
            # Reformat: list of {det: {cls: fpr}}
            save_cfar_per_class_figure(
                all_cfar_additive, agg_fig_dir,
                target_model='additive')

        save_auc_summary_figure(all_metrics, agg_fig_dir, n_budget)
        save_dr_at_fpr_figure(all_metrics, agg_fig_dir, n_budget)

    save_box_size_ablation_figure(
        all_metrics, agg_fig_dir,
        budgets=cfg['box_size_ablation'])

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min.", flush=True)
    print(f"All results: {run_dir}", flush=True)

    # ---- Dry-run checks ----
    if args.dry_run:
        all_pass = run_dry_run_checks(0, cfg['box_size_ablation'][0], run_dir)
        sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
