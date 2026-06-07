"""
run_thantd_ablation.py — THANTD hyper-param ablation vs spatial baselines.

Trains spatial baselines (CF-Attn, NeighborMLP, DSM, AMF) ONCE, then sweeps
all THANTD configs defined in thantd_configs list of the yaml.  Produces a
summary comparison table + bar chart.

Usage:
    .venv/bin/python -u experiments/spatial/run_thantd_ablation.py \
        --config experiments/spatial/thantd_ablation.yaml
"""

import argparse, os, sys, json, time, pickle
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, plant_targets,
)
from final_paper_experiments.baselines.detectors import (
    amf, reg_amf, dsm_additive, dsm_replacement, amf_replacement,
)
from final_paper_experiments.models.neighbor_adapted import extract_neighborhoods
from dsm_model import ScoreNet, dsm_loss
from cfattn_model import (
    CFAttnGaussianScoreNet, cfattn_dsm_loss,
    score_cfattn_additive, score_cfattn_replacement,
)
from neighbor_mlp_model import (
    NeighborMLPDenoiser, neighbor_mlp_dsm_loss,
    score_nmlp_additive, score_nmlp_replacement,
)
from thantd_model import THANTD, build_thantd_samples, train_thantd, score_thantd

CLS_NAMES = {0:'unlabeled', 1:'asphalt', 2:'meadows', 3:'gravel',
             4:'trees', 5:'metal_sheets', 6:'bare_soil', 7:'bitumen',
             8:'bricks', 9:'shadows'}

COLORS = {'CF-Attn':'#1f77b4','NeighborMLP':'#2ca02c','DSM':'#ff7f0e',
          'AMF':'#9467bd','Reg-AMF':'#aec7e8','AMF-rep':'#08306b'}
THANTD_CMAP = plt.cm.Reds


def auc_safe(lab, sc):
    try:    return float(roc_auc_score(lab, sc))
    except: return float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(_EXP, 'thantd_ablation.yaml'))
    args = p.parse_args()
    cfg  = yaml.safe_load(open(args.config))

    t0  = time.time()
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg['results_dir'], f'ablation_{ts}')
    fig_dir = os.path.join(run_dir, 'figures')
    mdl_dir = os.path.join(run_dir, 'models')
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)
    yaml.dump(cfg, open(os.path.join(run_dir, 'config.yaml'), 'w'), sort_keys=False)
    print(f"Run dir: {run_dir}", flush=True)

    seed = int(cfg['seed']); torch.manual_seed(seed)
    rng  = np.random.default_rng(seed)

    # ---- Data ----
    data_norm, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    H, W, D_raw = data_norm.shape
    all_flat = data_norm.reshape(-1, D_raw); gt_flat = gt.reshape(-1)
    print(f"Image {H}×{W}×{D_raw}  norm={cfg['norm_mode']}", flush=True)

    D   = cfg['latent_dim']
    pca = PCA(n_components=D, random_state=seed).fit(all_flat)
    pca_img = pca.transform(all_flat).reshape(H, W, D).astype(np.float32)

    # ---- Target signature ----
    r0,r1,c0,c1 = cfg['test_box']
    test_raw_box = all_flat.reshape(H,W,D_raw)[r0:r1,c0:c1].reshape(-1,D_raw)
    test_gt_box  = gt[r0:r1,c0:c1].reshape(-1)
    labeled = test_gt_box != 0
    cls_ids, cnts = np.unique(test_gt_box[labeled], return_counts=True)
    dom_cls  = int(cls_ids[cnts.argmax()])
    w_dom    = float(cfg.get('sig_dom_weight', 0.9))
    w_mean   = float(cfg.get('sig_mean_weight', 0.1))
    sig_raw  = (w_dom * test_raw_box[test_gt_box==dom_cls].mean(0).astype(np.float32)
              + w_mean * test_raw_box.mean(0).astype(np.float32))
    s_pca    = pca.transform(sig_raw[None]).flatten().astype(np.float32)
    dom_name = CLS_NAMES.get(dom_cls, f'cls{dom_cls}')
    print(f"Target: {w_dom}·{dom_name} + {w_mean}·test_mean", flush=True)

    # ---- Crop pixels ----
    k = cfg['k']
    def crop_box(box):
        r0,r1,c0,c1 = box
        sub = torch.tensor(pca_img[r0:r1,c0:c1,:], dtype=torch.float32)
        centers, nbrs = extract_neighborhoods(sub, k)
        return centers.numpy(), nbrs.numpy()

    tr_pca, tr_nbr = crop_box(cfg['train_box'])
    te_pca, te_nbr = crop_box(cfg['test_box'])

    def subsample(pix, nbr, n):
        if len(pix) <= n: return pix, nbr
        idx = rng.choice(len(pix), n, replace=False)
        return pix[idx], nbr[idx]

    tr_pca, tr_nbr = subsample(tr_pca, tr_nbr, cfg['train_n'])
    te_pca, te_nbr = subsample(te_pca, te_nbr, cfg['test_n'])
    print(f"train={len(tr_pca)} px  test={len(te_pca)} px", flush=True)

    def crop_raw(box):
        r0,r1,c0,c1 = box
        return data_norm[r0:r1,c0:c1].reshape(-1, D_raw)

    rng2 = np.random.default_rng(seed)
    tr_raw_box = crop_raw(cfg['train_box'])
    te_raw_box = crop_raw(cfg['test_box'])
    def _sub_raw(box_raw, n):
        if len(box_raw) <= n: return box_raw
        return box_raw[rng2.choice(len(box_raw), n, replace=False)]
    tr_raw = _sub_raw(tr_raw_box, cfg['train_n'])
    te_raw = _sub_raw(te_raw_box, cfg['test_n'])

    sigma = compute_sigma_from_data(pca.transform(all_flat), cfg['dsm_sigma_rho'])
    print(f"sigma={sigma:.5f}\n", flush=True)

    Xtr = torch.tensor(tr_pca, dtype=torch.float32)
    Ntr = torch.tensor(tr_nbr, dtype=torch.float32)
    P, bs = len(Xtr), cfg['batch_size']

    # ================================================================ #
    # 1. Train spatial baselines ONCE                                   #
    # ================================================================ #
    print("=" * 55, flush=True)
    print("Training spatial baselines (once) ...", flush=True)
    print("=" * 55, flush=True)

    # CF-Attn
    print("\n[CF-Attn] training ...", flush=True)
    cfattn = CFAttnGaussianScoreNet(D=D, h=cfg['cfattn_h'], K=cfg['cfattn_K'],
                                     sigma=sigma, eps=cfg.get('cfattn_eps', 1e-4))
    km = KMeans(n_clusters=cfg['cfattn_K'], init='k-means++',
                n_init=5, random_state=seed, max_iter=100).fit(tr_pca)
    cfattn.comp_mu.data.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
    opt = torch.optim.AdamW(cfattn.parameters(), lr=cfg['cfattn_lr'],
                            weight_decay=cfg['weight_decay'])
    for _ in tqdm(range(cfg['cfattn_epochs']), desc='CF-Attn', dynamic_ncols=True):
        perm = torch.randperm(P)
        for i in range(0, P, bs):
            sel = perm[i:i+bs]
            loss, _ = cfattn_dsm_loss(cfattn, Xtr[sel], Ntr[sel],
                                       lam_ent=cfg.get('lam_ent', 0.05),
                                       lam_div=cfg.get('lam_div', 0.05),
                                       lam_cov=cfg.get('lam_cov', 1e-5))
            opt.zero_grad(); loss.backward(); opt.step()
    cfattn.eval()

    # NeighborMLP
    print("\n[NeighborMLP] training ...", flush=True)
    nmlp = NeighborMLPDenoiser(D=D, d_lat=cfg['nmlp_d_lat'], K=cfg['nmlp_K'],
                                hidden=cfg['nmlp_hidden'], n_layers=cfg['nmlp_n_layers'],
                                sigma=sigma, activation=cfg['activation'])
    opt = torch.optim.AdamW(nmlp.parameters(), lr=cfg['nmlp_lr'],
                            weight_decay=cfg['weight_decay'])
    for _ in tqdm(range(cfg['nmlp_epochs']), desc='NeighborMLP', dynamic_ncols=True):
        perm = torch.randperm(P)
        for i in range(0, P, cfg['nmlp_batch']):
            sel = perm[i:i+cfg['nmlp_batch']]
            loss = neighbor_mlp_dsm_loss(nmlp, Xtr[sel], Ntr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
    nmlp.eval()

    # DSM
    print("\n[DSM] training ...", flush=True)
    dsm_net = ScoreNet(D, list(cfg['dsm_hidden']), cfg['activation'])
    opt = torch.optim.Adam(dsm_net.parameters(), lr=cfg['dsm_lr'],
                           weight_decay=cfg['weight_decay'])
    for _ in tqdm(range(cfg['dsm_epochs']), desc='DSM', dynamic_ncols=True):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = Xtr[perm[i:i+bs]]
            loss = dsm_loss(dsm_net, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
    dsm_net.eval()

    # ---- Evaluate baselines ----
    baseline_auc = {}
    for tm in ('additive', 'replacement'):
        planted_pca, labels, _ = plant_targets(
            te_pca, s_pca, cfg['amplitude'], cfg['target_fraction'],
            model=tm, seed=seed)
        te_nbr_f = te_nbr.astype(np.float32)
        tr_nbr_f = tr_nbr.astype(np.float32)

        if tm == 'additive':
            scores = {
                'CF-Attn':     score_cfattn_additive(cfattn, planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca),
                'NeighborMLP': score_nmlp_additive(nmlp,   planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca),
                'DSM':         dsm_additive(planted_pca, tr_pca, dsm_net, s_pca),
                'AMF':         amf(planted_pca, tr_pca, s_pca),
                'Reg-AMF':     reg_amf(planted_pca, tr_pca, s_pca, sigma),
            }
        else:
            scores = {
                'CF-Attn':     score_cfattn_replacement(cfattn, planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca),
                'NeighborMLP': score_nmlp_replacement(nmlp,   planted_pca, te_nbr_f, tr_pca, tr_nbr_f, s_pca),
                'DSM':         dsm_replacement(planted_pca, tr_pca, dsm_net, s_pca),
                'AMF':         amf(planted_pca, tr_pca, s_pca),
                'Reg-AMF':     reg_amf(planted_pca, tr_pca, s_pca, sigma),
                'AMF-rep':     amf_replacement(planted_pca, tr_pca, s_pca),
            }
        baseline_auc[tm] = {k: auc_safe(labels, v) for k, v in scores.items()}

    print("\nBaseline AUC:", flush=True)
    for tm in ('additive', 'replacement'):
        row = "  ".join(f"{k}={v:.3f}" for k, v in baseline_auc[tm].items())
        print(f"  [{tm}]  {row}", flush=True)

    # ================================================================ #
    # 2. THANTD ablation loop                                           #
    # ================================================================ #
    thantd_configs = cfg.get('thantd_configs', [])
    thantd_results = []   # list of {label, auc_add, auc_rep}

    for ti, tcfg in enumerate(thantd_configs):
        label = tcfg.get('label', f'THANTD-{ti}')
        print(f"\n{'='*55}", flush=True)
        print(f"[{ti+1}/{len(thantd_configs)}] {label}", flush=True)
        print(f"  m={tcfg.get('thantd_m',7)}  d={tcfg.get('thantd_d',64)}  "
              f"heads={tcfg.get('thantd_heads',4)}  "
              f"epochs={tcfg.get('thantd_epochs',300)}  "
              f"lr={tcfg.get('thantd_lr',1e-4)}", flush=True)

        rng_t = np.random.default_rng(seed)
        bkg_pool = tr_raw if cfg.get('thantd_use_secondary', False) else None
        a_smp, p_smp, n_smp = build_thantd_samples(
            tr_raw, sig_raw,
            alpha=cfg.get('thantd_alpha', 0.5),
            n_samples=cfg.get('thantd_n_pairs', 1024),
            rng=rng_t, bkg_pool=bkg_pool)

        torch.manual_seed(seed)
        mdl = THANTD(b=D_raw,
                     m=tcfg.get('thantd_m', 7),
                     d=tcfg.get('thantd_d', 64),
                     n_heads=tcfg.get('thantd_heads', 4))
        train_thantd(mdl, a_smp, p_smp, n_smp,
                     epochs=tcfg.get('thantd_epochs', 300),
                     batch_size=cfg.get('thantd_batch', 64),
                     lr=tcfg.get('thantd_lr', 1e-4),
                     margin=cfg.get('thantd_margin', 0.3),
                     lam=cfg.get('thantd_lambda', 0.5),
                     device='cpu')
        torch.save({'state_dict': mdl.state_dict(), 'cfg': tcfg},
                   os.path.join(mdl_dir, f'thantd_{label}.pt'))

        row = {}
        for tm in ('additive', 'replacement'):
            planted_raw, labels, _ = plant_targets(
                te_raw, sig_raw, cfg['amplitude'], cfg['target_fraction'],
                model=tm, seed=seed)
            sc = score_thantd(mdl, sig_raw, planted_raw)
            row[tm] = auc_safe(labels, sc)
        thantd_results.append({'label': label, **row})
        print(f"  add={row['additive']:.3f}  rep={row['replacement']:.3f}", flush=True)

    # ================================================================ #
    # 3. Summary table + figure                                         #
    # ================================================================ #
    all_results = {}
    for tm in ('additive', 'replacement'):
        all_results[tm] = dict(baseline_auc[tm])
        for r in thantd_results:
            all_results[tm][r['label']] = r[tm]

    print(f"\n{'='*55}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*55}", flush=True)
    for tm in ('additive', 'replacement'):
        print(f"\n[{tm}]")
        for k, v in sorted(all_results[tm].items(), key=lambda x: -x[1]):
            print(f"  {k:<25s}  {v:.3f}")

    json.dump({'baseline': baseline_auc,
               'thantd': thantd_results,
               'all': all_results,
               'dom_cls': dom_cls, 'dom_name': dom_name},
              open(os.path.join(run_dir, 'metrics.json'), 'w'), indent=2)

    # ---- Figure: grouped bar chart ----
    n_thantd = len(thantd_configs)
    thantd_colors = [THANTD_CMAP(0.4 + 0.5*i/max(n_thantd-1,1)) for i in range(n_thantd)]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, 3+len(all_results['additive'])*0.9), 5))
    for ax, tm in zip(axes, ('additive', 'replacement')):
        names = list(all_results[tm].keys())
        vals  = list(all_results[tm].values())
        bar_colors = []
        for n in names:
            if n in COLORS:
                bar_colors.append(COLORS[n])
            else:
                # find THANTD index
                idx = next((i for i,r in enumerate(thantd_results) if r['label']==n), 0)
                bar_colors.append(thantd_colors[idx])
        bars = ax.bar(names, vals, color=bar_colors)
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel('AUC')
        ax.set_title(f'{tm.capitalize()}')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, min(v+0.01, 1.00),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    fig.suptitle(f'Spatial ablation — target: {w_dom}·{dom_name} + {w_mean}·test_mean',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'ablation_bar.pdf'))
    plt.close(fig)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.  Results: {run_dir}", flush=True)


if __name__ == '__main__':
    main()
