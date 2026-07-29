"""tsp_repro.lrao_camera_ready — the definitive fixed-LRao run (camera-ready).

Trains LRao on all three spatial scenes (pavia4, sandiego, sandiego2), 5 seeds
each, and scores the full amplitude sweep. Everything is archived: a model
checkpoint every 10 epochs, best/final weights, loss histories, raw train/test
scores per (scene, seed, theta), and per-seed metrics.

Training = the LRao paper's PRESCRIBED usage: a validation split with early
stopping on the VALIDATION LFI cost (the camera-ready decision after the SD2
diagnosis, where the training cost decreases monotonically while detection
oscillates and collapses — no train-cost-based rule can select a good model):

  - 20% of the train-box pixels held out by a seeded permutation (the SDSM
    convention: val = first nv indices of default_rng(seed).permutation);
  - the model, its robust whitening, its scoring reference set and its CFAR
    thresholds use ONLY the remaining 80% (the sample-budget tradeoff to note
    in the paper);
  - stop after RECIPE['patience'] epochs without a new validation-cost
    minimum; keep the best-validation model.

Everything else follows the recovered July-7 recipe (lr 5e-4, wd 5e-5, batch
2048, sigma_cutoff 1e-22 in the loss, grad_clip 1.0) with hidden_dims=[128] —
the SAME architecture as DART (one hidden layer). As in the archive, the
robust normalization (mu=median, W=diag(1/(1.4826*MAD))) is a frozen whitening
layer INSIDE the ScoreNet (exactly how DART embeds its ZCA front-end), so the
LFI Jacobian runs through it and checkpoints are self-contained. Scoring uses
the archive's statistic: sigma_cutoff=1e-3, delta_theta=0.01, raw-space
signature.

Planting is bit-identical to tsp_repro.spatial_camera_ready.run_scene (same
plant_targets call, same seed, same edge_guard/fraction), so the labels in the
saved npz files match the archived sweep npz of the other detectors and the
rows can be merged into the existing tables.

Output layout (out_root, default results/lrao_camera_ready):
    ckpt/<scene>_seed<k>/epoch_0010.pt ... best.pt final.pt history.json
    scores/lrao_train__<scene>__seed<k>.npz          (clean train-box scores)
    scores/lrao__<scene>__seed<k>__additive__<th>.npz (labels + LRao)
    metrics__<scene>.json
"""

import copy
import json
import os

import numpy as np
import torch

import tsp_repro  # noqa: F401  (path shim)
from sklearn.metrics import roc_auc_score
from src.data import Whitening, plant_targets
from src.metrics import cfar_threshold, dr_at_fpr, partial_auc, per_class_fpr
from src.models import (ScoreNet, compute_lfi_detector_scores_mode2,
                        lfi_loss_mode2)
from tsp_repro import spatial_camera_ready as SC
from tsp_repro.iid_camera_ready import THETAS, zip_and_download  # noqa: F401
from tsp_repro.protocol import _load_pavia

SEEDS = [42, 43, 44, 45, 46]
SCENES = ['pavia4', 'sandiego', 'sandiego2']
ALPHA = 0.05

RECIPE = dict(
    hidden_dims=[128],            # DART's architecture (archive had [64, 64])
    activation='relu',
    input_norm='robust',          # median / (1.4826*MAD), frozen in-model
    delta_theta=0.01,
    train_sigma_cutoff=1e-22,     # effectively full pseudo-inverse in the loss
    score_sigma_cutoff=1e-3,      # the archive's scoring cutoff
    detach_sigma=True,
    lr=5e-4, weight_decay=5e-5, batch_size=2048,
    max_epochs=1000, grad_clip=1.0,
    val_fraction=0.2,             # held-out split for stopping (paper usage)
    patience=20,                  # epochs without a new val-cost minimum
    ckpt_every=10,
)


def robust_whitening(tr):
    """Whitening.robust from the archive: mu=median, W=diag(1/(1.4826*MAD))."""
    X = np.asarray(tr, np.float64)
    mu = np.median(X, axis=0)
    mad = np.median(np.abs(X - mu), axis=0) * 1.4826
    scale = np.sqrt(np.maximum(mad ** 2, 1e-22))
    return Whitening(mu.astype(np.float32),
                     np.diag(1.0 / scale).astype(np.float32))


def build_lrao(D, tr, device):
    net = ScoreNet(D, RECIPE['hidden_dims'], RECIPE['activation'],
                   whitening=robust_whitening(tr))
    return net.to(device)


def _val_split(n, seed):
    """SDSM convention: val = first nv indices of the seeded permutation."""
    idx = np.random.default_rng(seed).permutation(n)
    nv = max(1, int(n * RECIPE['val_fraction']))
    return idx[nv:], idx[:nv]                    # fit_idx, val_idx


def fit_lrao_cr(tr, seed, run_dir, device):
    """Train one LRao with validation early stopping (or resume from
    run_dir/best.pt). The returned net carries net._fit_idx — the 80% train
    subset that the model (whitening, scoring reference, thresholds) uses.
    Saves a checkpoint every RECIPE['ckpt_every'] epochs plus
    best/final/history (with both loss curves)."""
    os.makedirs(run_dir, exist_ok=True)
    tr = np.asarray(tr, np.float32)
    fit_idx, val_idx = _val_split(len(tr), seed)
    net = build_lrao(tr.shape[1], tr[fit_idx], device)
    best_p = os.path.join(run_dir, 'best.pt')
    if os.path.exists(best_p):
        blob = torch.load(best_p, map_location='cpu', weights_only=False)
        net.load_state_dict(blob['model'])
        net.to(device).eval()
        net._fit_idx = np.asarray(blob['fit_idx'])
        print(f'  resumed {best_p} (epoch {blob["epoch"]}, '
              f'val loss {blob["val_loss"]:.4f})', flush=True)
        return net

    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=RECIPE['lr'],
                           weight_decay=RECIPE['weight_decay'])
    Xf = torch.tensor(tr[fit_idx], device=device)
    Xv = torch.tensor(tr[val_idx], device=device)
    P, B = len(Xf), RECIPE['batch_size']
    tr_losses, val_losses = [], []
    best_val, best_state, best_epoch, bad = float('inf'), None, 0, 0
    stopped = RECIPE['max_epochs']
    for ep in range(1, RECIPE['max_epochs'] + 1):
        perm = torch.randperm(P, device=device)
        run, nb = 0.0, 0
        for i in range(0, P, B):
            batch = Xf[perm[i:i + B]]
            loss = lfi_loss_mode2(net, batch, RECIPE['delta_theta'],
                                  RECIPE['train_sigma_cutoff'],
                                  RECIPE['detach_sigma'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),
                                           RECIPE['grad_clip'])
            opt.step()
            run += float(loss.detach()); nb += 1
        tl = run / max(nb, 1)
        vl = float(lfi_loss_mode2(net, Xv, RECIPE['delta_theta'],
                                  RECIPE['train_sigma_cutoff'],
                                  RECIPE['detach_sigma']).detach())
        tr_losses.append(tl); val_losses.append(vl)
        if vl < best_val:
            best_val, best_epoch, bad = vl, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % RECIPE['ckpt_every'] == 0:
            torch.save({'model': {k: v.detach().cpu()
                                  for k, v in net.state_dict().items()},
                        'epoch': ep, 'train_loss': tl, 'val_loss': vl},
                       os.path.join(run_dir, f'epoch_{ep:04d}.pt'))
        if ep == 1 or ep % 50 == 0:
            print(f'    epoch {ep}/{RECIPE["max_epochs"]} train={tl:.4f} '
                  f'val={vl:.4f} best_val={best_val:.4f}@{best_epoch}',
                  flush=True)
        if bad >= RECIPE['patience']:
            stopped = ep
            print(f'    early stop at epoch {ep} (no new val minimum for '
                  f'{RECIPE["patience"]} epochs)', flush=True)
            break

    torch.save({'model': {k: v.detach().cpu()
                          for k, v in net.state_dict().items()},
                'epoch': stopped, 'train_loss': tr_losses[-1],
                'val_loss': val_losses[-1], 'fit_idx': fit_idx,
                'val_idx': val_idx}, os.path.join(run_dir, 'final.pt'))
    torch.save({'model': best_state, 'epoch': best_epoch,
                'train_loss': tr_losses[best_epoch - 1], 'val_loss': best_val,
                'fit_idx': fit_idx, 'val_idx': val_idx}, best_p)
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(dict(recipe={k: v for k, v in RECIPE.items()}, seed=seed,
                       n_fit=int(len(fit_idx)), n_val=int(len(val_idx)),
                       train_loss=tr_losses, val_loss=val_losses,
                       best_epoch=best_epoch, best_val_loss=best_val,
                       stopped_epoch=stopped), f)
    net.load_state_dict(best_state)
    net.eval()
    net._fit_idx = fit_idx
    print(f'  trained: best val epoch {best_epoch} (val {best_val:.4f}, '
          f'stopped {stopped}, fit {len(fit_idx)} px / val {len(val_idx)} px)',
          flush=True)
    return net


def score_lrao_cr(net, tr, X, sig):
    """The archive's detection statistic: raw inputs (whitening is inside the
    net), raw signature, sigma_cutoff=1e-3."""
    return compute_lfi_detector_scores_mode2(
        net, np.asarray(tr, np.float32), np.asarray(X, np.float32),
        np.asarray(sig, np.float32), RECIPE['delta_theta'],
        RECIPE['score_sigma_cutoff'])


def _pavia_test_gt():
    _, gt = _load_pavia()
    return np.asarray(gt, int)[419:508, 250:334].ravel()


def _metrics_row(labels, sc, thr, te_gt=None):
    y, sc = np.asarray(labels), np.asarray(sc, float)
    r = dict(auc=float(roc_auc_score(y, sc)),
             pauc=float(partial_auc(y, sc, fpr_max=ALPHA)),
             pd05=float(dr_at_fpr(y, sc, fpr_list=(ALPHA,))[str(ALPHA)]),
             pd_cfar=float((sc[y == 1] > thr).mean()),
             pfa=float((sc[y == 0] > thr).mean()))
    if te_gt is not None:
        pcf = per_class_fpr(sc, y, te_gt, thr)
        vals = list(pcf.values())
        r.update(pfa_avg=float(np.nanmean(vals)),
                 pfa_max=float(np.nanmax(vals)),
                 unlab=float(pcf.get('unlabeled', 0.0)),
                 asph=float(pcf.get('asphalt', 0.0)),
                 trees=float(pcf.get('trees', 0.0)))
    return r


def run_scene(scene_name, seeds=SEEDS, thetas=None,
              out_root='results/lrao_camera_ready', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    thetas = list(thetas or THETAS)
    cfg = SC.spatial_cfg(device)
    sc_dir = os.path.join(out_root, 'scores')
    os.makedirs(sc_dir, exist_ok=True)
    scene = SC.build_scene(scene_name)
    te_gt = _pavia_test_gt() if scene_name == 'pavia4' else None
    frac = float(cfg.get('target_fraction', 0.10))
    guard = int(cfg.get('edge_guard', 3))

    metrics = {}
    for seed in seeds:
        print(f'[{scene_name}] seed {seed}', flush=True)
        run_dir = os.path.join(out_root, 'ckpt', f'{scene_name}_seed{seed}')
        net = fit_lrao_cr(scene['tr'], seed, run_dir, device)
        tr_fit = scene['tr'][net._fit_idx]     # the model's own 80% subset

        tr_sc = score_lrao_cr(net, tr_fit, tr_fit, scene['sig'])
        np.savez_compressed(
            os.path.join(sc_dir, f'lrao_train__{scene_name}__seed{seed}.npz'),
            train_scores=np.asarray(tr_sc, np.float32),
            fit_idx=np.asarray(net._fit_idx, np.int32))
        thr = cfar_threshold(np.asarray(tr_sc, float), target_fpr=ALPHA)

        for th in thetas:
            planted, labels, _ = plant_targets(
                scene['te'], scene['sig'], float(th), frac, model='additive',
                seed=int(seed), spatial_shape=scene['te_shape'],
                edge_guard=guard)
            sc = score_lrao_cr(net, tr_fit, planted.astype(np.float32),
                               scene['sig'])
            np.savez_compressed(
                os.path.join(
                    sc_dir,
                    f'lrao__{scene_name}__seed{seed}__additive__{th}.npz'),
                labels=np.asarray(labels, np.int8),
                LRao=np.asarray(sc, np.float32))
            row = _metrics_row(labels, sc, thr, te_gt)
            metrics.setdefault(str(th), {})[str(seed)] = row
            print(f'[{scene_name}] seed{seed} th={th}: AUC={row["auc"]:.4f} '
                  f'Pd={row["pd05"]:.3f}', flush=True)
            with open(os.path.join(out_root,
                                   f'metrics__{scene_name}.json'), 'w') as f:
                json.dump(dict(scene=scene_name, thetas=thetas,
                               alpha=ALPHA, recipe=RECIPE, rows=metrics),
                          f, indent=1)
    return metrics


def summarize(out_root='results/lrao_camera_ready', scenes=SCENES,
              table_theta=0.15):
    """Prints the AUC-vs-theta grid per scene and the Table row at
    theta=table_theta (mean over seeds)."""
    for scn in scenes:
        p = os.path.join(out_root, f'metrics__{scn}.json')
        if not os.path.exists(p):
            print(f'[{scn}] no metrics yet'); continue
        m = json.load(open(p))
        rows = m['rows']
        ths = [str(t) for t in m['thetas'] if str(t) in rows]
        print(f'\n=== {scn} — LRao AUC (mean±std over seeds) ===')
        print('| ' + ' | '.join(f'θ={t}' for t in ths) + ' |')
        print('|' + '---|' * len(ths))
        cells = []
        for t in ths:
            v = [r['auc'] for r in rows[t].values()]
            cells.append(f'{np.mean(v):.3f}±{np.std(v):.3f}')
        print('| ' + ' | '.join(cells) + ' |')
        t = str(table_theta)
        if t in rows:
            keys = list(next(iter(rows[t].values())).keys())
            agg = {k: float(np.mean([r[k] for r in rows[t].values()]))
                   for k in keys}
            print(f'Table row (θ={table_theta}): '
                  + '  '.join(f'{k}={v:.3f}' for k, v in agg.items()))


def make_zip(out_root='results/lrao_camera_ready',
             zip_name='lrao_camera_ready.zip'):
    zip_and_download([out_root], zip_name)
