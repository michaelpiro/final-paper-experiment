"""tsp_repro.iid_camera_ready — camera-ready IID experiments.

Two pieces on top of the ORIGINAL src/iid.py pipeline (vendored verbatim on
this branch):

1. apply_robust_lrao(): the FIXED LRao. Replaces the ZCA whitening front layer
   of the LRao score nets with a robust normalization (median / IQR per band),
   which removed the training instability observed with the original recipe.
   Implemented by wrapping src.iid.train_lrao_local and swapping the module's
   _make_whitening only for the duration of LRao training — DART / L-DART keep
   their ZCA whitening untouched. The robust layer reuses the existing
   Whitening module (mu = median, W = diag(1/IQR)), so checkpoints load
   through the normal path.

2. theta_sweep(): amplitude sweep at the largest training size. Reuses the
   models trained by run_iid (per-seed checkpoints at tag n<max(n_list)>), rebuilds
   the identical train/test split from the seed, replants the clean test
   background at each theta with the ORIGINAL plant_targets, and rescores all
   detectors (classical detectors recomputed fresh; AMF is the pure
   unregularized version, as in the IID path). Saves raw scores per
   (seed, theta), an aggregate json, and AUC / Pd@Pfa vs-theta figures.

THETAS (user-specified, ascending):
    0.03, 0.075, 0.15, 0.225, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95
"""

import glob
import json
import os
import zipfile

import numpy as np
import torch

import tsp_repro  # noqa: F401  (path shim -> vendored src)
import src.iid as iid
from src.data import Whitening, load_hsi, placeholder_whitening, plant_targets
from src.iid import (build_pools, run_classical_additive, score_dsm_add,
                     score_lrao)
from src.models import ScoreNet

THETAS = [0.03, 0.075, 0.15, 0.225, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

DEEP_BASELINES = ('THANTD', 'HTDNet', 'TSTTD', 'OSVAE')
# paper-recipe budgets; smoke tests may shrink these
DEEP_BUDGET = dict(thantd_epochs=300, uae_epochs=150, sdcnn_epochs=30,
                   sdcnn_pairs=100_000, tsttd_epochs=20, osvae_epochs=200)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Fixed LRao (robust normalization front layer)
# ---------------------------------------------------------------------------
def _robust_whitening(train_raw, cfg):
    """Whitening module with mu = per-band median, W = diag(1/IQR)."""
    X = np.asarray(train_raw, dtype=np.float64)
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr = np.where(iqr > 1e-8, iqr, 1.0)
    return Whitening(med.astype(np.float32),
                     np.diag(1.0 / iqr).astype(np.float32))


_ORIG_TRAIN_LRAO = iid.train_lrao_local
_ORIG_MAKE_WHITENING = iid._make_whitening


def _train_lrao_robust(train_raw, cfg, seed, label):
    # L-LRao must be LINEAR. run_iid trains it with cfg['hidden_dims'], which
    # in the MULTI config is [128] (the DSM-MLP arch) — making "L-LRao" a
    # bit-identical duplicate of the MLP LRao (same arch, same seed). The MLP
    # LRao labels always carry an 'mlp' prefix, so force hidden_dims=[] for
    # every non-mlp label (a no-op in single mode, where it is already []).
    if not str(label).startswith('mlp'):
        cfg = {**cfg, 'hidden_dims': []}
    iid._make_whitening = _robust_whitening
    try:
        return _ORIG_TRAIN_LRAO(train_raw, cfg, seed, label)
    finally:
        iid._make_whitening = _ORIG_MAKE_WHITENING


def apply_robust_lrao():
    """Activate the fixed LRao for every subsequent run_iid / theta_sweep."""
    iid.train_lrao_local = _train_lrao_robust
    print("[iid_camera_ready] FIXED LRao active: robust (median/IQR) input "
          "normalization for LRao nets; L-LRao forced LINEAR (multi config "
          "would otherwise duplicate the MLP LRao); DART keeps ZCA whitening.",
          flush=True)


# ---------------------------------------------------------------------------
# 2. Amplitude sweep at n = max(n_list), reusing the run_iid checkpoints
# ---------------------------------------------------------------------------
def _rebuild_split(cfg, mode, seed):
    """EXACT replica of run_iid's pool construction and split (lines 594-620)."""
    rng = np.random.default_rng(seed)
    data, gt = load_hsi(cfg['dataset'])
    gt_flat = gt.flatten()
    bkg_raw, tgt_raw = build_pools(data, gt_flat, cfg, mode)
    s_raw = tgt_raw.mean(axis=0).astype(np.float32)
    if cfg.get('normalize_signature', False):
        s_raw = (s_raw / (np.linalg.norm(s_raw) + 1e-12)).astype(np.float32)
    idx = np.arange(len(bkg_raw)); rng.shuffle(idx)
    bkg_shuf = bkg_raw[idx]
    n_list = sorted(set(int(n) for n in cfg['n_train_list']))
    n_fixed = int(cfg.get('n_fixed_for_rho', max(n_list)))
    max_n = max(max(n_list), n_fixed)
    test_size = int(cfg['test_size'])
    train_pool = bkg_shuf[:max_n].astype(np.float32)
    test_bkg = bkg_shuf[-test_size:].astype(np.float32)
    n_sweep = max(n_list)      # the vs-n checkpoints exist at tag n{n_sweep}
    return train_pool, test_bkg, s_raw, n_sweep


def _arch_table(cfg):
    """Replica of run_iid's architecture-selection logic (lines 636-656):
    {model file stem -> (detector label, hidden_dims, activation)}."""
    out = {}
    h1 = list(cfg.get('hidden_dims', []) or [])
    h2 = list(cfg.get('hidden_dims_2', []) or []) \
        if cfg.get('hidden_dims_2') is not None else None
    d1 = cfg.get('dsm_label', 'DART')
    out[d1] = (d1, h1, cfg['activation'])
    if h2 is not None:
        d2 = cfg.get('dsm2_label', 'DART' if h2 else 'L-DART')
        out[d2] = (d2, h2, cfg.get('activation_2', cfg['activation']))
    # LRao arches
    out['lrao'] = ('L-LRao', h1, cfg['activation'])
    if cfg.get('lrao_mlp_hidden') is not None:
        mlp_h, mlp_a = list(cfg['lrao_mlp_hidden']), \
            cfg.get('lrao_mlp_activation', cfg['activation'])
    elif h2:
        mlp_h, mlp_a = h2, cfg.get('activation_2', cfg['activation'])
    elif h1:
        mlp_h, mlp_a = h1, cfg['activation']
    else:
        mlp_h, mlp_a = [], cfg['activation']
    if bool(cfg.get('run_lrao_mlp', False)) and len(mlp_h) > 0:
        out['lrao_mlp'] = ('LRao', mlp_h, mlp_a)
    return out


def _load_net(path, D, hidden, activation, device):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    sd = blob['state_dict']
    # infer the architecture from the checkpoint itself (robust to the
    # L-LRao linear fix and to config/label mismatches)
    w = [v for k, v in sd.items()
         if k.endswith('.weight') and v.ndim == 2 and 'whiten' not in k]
    hidden_ck = [int(x.shape[0]) for x in w[:-1]]
    net = ScoreNet(D, hidden_ck, activation,
                   whitening=placeholder_whitening(D))
    net.load_state_dict(sd)
    net.to(torch.device(device)).eval()
    return net


def _seed_model_dir(agg_dir, mode, seed):
    hits = sorted(glob.glob(os.path.join(agg_dir, f'seed_{seed}',
                                         f'iid_{mode}_*', 'models')))
    if not hits:
        raise FileNotFoundError(
            f'no models dir for seed {seed} under {agg_dir}')
    return hits[-1]


def theta_sweep(agg_dir, cfg, mode, thetas=None, out_dir=None, device=None,
                plant_model='additive', deep=()):
    """Score every detector on planted test sets at each theta, reusing the
    n=max(n_list) checkpoints of the multi-seed run at `agg_dir`.

    deep : iterable of names from DEEP_BASELINES. Each is trained per seed on
           the SAME n=max(n_list) pixels + signature (its own paper recipe,
           checkpoint-resumable under <out_dir>/ckpt_deep) and scored on the
           same planted sets."""
    thetas = list(thetas or THETAS)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = out_dir or os.path.join(agg_dir, 'theta_sweep')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    deep = [d for d in deep if d in _DEEP]
    if deep:
        os.makedirs(os.path.join(out_dir, 'ckpt_deep'), exist_ok=True)
    seeds = list(cfg['seed']) if isinstance(cfg['seed'], (list, tuple)) \
        else [cfg['seed']]
    cfg = {**cfg, 'device': device}
    pfa = float(cfg.get('pfa', 0.1))
    pauc_fpr = float(cfg.get('pauc_fpr', 0.1))

    arch = _arch_table(cfg)
    results = {}          # det -> theta-key -> [auc per seed]
    results_pd = {}
    for seed in seeds:
        train_pool, test_bkg, s_raw, n_sweep = _rebuild_split(cfg, mode, int(seed))
        D = train_pool.shape[1]
        tr = train_pool[:n_sweep]
        mdl_dir = _seed_model_dir(agg_dir, mode, seed)
        tag = f'n{n_sweep}'
        nets = {}
        for stem, (det, hidden, act) in arch.items():
            p = os.path.join(mdl_dir, f'{stem}_{tag}.pt')
            if os.path.exists(p):
                nets[det] = (_load_net(p, D, hidden, act, device),
                             'lrao' in stem)
            else:
                print(f'  [warn] missing checkpoint {p} - {det} skipped',
                      flush=True)

        from src.data import compute_sigma_from_data
        rho_fixed = float(cfg['dsm_sigma_rho']) \
            if not isinstance(cfg['dsm_sigma_rho'], str) else 0.1
        reg_sigma = compute_sigma_from_data(tr, rho_fixed)

        # ---- deep baselines: fit once per seed on the SAME tr + signature ----
        dstag = os.path.splitext(os.path.basename(cfg['dataset']))[0]
        deep_states = {}
        for name in deep:
            fit_fn, _ = _DEEP[name]
            ck = os.path.join(out_dir, 'ckpt_deep',
                              f'{name}__{dstag}__{mode}__seed{seed}.pt')
            print(f'[{mode}] seed{seed} fitting {name} on {len(tr)} px ...',
                  flush=True)
            deep_states[name] = fit_fn(tr.astype(np.float64), s_raw, int(seed),
                                       ck, device)

        for th in thetas:
            planted, labels, _ = plant_targets(
                test_bkg, s_raw, float(th), cfg['target_fraction'],
                model=plant_model, seed=int(seed))
            planted = planted.astype(np.float32)
            det_scores = dict(run_classical_additive(
                tr, planted, s_raw, reg_sigma, cfg, mode))
            for det, (net, is_lrao) in nets.items():
                if is_lrao:
                    det_scores[det] = score_lrao(net, tr, planted, s_raw, cfg)
                else:
                    det_scores[det] = score_dsm_add(net, tr, planted, s_raw)
            for name, state in deep_states.items():
                _, score_fn = _DEEP[name]
                det_scores[name] = score_fn(state,
                                            planted.astype(np.float64),
                                            s_raw, device)
            key = f'{plant_model}|{th}'
            np.savez_compressed(
                os.path.join(out_dir,
                             f'scores__{mode}__seed{seed}__{plant_model}__{th}.npz'),
                labels=labels.astype(np.int8),
                **{d: np.asarray(s, np.float32) for d, s in det_scores.items()})
            for det, sc in det_scores.items():
                au = iid._auc(labels, sc)
                pd_ = iid._pd_at_fa(labels, sc, pfa)
                results.setdefault(det, {}).setdefault(key, []).append(au)
                results_pd.setdefault(det, {}).setdefault(key, []).append(pd_)
                print(f'[{mode}] seed{seed} {det:8s} th={th}: '
                      f'AUC={au:.4f} Pd@{pfa}={pd_:.3f}', flush=True)

        with open(os.path.join(out_dir, 'theta_results.json'), 'w') as f:
            json.dump(dict(mode=mode, thetas=thetas, n_sweep=n_sweep, pfa=pfa,
                           pauc_fpr=pauc_fpr, plant_model=plant_model,
                           auc=results, pd=results_pd), f, indent=1)

    # ---- aggregate figures (reuse the original plot helper) ----
    keys = [f'{plant_model}|{th}' for th in thetas]
    auc_mu = {d: [float(np.mean(results[d][k])) for k in keys] for d in results}
    auc_sd = {d: [float(np.std(results[d][k])) for k in keys] for d in results}
    pd_mu = {d: [float(np.mean(results_pd[d][k])) for k in keys]
             for d in results_pd}
    pd_sd = {d: [float(np.std(results_pd[d][k])) for k in keys]
             for d in results_pd}
    iid._plot_vs(thetas, auc_mu, r'target amplitude  $\theta$', 'AUC',
                 f'AUC vs amplitude  ({mode}, n={n_sweep}, {plant_model})',
                 os.path.join(fig_dir, 'auc_vs_theta.pdf'), logx=True,
                 series_std=auc_sd)
    iid._plot_vs(thetas, pd_mu, r'target amplitude  $\theta$',
                 f'Pd @ Pfa={pfa}',
                 f'Pd @ Pfa={pfa} vs amplitude  ({mode}, n={n_sweep})',
                 os.path.join(fig_dir, 'pd_at_fa_vs_theta.pdf'), logx=True,
                 series_std=pd_sd)
    print(f'\ntheta sweep done -> {out_dir}', flush=True)
    return results


# ---------------------------------------------------------------------------
# 2b. Deep baselines in the amplitude sweep (paper recipes, same 2000 pixels)
# ---------------------------------------------------------------------------
def _fit_thantd(tr, sig, seed, ck, device):
    from thantd_model import THANTD, build_thantd_samples, train_thantd
    m = THANTD(b=tr.shape[1])
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False))
        m.to(device).eval(); print('  resumed', ck, flush=True); return m
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    a, p, n = build_thantd_samples(tr, sig, alpha=0.5, n_samples=1024,
                                   rng=rng, bkg_pool=tr)
    m.to(device)
    train_thantd(m, a, p, n, epochs=DEEP_BUDGET['thantd_epochs'],
                 batch_size=64, lr=1e-4, margin=0.3, device=device)
    torch.save(m.state_dict(), ck)
    return m


def _score_thantd(m, planted, sig, device):
    from thantd_model import score_thantd
    return score_thantd(m, sig, planted, device=device)


def _fit_htdnet(tr, sig, seed, ck, device):
    from colab_deep import htdnet_model as H
    if os.path.exists(ck):
        blob = torch.load(ck, map_location='cpu', weights_only=False)
        sd = H.SDCNN(); sd.load_state_dict(blob['sdcnn'])
        sd._scale = blob['scale']; sd.to(device).eval()
        print('  resumed', ck, flush=True)
        return dict(sd=sd, gen_t=blob['gen_t'], bkg=blob['bkg'])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    uae, sc = H.train_uae(tr, epochs=DEEP_BUDGET['uae_epochs'],
                          device=device, seed=seed)
    gen_t = H.generate_targets(uae, sc, sig, tr, n_samples=1000,
                               device=device, rng=rng)
    bkg = H.lp_background_selection(tr, sig, n_direct=60, n_total=500)
    sd = H.train_sdcnn(gen_t, bkg, H.ACELabeler(tr),
                       epochs=DEEP_BUDGET['sdcnn_epochs'],
                       pairs_per_epoch=DEEP_BUDGET['sdcnn_pairs'],
                       device=device, seed=seed, log_every=10)
    torch.save(dict(sdcnn=sd.state_dict(), scale=sd._scale,
                    gen_t=gen_t, bkg=bkg), ck)
    return dict(sd=sd, gen_t=gen_t, bkg=bkg)


def _score_htdnet(state, planted, sig, device):
    from colab_deep import htdnet_model as H
    return H.htdnet_detect(state['sd'], planted, state['gen_t'],
                           state['bkg'], device=device)


def _tsttd_modules():
    vend = os.path.join(_REPO, 'colab_deep', 'vendor_tsttd')
    if vend not in os.sys.path:
        os.sys.path.insert(0, vend)
    cwd = os.getcwd(); os.chdir(vend)
    try:
        import Train_eval as TE
        from Model import SpectralGroupAttention
    finally:
        os.chdir(cwd)
    # vendor hardcodes num_workers=4; worker processes can't pickle our
    # closure-defined Dataset (and hang on macOS spawn). 0 workers is also
    # faster for these small in-memory datasets.
    if not getattr(TE, '_dl_patched', False):
        _DL = TE.DataLoader
        TE.DataLoader = lambda *a, **k: _DL(*a, **{**k, 'num_workers': 0})
        TE._dl_patched = True
    return TE, SpectralGroupAttention, vend


def _fit_tsttd(tr, sig, seed, ck, device):
    TE, SGA, vend = _tsttd_modules()
    BAND = tr.shape[1]
    dev = 'cuda:0' if str(device).startswith('cuda') else 'cpu'
    model = SGA(band=BAND, m=20, d=128, depth=4, heads=4, dim_head=64,
                mlp_dim=64, adjust=False).to(dev)
    mn, mx = float(tr.min()), float(tr.max())      # frozen normalization
    model._norm = (mn, mx); model._band = BAND; model._dev = dev
    if os.path.exists(ck):
        blob = torch.load(ck, map_location='cpu', weights_only=False)
        model.load_state_dict(blob['model']); model.to(dev).eval()
        model._norm = tuple(blob['norm'])
        print('  resumed', ck, flush=True); return model
    S = lambda x: ((np.asarray(x) - mn) / (mx - mn + 1e-12)).astype(np.float32)

    class OurData(torch.utils.data.Dataset):       # vendor recipe; bkg = tr
        def __init__(self, path=None):
            bkg = S(tr); ts = S(sig)[None, :]
            al = np.random.uniform(0, 0.1, (len(bkg), 1)).astype(np.float32)
            self.target_samples = al * bkg + (1 - al) * ts
            self.background_samples = bkg
            self.target_spectrum, self.nums = ts, len(bkg)
        def __getitem__(self, i):
            return self.target_samples[i], self.background_samples[i]
        def __len__(self):
            return self.nums

    cfg = {"state": "train", "epoch": DEEP_BUDGET['tsttd_epochs'],
           "band": BAND, "multiplier": 2, "seed": int(seed),
           "batch_size": min(64, len(tr)),
           "group_length": 20, "depth": 4, "heads": 4, "dim_head": 64,
           "mlp_dim": 64, "adjust": False, "channel": 128, "lr": 1e-4,
           "epision": 5, "grad_clip": 1., "device": dev,
           "training_load_weight": None,
           "save_dir": f"./Ckpt_iid_s{seed}_{os.path.basename(ck)}/",
           "test_load_weight": None, "path": "ours"}
    np.random.seed(seed); TE.Data = OurData
    cwd = os.getcwd(); os.chdir(vend)
    try:
        os.makedirs(cfg['save_dir'] + '/ours/', exist_ok=True)
        TE.train(cfg)
        ckdir = cfg['save_dir'] + '/ours/'
        last = max(os.listdir(ckdir),
                   key=lambda s: int(''.join(filter(str.isdigit, s)) or -1))
        model.load_state_dict(torch.load(ckdir + last, map_location=dev, weights_only=False))
    finally:
        os.chdir(cwd)
    model.eval()
    torch.save(dict(model=model.state_dict(), norm=[mn, mx]), ck)
    return model


def _score_tsttd(model, planted, sig, device):
    TE, _, _ = _tsttd_modules()
    mn, mx = model._norm
    S = lambda x: ((np.asarray(x) - mn) / (mx - mn + 1e-12)).astype(np.float32)
    dev = model._dev
    tf = model(TE.spectral_group(S(sig)[None, :], model._band, 20).to(dev)).detach()
    out = []
    X = S(planted)
    for i in range(0, len(X), 512):
        f = model(TE.spectral_group(X[i:i + 512], model._band, 20).to(dev)).detach()
        out.append(torch.nn.functional.cosine_similarity(
            f, tf.expand_as(f), -1).cpu().numpy())
    return np.concatenate(out)


def _fit_osvae(tr, sig, seed, ck, device):
    from colab_deep.osvae_model import CVAE, fit_osvae
    if os.path.exists(ck):
        m = CVAE(tr.shape[1])
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False))
        m.to(device).eval(); m._device = str(device)
        print('  resumed', ck, flush=True); return m
    m = fit_osvae(tr, sig, seed=seed, epochs=DEEP_BUDGET['osvae_epochs'],
                  device=str(device))
    torch.save(m.state_dict(), ck)
    return m


def _score_osvae(m, planted, sig, device):
    from colab_deep.osvae_model import score_osvae
    return score_osvae(m, planted, sig)


_DEEP = {'THANTD': (_fit_thantd, _score_thantd),
         'HTDNet': (_fit_htdnet, _score_htdnet),
         'TSTTD': (_fit_tsttd, _score_tsttd),
         'OSVAE': (_fit_osvae, _score_osvae)}

iid.DETECTOR_COLORS.update({'THANTD': '#8c564b', 'HTDNet': '#e377c2',
                            'TSTTD': '#17becf', 'OSVAE': '#bcbd22'})


# ---------------------------------------------------------------------------
# 3. Packaging
# ---------------------------------------------------------------------------
def zip_and_download(paths, zip_name='iid_camera_ready.zip'):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for d in paths:
            if os.path.isfile(d):
                z.write(d)
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    z.write(os.path.join(root, fn))
    print('zipped ->', zip_name, flush=True)
    try:
        from google.colab import files
        files.download(zip_name)
    except Exception:
        print('(not on Colab - zip left on disk)')
