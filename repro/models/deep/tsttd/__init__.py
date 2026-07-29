"""TSTTD wrapper. The authors' code is kept VERBATIM in vendor/ (Train_eval,
Model, ...); this wrapper only supplies the dataset, the config dict, and two
runtime patches the vendor needs outside its repo:
  - DataLoader num_workers=0 (the vendor hardcodes 4; worker processes cannot
    pickle our closure-defined Dataset and hang on macOS spawn),
  - chdir into vendor/ around train() (it writes relative checkpoint paths).
The vendor's train() calls its own seed_torch(cfg['seed']) before building the
model, so the trained network satisfies the seeding contract; the shell model
built here only receives the trained weights via load_state_dict."""
import os
import sys

import numpy as np
import torch

_VEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')


def _modules():
    if _VEND not in sys.path:
        sys.path.insert(0, _VEND)
    cwd = os.getcwd()
    os.chdir(_VEND)
    try:
        import Train_eval as TE
        from Model import SpectralGroupAttention
    finally:
        os.chdir(cwd)
    if not getattr(TE, '_dl_patched', False):
        _DL = TE.DataLoader
        TE.DataLoader = lambda *a, **k: _DL(*a, **{**k, 'num_workers': 0})
        TE._dl_patched = True
    return TE, SpectralGroupAttention


def _fit(tr, sig, cfg, seed, ckpt, device):
    TE, SGA = _modules()
    BAND = tr.shape[1]
    dev = 'cuda:0' if str(device).startswith('cuda') else 'cpu'
    model = SGA(band=BAND, m=20, d=128, depth=4, heads=4, dim_head=64,
                mlp_dim=64, adjust=False).to(dev)
    mn, mx = float(tr.min()), float(tr.max())        # frozen normalization
    model._norm = (mn, mx); model._band = BAND; model._dev = dev
    if ckpt and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(blob['model']); model.to(dev).eval()
        model._norm = tuple(blob['norm'])
        print('    resumed', ckpt, flush=True)
        return model
    S = lambda x: ((np.asarray(x) - mn) / (mx - mn + 1e-12)).astype(np.float32)

    class OurData(torch.utils.data.Dataset):         # vendor recipe; bkg = tr
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

    tcfg = {"state": "train", "epoch": int(cfg['tsttd_epochs']), "band": BAND,
            "multiplier": 2, "seed": int(seed),
            "batch_size": min(64, len(tr)), "group_length": 20, "depth": 4,
            "heads": 4, "dim_head": 64, "mlp_dim": 64, "adjust": False,
            "channel": 128, "lr": 1e-4, "epision": 5, "grad_clip": 1.,
            "device": dev, "training_load_weight": None,
            "save_dir": f"./Ckpt_s{seed}_{os.path.basename(str(ckpt))}/",
            "test_load_weight": None, "path": "ours"}
    np.random.seed(seed)
    TE.Data = OurData
    cwd = os.getcwd()
    os.chdir(_VEND)
    try:
        os.makedirs(tcfg['save_dir'] + '/ours/', exist_ok=True)
        TE.train(tcfg)
        ckdir = tcfg['save_dir'] + '/ours/'
        last = max(os.listdir(ckdir),
                   key=lambda s: int(''.join(filter(str.isdigit, s)) or -1))
        model.load_state_dict(torch.load(ckdir + last, map_location=dev,
                                         weights_only=False))
    finally:
        os.chdir(cwd)
    model.eval()
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(dict(model=model.state_dict(), norm=[mn, mx]), ckpt)
    return model


def _score(model, planted, sig, device):
    TE, _ = _modules()
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


class TSTTD:
    """Uniform detector API: TSTTD(cfg).fit(tr, sig, seed, device, ckpt).score(...)."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.state = None

    def fit(self, tr, sig, seed, device, ckpt=None):
        self.state = _fit(tr, sig, self.cfg, seed, ckpt, device)
        return self

    def score(self, planted, sig, device):
        return _score(self.state, planted, sig, device)
