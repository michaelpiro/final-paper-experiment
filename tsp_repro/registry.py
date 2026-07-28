"""tsp_repro.registry — one entry per detector.

Adding a baseline = adding one ``DetectorSpec``:

    fit(scene, seed, ckpt_path, device) -> state          (None if not trainable)
    score(state, scene, planted, device) -> np.ndarray    (higher = more target)

``state`` is whatever fit returns (a model, a dict, or None). Trainable
detectors must save/resume from ``ckpt_path`` so Colab runs survive restarts.
All our trainable detectors use the canonical TSP budget (protocol.EPOCHS)
and the mse_optimal sigma rule.

Detector list (paper order):
  DART, DART-CFAR, DARTS, DARTS-CFAR      ours (CFAR variants derive from base)
  AMF, AMF-local, GMM-Levin, LRao         classical / learned baselines
  THANTD, HTDNet                          deep (paper-faithful ports, GPU)
  OSVAE, TSTTD                            deep (artifact-only for now: scored
                                          from the archived raw scores; their
                                          runnable ports are a Phase-B item)
"""

import copy
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

import tsp_repro  # noqa: F401  (path shim)
from tsp_repro import protocol as PR

from src.data import Whitening, extract_neighborhoods
from src.detectors import (amf, amf_local, dsm_additive,
                           gmm_glrt_levin_additive)
from src.models import (NeighborMLPDenoiser, ScoreNet,
                        compute_lfi_detector_scores_mode2, dsm_loss,
                        lfi_loss_mode2, neighbor_mlp_dsm_loss,
                        score_nmlp_additive)
from src.spatial import _cfar_normalize_map, _knn_fisher_normalize

# --------------------------------------------------------------------------
# Hyperparameters (paper Table-1 values, epochs raised to the TSP budget)
# --------------------------------------------------------------------------
HP = dict(
    dsm_hidden=[128], dsm_lr=5e-4, batch_size=256, weight_decay=1e-5,
    activation="relu", whiten_eig_floor=1e-5,
    nmlp_d_lat=16, nmlp_K=7, nmlp_enc_hidden=[64, 32], nmlp_score_hidden=[128],
    nmlp_lr=3e-4, nmlp_batch=512, k=7,
    gmm_K=9, gmm_steps=50,
    cfar_lam=0.1, dart_cfar_window=5, dart_cfar_guard=1, darts_cfar_guard=1,
    lrao_lr=3e-4, lrao_hidden=[128], lrao_delta_theta=0.01,
)


def _whiten(tr_raw, device):
    return Whitening.from_data(np.asarray(tr_raw, np.float32),
                               eig_floor=HP["whiten_eig_floor"]).to(device)


def _train_loop(net, params, data_tensors, loss_fn, epochs, lr, batch,
                weight_decay, desc, grad_clip=None):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    P = len(data_tensors[0])
    best_loss, best_state = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(P, device=data_tensors[0].device)
        ep_loss, nb = 0.0, 0
        for i in range(0, P, batch):
            sel = perm[i:i + batch]
            loss = loss_fn(*[t[sel] for t in data_tensors])
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params if isinstance(params, list)
                                               else net.parameters(), grad_clip)
            opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        if last < best_loss:
            best_loss, best_state = last, copy.deepcopy(net.state_dict())
        if ep == 0 or (ep + 1) % max(epochs // 10, 1) == 0:
            print(f"    [{desc}] epoch {ep+1}/{epochs} loss={last:.4f} "
                  f"best={best_loss:.4f}", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _resume(build_fn, ckpt, device):
    if ckpt and os.path.exists(ckpt):
        state = build_fn()
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state.load_state_dict(blob["model"])
        state.to(device).eval()
        state.sigma = blob.get("sigma", getattr(state, "sigma", None))
        print(f"    resumed {ckpt}", flush=True)
        return state
    return None


def _save(net, ckpt, sigma=None):
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save({"model": net.state_dict(), "sigma": sigma}, ckpt)


def _test_windows(scene, planted, k, device):
    H, W = scene["te_shape"]
    img = torch.tensor(planted.reshape(H, W, -1), dtype=torch.float32,
                       device=device)
    pix, nbr = extract_neighborhoods(img, k)
    return pix.cpu().numpy(), nbr.cpu().numpy()


# --------------------------------------------------------------------------
# Our detectors
# --------------------------------------------------------------------------
def fit_dart(scene, seed, ckpt, device):
    tr = np.asarray(scene["tr"], np.float32)
    D = tr.shape[1]
    Wh = _whiten(tr, device)
    tr_w = Wh(torch.tensor(tr, device=device)).detach()
    sigma = PR.sigma_mse_optimal(tr_w.cpu().numpy())

    def build():
        net = ScoreNet(D, HP["dsm_hidden"], HP["activation"], whitening=Wh)
        net.sigma = sigma
        return net

    net = _resume(build, ckpt, device)
    if net is not None:
        return net
    torch.manual_seed(seed); np.random.seed(seed)
    net = build().to(device)
    Xtr = torch.tensor(tr, dtype=torch.float32, device=device)
    net = _train_loop(net, net.parameters(), (Xtr,),
                      lambda b: dsm_loss(net, b, sigma), PR.EPOCHS,
                      HP["dsm_lr"], HP["batch_size"], HP["weight_decay"],
                      f"DART s{seed}")
    _save(net, ckpt, sigma)
    return net


def score_dart(net, scene, planted, device):
    return dsm_additive(planted.astype(np.float32),
                        np.asarray(scene["tr"], np.float32), net,
                        scene["_sig"])


def score_dart_cfar(net, scene, planted, device):
    flat = score_dart(net, scene, planted, device)
    return _cfar_normalize_map(flat, scene["te_shape"],
                               bg=HP["dart_cfar_window"],
                               guard=HP["dart_cfar_guard"],
                               cfar_lam=HP["cfar_lam"])


def fit_darts(scene, seed, ckpt, device):
    tr = np.asarray(scene["tr"], np.float32)
    D = tr.shape[1]
    Wh = _whiten(tr, device)
    tr_w = Wh(torch.tensor(tr, device=device)).detach()
    sigma = PR.sigma_mse_optimal(tr_w.cpu().numpy())

    # training windows come from the TRAIN box image
    Hb = scene["boxes"]["train"]
    Ht, Wt = Hb[1] - Hb[0], Hb[3] - Hb[2]
    img = torch.tensor(tr.reshape(Ht, Wt, D), dtype=torch.float32,
                       device=device)
    pix, nbr = extract_neighborhoods(img, HP["k"])

    def build():
        return NeighborMLPDenoiser(D=D, d_lat=HP["nmlp_d_lat"], K=HP["nmlp_K"],
                                   enc_hidden=HP["nmlp_enc_hidden"],
                                   score_hidden=HP["nmlp_score_hidden"],
                                   sigma=sigma, activation=HP["activation"],
                                   whitening=Wh)

    net = _resume(build, ckpt, device)
    if net is not None:
        net._tr_pix, net._tr_nbr = pix, nbr
        return net
    torch.manual_seed(seed); np.random.seed(seed)
    net = build().to(device)
    net = _train_loop(net, net.parameters(), (pix, nbr),
                      lambda p, n: neighbor_mlp_dsm_loss(net, p, n), PR.EPOCHS,
                      HP["nmlp_lr"], HP["nmlp_batch"], HP["weight_decay"],
                      f"DARTS s{seed}")
    _save(net, ckpt, sigma)
    net._tr_pix, net._tr_nbr = pix, nbr
    return net


def score_darts(net, scene, planted, device):
    pix, nbr = _test_windows(scene, planted, HP["k"], device)
    tr_pix = net._tr_pix.cpu().numpy()
    tr_nbr = net._tr_nbr.cpu().numpy()
    return score_nmlp_additive(net, pix, nbr, tr_pix, tr_nbr, scene["_sig"])


def score_darts_cfar(net, scene, planted, device):
    pix, nbr = _test_windows(scene, planted, HP["k"], device)
    tr_pix = net._tr_pix.cpu().numpy()
    tr_nbr = net._tr_nbr.cpu().numpy()
    flat = score_nmlp_additive(net, pix, nbr, tr_pix, tr_nbr, scene["_sig"])
    return _knn_fisher_normalize(flat, net, pix, nbr, scene["te_shape"],
                                 HP["k"], cfar_lam=HP["cfar_lam"],
                                 guard=HP["darts_cfar_guard"])


# --------------------------------------------------------------------------
# Classical / learned baselines
# --------------------------------------------------------------------------
def score_amf(state, scene, planted, device):
    return amf(planted, scene["tr"], scene["_sig"], eig_floor=0.0)


def amf_local_window(D):
    """Dimension-aware local-SCM window: smallest odd k with k^2-1 >= 2D, so
    the window supplies at least twice as many samples as bands (n/D >= 2).
    Reproduces the paper's 15x15 on Pavia (D=103) and gives 21x21 on San
    Diego (D=189), where the fixed 15x15 window left the unloaded local SCM
    sample-starved (n/D=1.2, AUC 0.59 -> 0.99 at n/D=2.3)."""
    k = int(np.ceil(np.sqrt(2 * D + 1)))
    return k + 1 if k % 2 == 0 else k


def score_amf_local(state, scene, planted, device):
    win = amf_local_window(planted.shape[1])
    pix, nbr = _test_windows(scene, planted, win, device)
    return amf_local(pix, nbr, scene["_sig"], device=device, loading=0.0)


def score_gmm_levin(state, scene, planted, device):
    return gmm_glrt_levin_additive(planted, scene["tr"], scene["_sig"],
                                   p_steps=HP["gmm_steps"])


class _RobustNorm:
    """Median/IQR input normalization — the stabilization that fixed LRao.
    TODO(verify vs. archived pavia4_lrao checkpoint before quoting numbers)."""

    def __init__(self, X):
        self.med = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        self.scale = np.where(iqr > 1e-8, iqr, 1.0)

    def __call__(self, X):
        return (np.asarray(X, np.float32) - self.med) / self.scale


def fit_lrao(scene, seed, ckpt, device):
    tr = np.asarray(scene["tr"], np.float32)
    D = tr.shape[1]
    norm = _RobustNorm(tr)
    tr_n = norm(tr)

    def build():
        net = ScoreNet(D, HP["lrao_hidden"], HP["activation"])
        net._robust_norm = norm
        return net

    net = _resume(build, ckpt, device)
    if net is not None:
        net._robust_norm = norm
        return net
    torch.manual_seed(seed); np.random.seed(seed)
    net = build().to(device)
    Xtr = torch.tensor(tr_n, dtype=torch.float32, device=device)
    net = _train_loop(net, net.parameters(), (Xtr,),
                      lambda b: lfi_loss_mode2(net, b,
                                               HP["lrao_delta_theta"],
                                               detach_sigma=True),
                      PR.EPOCHS, HP["lrao_lr"], HP["batch_size"],
                      HP["weight_decay"], f"LRao s{seed}", grad_clip=1.0)
    _save(net, ckpt)
    return net


def score_lrao(net, scene, planted, device):
    norm = net._robust_norm
    tr_n = norm(scene["tr"])
    te_n = norm(planted)
    # the signature is a direction: scale only (no median shift)
    sig_n = np.asarray(scene["_sig"], np.float32) / norm.scale
    return compute_lfi_detector_scores_mode2(net, tr_n, te_n, sig_n,
                                             HP["lrao_delta_theta"])


# --------------------------------------------------------------------------
# Deep baselines (paper-faithful ports from the colab-deep-baselines branch)
# --------------------------------------------------------------------------
def fit_thantd(scene, seed, ckpt, device):
    from thantd_model import THANTD, build_thantd_samples, train_thantd
    tr, sig = np.asarray(scene["tr"]), scene["_sig"]

    def build():
        return THANTD(b=tr.shape[1])

    m = _resume(build, ckpt, device)
    if m is not None:
        return m
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    a, p, n = build_thantd_samples(tr, sig, alpha=0.5, n_samples=1024,
                                   rng=rng, bkg_pool=tr)
    m = build().to(device)
    train_thantd(m, a, p, n, epochs=300, batch_size=64, lr=1e-4, margin=0.3,
                 device=device)
    _save(m, ckpt)
    return m


def score_thantd(m, scene, planted, device):
    from thantd_model import score_thantd as _score
    return _score(m, scene["_sig"], planted, device=device)


def fit_htdnet(scene, seed, ckpt, device):
    from colab_deep import htdnet_model as H
    tr, sig = np.asarray(scene["tr"]), scene["_sig"]
    if ckpt and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = H.SDCNN(); sd.load_state_dict(blob["sdcnn"])
        sd._scale = blob["scale"]; sd.to(device).eval()
        print(f"    resumed {ckpt}", flush=True)
        return dict(sd=sd, gen_t=blob["gen_t"], bkg=blob["bkg"])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    uae, sc = H.train_uae(tr, epochs=150, device=device, seed=seed)
    gen_t = H.generate_targets(uae, sc, sig, tr, n_samples=1000,
                               device=device, rng=rng)
    bkg = H.lp_background_selection(tr, sig, n_direct=60, n_total=500)
    sd = H.train_sdcnn(gen_t, bkg, H.ACELabeler(tr), epochs=30,
                       pairs_per_epoch=100_000, device=device, seed=seed,
                       log_every=10)
    if ckpt:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        torch.save(dict(sdcnn=sd.state_dict(), scale=sd._scale,
                        gen_t=gen_t, bkg=bkg), ckpt)
    return dict(sd=sd, gen_t=gen_t, bkg=bkg)


def score_htdnet(state, scene, planted, device):
    from colab_deep import htdnet_model as H
    return H.htdnet_detect(state["sd"], planted, state["gen_t"],
                           state["bkg"], device=device)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
@dataclass
class DetectorSpec:
    name: str
    score: Callable
    fit: Optional[Callable] = None      # None = training-free
    needs_gpu: bool = False
    artifact_only: bool = False         # scored from archived raw scores only
    notes: str = ""


REGISTRY = {s.name: s for s in [
    DetectorSpec("DART", score_dart, fit_dart),
    DetectorSpec("DART-CFAR", score_dart_cfar, fit_dart,
                 notes="derived from the DART checkpoint"),
    DetectorSpec("DARTS", score_darts, fit_darts, needs_gpu=True),
    DetectorSpec("DARTS-CFAR", score_darts_cfar, fit_darts, needs_gpu=True,
                 notes="derived from the DARTS checkpoint"),
    DetectorSpec("AMF", score_amf),
    DetectorSpec("AMF-local", score_amf_local),
    DetectorSpec("GMM-Levin", score_gmm_levin),
    DetectorSpec("LRao", score_lrao, fit_lrao,
                 notes="robust-normalization stabilized variant"),
    DetectorSpec("THANTD", score_thantd, fit_thantd, needs_gpu=True),
    DetectorSpec("HTDNet", score_htdnet, fit_htdnet, needs_gpu=True),
    DetectorSpec("OSVAE", None, None, artifact_only=True,
                 notes="Phase B: re-create the OS-VAE port as a module"),
    DetectorSpec("TSTTD", None, None, artifact_only=True,
                 notes="vendor code in colab_deep/vendor_tsttd; wire in Phase B"),
]}

OUR_DETECTORS = ["DART", "DART-CFAR", "DARTS", "DARTS-CFAR"]
CLASSICAL = ["AMF", "AMF-local", "GMM-Levin", "LRao"]
DEEP = ["THANTD", "HTDNet", "OSVAE", "TSTTD"]

# fit-state sharing: CFAR variants reuse their base detector's checkpoint
CKPT_ALIAS = {"DART-CFAR": "DART", "DARTS-CFAR": "DARTS"}
