"""tsp_repro.robust_leakage — L1 vs L2 DSM training under target leakage.

Hypothesis: the L2 DSM regression matches the conditional MEAN of the
denoising target, so a few strongly contaminated training windows pull the
learned score toward the leaked targets (the absorption mechanism of the
leakage study). L1 matches the conditional MEDIAN, so it should tolerate a
contaminated minority at the price of a small clean-data bias.

Protocol (mirrors the leakage study, on the TSP pavia4 protocol):
  - scene pavia4, bitumen signature; train image = side-cropped train box.
  - contamination: replacement theta=0.95 planted at rate r in {0,1,5,10}%
    of TRAIN-IMAGE pixels (fixed planting seed), neighborhoods re-extracted
    from the contaminated image, whitening fit on the contaminated pixels
    (the realistic contam/contam cell).
  - train DARTS with loss in {l2, l1}, same budget for both.
  - evaluate: weak test (theta=0.15 additive), strong test (theta=0.95
    replacement), and the detect-own diagnostic (AUC of the model's own
    statistic on its training set vs the leaked-pixel labels; ~1 = still
    sees the leaks, ~0.5 = absorbed).

Run:  .venv/bin/python -m tsp_repro.robust_leakage --epochs 1000 --out results_robust_leakage
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import tsp_repro  # noqa: F401
from tsp_repro import protocol as PR
from tsp_repro.registry import HP, _whiten

from src.data import extract_neighborhoods
from src.models import NeighborMLPDenoiser, score_nmlp_additive
from src.spatial import _knn_fisher_normalize


def dsm_loss_generic(model, x, neighbors, loss_type="l2"):
    """neighbor_mlp_dsm_loss with a selectable residual penalty."""
    sigma = model.sigma
    x_w = model.whiten(x)
    nbr_w = model.whiten(neighbors)
    eps = torch.randn_like(x_w) * sigma
    y = x_w + eps
    target = -eps / (sigma ** 2)
    score = model._forward_inner(y, nbr_w)
    r = score - target
    if loss_type == "l2":
        return (r ** 2).sum(-1).mean()
    if loss_type == "l1":
        return r.abs().sum(-1).mean()
    if loss_type == "huber":          # per-coordinate Huber, delta in target units
        return torch.nn.functional.huber_loss(score, target, delta=1.0,
                                              reduction="none").sum(-1).mean()
    raise ValueError(loss_type)


def contaminate_train_image(tr, shape, sig, rate, seed=0):
    """Replacement-plant theta=0.95 targets at `rate` of train-image pixels."""
    rng = np.random.RandomState(seed)
    n = len(tr)
    labels = np.zeros(n, dtype=np.int8)
    img = tr.copy()
    if rate > 0:
        idx = rng.choice(n, size=int(round(rate * n)), replace=False)
        img[idx] = (1 - 0.95) * img[idx] + 0.95 * sig[None, :]
        labels[idx] = 1
    return img, labels


def train_darts(tr_img_flat, shape, seed, epochs, loss_type, device="cpu"):
    D = tr_img_flat.shape[1]
    Wh = _whiten(tr_img_flat, device)
    tr_w = Wh(torch.tensor(tr_img_flat, dtype=torch.float32,
                           device=device)).detach()
    sigma = PR.sigma_mse_optimal(tr_w.cpu().numpy())
    img = torch.tensor(tr_img_flat.reshape(*shape, D), dtype=torch.float32,
                       device=device)
    pix, nbr = extract_neighborhoods(img, HP["k"])

    torch.manual_seed(seed); np.random.seed(seed)
    net = NeighborMLPDenoiser(D=D, d_lat=HP["nmlp_d_lat"], K=HP["nmlp_K"],
                              enc_hidden=HP["nmlp_enc_hidden"],
                              score_hidden=HP["nmlp_score_hidden"],
                              sigma=sigma, activation=HP["activation"],
                              whitening=Wh).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=HP["nmlp_lr"],
                            weight_decay=HP["weight_decay"])
    P = len(pix)
    best_loss, best_state = float("inf"), None
    for ep in range(epochs):
        perm = torch.randperm(P, device=device)
        ep_loss, nb = 0.0, 0
        for i in range(0, P, HP["nmlp_batch"]):
            sel = perm[i:i + HP["nmlp_batch"]]
            loss = dsm_loss_generic(net, pix[sel], nbr[sel], loss_type)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        last = ep_loss / max(nb, 1)
        if last < best_loss:
            best_loss, best_state = last, copy.deepcopy(net.state_dict())
        if ep == 0 or (ep + 1) % max(epochs // 10, 1) == 0:
            print(f"    [{loss_type} s{seed}] ep {ep+1}/{epochs} "
                  f"loss={last:.4f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    net._tr_pix, net._tr_nbr = pix, nbr
    return net


def evaluate(net, scene, sig, device="cpu"):
    from sklearn.metrics import roc_auc_score
    tr_pix = net._tr_pix.cpu().numpy()
    tr_nbr = net._tr_nbr.cpu().numpy()
    out = {}
    for tag, model, th in (("weak", "additive", 0.15),
                           ("strong", "replacement", 0.95)):
        planted, lab = PR.plant(scene, sig, th, model=model, seed=42)
        H, W = scene["te_shape"]
        img = torch.tensor(planted.reshape(H, W, -1), dtype=torch.float32,
                           device=device)
        pix, nbr = extract_neighborhoods(img, HP["k"])
        pix, nbr = pix.cpu().numpy(), nbr.cpu().numpy()
        flat = score_nmlp_additive(net, pix, nbr, tr_pix, tr_nbr, sig)
        out[f"darts_{tag}"] = float(roc_auc_score(lab, flat))
        cf = _knn_fisher_normalize(flat, net, pix, nbr, scene["te_shape"],
                                   HP["k"], cfar_lam=HP["cfar_lam"],
                                   guard=HP["darts_cfar_guard"])
        out[f"cfar_{tag}"] = float(roc_auc_score(lab, cf))
    return out


def detect_own(net, leak_labels, sig):
    """AUC of the model's own statistic on its (contaminated) training set."""
    from sklearn.metrics import roc_auc_score
    if leak_labels.sum() == 0:
        return None
    tr_pix = net._tr_pix.cpu().numpy()
    tr_nbr = net._tr_nbr.cpu().numpy()
    flat = score_nmlp_additive(net, tr_pix, tr_nbr, tr_pix, tr_nbr, sig)
    return float(roc_auc_score(leak_labels, flat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.0, 0.01, 0.05, 0.10])
    ap.add_argument("--losses", nargs="+", default=["l2", "l1"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_robust_leakage")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = PR.build_scene("pavia4")
    sig = scene["sigs"]["bitumen"]
    tb = scene["boxes"]["train"]
    shape = (tb[1] - tb[0], tb[3] - tb[2])
    tr = np.asarray(scene["tr"], np.float64)

    results = {}
    for rate in args.rates:
        img, leak_lab = contaminate_train_image(tr, shape, sig, rate, seed=0)
        for loss in args.losses:
            key = f"{loss}_r{rate:g}"
            ck = os.path.join(args.out, f"darts_{key}_s{args.seed}.pt")
            t0 = time.time()
            net = train_darts(img, shape, args.seed, args.epochs, loss,
                              args.device)
            torch.save({"model": net.state_dict(), "sigma": net.sigma}, ck)
            res = evaluate(net, scene, sig, args.device)
            res["detect_own"] = detect_own(net, leak_lab, sig)
            res["train_minutes"] = round((time.time() - t0) / 60, 1)
            results[key] = res
            print(f"== {key}: {res}", flush=True)
            with open(os.path.join(args.out, "results.json"), "w") as f:
                json.dump(dict(epochs=args.epochs, seed=args.seed,
                               protocol="pavia4/bitumen contam-image "
                                        "replacement0.95 weak0.15",
                               results=results), f, indent=1)

    print("\n=== summary (weak / strong / detect-own) ===")
    for k, r in results.items():
        print(f"  {k:10s} darts {r['darts_weak']:.3f}/{r['darts_strong']:.3f} "
              f"cfar {r['cfar_weak']:.3f}/{r['cfar_strong']:.3f} "
              f"own={r['detect_own']}")


if __name__ == "__main__":
    main()
