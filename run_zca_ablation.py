#!/usr/bin/env python3
"""DART ZCA whitening-floor ablation (IID multi, retrained).

The FAIR version of the floor question: for every floor value the model is
retrained from scratch with that whitening, and scored through the same
whitening — no train/score mismatch.

Grid: floors = [1e-3, 1e-5, 1e-6, 1e-7, 1e-9, 1e-12, 1e-18, 1e-22, none]
(relative to lambda_max; *none* = 1/sqrt(lambda) wherever lambda>0, null
directions dropped). n = [20, 40, 60, 100, 200, 500, 1000, 2000, 4000].
rho=0.1, 3000 epochs, 5 seeds (42-46), multi-class = 405 trainings.

Recipe is the published multi DART otherwise: [128] ReLU, batch 512,
Adam 5e-4 / wd 5e-7, sigma=sqrt(rho) in whitened space. Reported model = the
published selection rule (best train loss over epochs %100, plus final); test
Pd/AUC also snapshotted every 100 epochs. Divergence/NaN runs are recorded as
NaN, not crashes. Resume-safe: rerun after any interrupt.

Local port of RunZCAAblation.ipynb: no Colab clone/download, runs from the repo
root, and uses Apple MPS when available (set env DEVICE=cpu to force CPU).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import copy
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from repro.protocols.iid import load_hsi, build_pools, _pd_at_fa, _auc, _pauc
from repro.core.data import plant_targets, Whitening
from repro.core.models import ScoreNet, dsm_loss
from repro.core.detectors import dsm_additive


def pick_device():
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = pick_device()
print("device:", DEVICE)
assert os.path.exists("repro/data/pavia-u.mat"), "missing repro/data/pavia-u.mat"

# ----------------- knobs -----------------
FLOORS = [1e-3, 1e-5, 1e-6, 1e-7, 1e-9, 1e-12, 1e-18, 1e-22, "none"]
N_LIST = [20, 40, 60, 100, 200, 500, 1000, 2000, 4000]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 3000
RHO = 0.1
EVAL_EVERY = 100                      # test-metric snapshots (diagnostic)
OUT = "results/zca_ablation"


def flab(fl):
    return "none" if fl == "none" else f"{fl:.0e}"


# Published multi reference lines (mean of 5 seeds, Pd@Pfa=0.1; n<=2000 only)
REF_N = [20, 40, 60, 100, 200, 500, 1000, 2000]
REF_DART = [0.287, 0.300, 0.327, 0.390, 0.402, 0.447, 0.537, 0.595]
REF_LRAO = [0.329, 0.367, 0.465, 0.501, 0.509, 0.565, 0.561, 0.553]
REF_LRAO1E5 = [0.287, 0.440, 0.436, 0.534, 0.591, 0.661, 0.690, 0.703]

# ----------------- protocol + trainer -----------------
cfg = yaml.safe_load(open("repro/configs/iid_multi.yaml"))
cfg.update(dataset="repro/data/pavia-u.mat", dsm_sigma_rho=RHO)


def build_data(seed):
    """run_iid pool/split/plant with max_n extended to 4000: the first-2000
    prefix and the test set are bit-identical to the published runs."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data, gt = load_hsi(cfg["dataset"])
    bkg_raw, tgt_raw = build_pools(data, gt.flatten(), cfg, "multi")
    s_raw = tgt_raw.mean(axis=0).astype(np.float32)
    idx = np.arange(len(bkg_raw)); rng.shuffle(idx)
    bkg_shuf = bkg_raw[idx]
    max_n = max(N_LIST)
    test_size = int(cfg["test_size"])
    assert len(bkg_shuf) >= max_n + test_size
    pool = bkg_shuf[:max_n].astype(np.float32)
    test_bkg = bkg_shuf[-test_size:].astype(np.float32)
    planted, labels, _ = plant_targets(test_bkg, s_raw, cfg["amplitude"],
                                       cfg["target_fraction"],
                                       model="additive", seed=seed)
    return pool, planted.astype(np.float32), labels, s_raw


def zca_with_floor(tr, fl):
    """Whitening.from_data ops with a PURE relative floor (no absolute eps).
    fl='none' -> 1/sqrt(eval) wherever eval > 0, null directions dropped."""
    X = np.asarray(tr, np.float64)
    mu = X.mean(0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / max(len(X) - 1, 1)
    Sigma = (Sigma + Sigma.T) / 2
    evals, evecs = np.linalg.eigh(Sigma)
    if fl == "none":
        inv_sqrt = np.where(evals > 0, 1.0 / np.sqrt(np.abs(evals)), 0.0)
    else:
        inv_sqrt = 1.0 / np.sqrt(np.clip(evals, float(evals[-1]) * float(fl),
                                         None))
    W = evecs @ np.diag(inv_sqrt) @ evecs.T
    return Whitening(mu.astype(np.float32), W.astype(np.float32))


def train_dart_floor(tr, fl, seed, label, planted, labels, s_raw):
    """train_dsm_local (published recipe) with the floor-parameterized ZCA;
    published best-train-loss selection + per-EVAL_EVERY test snapshots."""
    torch.manual_seed(seed)
    D = tr.shape[1]
    W = zca_with_floor(tr, fl)
    sigma = float(np.sqrt(RHO))
    model = ScoreNet(D, list(cfg["hidden_dims"]), cfg["activation"],
                     whitening=W).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    X = torch.tensor(np.asarray(tr, np.float32)).to(DEVICE)
    N = len(X); bs = min(int(cfg["batch_size"]), N)
    best, best_state = float("inf"), None
    traj = []
    pbar = tqdm(range(1, EPOCHS + 1), desc=label, dynamic_ncols=True,
                leave=False)
    for ep in pbar:
        model.train()
        perm = torch.randperm(N)
        tot, nb = None, 0
        for i in range(0, N, bs):
            loss = dsm_loss(model, X[perm[i:i + bs]], sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot = loss.detach() if tot is None else tot + loss.detach()
            nb += 1
        ep_loss = float(tot.item()) / max(nb, 1)
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            if np.isfinite(ep_loss) and ep_loss < best:
                best = ep_loss
                best_state = copy.deepcopy(model.state_dict())
            model.eval()
            try:
                sc = dsm_additive(planted, tr, model, s_raw)
                traj.append({"epoch": ep, "pd": _pd_at_fa(labels, sc,
                                                          cfg["pfa"]),
                             "auc": _auc(labels, sc), "loss": ep_loss})
            except Exception:
                traj.append({"epoch": ep, "pd": float("nan"),
                             "auc": float("nan"), "loss": ep_loss})
            pbar.set_postfix(loss=f"{ep_loss:.1f}", pd=f"{traj[-1]['pd']:.3f}")
    # metrics of the published-rule model (best train loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    try:
        sc = dsm_additive(planted, tr, model, s_raw)
        pd_sel = _pd_at_fa(labels, sc, cfg["pfa"])
        auc_sel = _auc(labels, sc)
    except Exception:
        pd_sel = auc_sel = float("nan")
    return model, dict(pd_sel=pd_sel, auc_sel=auc_sel, traj=traj)


def run_sweep():
    """The sweep (resume-safe)."""
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    met_path = os.path.join(OUT, "metrics.json")
    rec = json.load(open(met_path)) if os.path.exists(met_path) else {}
    rec["_meta"] = dict(floors=[flab(f) for f in FLOORS], n_list=N_LIST,
                        seeds=SEEDS, epochs=EPOCHS, rho=RHO, mode="multi")
    t0 = time.time()
    for seed in SEEDS:
        pool, planted, labels, s_raw = build_data(seed)
        for n in N_LIST:
            tr = pool[:n]
            for fl in FLOORS:
                key = f"fl{flab(fl)}_n{n}_s{seed}"
                if rec.get(key, {}).get("done"):
                    continue
                t1 = time.time()
                net, r = train_dart_floor(tr, fl, seed, key, planted, labels,
                                          s_raw)
                torch.save({"state_dict": {k: v.cpu() for k, v in
                                           net.state_dict().items()}},
                           os.path.join(OUT, "models", key + ".pt"))
                r["done"] = True
                r["sec"] = round(time.time() - t1, 1)
                rec[key] = r
                with open(met_path, "w") as f:
                    json.dump(rec, f)
                print(f"{key}: Pd(sel)={r['pd_sel']:.3f} ({r['sec']}s)",
                      flush=True)
    print(f"TOTAL {(time.time() - t0) / 3600:.2f} h")


def analyze():
    """Analysis + figures + summary."""
    FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
    rec = json.load(open(os.path.join(OUT, "metrics.json")))
    FL = [flab(f) for f in FLOORS]

    P = np.full((len(FL), len(N_LIST), len(SEEDS)), np.nan)   # pd_sel
    O = np.full((len(FL), len(N_LIST), len(SEEDS)), np.nan)   # oracle over epochs
    for i, fl in enumerate(FL):
        for j, n in enumerate(N_LIST):
            for k, s in enumerate(SEEDS):
                r = rec.get(f"fl{fl}_n{n}_s{s}")
                if not r or not r.get("done"):
                    continue
                P[i, j, k] = r["pd_sel"]
                pds = [t["pd"] for t in r["traj"] if np.isfinite(t["pd"])]
                O[i, j, k] = max(pds) if pds else np.nan
    Pm, Om = np.nanmean(P, axis=2), np.nanmean(O, axis=2)

    # --- figure 1: Pd vs n per floor ---
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(FL)))
    for i, fl in enumerate(FL):
        lw = 2.4 if fl == "1e-05" else 1.4
        ax.plot(N_LIST, Pm[i], "o-", color=colors[i], lw=lw,
                label=f"floor {fl}" + (" (~published)" if fl == "1e-05" else ""))
    ax.plot(REF_N, REF_DART, "s--", color="tab:red", lw=1.6,
            label="DART (published, eps floor, 2000ep)")
    ax.plot(REF_N, REF_LRAO, "x", color="k", ms=6, label="LRao (published)")
    ax.plot(REF_N, REF_LRAO1E5, "^-", color="tab:green", lw=1.2,
            label="LRao cutoff 1e-5")
    ax.set_xscale("log"); ax.set_xticks(N_LIST); ax.set_xticklabels(N_LIST)
    ax.minorticks_off(); ax.grid(alpha=0.3)
    ax.set_xlabel("n (training pixels)")
    ax.set_ylabel(r"$P_d$ @ $P_{fa}=0.1$ (published selection rule)")
    ax.set_title("IID multi: DART ZCA-floor ablation (retrained; "
                 f"rho={RHO}, {EPOCHS}ep, {len(SEEDS)} seeds)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out1 = os.path.join(FIG, "zca_floor_vs_n.png")
    fig.savefig(out1, dpi=200)
    fig.savefig(out1.replace(".png", ".pdf"))
    plt.close(fig); print("wrote", out1)

    # --- figure 2: heatmap + best floor per n ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    im = axes[0].imshow(Pm, aspect="auto", cmap="viridis")
    axes[0].set_xticks(range(len(N_LIST))); axes[0].set_xticklabels(N_LIST)
    axes[0].set_yticks(range(len(FL))); axes[0].set_yticklabels(FL, fontsize=8)
    axes[0].set_xlabel("n"); axes[0].set_ylabel("floor")
    axes[0].set_title("mean Pd@0.1 (selection rule)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    best_i = np.nanargmax(np.where(np.isnan(Pm), -1, Pm), axis=0)
    axes[1].plot(N_LIST, [FL.index(FL[b]) for b in best_i], "s-")
    axes[1].set_xscale("log"); axes[1].set_yticks(range(len(FL)))
    axes[1].set_yticklabels(FL, fontsize=8)
    axes[1].set_xlabel("n"); axes[1].set_title("best floor per n")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out2 = os.path.join(FIG, "zca_floor_heatmap.png")
    fig.savefig(out2, dpi=200)
    fig.savefig(out2.replace(".png", ".pdf"))
    plt.close(fig); print("wrote", out2)

    # --- summary table ---
    lines = ["# ZCA floor ablation — Pd@0.1 (published selection; mean over seeds)",
             "", "| floor | " + " | ".join(f"n={n}" for n in N_LIST) + " |",
             "|" + "---|" * (len(N_LIST) + 1)]
    for i, fl in enumerate(FL):
        lines.append(f"| {fl} | " + " | ".join(
            f"{v:.3f}" if np.isfinite(v) else "nan" for v in Pm[i]) + " |")
    lines += ["", "| best floor per n | " + " | ".join(
        FL[b] for b in best_i) + " |",
        "", "Oracle-over-epochs (diagnostic) max gain vs selection: "
        f"{np.nanmax(Om - Pm):.3f}"]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def package():
    """Package the results (light = no per-model checkpoints; full = all)."""
    import zipfile

    def _zip(path, root, skip=None):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for r, _, files in os.walk(root):
                if skip and skip in r.split(os.sep):
                    continue
                for fn in files:
                    z.write(os.path.join(r, fn))

    _zip("zca_ablation_light.zip", "results/zca_ablation", skip="models")
    _zip("zca_ablation_full.zip", "results/zca_ablation")
    print("zipped -> zca_ablation_light.zip, zca_ablation_full.zip")


def main():
    run_sweep()
    analyze()
    package()


if __name__ == "__main__":
    main()
