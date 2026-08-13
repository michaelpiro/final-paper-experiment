#!/usr/bin/env python3
"""DART envelope study — best rho and best epoch per n (IID).

For every n in the published IID protocol (single and multi), train DART with
rho in {0.01, 0.08, 0.1, 0.5, 1.0} for 3000 epochs, snapshotting every 50
epochs (5 seeds, 42-46). Every snapshot is scored on the published planted test
set (additive theta=0.15), so afterwards we pick the best epoch for detection
per (n, rho) and draw the oracle envelope over (rho, epoch) that bounds DART's
performance.

Data pools, splits, planting, whitening, architecture ([128] ReLU), optimizer
and batch schedule are the verbatim published run_iid recipe. Outputs land
under results/rho_envelope/<mode>/ (resume-safe, zipped at the end).

Budget: 2 modes x 8 n x 5 rho x 5 seeds = 400 trainings x 3000 epochs.
Set SAVE_CKPTS=False for a metrics-only run. Interrupt + rerun any time —
finished (n, rho, seed) combos are skipped.

Local port of RunRhoEnvelope.ipynb: no Colab clone/download, runs from the repo
root, and uses Apple MPS when available (set env DEVICE=cpu to force CPU).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from repro.protocols.iid import (load_hsi, build_pools, _make_whitening,
                                 _auc, _pauc, _pd_at_fa)
from repro.core.data import plant_targets
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

# ----------------- experiment knobs -----------------
MODES = ["single", "multi"]
RHOS = [0.01, 0.08, 0.1, 0.5, 1.0]     # 0.1 = the published DART value
EPOCHS = 3000                          # published budget was 2000
SNAP_EVERY = 50
SEEDS = [42, 43, 44, 45, 46]           # published seeds
N_LIST = None                          # None -> config n_train_list [20..2000]
SAVE_CKPTS = True                      # fp16 net-only snapshots (~1.3 GB total)
OUT = "results/rho_envelope"

# Published reference lines (verified camera-ready IID run, mean over 5 seeds).
REF = {
 "single": {
  "n": [20, 40, 60, 100, 200, 500, 1000, 2000],
  "pd": {
   "DART":      [0.539, 0.492, 0.500, 0.493, 0.522, 0.655, 0.793, 0.873],
   "L-DART":    [0.516, 0.492, 0.483, 0.535, 0.609, 0.671, 0.706, 0.715],
   "LRao":      [0.572, 0.617, 0.546, 0.539, 0.673, 0.712, 0.768, 0.791],
   "L-LRao":    [0.523, 0.486, 0.438, 0.512, 0.579, 0.667, 0.693, 0.707],
   "AMF":       [None,  None,  None,  None,  0.516, 0.650, 0.682, 0.708],
  },
  "auc": {
   "DART":      [0.845, 0.828, 0.822, 0.825, 0.844, 0.890, 0.935, 0.954],
   "L-DART":    [0.843, 0.829, 0.817, 0.835, 0.870, 0.897, 0.910, 0.916],
   "LRao":      [0.843, 0.857, 0.814, 0.807, 0.879, 0.900, 0.919, 0.931],
   "L-LRao":    [0.826, 0.815, 0.793, 0.835, 0.867, 0.898, 0.909, 0.915],
   "AMF":       [None,  None,  None,  None,  0.837, 0.891, 0.906, 0.915],
  },
 },
 "multi": {
  "n": [20, 40, 60, 100, 200, 500, 1000, 2000],
  "pd": {
   "DART":      [0.287, 0.300, 0.327, 0.390, 0.402, 0.447, 0.537, 0.595],
   "L-DART":    [0.279, 0.305, 0.318, 0.395, 0.411, 0.422, 0.457, 0.471],
   "LRao":      [0.329, 0.367, 0.465, 0.501, 0.509, 0.565, 0.561, 0.553],
   "L-LRao":    [0.312, 0.337, 0.332, 0.354, 0.366, 0.361, 0.369, 0.382],
   "AMF":       [None,  None,  None,  0.135, 0.349, 0.426, 0.463, 0.493],
   "GMM-Levin": [0.152, 0.185, 0.278, 0.202, 0.596, 0.757, 0.757, 0.782],
  },
  "auc": {
   "DART":      [0.714, 0.727, 0.738, 0.752, 0.771, 0.803, 0.841, 0.853],
   "L-DART":    [0.696, 0.721, 0.731, 0.759, 0.768, 0.787, 0.800, 0.807],
   "LRao":      [0.710, 0.755, 0.789, 0.804, 0.804, 0.821, 0.830, 0.828],
   "L-LRao":    [0.686, 0.703, 0.700, 0.702, 0.714, 0.714, 0.718, 0.727],
   "AMF":       [None,  None,  None,  0.548, 0.742, 0.786, 0.801, 0.808],
   "GMM-Levin": [0.631, 0.698, 0.739, 0.633, 0.847, 0.895, 0.909, 0.908],
  },
 },
}


# ---- published protocol pieces (verbatim imports from the repro package) ----
def build_mode_data(cfg, mode, seed):
    """Faithful replica of run_iid's pool/split/plant — bit-identical train
    pools and planted test sets to the published runs."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data, gt = load_hsi(cfg["dataset"])
    gt_flat = gt.flatten()
    bkg_raw, tgt_raw = build_pools(data, gt_flat, cfg, mode)
    s_raw = tgt_raw.mean(axis=0).astype(np.float32)
    if cfg.get("normalize_signature", False):
        s_raw = (s_raw / (np.linalg.norm(s_raw) + 1e-12)).astype(np.float32)
    idx = np.arange(len(bkg_raw)); rng.shuffle(idx)
    bkg_shuf = bkg_raw[idx]
    n_list = [int(n) for n in cfg["n_train_list"]]
    max_n = max(max(n_list), int(cfg["n_fixed_for_rho"]))
    test_size = int(cfg["test_size"])
    assert len(bkg_shuf) >= max_n + test_size
    train_pool = bkg_shuf[:max_n].astype(np.float32)
    test_bkg = bkg_shuf[-test_size:].astype(np.float32)
    test_planted, labels, _ = plant_targets(
        test_bkg, s_raw, cfg["amplitude"], cfg["target_fraction"],
        model="additive", seed=seed)
    return train_pool, test_planted.astype(np.float32), labels, s_raw, n_list


def dart_arch(cfg, mode):
    """The mode's MLP DART architecture (paper: [128] ReLU in both modes)."""
    if mode == "single":
        return list(cfg["hidden_dims_2"]), cfg.get("activation_2", cfg["activation"])
    return list(cfg["hidden_dims"]), cfg["activation"]


def train_dart_snapshots(tr, cfg, hidden, act, rho, seed, label,
                         test_planted, labels, s_raw, device,
                         epochs, snap_every, ckpt_dir=None):
    """The published train_dsm_local loop + per-snapshot test evaluation."""
    torch.manual_seed(seed)                       # as in train_dsm_local
    D = tr.shape[1]
    W = _make_whitening(tr, cfg)
    sigma = float(np.sqrt(rho))
    model = ScoreNet(D, list(hidden), act, whitening=W).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    X = torch.tensor(np.asarray(tr, np.float32)).to(device)
    N = len(X); bs = min(int(cfg["batch_size"]), N)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()
                    if k.startswith("whitening")},
                   os.path.join(ckpt_dir, "whitening.pt"))
    rec = {"epochs": [], "auc": [], "pauc": [], "pd": [], "loss_at_snap": []}
    loss_hist = []
    best = {"pd": -1.0, "epoch": None, "scores": None}
    pbar = tqdm(range(1, epochs + 1), desc=label, dynamic_ncols=True, leave=False)
    for ep in pbar:
        model.train()
        perm = torch.randperm(N)
        tot = None; nb = 0
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot = loss.detach() if tot is None else tot + loss.detach()
            nb += 1
        ep_loss = float(tot.item()) / max(nb, 1)
        loss_hist.append(round(ep_loss, 4))
        if ep % snap_every == 0:
            sc = dsm_additive(test_planted, tr, model, s_raw)  # @no_grad inside
            au = _auc(labels, sc)
            pa = _pauc(labels, sc, cfg["pauc_fpr"])
            pd = _pd_at_fa(labels, sc, cfg["pfa"])
            rec["epochs"].append(ep); rec["auc"].append(au)
            rec["pauc"].append(pa);   rec["pd"].append(pd)
            rec["loss_at_snap"].append(ep_loss)
            if pd > best["pd"]:
                best.update(pd=pd, epoch=ep, scores=np.asarray(sc, np.float32))
            if ckpt_dir:
                torch.save({k: v.detach().cpu().half()
                            for k, v in model.state_dict().items()
                            if k.startswith("net")},
                           os.path.join(ckpt_dir, f"ep{ep:04d}.pt"))
            pbar.set_postfix(loss=f"{ep_loss:.1f}", pd=f"{pd:.3f}",
                             best=f"{best['pd']:.3f}@{best['epoch']}")
    rec["loss_hist"] = loss_hist
    rec["best_epoch"] = best["epoch"]
    rec["best_pd"] = best["pd"]
    return rec, best["scores"]


def run_sweep():
    """The sweep (resume-safe: rerun after any interrupt)."""
    t_start = time.time()
    for mode in MODES:
        cfg = yaml.safe_load(open(f"repro/configs/iid_{mode}.yaml"))
        cfg.update(dataset="repro/data/pavia-u.mat", device=DEVICE)
        hidden, act = dart_arch(cfg, mode)
        out_m = os.path.join(OUT, mode)
        os.makedirs(os.path.join(out_m, "scores_best"), exist_ok=True)
        met_path = os.path.join(out_m, "metrics.json")
        rec_all = json.load(open(met_path)) if os.path.exists(met_path) else {}
        rec_all["_meta"] = dict(mode=mode, rhos=RHOS, seeds=SEEDS, epochs=EPOCHS,
                                snap_every=SNAP_EVERY, hidden=list(hidden), act=act,
                                amplitude=float(cfg["amplitude"]),
                                pfa=float(cfg["pfa"]), batch=int(cfg["batch_size"]),
                                lr=float(cfg["lr"]), wd=float(cfg["weight_decay"]))
        for seed in SEEDS:
            pool, test_planted, labels, s_raw, n_cfg = build_mode_data(cfg, mode, seed)
            n_list = N_LIST or n_cfg
            np.savez_compressed(os.path.join(out_m, f"testset_s{seed}.npz"),
                                labels=labels.astype(np.int8))
            for n in n_list:
                tr = pool[:n]
                for rho in RHOS:
                    key = f"n{n}_rho{rho}_s{seed}"
                    if rec_all.get(key, {}).get("done"):
                        continue
                    ck = os.path.join(out_m, "ckpt", key) if SAVE_CKPTS else None
                    t0 = time.time()
                    r, best_scores = train_dart_snapshots(
                        tr, cfg, hidden, act, rho, seed, f"{mode} {key}",
                        test_planted, labels, s_raw, DEVICE, EPOCHS, SNAP_EVERY, ck)
                    r["done"] = True
                    r["sec"] = round(time.time() - t0, 1)
                    if best_scores is not None:
                        np.savez_compressed(
                            os.path.join(out_m, "scores_best", key + ".npz"),
                            scores=best_scores, labels=labels.astype(np.int8),
                            epoch=np.int32(r["best_epoch"]))
                    rec_all[key] = r
                    with open(met_path, "w") as f:
                        json.dump(rec_all, f)
                    print(f"[{mode}] {key}: best Pd={r['best_pd']:.3f} "
                          f"@ep{r['best_epoch']} ({r['sec']}s)", flush=True)
    print(f"TOTAL {(time.time() - t_start) / 3600:.2f} h")


# ----------------- envelope analysis + figures + summary.md -----------------
def assemble(mode):
    """Z[metric][i_n, i_r, i_s, i_t] over snapshots; plus loss curves."""
    rec = json.load(open(os.path.join(OUT, mode, "metrics.json")))
    meta = rec["_meta"]
    rhos, seeds = meta["rhos"], meta["seeds"]
    n_list = sorted({int(k.split("_")[0][1:]) for k in rec if k != "_meta"})
    snaps = None
    Z = {}
    for m in ("pd", "auc", "pauc"):
        Z[m] = np.full((len(n_list), len(rhos), len(seeds), 0), np.nan)
    L = {}
    for i, n in enumerate(n_list):
        for j, r in enumerate(rhos):
            for k, s in enumerate(seeds):
                rr = rec.get(f"n{n}_rho{r}_s{s}")
                if not rr or not rr.get("done"):
                    continue
                if snaps is None:
                    snaps = rr["epochs"]
                    for m in Z:
                        Z[m] = np.full((len(n_list), len(rhos), len(seeds),
                                        len(snaps)), np.nan)
                for m in Z:
                    Z[m][i, j, k] = rr[m]
                L[(n, r, s)] = rr["loss_hist"]
    return meta, n_list, rhos, seeds, np.asarray(snaps), Z, L


def published_rule_line(Z, L, n_list, rhos, seeds, snaps, budget=2000):
    """Replicate the published selection (best train loss over epochs %100==0,
    within `budget` epochs) at rho=0.1 — must land on the published DART line."""
    j = rhos.index(0.1)
    snap_set = set(int(e) for e in snaps)
    sel = np.full((len(n_list), len(seeds)), np.nan)
    for i, n in enumerate(n_list):
        for k, s in enumerate(seeds):
            lh = L.get((n, 0.1, s))
            if lh is None:
                continue
            ck_eps = [e for e in range(100, budget + 1, 100)
                      if e <= len(lh) and e in snap_set]
            if not ck_eps:
                continue
            e_star = min(ck_eps, key=lambda e: lh[e - 1])
            t = int(np.where(snaps == e_star)[0][0])
            sel[i, k] = Z["pd"][i, j, k, t]
    return np.nanmean(sel, axis=1)


def analyze():
    FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
    summary_lines = ["# DART rho x n x epoch envelope — summary\n"]

    for mode in MODES:
        meta, n_list, rhos, seeds, snaps, Z, L = assemble(mode)
        P = Z["pd"]
        ref_n = np.asarray(REF[mode]["n"], float)

        # --- the two envelope notions ---
        # (a) per-seed oracle: max over (rho, epoch) per seed, then mean -> BOUND
        per_seed_max = np.nanmax(
            P.transpose(0, 2, 1, 3).reshape(P.shape[0], P.shape[2], -1), axis=2)
        oracle_mean = np.nanmean(per_seed_max, axis=1)
        oracle_std = np.nanstd(per_seed_max, axis=1)
        # (b) shared-config: mean over seeds first, then max over (rho, epoch)
        Pm = np.nanmean(P, axis=2)                     # (n, rho, t)
        flat = Pm.reshape(Pm.shape[0], -1)
        shared = np.nanmax(flat, axis=1)
        arg = np.nanargmax(flat, axis=1)
        best_rho = [rhos[a // Pm.shape[2]] for a in arg]
        best_ep = [int(snaps[a % Pm.shape[2]]) for a in arg]

        # published-rule sanity replica (rho=0.1, budget 2000) + budget 3000
        rule2000 = published_rule_line(Z, L, n_list, rhos, seeds, snaps, budget=2000)
        rule3000 = published_rule_line(Z, L, n_list, rhos, seeds, snaps, budget=3000)

        # ---------- figure 1: the envelope vs published lines ----------
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.fill_between(n_list, oracle_mean - oracle_std, oracle_mean + oracle_std,
                        alpha=0.15, color="crimson")
        ax.plot(n_list, oracle_mean, "o--", color="crimson", lw=2,
                label=r"DART bound: oracle over $(\rho,$ epoch$)$ per seed")
        ax.plot(n_list, shared, "s-", color="darkred", lw=2,
                label=r"DART: best shared $(\rho^{*},$ ep$^{*})$ per $n$")
        ax.plot(n_list, rule2000, ":", color="gray", lw=1.5,
                label="sanity: published rule replicated")
        ax.plot(n_list, rule3000, "-.", color="dimgray", lw=1.2,
                label=r"published rule, 3000-ep budget, $\rho=0.1$")
        styles = {"DART": ("tab:red", "-"), "L-DART": ("tab:orange", "-"),
                  "LRao": ("tab:green", "-"), "L-LRao": ("tab:olive", "--"),
                  "AMF": ("tab:blue", "-"), "GMM-Levin": ("tab:purple", "-")}
        for det, pd_line in REF[mode]["pd"].items():
            c, ls = styles.get(det, ("k", "-"))
            ax.plot(ref_n, np.asarray(pd_line, float), ls, color=c, alpha=0.85,
                    lw=1.4, label=f"{det} (published)")
        ax.set_xscale("log"); ax.set_xticks(n_list)
        ax.set_xticklabels(n_list); ax.minorticks_off()
        ax.set_xlabel("n (training pixels)")
        ax.set_ylabel(r"$P_d$ @ $P_{fa}=0.1$")
        ax.set_title(f"IID {mode}: DART envelope over " r"$\rho \times$ epoch"
                     f"  ({len(seeds)} seeds)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=2)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"envelope_{mode}.{ext}"), dpi=200)
        plt.close(fig)

        # ---------- figure 2: best rho / best epoch per n ----------
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
        axes[0].plot(n_list, best_rho, "o-")
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel("n"); axes[0].set_ylabel(r"best $\rho$")
        axes[0].axhline(0.1, color="gray", ls=":", label=r"published $\rho=0.1$")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        axes[1].plot(n_list, best_ep, "s-")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("n"); axes[1].set_ylabel("best epoch")
        axes[1].axhline(2000, color="gray", ls=":", label="published budget")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.suptitle(f"IID {mode}: shared-config argmax per n")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"best_rho_epoch_{mode}.{ext}"), dpi=200)
        plt.close(fig)

        # ---------- figure 3: Pd(rho, epoch) heatmaps per n ----------
        ncol = 4; nrow = int(np.ceil(len(n_list) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.6 * nrow),
                                 squeeze=False)
        for i, n in enumerate(n_list):
            a = axes[i // ncol][i % ncol]
            im = a.imshow(Pm[i], aspect="auto", origin="lower", cmap="viridis",
                          extent=[snaps[0], snaps[-1], -0.5, len(rhos) - 0.5])
            a.set_yticks(range(len(rhos))); a.set_yticklabels(rhos, fontsize=7)
            a.set_title(f"n={n}", fontsize=9)
            a.set_xlabel("epoch", fontsize=7)
            plt.colorbar(im, ax=a, fraction=0.046)
        for i in range(len(n_list), nrow * ncol):
            axes[i // ncol][i % ncol].axis("off")
        fig.suptitle(f"IID {mode}: mean Pd@0.1 over seeds — rho (y) x epoch (x)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"heatmaps_{mode}.{ext}"), dpi=200)
        plt.close(fig)

        # ---------- summary table ----------
        ref_pd = {d: np.asarray(v, float) for d, v in REF[mode]["pd"].items()}
        summary_lines.append(f"\n## {mode}\n")
        summary_lines.append("| n | oracle bound | shared best | rho* | ep* | "
                             "published DART | LRao | bound - LRao |")
        summary_lines.append("|---|---|---|---|---|---|---|---|")
        for i, n in enumerate(n_list):
            ri = list(ref_n).index(n) if n in ref_n else None
            pub_d = ref_pd["DART"][ri] if ri is not None else np.nan
            pub_l = ref_pd["LRao"][ri] if ri is not None else np.nan
            summary_lines.append(
                f"| {n} | {oracle_mean[i]:.3f}±{oracle_std[i]:.3f} | {shared[i]:.3f} "
                f"| {best_rho[i]} | {best_ep[i]} | {pub_d:.3f} | {pub_l:.3f} "
                f"| {oracle_mean[i] - pub_l:+.3f} |")
        # automatic findings
        low = [i for i, n in enumerate(n_list) if n <= 100 and n in ref_n]
        if low:
            gaps = [oracle_mean[i] - ref_pd["LRao"][list(ref_n).index(n_list[i])]
                    for i in low]
            beats = sum(g > 0 for g in gaps)
            summary_lines.append(
                f"\n- Low-sample regime (n<=100): the DART envelope beats published "
                f"LRao at {beats}/{len(low)} points "
                f"(mean gap {np.mean(gaps):+.3f}). If positive, LRao's low-n win "
                f"is a model-selection / training-budget artifact, not an information gap; "
                f"if negative, LRao's objective genuinely extracts more.")
        common = [i for i, n in enumerate(n_list) if n in ref_n]
        if common:
            ref_dart_c = np.array([ref_pd["DART"][list(ref_n).index(n_list[i])]
                                   for i in common])
            gap = np.nanmax(np.abs(rule2000[common] - ref_dart_c))
            summary_lines.append(
                f"- Sanity: replicated published rule vs published DART line — "
                f"max abs gap {gap:.3f} (should be ~0).")

    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))


def low_n_dynamics():
    """Low-n training dynamics (why does LRao win at low n?)."""
    FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
    for mode in MODES:
        meta, n_list, rhos, seeds, snaps, Z, L = assemble(mode)
        Pm = np.nanmean(Z["pd"], axis=2)
        ref_n = list(REF[mode]["n"])
        show_n = [n for n in n_list if n <= 200]
        fig, axes = plt.subplots(1, len(show_n), figsize=(3.2 * len(show_n), 3.4),
                                 squeeze=False)
        for i, n in enumerate(show_n):
            a = axes[0][i]
            for j, r in enumerate(rhos):
                a.plot(snaps, Pm[n_list.index(n), j], lw=1.3, label=f"ρ={r}")
            if n in ref_n:
                ri = ref_n.index(n)
                lr_ref = REF[mode]["pd"]["LRao"][ri]
                da_ref = REF[mode]["pd"]["DART"][ri]
                if lr_ref is not None:
                    a.axhline(lr_ref, color="tab:green", ls="--", lw=1.2,
                              label="LRao (published)")
                if da_ref is not None:
                    a.axhline(da_ref, color="tab:red", ls=":", lw=1.2,
                              label="DART (published)")
            a.set_title(f"n={n}", fontsize=10)
            a.set_xlabel("epoch"); a.grid(alpha=0.3)
            if i == 0:
                a.set_ylabel("Pd@0.1 (mean over seeds)")
                a.legend(fontsize=6.5)
        fig.suptitle(f"IID {mode}: DART detection vs training epoch at low n")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"low_n_dynamics_{mode}.{ext}"), dpi=200)
        plt.close(fig)


def dynamics_all_n():
    """Detection vs epoch, ALL n."""
    for mode in ("single", "multi"):
        path = os.path.join(OUT, mode, "metrics.json")
        if not os.path.exists(path):
            print("skip", mode); continue
        rec = json.load(open(path))
        meta = rec["_meta"]; rhos, seeds = meta["rhos"], meta["seeds"]
        n_list = sorted({int(k.split("_")[0][1:]) for k in rec if k != "_meta"})
        snaps, P = None, None
        for i, n in enumerate(n_list):
            for j, r in enumerate(rhos):
                for k, s in enumerate(seeds):
                    rr = rec.get(f"n{n}_rho{r}_s{s}")
                    if not rr:
                        continue
                    if snaps is None:
                        snaps = np.asarray(rr["epochs"])
                        P = np.full((len(n_list), len(rhos), len(seeds),
                                     len(snaps)), np.nan)
                    P[i, j, k] = rr["pd"]
        Pm = np.nanmean(P, axis=2)
        ref = {"n": REF[mode]["n"], "DART": REF[mode]["pd"]["DART"],
               "LRao": REF[mode]["pd"]["LRao"]}
        ncol = 4; nrow = int(np.ceil(len(n_list) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                                 squeeze=False)
        for i, n in enumerate(n_list):
            a = axes[i // ncol][i % ncol]
            for j, r in enumerate(rhos):
                a.plot(snaps, Pm[i, j], lw=1.2, label=f"rho={r}")
            if n in ref["n"]:
                ri = ref["n"].index(n)
                a.axhline(ref["LRao"][ri], color="tab:green", ls="--", lw=1.1,
                          label="LRao (published)")
                a.axhline(ref["DART"][ri], color="tab:red", ls=":", lw=1.1,
                          label="DART (published)")
            a.set_title(f"n={n}", fontsize=10); a.grid(alpha=0.3)
            a.set_xlabel("epoch", fontsize=8)
            if i % ncol == 0:
                a.set_ylabel("Pd@0.1 (mean over seeds)", fontsize=8)
            if i == 0:
                a.legend(fontsize=6)
        for i in range(len(n_list), nrow * ncol):
            axes[i // ncol][i % ncol].axis("off")
        fig.suptitle(f"IID {mode}: DART detection vs training epoch — all n")
        fig.tight_layout()
        FIG_DIR = os.path.join(OUT, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, f"dynamics_all_n_{mode}.png")
        fig.savefig(out, dpi=200)
        fig.savefig(out.replace(".png", ".pdf"))
        plt.close(fig)


def package():
    """Package everything (light = no checkpoints; full = all)."""
    import zipfile

    def _zip(path, root, skip=None):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for r, _, files in os.walk(root):
                if skip and skip in r.split(os.sep):
                    continue
                for fn in files:
                    z.write(os.path.join(r, fn))

    _zip("rho_envelope_light.zip", "results/rho_envelope", skip="ckpt")
    _zip("rho_envelope_full.zip", "results/rho_envelope")
    print("zipped -> rho_envelope_light.zip, rho_envelope_full.zip")
    import glob
    figs = sorted(glob.glob("results/rho_envelope/figures/*.png"))
    for p in figs:
        print(p)


def main():
    run_sweep()
    analyze()
    low_n_dynamics()
    dynamics_all_n()
    package()


if __name__ == "__main__":
    main()
