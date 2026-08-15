"""paper_protocol.py — the paper's Table-1 spatial protocol, self-contained for Colab.

Scene: Pavia University, scenario 4 of the paper (train box [85,193,207,306],
test box [419,508,250,334]); training pixels = contiguous centered side-crop of
the train box to ~4000 px (n_budget=4000, the PAPER value); prior signature =
foreign class 7 (bitumen): global class mean scaled to the mean test-patch pixel
norm; planting: additive theta in {0.075, 0.15, 0.225} into 10% of test pixels
(edge_guard=3) + replacement theta=0.95 (strong); seeds 42-46.

Deep baselines TRAIN exactly as in their papers (their own sample construction
and losses) with the SAME inputs our detectors get: the prior signature + the
target-free secondary pixels. Only the TEST regime (weak subpixel targets)
differs from their papers.
"""

import os, json, zipfile, urllib.request
import numpy as np
import scipy.io as sio

TRAIN_BOX = [85, 193, 207, 306]
TEST_BOX = [419, 508, 250, 334]
FOREIGN_CLS = 7
N_BUDGET = 4000
THETAS_ADD = [0.075, 0.15, 0.225]
THETA_REPL = 0.95
SEEDS = [42, 43, 44, 45, 46]
EDGE_GUARD = 3
TGT_FRACTION = 0.10

EHU = "https://www.ehu.eus/ccwintco/uploads"
PAVIA_URLS = [f"{EHU}/e/ee/PaviaU.mat", f"{EHU}/5/50/PaviaU_gt.mat"]


# --------------------------------------------------------------------------- #
def get_pavia(data_dir="data_dl"):
    """Return (data HxWxD float64 raw radiances, gt HxW int). Uses a local
    pavia-u.mat (fields data/map) when available, else downloads the public
    EHU PaviaU (with a browser User-Agent; EHU 403s the urllib default)."""
    for local in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pavia-u.mat"),
                  "repro/data/pavia-u.mat", "pavia-u.mat"):
        if os.path.exists(local):
            m = sio.loadmat(local)
            if "data" in m and "map" in m:
                return m["data"].astype(np.float64), m["map"].astype(int)
    os.makedirs(data_dir, exist_ok=True)
    paths = []
    for url in PAVIA_URLS:
        p = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(p):
            print("downloading", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(p, "wb") as f:
                f.write(r.read())
        paths.append(p)
    data = sio.loadmat(paths[0])["paviaU"].astype(np.float64)
    gt = sio.loadmat(paths[1])["paviaU_gt"].astype(int)
    return data, gt


def side_crop(box, budget=N_BUDGET):
    r0, r1, c0, c1 = box
    H, W = r1 - r0, c1 - c0
    if not budget or H * W <= budget:
        return list(box)
    s = (budget / (H * W)) ** 0.5
    nH, nW = max(int(round(H * s)), 1), max(int(round(W * s)), 1)
    dr, dc = (H - nH) // 2, (W - nW) // 2
    return [r0 + dr, r0 + dr + nH, c0 + dc, c0 + dc + nW]


def crop(data, box):
    r0, r1, c0, c1 = box
    return data[r0:r1, c0:c1].reshape(-1, data.shape[-1])


def foreign_signature(data, gt, te_pix, cls=FOREIGN_CLS):
    D = data.shape[-1]
    mu = data.reshape(-1, D)[gt.ravel() == cls].mean(axis=0)
    scalar = float(np.linalg.norm(te_pix, axis=1).mean())
    return (mu / (np.linalg.norm(mu) + 1e-12) * scalar)


def plant_targets(test_bkg, s, amplitude, tgt_fraction, model="additive",
                  seed=0, spatial_shape=None, edge_guard=0):
    """Portable re-implementation of the paper planting (random cells, additive
    y=w+theta*s or replacement y=(1-theta)*w+theta*s, optional edge guard)."""
    rng = np.random.RandomState(seed)
    N = len(test_bkg)
    n_t = int(round(tgt_fraction * N))
    if spatial_shape is not None and edge_guard > 0:
        H, W = spatial_shape
        rows, cols = np.unravel_index(np.arange(N), (H, W))
        ok = ((rows >= edge_guard) & (rows < H - edge_guard) &
              (cols >= edge_guard) & (cols < W - edge_guard))
        pool = np.where(ok)[0]
    else:
        pool = np.arange(N)
    idx = rng.choice(pool, size=min(n_t, len(pool)), replace=False)
    planted = test_bkg.copy()
    if model == "additive":
        planted[idx] = planted[idx] + amplitude * s[None, :]
    else:
        planted[idx] = (1 - amplitude) * planted[idx] + amplitude * s[None, :]
    labels = np.zeros(N, dtype=np.int8); labels[idx] = 1
    return planted, labels


def auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------------- #
def build_protocol():
    """Returns dict with tr (secondary pixels), te (clean test pixels),
    te_shape, sig."""
    data, gt = get_pavia()
    tb = side_crop(TRAIN_BOX)
    tr = crop(data, tb)
    te = crop(data, TEST_BOX)
    te_shape = (TEST_BOX[1] - TEST_BOX[0], TEST_BOX[3] - TEST_BOX[2])
    sig = foreign_signature(data, gt, te)
    print(f"protocol: train {tb}={len(tr)}px  test {len(te)}px  ||s||={np.linalg.norm(sig):.4g}",
          flush=True)
    return dict(tr=tr, te=te, te_shape=te_shape, sig=sig, data=data, gt=gt)


def eval_cells(score_fn, proto, seeds=SEEDS, out_dir="results_deep", tag="model"):
    """score_fn(planted_pixels) -> scores. Evaluates all planted cells for one
    trained model instance (call once per seed with that seed's model)."""
    raise NotImplementedError  # per-model runners below drive this inline


def run_detector(name, fit_fn, score_fn_builder, proto, seeds=SEEDS,
                 out_dir="results_deep", ckpt_dir="ckpt_deep"):
    """Generic runner: per seed -> fit (with checkpoint resume) -> score all
    planted cells (additive theta sweep + replacement strong) -> archive.

    fit_fn(tr, sig, seed, ckpt_path) -> model_state (trains or resumes)
    score_fn_builder(model_state, proto, seed) -> callable(planted)->scores
    """
    os.makedirs(out_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    results = {}
    for seed in seeds:
        ck = os.path.join(ckpt_dir, f"{name}_seed{seed}.pt")
        state = fit_fn(proto["tr"], proto["sig"], seed, ck)
        score = score_fn_builder(state, proto, seed)
        cells = [("additive", th) for th in THETAS_ADD] + [("replacement", THETA_REPL)]
        for model, th in cells:
            pl, lab = plant_targets(proto["te"], proto["sig"], th, TGT_FRACTION,
                                    model=model, seed=seed,
                                    spatial_shape=proto["te_shape"],
                                    edge_guard=EDGE_GUARD)
            sc = np.asarray(score(pl), np.float32)
            a = auc(lab, sc)
            results.setdefault(f"{model}|{th}", []).append(a)
            np.savez_compressed(os.path.join(
                out_dir, f"scores_{name}_seed{seed}_{model}_{th}.npz"),
                scores=sc, labels=lab)
            print(f"[{name}] seed{seed} {model} th={th}: AUC={a:.4f}", flush=True)
    with open(os.path.join(out_dir, f"{name}_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n=== {name} summary (mean+/-std over {len(seeds)} seeds) ===")
    for k, v in results.items():
        print(f"  {k}: {np.mean(v):.3f}+/-{np.std(v):.3f}")
    return results


def zip_and_download(dirs=("results_deep", "ckpt_deep"), zip_name="deep_baselines_results.zip"):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    z.write(os.path.join(root, fn))
    print("zipped ->", zip_name, flush=True)
    try:
        from google.colab import files
        files.download(zip_name)
    except Exception:
        print("(not on Colab - zip left on disk)")


# --------------------------------------------------------------------------- #
# San Diego validation helpers (real full-pixel targets - the models' regime)
# --------------------------------------------------------------------------- #
def get_sandiego(path="colab_deep/data/sandiego.mat"):
    m = sio.loadmat(path)
    data = m["data"].astype(np.float64)          # (100,100,189)
    gtmap = (m["map"] > 0).astype(int)           # (100,100)
    return data, gtmap


def sandiego_validation_sets(n_known=5, seed=0):
    """Known targets = n_known random GT target pixels (their papers' setting);
    detect over the whole image; AUC against the full GT map."""
    data, gtmap = get_sandiego()
    D = data.shape[-1]
    X = data.reshape(-1, D)
    y = gtmap.ravel()
    rng = np.random.default_rng(seed)
    tgt_idx = np.where(y == 1)[0]
    known = X[rng.choice(tgt_idx, n_known, replace=False)]
    sig = known.mean(axis=0)
    return X, y, sig, known
