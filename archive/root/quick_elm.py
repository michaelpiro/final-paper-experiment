"""
Quick ELM normalization test — 1 seed, n=2000.

Empirical Line Method (ELM): per-band linear map DN→[0,1] using a dark and
a bright reference.  Without actual calibration panels we approximate:
  dark_ref[k]   = 1st  percentile of background pixels in band k
  bright_ref[k] = 99th percentile of background pixels in band k

This is more robust than per-band min-max (absolute extremes, sensitive to
outliers) and closer to what real ELM would give if calibration panels were
placed at those spectral levels.

If AUC differences vs minmax are negligible, it confirms our claim that the
detector is invariant to the normalization choice (Proposition 1).

Compare: per_band_std | per_band_minmax | elm (1st/99th pct)
Models:  additive + replacement DSM + AMF reference
Setup:   single-class (bkg=2, tgt=1), d=5, n=2000, 1 seed, θ=0.15
"""
import numpy as np, torch, sys, scipy.io
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '.'); sys.path.insert(0, 'experiments/honest_pipeline')
from dsm_model import ScoreNet, dsm_loss, compute_scores
from final_paper_experiments.data_utils import compute_sigma_from_data
from pipeline import HonestDetectionPipeline, amf_score, amf_replacement_score

# ── config ───────────────────────────────────────────────────────────────────
D, N, SEED, RHO, EP, LR = 5, 2000, 42, 0.001, 2000, 5e-4
BKG_CLS, TGT_CLS, AMP = 2, 1, 0.15

# ── data ─────────────────────────────────────────────────────────────────────
mat  = scipy.io.loadmat('data/pavia-u.mat')
flat = mat['data'].astype(np.float32).reshape(-1, 103)
gt   = mat['map'].astype(int).reshape(-1)

bkg = flat[gt == BKG_CLS]
t   = flat[gt == TGT_CLS].mean(0).astype(np.float32)  # raw class-mean, no ℓ2-norm

rng    = np.random.default_rng(SEED)
idx    = rng.permutation(len(bkg))
tr_raw = bkg[idx[:N]]
te_raw = bkg[idx[N:N+2000]]
pos    = rng.choice(2000, 200, replace=False)
lab    = np.zeros(2000, dtype=int); lab[pos] = 1
te_add = te_raw.copy(); te_add[pos] += AMP * t
te_rep = te_raw.copy(); te_rep[pos]  = (1-AMP)*te_raw[pos] + AMP*t


# ── helpers ───────────────────────────────────────────────────────────────────
def train_dsm(tr, sigma):
    torch.manual_seed(SEED)
    m = ScoreNet(D, [64, 64], 'silu')
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    X = torch.tensor(tr, dtype=torch.float32)
    Nb, bs = len(X), 256
    for _ in range(EP):
        p = torch.randperm(Nb)
        for i in range(0, Nb, bs):
            b = X[p[i:i+bs]]
            L = dsm_loss(m, b, sigma)
            opt.zero_grad(); L.backward(); opt.step()
    m.eval(); return m


def auc_add(model, tr, te, s):
    zt = compute_scores(model, tr); ze = compute_scores(model, te)
    zb = zt.mean(0); C = np.cov(zt, rowvar=False)
    if C.ndim == 0: C = np.array([[float(C)]])
    norm = float(np.sqrt(max(float(s @ C @ s), 1e-12)))
    return roc_auc_score(lab, -((ze - zb) @ s) / norm)


def auc_rep(model, tr, te, s_rep):
    pt = compute_scores(model, tr); pe = compute_scores(model, te)
    pb = pt.mean(0)
    r_tr = ((pt - pb) * (tr - s_rep)).sum(1)
    r_bar, r_std = r_tr.mean(), r_tr.std() + 1e-12
    r_te = ((pe - pb) * (te - s_rep)).sum(1)
    return roc_auc_score(lab, (r_te - r_bar) / r_std)


def run_norm(label, norm_mode):
    pipe = HonestDetectionPipeline(D, norm_mode).fit(tr_raw)
    tr  = pipe.project(tr_raw)
    tae = pipe.project(te_add)
    tre = pipe.project(te_rep)
    sa  = pipe.signature_additive(t)
    sr  = pipe.signature_replacement(t)
    sig = compute_sigma_from_data(tr, RHO)

    print(f"\n[{label}]  sigma={sig:.5f}")
    m = train_dsm(tr, sig)
    dsm_a = auc_add(m, tr, tae, sa)
    dsm_r = auc_rep(m, tr, tre, sr)
    amf_a = roc_auc_score(lab, amf_score(tr, tae, sa))
    amf_r = roc_auc_score(lab, amf_replacement_score(tr, tre, sr))
    print(f"  DSM add={dsm_a:.4f}  rep={dsm_r:.4f}")
    print(f"  AMF add={amf_a:.4f}  rep={amf_r:.4f}  (reference)")
    return dsm_a, dsm_r, amf_a, amf_r


# ── run ───────────────────────────────────────────────────────────────────────
print("="*60)
print(f"ELM normalization quick test")
print(f"bkg=cls{BKG_CLS}  tgt=cls{TGT_CLS}  d={D}  n={N}  seed={SEED}  θ={AMP}")
print("="*60)

results = {}
for label, mode in [('per_band_std',    'per_band_std'),
                    ('per_band_minmax',  'per_band_minmax'),
                    ('ELM (p1/p99)',     'elm')]:
    results[label] = run_norm(label, mode)

print("\n" + "="*60)
print("Summary  (DSM add | DSM rep | AMF add | AMF rep)")
print("="*60)
for label, (da, dr, aa, ar) in results.items():
    print(f"  {label:<22} {da:.4f}   {dr:.4f}   {aa:.4f}   {ar:.4f}")

base_da, base_dr = results['per_band_minmax'][0], results['per_band_minmax'][1]
elm_da,  elm_dr  = results['ELM (p1/p99)'][0],   results['ELM (p1/p99)'][1]
print(f"\n  Δ ELM vs minmax  add={elm_da-base_da:+.4f}  rep={elm_dr-base_dr:+.4f}")
