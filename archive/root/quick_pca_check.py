"""
PCA contamination check.

Question: does planting a fraction of target pixels into the PCA fitting set
improve rho_d (the fraction of target-direction energy captured by the top-d
PCA subspace)?

We compare:
  - PCA on background only (current baseline)
  - PCA on background + f% target pixels, for f in {5, 10, 20, 50}

Only PCA is analysed here — no DSM training.
Metrics:
  rho_d        : fraction of AMF deflection captured in top-d dims (higher = better)
  cos(v1, t_n) : cosine between top PC and the normalized target direction
  eigenvalue spectrum: how much variance is in each PC
"""
import numpy as np, scipy.io, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments/honest_pipeline')
from pipeline import HonestDetectionPipeline

# ── config ────────────────────────────────────────────────────────────────────
D      = 5
N_TR   = 2000
SEED   = 42
BKG    = None   # None = all non-target classes (multi-class)
TGT    = 1
FRACS  = [0.0, 0.05, 0.10, 0.20, 0.50]   # fraction of target pixels to mix in

# ── data ──────────────────────────────────────────────────────────────────────
mat  = scipy.io.loadmat('data/pavia-u.mat')
flat = mat['data'].astype(np.float32).reshape(-1, 103)
gt   = mat['map'].astype(int).reshape(-1)

bkg_all = flat[(gt != 0) & (gt != TGT)] if BKG is None else flat[gt == BKG]
tgt_all = flat[gt == TGT]
bkg_desc = 'all-non-target' if BKG is None else f'cls{BKG}'
t_raw   = tgt_all.mean(0).astype(np.float32)

rng    = np.random.default_rng(SEED)
idx    = rng.permutation(len(bkg_all))
tr_bkg = bkg_all[idx[:N_TR]]

print(f"Background train pixels : {N_TR}")
print(f"Target class pixels     : {len(tgt_all)}")
print(f"Target signature norm   : {np.linalg.norm(t_raw):.1f}")
print(f"d={D}\n")
print(f"{'Contamination':>15}  {'rho_d':>7}  {'cos(v1,t)':>10}  "
      f"{'lambda_1':>10}  {'lambda_d':>10}  {'lambda_1/sum':>12}")
print("-"*70)

for frac in FRACS:
    # build training set for PCA
    n_tgt = int(N_TR * frac / (1 - frac + 1e-9))   # so tgt/(bkg+tgt) = frac
    n_tgt = min(n_tgt, len(tgt_all))
    t_idx = rng.choice(len(tgt_all), n_tgt, replace=False)
    if n_tgt > 0:
        pca_train = np.vstack([tr_bkg, tgt_all[t_idx]])
    else:
        pca_train = tr_bkg

    # fit pipeline on this (possibly contaminated) set
    pipe = HonestDetectionPipeline(D, 'per_band_minmax').fit(pca_train)

    # rho_d: fraction of AMF deflection in top-d subspace
    rho_d = pipe.rho_d(t_raw)

    # cosine between top PC and target direction in normalized space
    t_n   = (t_raw * pipe.A).astype(np.float32)
    t_n   = t_n / (np.linalg.norm(t_n) + 1e-12)
    cos_v1 = abs(float(pipe.eigvecs[:, 0] @ t_n))

    lam = pipe.eigvals
    print(f"{frac:>14.0%}  {rho_d:>7.4f}  {cos_v1:>10.4f}  "
          f"{lam[0]:>10.4f}  {lam[D-1]:>10.4f}  {lam[0]/lam.sum():>12.4f}")

print()
print("rho_d   : fraction of target-direction AMF energy in top-d PCA dims")
print("cos(v1) : alignment between 1st PC and target direction (in norm. space)")
print("Ideally rho_d → 1 so no detection power is lost by PCA projection.")
