# Local runner archive (NOT standalone-runnable)

These scripts ran on the local machine from the `pythonProject` cwd against
the sibling `camera_ready/diagnostics/` tree; several import from sibling
experiment dirs (`kfree_wmw/run_kfree.py`, `zcainit_ladder/...`) that are not
part of this repo. They are archived here as the exact provenance of the
results JSONs in `../results/`. The RUNNABLE form of every recipe is the
three notebooks at the repo root (see `../RECIPE.md`).

---

# NOISE-BEFORE-FRONT program — consolidated atlas (2026-08-11/12)

Standing user directive: DSM noise added BEFORE the front (raw isotropic,
sigma^2 = rho * mean band variance); best epoch by DETECTION (curves every
50-100 ep saved in every JSON). All runs 3 seeds unless noted; IID budget
15000 ep; DARTS 3000-5000 ep. Fronts: std (per-band), lwdiag (LW-diag
scales — ties std at n=2048, alpha=.005), stdlin (std -> trainable
Linear(D,D) init=I, trained jointly — the convention's key discovery:
noise-through-the-layer makes the learnable front WORK, unlike
noise-after where it was provably destructive).

## Best recipes per setting (this convention)

| setting | recipe | best | vs noise-after champion |
|---|---|---|---|
| multi IID | **stdlin, rho=.1** | **.817** (std rho=.003: .797) | WINS (.776 wmw) |
| single IID | std, rho=.01 | .745 | loses (.885 BIC-gate ZCA) |
| pavia4 DART | std, rho=.003 | .769 AUC | ~ties (.793 pub) |
| pavia4 DARTS | **std+lin graft, rho=.03, 5000ep** | **.908/.547** | ties best-ever (.913 gmm) |
| sandiego DART | std, rho=.003 | .974/.936 | ties (.982) |
| sandiego DARTS | std, rho=.1 | .996/.989 | ties (.998) |
| sandiego2 DART | std, rho=.1 | .839/.375 | loses (.917 gmm) |
| sandiego2 DARTS | std, rho=.01 | .922/.678 | loses (.974 gmm) |

## n-sweep (paper grid, best of rho {.01,.1}, Pd@0.1)

multi std: .58/.60/.63/.71/.70/.73/.79/.78 — DOUBLE the published DART
(.29..0.59) at every n. single std: .68/.79/.71/.85/.79/.71/.79/.79 —
beats published at n<=1000, loses at 2000 (.79 vs .873).

## Key laws of this convention (all measured)

1. rho-epoch coupling: best epoch scales ~1/rho (rho=.3 peaks ~600 ep,
   rho=.003 ~13-15k); epoch-by-detection selection is MANDATORY (peaks
   decay .03-.05 by final epoch).
2. Off-diagonal whitening always hurts: zca/wmw collapse; fixed-alpha
   correlation blends monotone-worse (alpha .7->.99: .70->.78 < std .79);
   LW's own alpha (.005) = near-full decorrelation = collapse. Only the
   LEARNED off-diagonals (stdlin) help, and only on mixtures (multi
   +.02, pavia4-DARTS +.05); stdlin does NOT help single/SDs.
3. lwdiag == std at n=2048 (LW alpha .005); would differentiate small-n.
4. sd2 is the convention's weak spot (-.05..-.08 vs noise-after gmm).

## Convention verdict (for the paper)

Noise-before (isotropic, physically clean): wins multi decisively
(.817 vs .776), ties pavia4-DARTS/SD1, loses single (-.14) and sd2
(-.05..-.08). Noise-after (= covariance-shaped KDE bandwidth): the WMW/
GMM-Sigma_w champions. The conventions are a genuine trade, not a
dominance; the noise-before story pairs with best-epoch selection and a
learnable linear on mixtures.

Files: results_nb_rho.json (multi 6k), results_nb_rho_sp.json,
results_nb_long{,30}.json, results_nb_extras.json (stdlw/stdlin),
results_nb_lwdiag.json, results_nb_lwalpha.json (fixed-alpha diag),
results_nb_alpha.json (fixed-alpha corr), results_nb_darts_1000ep.json,
results_nb_darts_lin.json (pavia4 graft), results_ov_{pavia_rho,pavia_n,
sd_dart,sd_darts}.json (overnight), fig_nb_std_curves.*, runners run_*.py.
