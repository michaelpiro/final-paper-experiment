# The NOISE-BEFORE recipe — camera-ready version (2026-08-13)

This branch (`camera-ready`) packages the updated DART/DARTS/LRao recipes
developed 2026-08-11..13 under the **noise-before** convention, together
with the experiments that froze each choice. The published-paper recipes
remain untouched on branch `rebuttal`; the `repro/` package is identical
on both branches (the notebooks may clone either).

Runnable form: the three notebooks at the repo root (below).
Archival form: `noise_before/local_runners/` (scripts as run locally).
Data: `noise_before/results/` (all result JSONs with full curves + figures).

---

## 1. The convention

DSM noise is added **before** the whitening front, isotropic in RAW space:

    sigma^2 = rho * mean(per-band variance of the training pool)
    loss    = || psi(x + eps) + eps / sigma^2 ||^2 ,   eps ~ N(0, sigma^2 I)

`psi` is the data-space score (`ScoreNet.forward` = W^T net(W(x-mu))).
Motivation: the paper states the DSM noise is isotropic; adding it after
the front makes it non-isotropic in raw space. Under this convention the
front ordering flips relative to the published (noise-after) results.

## 2. The front

**Per-band std** (mean-center, divide by per-band std) is the default
front everywhere. Measured hierarchy under noise-before:

- std: the workhorse (all tables below).
- robust median/MAD: +.04 on multi-IID only (mixture protection);
  loses on single and pavia4. Optional.
- ZCA / WMW (off-diagonal whitening): **collapse** (raw isotropic noise is
  amplified by 1/lambda along stretched directions — multi@2048 ZCA tops
  out ~.50 even at 50k epochs with every optimizer aid). Exception:
  near-unimodal scenes (pavia4 metal cell: ZCA .857) — mixture geometry
  is what off-diagonal whitening destroys.
- learned linear front: works ONLY stacked on a frozen std front
  ("stdlin", +.02 multi / +.05 pavia4-DARTS); alone (any init) it falls
  into the trivial zero-score minimum. The frozen W^T back-map is
  load-bearing.

## 3. Optimizer law (DART / score nets)

    Adam, lr 5e-4, batch 512, weight decay = 0, grad clip = 1.0
    NO other clipping of any kind (no eigen floors, no MAD floors,
    no statistic guards) — user directive, enforced in all runners.

Measured ablation (multi@2048, rho=.003, 50k epochs, seed 42):
baseline(15k) .835 | clip-only .850 (flat tail, final .800) |
wd-only .435 (loss freezes at the shrinkage equilibrium) | both = wd-only
bit-for-bit. **Weight decay kills DART under noise-before; the clip is
harmless-to-stabilizing.** DARTS is the opposite: keep its published
AdamW + wd 1e-5 (helps at long budgets).
Figure: `results/fig_nb_wd_ablation.png`.

## 4. Budget & selection laws

- **best-epoch ~ 1/rho** (peak location scales inversely with rho).
- The rho-optimum drifts to smaller rho as budget grows.
- Oracle best-epoch on a noisy Pd curve inflates by roughly
  sigma_eval * sqrt(2 ln N_evals) (~ +.09 at 100 evals, Pd noise ±.03 on
  200 targets). Trust best-epoch only where peak locations cluster
  across seeds; otherwise read the final/plateau level.
- multi@2048, 30k epochs: broad plateau rho .003–.1
  (best .77–.82, final .71–.77; peak at rho=.003: best .815±.030 /
  final .768). Final-epoch loses only ~.04 on the plateau -> deployable
  without selection.
- single@2048 was **budget-limited, not geometry-limited**: at 30k epochs
  rho .01–.1 reaches best ~.81 mean with breakout seeds .87–.945 (peaks
  still at the 30k cap; published noise-after DART = .873). The old .745
  ceiling was a 15k artifact. Seed behavior is bimodal (breakout vs
  plateau seeds) — open item: 100k confirmation.
- Small-n single (n<=512) at 5k epochs shows an early peak + decay that
  REVERSES with budget: at 30k the final-epoch models reach .68–.73,
  above the 5k oracle numbers.

### Deployable rho(n) (std front, final-epoch)

| n          | rho          |
|------------|--------------|
| <= 128     | 0.03 – 0.1   |
| 256 – 512  | 0.03         |
| >= 512     | 0.003 – 0.005|
(and budget 30k+; multi final-epoch envelope beats published DART at
every n: .57/.63/.71/.71/.74/.76/.77 for n=32..2048.)

## 5. LRao strong recipe

    no cutoff (full SVD pseudo-inverse) + robust median/IQR input norm
    + detach_sigma + FULL-batch steps + [128] ReLU MLP + Adam 5e-4
    (wd 0, no clip in the no-reg protocol)

- The batch IS the regularizer: no-cutoff + mini-batch is the unstable
  worst cell; cutoff and batch are redundant regularizers — use exactly
  one (we use batch).
- Val-ES (patience 3 / check-every-1) is broken at small n (stops at
  epoch 4–12); use fixed-budget training.
- multi: LRao diverges after its peak -> REQUIRES selection (val-LFI or
  best-epoch); final-epoch reading is its worst case.
- single: curves hold late; final-epoch is fine. LRao owns single-class
  mid-n (best-epoch .71/.81 at n=128/512 vs DART .62-.73); DART owns
  multi at every n.
- CONFIG TRAP: `iid_multi.yaml` and `iid_single.yaml` swap the roles of
  `hidden_dims` / `hidden_dims_2` — always set the LRao MLP explicitly
  to `[128]` (an early GrandSweep multi session silently trained the
  LINEAR LRao; those rows are L-LRao numbers: final .14/.11/.20/.27/.43/
  .47 for n=32..1024 — kept as a free L-LRao baseline).

## 6. Key result tables (all in `noise_before/results/`)

- **Repro n-sweeps** (std, best-epoch, 5 seeds, 15k, fixed rho):
  multi rho=.003: .44/.43/.43/.45/.41/.42/.66/**.800±.03** (n=20..2000)
  single rho=.01: .58/.58/.58/.59/.57/.59/.71/**.732±.02**
  -> `results_repro_std.json` (checkpoints local: ckpt_repro/, not in git)
- **30k rho-responses @2048** (5 seeds, best+final+curves):
  -> `results_rho_sweep_30k.json` (5k partial: `results_rho_sweep.json`)
- **Overnight atlas** (fronts x rho x n x scenes, 372 runs, 08-12):
  -> `results_ov_*.json`; consolidated in `local_runners/README.md`
- **Metal-target (pavia4 cls-5, theta=.075) leaderboard**: DARTS-robust
  .906/.897 > AMF-local .888 > ZCA-DART .857 (still climbing @50k) >
  LRao-noES .833 > DART-robust ~.80 > Levin .781 > AMF-global .77
  -> `results_nb_robust_p4.json`, `results_nb_darts_robust.json`
- **wd/clip ablation** -> `results_nb_std_{long,wdonly,cliponly}.json`
- **LRao batch A/B** (multi n=1024, full vs 512) -> `results_lrao_ab.json`

## 7. Which notebook reproduces what

| notebook | reproduces |
|---|---|
| `RunGrandSweep.ipynb` | n x rho grids (multi/single/pavia4) + LRao lines, 30k final-epoch protocol, restartable ckpts every 100 ep, plots 1–4, archive |
| `RunIIDConfigurable.ipynb` | single-config IID runs, best-epoch protocol, per-run ckpt+scores; baselines cells (AMF/Levin/LRao) pasted per session |
| `RunDARTSMetal.ipynb` | pavia4 metal-target suite (DARTS sweep + baselines + LRao no-ES + DART-ZCA) |

All three clone the repo (`-b rebuttal` or `-b camera-ready`; `repro/` is
identical), write their runner via `%%writefile`, and are resume-safe.

## 8. Open items

- single-class 100k confirmation (do breakout seeds converge ~.9?).
- pavia4 GrandSweep session (full 16-rho ladder at the scene pool).
- deployable epoch/rho selection without oracle (val-based; the flat
  final-epoch tails make this mostly moot for DART on the plateau).
- DARTS under the full no-floor protocol on SD scenes.
