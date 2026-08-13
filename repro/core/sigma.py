"""sigma.py — choosing the DSM noise level.

DSM at noise sigma learns the score of  p * N(0, sigma^2 I): sigma decides WHICH
(blurred) density the score net estimates, so it is a real modelling choice, not
a nuisance constant. Everything that picks it lives here, behind one entry point:

    resolve_sigma(Zw, cfg, ...)      Zw = the WHITENED training pixels

`cfg['dsm_sigma_rho']` is either a NUMBER (sigma = sqrt(rho), the published
behaviour) or one of these strings:

    'dn16'      (D/n)^(1/6) on the ambient D
    'eff'       s_bar * (d_eff/n)^(1/6)
    'eff2'      c * s_bar * (d_eff^2/n)^(1/6)      <- best closed form measured
    'n16'       c * s_bar * n^(-1/6)
    'lw'        Ledoit-Wolf shrinkage loading
    'auto'      Parzen (KDE) bandwidth, LOO log-likelihood
    'auto-sm'   Parzen (KDE) bandwidth, LOO score-matching
    'detval'    held-out detection AUC (the only budget-aware option)

Every rule above reads the TRAINING pixels only, so none can see how far the test
data sits from them — which is why the best rule on the IID protocol is the worst
on the spatial one. When a disjoint validation region is available it is passed in
as `Zval` and sigma is widened in quadrature by the train->val displacement
(`transfer_gap`); with delta ~ 0, as in IID, that is a no-op. Switch it off with
`dsm_sigma_transfer: false`. A numeric dsm_sigma_rho is never affected.

WHAT THE MEASUREMENTS SAY (Pavia-U multi, MLP[128], converged at 30k steps,
sigma* found per cell by an explicit sweep):

  * sigma* is essentially FLAT in n for front-ends that preserve the background's
    mode structure (normalize: .171/.087/.170, vicreg: .091/.099/.110 at
    n=200/700/2000) and only falls with n for ZCA (.870/.606/.294), which
    whitens that structure away. So the textbook n-scaling describes the regime
    you do not want to be in.
  * sigma* depends on the TRAINING BUDGET (and on the learning rate, not just
    the step count): an under-trained net cannot represent a sharp score, so it
    prefers a large one. Measured at n=1000 on 'normalize', 8000 steps in both
    cases: lr 5e-4 -> sigma* 0.245, lr 1e-3 -> sigma* 0.141. No (d, n) rule can
    see that axis; 'detval' can.
  * AUC given up versus the per-cell optimum, 14 cells over 5 front-ends:
        dn16 .0409 | n16 .0267 | eff .0135 | eff2 .0119
    and on the two fronts worth using, a fitted CONSTANT beat all of them
    (.0158). Treat these rules as a good starting point, not an oracle.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Effective dimension
# ---------------------------------------------------------------------------

def effective_dim(Z: np.ndarray, method: str = 'pr') -> float:
    """Effective (not ambient) dimension of whitened data Z (n, D).

    HSI bands are strongly correlated, so the ambient D (103 / 189) badly
    overstates how many directions the density occupies -- and the count depends
    on the FRONT-END: measured on Pavia multi (n=1000) the participation ratio is
    ~66 after ZCA (which spreads variance over every direction by construction)
    but ~2 after 'normalize' (which does not decorrelate). Any sigma rule
    carrying an ambient D therefore mis-scales the moment the front-end changes.

      'pr'      participation ratio (sum lam)^2 / sum lam^2 (DEFAULT)
      'entropy' effective rank exp(-sum p log p), p = lam / sum lam
      'var95'   number of eigenvalues needed to reach 95% of the variance
    """
    Z = np.asarray(Z, dtype=np.float64)
    n = len(Z)
    Zc = Z - Z.mean(0)
    lam = np.clip(np.linalg.eigvalsh(Zc.T @ Zc / max(n - 1, 1)), 0.0, None)
    tot = float(lam.sum())
    if tot <= 0:
        return float(Z.shape[1])
    if method == 'entropy':
        p = lam / tot
        p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))
    if method == 'var95':
        return float(np.searchsorted(np.cumsum(np.sort(lam)[::-1]) / tot, 0.95) + 1)
    return float(tot ** 2 / float(np.sum(lam ** 2)))


def _scale(Z: np.ndarray) -> float:
    """RMS per-coordinate std of the whitened data.

    Dimensional analysis, not a fitted term: scaling the whitened space by k must
    scale sigma by k, so every rule below carries s_bar with exponent 1. The
    published "mean_std = 1 in whitened space" assumption is false in practice --
    ZCA's eigenvalue floor leaves ~0.75, WMW ~1.6, 'normalize' exactly 1.
    """
    return float(np.sqrt(np.mean(np.asarray(Z, dtype=np.float64).var(axis=0))))


# ---------------------------------------------------------------------------
# Closed-form rules
# ---------------------------------------------------------------------------

def sigma_dn16(n: int, D: int) -> float:
    """Textbook score-MSE scale sigma = (D/n)^(1/6) on the AMBIENT D.

    Kept for reference/ablation: it is the rule the literature states, and the
    worst performer measured here (mean .0409 AUC lost) because the ambient D is
    far too large for every front-end tried.
    """
    return float((float(D) / max(int(n), 1)) ** (1.0 / 6.0))


def sigma_effective(Z: np.ndarray, method: str = 'pr') -> float:
    """s_bar * (d_eff/n)^(1/6) -- ambient D swapped for the effective dimension."""
    return float(_scale(Z) * (effective_dim(Z, method) / max(len(Z), 1)) ** (1.0 / 6.0))


def sigma_eff2(Z: np.ndarray, c: float = 0.755, method: str = 'pr') -> float:
    """c * s_bar * (d_eff^2 / n)^(1/6) -- the best closed form measured.

    A free 3-parameter fit over 5 front-ends x 3 sample sizes gave
    sigma*/s_bar = 0.836 * d_eff^(+0.327) * n^(-0.183); pinning the exponents to
    the nearby rationals 1/3 and -1/6 (i.e. d_eff SQUARED inside the sixth root)
    costs nothing (.0119 either way). So the n-scaling is the theoretical -1/6,
    but d_eff enters with twice the exponent the plain (d_eff/n)^(1/6) assumes.

    Caveats: `c` is empirical and moves with the optimizer (~0.755 at lr 5e-4,
    ~0.37 at lr 1e-3); and the d_eff exponent is pinned mostly by ZCA, the only
    high-d_eff front in the fit.
    """
    return float(c * _scale(Z) * (effective_dim(Z, method) ** 2 / max(len(Z), 1)) ** (1.0 / 6.0))


def sigma_n16(Z: np.ndarray, c: float = 0.76) -> float:
    """c * s_bar * n^(-1/6) -- single-front-end calibration, no d_eff term."""
    return float(c * _scale(Z) * max(len(Z), 1) ** (-1.0 / 6.0))


def sigma_ledoitwolf(Z: np.ndarray, seed: int = 0) -> float:
    """Ledoit-Wolf shrinkage loading, used as the noise level.

    For a Gaussian, DSM at sigma estimates the score of the diagonally-loaded
    covariance (Sigma + sigma^2 I) -- so sigma IS a shrinkage parameter and the
    optimal loading is a principled choice for it. Elegant, but it came last of
    the data-driven options here (mean .0428 AUC lost): its sigma falls like
    n^(-0.54) while the measured optimum is flat, because the shrinkage view only
    captures sigma's second-order role and misses its job of preserving the
    non-Gaussian mode structure DART actually exploits.
    """
    from sklearn.covariance import LedoitWolf
    X = np.asarray(Z, dtype=np.float64)
    n, d = X.shape
    if n < 3:
        return 1.0
    rho = float(np.clip(LedoitWolf().fit(X).shrinkage_, 0.0, 1.0))
    Xc = X - X.mean(0)
    mu_scale = float(np.trace(Xc.T @ Xc) / max(n - 1, 1) / d)
    return float(np.sqrt(max(rho * mu_scale, 1e-12)))


def sigma_parzen(Z: np.ndarray, sigma_lo: float = 0.02, sigma_hi: float = 5.0,
                 seed: int = 0, criterion: str = 'loglik') -> float:
    """Parzen (KDE) bandwidth by leave-one-out CV -- no network training.

    The infinite-capacity DSM minimizer is exactly the score of a Gaussian KDE
    with bandwidth sigma, so picking sigma is a bandwidth-selection problem.

      criterion='loglik'     maximise the LOO Parzen log-likelihood
      criterion='scorematch' minimise the LOO implicit-score-matching loss
                             (the objective DSM itself optimises)

    NOTE this standardises internally, so it returns a bandwidth in STANDARDISED
    units; `resolve_sigma` rescales by s_bar to put it back in whitened units.
    That correction is a no-op for a 'normalize' front (s_bar == 1) and matters
    for ZCA and WMW.
    """
    from scipy.special import logsumexp
    W = np.asarray(Z, dtype=np.float64)
    mu, std = W.mean(0), W.std(0) + 1e-8
    U = (W - mu) / std
    n, d = U.shape
    if n < 3:
        return 1.0
    sqn = np.einsum('ij,ij->i', U, U)
    Dm = np.maximum(sqn[:, None] + sqn[None, :] - 2.0 * (U @ U.T), 0.0)
    diag = np.arange(n)
    Dself0 = Dm.copy()
    Dm[diag, diag] = np.inf

    def negobj(log_sigma):
        s2 = np.exp(2.0 * log_sigma)
        logk = -Dm / (2.0 * s2)
        lse = logsumexp(logk, axis=1)
        if criterion == 'loglik':
            ll = lse - np.log(n - 1) - 0.5 * d * np.log(2 * np.pi * s2)
            return -ll.mean()
        r = np.exp(logk - lse[:, None])
        m = r @ U - U
        m2 = np.einsum('ij,ij->i', m, m)
        rD = np.einsum('ij,ij->i', r, Dself0)
        return (0.5 * (m2 / s2 ** 2) + (-d + (rD - m2) / s2) / s2).mean()

    gr = (np.sqrt(5) - 1) / 2
    a, b = np.log(sigma_lo), np.log(sigma_hi)
    c, dd = b - gr * (b - a), a + gr * (b - a)
    fc, fd = negobj(c), negobj(dd)
    for _ in range(40):
        if fc < fd:
            b, dd, fd = dd, c, fc
            c = b - gr * (b - a)
            fc = negobj(c)
        else:
            a, c, fc = c, dd, fd
            dd = a + gr * (b - a)
            fd = negobj(dd)
        if (b - a) < 1e-24:
            break
    return float(np.exp((a + b) / 2))


# ---------------------------------------------------------------------------
# Budget-aware: held-out detection AUC
# ---------------------------------------------------------------------------

def sigma_detval(train_raw: np.ndarray, cfg: dict, seed: int, s_raw: np.ndarray,
                 trainer, scorer, label: str = '') -> float:
    """Pick sigma by held-out DETECTION AUC -- the only budget-aware option.

    Every closed-form rule above is a function of (d_eff, n) only, but the
    optimum also moves with the training budget and learning rate (see the module
    docstring). This measures the thing we care about instead: hold out
    background, plant the KNOWN signature on it, train at each candidate rho, and
    keep the rho with the best held-out AUC. The learned-Rao trainer already uses
    this trick for its early stopping.

    trainer(fit_np, cfg, seed, label) -> model     (cfg carries a numeric rho)
    scorer(model, fit_np, val_planted, s_raw)      -> scores

    cfg: detval_grid (candidate RHOs), detval_val_frac (0.25),
         detval_n_correct (True -> rescale by (n_fit/n)^(1/6), since the
         selection trains on a smaller split and smaller n prefers a larger sigma).
    Cost: len(grid) trainings, plus the caller's final fit.
    """
    from sklearn.metrics import roc_auc_score
    from repro.core.data import plant_targets

    grid = list(cfg.get('detval_grid') or [0.005, 0.02, 0.05, 0.15, 0.5])
    vf = float(cfg.get('detval_val_frac', 0.25))
    X = np.asarray(train_raw, np.float32)
    n = len(X)
    n_val = int(np.clip(round(vf * n), 8, max(8, n - 8)))
    perm = np.random.default_rng(seed).permutation(n)
    fit_np, val_np = X[perm[n_val:]], X[perm[:n_val]]
    v_pl, v_lab, _ = plant_targets(val_np, s_raw, float(cfg['amplitude']),
                                   float(cfg['target_fraction']),
                                   model='additive', seed=seed)
    v_pl = v_pl.astype(np.float32)

    best_rho, best_auc = grid[0], -np.inf
    for rho in grid:
        c = {**cfg, 'dsm_sigma_rho': float(rho)}
        try:
            model = trainer(fit_np, c, seed, f'{label}/detval-rho{rho}')
            auc = float(roc_auc_score(v_lab, scorer(model, fit_np, v_pl, s_raw)))
        except Exception as exc:
            print(f'      [detval] rho={rho} failed: {exc}', flush=True)
            continue
        print(f'      [detval] rho={rho:<8g} sigma={np.sqrt(rho):.4f}  '
              f'held-out AUC={auc:.4f}', flush=True)
        if auc > best_auc:
            best_rho, best_auc = rho, auc
    sigma = float(np.sqrt(best_rho))
    if bool(cfg.get('detval_n_correct', True)):
        sigma *= float((len(fit_np) / max(n, 1)) ** (1.0 / 6.0))
    print(f'    [detval] {label}: rho={best_rho:g} -> sigma={sigma:.4f} '
          f'(held-out AUC={best_auc:.4f})', flush=True)
    return sigma


# ---------------------------------------------------------------------------
# The single entry point
# ---------------------------------------------------------------------------

_ALIASES = {
    'dn16': 'dn16', 'auto-dn': 'dn16', 'auto-dn16': 'dn16', 'dn': 'dn16',
    'eff': 'eff', 'auto-eff': 'eff', 'dneff': 'eff',
    'eff2': 'eff2', 'auto-eff2': 'eff2',
    'n16': 'n16', 'caln16': 'n16', 'cal': 'n16',
    'lw': 'lw', 'ledoit': 'lw', 'ledoitwolf': 'lw',
    'detval': 'detval', 'auto-detval': 'detval',
    'auto': 'parzen-loglik', 'auto-sm': 'parzen-scorematch',
    'g': 'g', 'rel': 'g', 'relative': 'g',
    'parzen': 'parzen-loglik', 'parzen-sm': 'parzen-scorematch',
}


def transfer_gap(Zw_train: np.ndarray, Zw_val: np.ndarray,
                 unbiased: bool = True) -> float:
    """delta — how far the data the detector is APPLIED to sits from the data it
    was FIT on: the RMS per-coordinate mean displacement, in the same whitened
    units as sigma.

    WHY THIS EXISTS. Every closed-form rule above is a function of the TRAINING
    pixels alone — (n, d_eff, s_bar) — so none of them can see that quantity, and
    it is what makes the two protocols want different sigma. Measured on Pavia-U
    at n=4032 (front / delta / measured sigma*):

        IID        normalize  0.018   0.100      train and test are one shuffled
        IID        zca        0.020   0.227      pool, so delta ~ 0
        pavia_mix  normalize  0.434   0.500      disjoint boxes, so delta is 20x
        pavia_mix  zca        0.217   0.316      larger and dominates sigma*

    It is also why the rule RANKINGS invert between the protocols: 'auto' is the
    most accurate rule on IID-normalize (0.9x) and the least accurate on
    spatial-normalize (0.16x), because it models only the intrinsic term.

    LEGITIMACY. delta uses pixel MEANS of an unlabelled held-out region — no
    labels, no target signature — so measuring it on the VALIDATION region is
    ordinary model selection. Measuring it on the TEST region would be leakage.

    HOW WELL THE CORRECTION IS VALIDATED — read before trusting it. The
    quadrature form sigma = sqrt(sigma_rule^2 + delta^2) is the natural one (the
    blur has to cover the displacement, and it is exactly neutral when delta ~ 0,
    which is the IID case) and it is consistent for the zca front: the implied
    intrinsic term is 0.226 on IID vs 0.229 on pavia_mix. It does NOT close the
    gap for the normalize front (0.098 vs 0.248) even though n, d_eff and s_bar
    are identical there, so something beyond a mean displacement is also moving
    sigma*. Four measured optima on a 5-point grid cannot separate quadrature
    from a linear form. Treat this as a measured improvement, not a law.
    """
    a = np.asarray(Zw_train, dtype=np.float64)
    b = np.asarray(Zw_val, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or len(b) < 2:
        return 0.0
    diff = b.mean(0) - a.mean(0)
    d2 = float(diff @ diff) / a.shape[1]          # mean square per coordinate
    if unbiased:
        # SUBTRACT THE SAMPLING FLOOR. Two sample means differ even when the two
        # distributions are IDENTICAL, by about s_bar*sqrt(1/n_tr + 1/n_val), and
        # that floor grows as n shrinks — exactly where it would distort sigma
        # most. Measured on the IID pool before this correction: delta came out
        # 0.033-0.094 at n=300 (vs 0.019 at n=4032) purely from estimation noise,
        # varying seed to seed, which injected seed-dependent sigma jitter into a
        # protocol that has no shift at all. Removing the floor leaves the real
        # displacement: ~0 for IID, essentially unchanged (-0.1%) for disjoint
        # boxes, where delta is 20x the floor.
        var = 0.5 * float(a.var(0).mean() + b.var(0).mean())
        floor2 = var * (1.0 / len(a) + 1.0 / len(b))
        # Subtracting the floor is unbiased in delta^2, but delta^2 itself is
        # noisy, and sqrt(max(.,0)) then biases the result UP whenever the truth
        # is 0 -- measured: 0.047-0.071 still leaked through at n=300. So first
        # require the displacement to be resolvable at all: at least 2x the noise
        # floor (4x in squared terms). Below that it is reported as no shift,
        # which is the correct answer for identically-distributed pools at every
        # n. Real region shift clears this by a wide margin (pavia_mix: 377x).
        if d2 < 4.0 * floor2:
            return 0.0
        d2 -= floor2
    return float(np.sqrt(max(d2, 0.0)))


def resolve_sigma(Zw: np.ndarray, cfg: dict, seed: int = 0, label: str = '',
                  D: int = None, train_raw: np.ndarray = None,
                  s_raw: np.ndarray = None, trainer=None, scorer=None,
                  Zval: np.ndarray = None, verbose: bool = True) -> float:
    """Turn cfg['dsm_sigma_rho'] into a sigma, in WHITENED-space units.

    Zw : the whitened training pixels (only needed for the data-driven rules).
    A numeric dsm_sigma_rho short-circuits to sqrt(rho) and Zw may be None.
    """
    rho = cfg['dsm_sigma_rho']
    if not isinstance(rho, str):
        return float(np.sqrt(rho))

    key = _ALIASES.get(rho.lower())
    if key is None:
        raise ValueError(
            f"unknown dsm_sigma_rho {rho!r}; use a number or one of "
            f"{sorted(set(_ALIASES))}")
    if key == 'detval':
        if trainer is None or scorer is None or s_raw is None:
            raise ValueError("dsm_sigma_rho='detval' needs train_raw, s_raw, "
                             "trainer and scorer")
        return sigma_detval(train_raw, cfg, seed, s_raw, trainer, scorer, label)

    Zw = np.asarray(Zw, dtype=np.float64)
    if key == 'g':
        # sigma = g * s_bar: the DIMENSIONLESS parameterisation. sigma is only
        # meaningful relative to the whitened scale, so quoting a bare rho couples
        # it to the front-end -- rho=0.1 is g=0.38 after ZCA (s_bar .84) but g=0.98
        # after corr-shrink (s_bar .32), a 2.6x different amount of noise. With `g`
        # the front-end can change (or be auto-selected) and sigma follows.
        # Measured optima: IID g ~ 0.13-0.3; SPATIAL g ~ 1.0-1.7, because spatial
        # has train/test distribution shift and blur is what makes the score
        # transfer (same-box optimum g=0.53 vs test-box optimum g=1.69).
        gval = float(cfg.get('dsm_sigma_g', 0.3))
        sigma, how = gval * _scale(Zw), f'g={gval} x s_bar'
    elif key == 'dn16':
        sigma, how = sigma_dn16(len(Zw), D or Zw.shape[1]), '(D/n)^(1/6)'
    elif key == 'eff':
        m = str(cfg.get('dsm_eff_dim_method', 'pr'))
        sigma, how = sigma_effective(Zw, m), 's_bar*(d_eff/n)^(1/6)'
    elif key == 'eff2':
        m = str(cfg.get('dsm_eff_dim_method', 'pr'))
        c = float(cfg.get('dsm_sigma_c', 0.755))
        sigma, how = sigma_eff2(Zw, c, m), f'{c}*s_bar*(d_eff^2/n)^(1/6)'
    elif key == 'n16':
        c = float(cfg.get('dsm_sigma_c', 0.76))
        sigma, how = sigma_n16(Zw, c), f'{c}*s_bar*n^(-1/6)'
    elif key == 'lw':
        sigma, how = sigma_ledoitwolf(Zw, seed), 'ledoit-wolf'
    else:                                            # parzen-*
        crit = key.split('-', 1)[1]
        sigma = sigma_parzen(Zw, seed=seed, criterion=crit)
        if bool(cfg.get('parzen_rescale', True)):    # standardised -> whitened units
            sigma *= _scale(Zw)
        how = f'parzen-{crit}'
    # ---- transfer correction ------------------------------------------------
    # Widen sigma by the train->val displacement, so the same rule can serve a
    # protocol whose test data is identically distributed with training (IID,
    # delta ~ 0 -> unchanged) and one whose test data is a different region
    # (spatial, delta dominates). See transfer_gap() for the evidence and its
    # limits. 'detval' is excluded: it already picks sigma by measured held-out
    # detection, so adding delta on top double-counts.
    if Zval is not None and bool(cfg.get('dsm_sigma_transfer', True)):
        delta = transfer_gap(Zw, Zval)
        if delta > 0.0:
            sigma = float(np.hypot(sigma, delta))
            how += f' (+) delta={delta:.3f}'
        else:
            # Say so explicitly. A silent no-op reads exactly like a broken knob.
            how += ' (delta=0: val is not measurably shifted from train)'
    elif Zval is None and cfg.get('dsm_sigma_transfer'):
        # Explicitly asked for, but no held-out region reached this call. Say so:
        # a silently ignored knob is worse than no knob. The IID protocol's
        # class-labelled pools have no disjoint region by construction (train and
        # test are one shuffled pool), so there is genuinely nothing to correct;
        # its cfg['scene'] mode does, and passes it.
        print(f'    [sigma] {label}: dsm_sigma_transfer is set but no validation '
              f'region was passed — no delta applied.', flush=True)
    if verbose:
        print(f'    [sigma] {label}: {sigma:.4f} (rho={sigma ** 2:.4g})  [{how}]',
              flush=True)
    return float(sigma)
