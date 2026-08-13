"""normalization.py — the frozen linear FRONT-ENDS the score nets sit behind.

Every front-end returns a `Whitening(mu, W)`: z = (x - mu) W^T, square D->D, so
they are drop-in for one another and a signature transforms exactly as s -> W s.
`make_frontend(X, cfg)` is the single dispatcher; `whiten_mode` picks one:

  zca | pca | cholesky | normalize   closed forms on core.data.Whitening
  iqr | mad                          the robust/published LRao front-ends
  shrink                             regularised whitening (shrink_space/lam)
  wmw                                within-mode whitening
  vicreg                             a LEARNED linear front-end

WHICH ONE MATTERS (Pavia-U multi, n=1000, converged; best AUC over a sigma sweep):

    normalize .885 | wmw .885 | vicreg .882 | zca .828 | mad .777

The winners all PRESERVE per-band scaling and differ only in how much they
decorrelate; ZCA loses because forcing cov(z)=I amplifies the low-variance,
noise-dominated bands and flattens the multi-modal background structure a score
detector exploits, and `mad` loses because a single global scalar throws the
per-band scaling away. Choosing the front-end is worth ~.10 AUC here -- far more
than choosing sigma (~.01-.04), so start here.
"""
import numpy as np

from .data import Whitening


def robust_whitening_iqr(train_raw, cfg=None):
    X = np.asarray(train_raw, dtype=np.float64)
    med = np.median(X, axis=0)
    iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    iqr = np.where(iqr > 1e-8, iqr, 1.0)
    return Whitening(med.astype(np.float32), np.diag(1.0 / iqr).astype(np.float32))


def robust_whitening_mad(train_raw, cfg=None):
    X = np.asarray(train_raw, dtype=np.float64)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = np.sqrt(np.maximum(mad ** 2, 1e-22))
    return Whitening(med.astype(np.float32), np.diag(1.0 / scale).astype(np.float32))


def global_mad_whitening(train_raw, cfg=None):
    """The LRao paper's real-data preprocessing: a single GLOBAL scalar
    location/scale (not per-band), sigma_mad = 1.4826 * median(|x - median(x)|)
    over ALL values, applied uniformly to every band. Expressed as a Whitening
    module (mu = median*1, W = (1/sigma_mad) * I) so it plugs into ScoreNet /
    TrafoScore and un-whitens like any other front-end.

    A single scalar keeps the CNN's translation-equivariance across the
    sequence intact (per-band scaling would break it), matching the paper's
    `sigma_mad = 1.483*np.median(np.abs(data-np.median(data)))`.
    """
    X = np.asarray(train_raw, dtype=np.float64)
    D = X.shape[1]
    med = float(np.median(X))
    mad = float(np.median(np.abs(X - med)) * 1.4826)
    scale = float(np.sqrt(max(mad ** 2, 1e-22)))
    mu = np.full(D, med, dtype=np.float32)
    W = (np.eye(D, dtype=np.float32) / scale)
    return Whitening(mu, W.astype(np.float32))


def wmw_whitening(train_raw, cfg=None, seed: int = 0):
    """Within-Mode Whitening (WMW) front-end — see WMW_HANDOFF.md.

    "Fix the SCALE, never the SHAPE": whiten only the WITHIN-mode clutter and
    leave the between-mode geometry (the non-Gaussian structure a score detector
    exploits) untouched. ZCA of the TOTAL covariance flattens the between-mode
    structure; per-band std leaves the clutter correlated. WMW whitens the pooled
    within-cluster covariance instead:

        mu, std = column mean / std of X
        labels  = kmeans((X - mu)/std, k)                 # std space: distances
        Sigma_w = pooled within-cluster covariance in RAW space
        Sigma_w = V diag(lam) V^T,  lam = clip(lam, floor, None)
        W       = V diag(1/sqrt(lam)) V^T                 # symmetric (ZCA-style)
    Front: z = W (x - mu);  the signature transforms exactly as s -> W s.

    After the front Cov(z) = I + W Sigma_b W^T: within-mode clutter is a unit
    sphere (isotropic DSM noise is bandwidth-matched) while modes stay separated.
    On a unimodal scene the clusters coincide, Sigma_w -> Sigma_total, and WMW
    smoothly becomes ZCA.

    cfg knobs (all optional):
      wmw_k          cluster count; None (default) -> deterministic elbow rule.
      wmw_kmax       elbow search cap (default 16).
      wmw_elbow_tol  stop when trace(Sigma_w) improves by < this (default 0.03).
      wmw_floor_rel  relative eigenvalue floor x lam_max (default 1e-5).
      wmw_floor_abs  absolute eigenvalue floor (default 100 — raw radiance).
      wmw_kmeans_n_init  KMeans restarts (default 10).

    SCOPE: WMW is for DART (the pointwise score net). Do NOT use it for DARTS or
    LRao — it is measured harmful there (handoff Sec. 2.6).
    """
    from sklearn.cluster import KMeans
    cfg = cfg or {}
    X = np.asarray(train_raw, dtype=np.float64)
    n, D = X.shape
    mu = X.mean(0)
    std = X.std(0) + 1e-9
    Z = (X - mu) / std                                    # std space: distances only

    k          = cfg.get('wmw_k', None)
    kmax       = int(cfg.get('wmw_kmax', 16))
    tol        = float(cfg.get('wmw_elbow_tol', 0.03))
    floor_rel  = float(cfg.get('wmw_floor_rel', 1e-5))
    floor_abs  = float(cfg.get('wmw_floor_abs', 100.0))
    n_init     = int(cfg.get('wmw_kmeans_n_init', 10))

    _cache = {}
    def pooled_sw(kk):
        """Pooled within-cluster covariance in RAW space (kk=1 -> total cov)."""
        if kk in _cache:
            return _cache[kk]
        if kk <= 1:
            Sw = np.cov(X, rowvar=False)
        else:
            lab = KMeans(kk, n_init=n_init, random_state=seed).fit_predict(Z)
            Sw = np.zeros((D, D))
            for c in range(kk):
                Xc = X[lab == c]
                if len(Xc) > 1:
                    d = Xc - Xc.mean(0)
                    Sw += d.T @ d
            Sw = Sw / max(n - kk, 1)
        _cache[kk] = Sw
        return Sw

    if k is None:                                         # deterministic elbow rule
        prev, k = None, kmax
        for kk in range(1, kmax + 1):
            tr = float(np.trace(pooled_sw(kk)))
            if prev is not None and prev > 0 and (prev - tr) / prev < tol:
                k = max(kk - 1, 1)
                break
            prev = tr
    Sw = pooled_sw(int(k))
    lam, V = np.linalg.eigh((Sw + Sw.T) / 2.0)
    lam = np.clip(lam, max(float(lam[-1]) * floor_rel, floor_abs), None)
    W = V @ np.diag(1.0 / np.sqrt(lam)) @ V.T             # symmetric within-mode root
    return Whitening(mu.astype(np.float32), W.astype(np.float32))


def shrinkage_whitening(train_raw, cfg=None):
    """Regularized ("shrinkage") whitening: W = (Sigma + lam * tr(Sigma)/D * I)^(-1/2).

    The closed-form standard-statistics answer to what VICReg turns out to be
    doing. Full ZCA forces cov(Z) = I, which means amplifying every low-variance
    direction to unit variance -- in HSI those are mostly sensor noise, so ZCA
    boosts noise to parity with signal and flattens the between-mode structure a
    score detector needs. Adding lam*mean-eigenvalue to the covariance before
    inverting leaves the noise-dominated directions alone.

    Measured on Pavia multi (n=1000), this reproduces the learned front-ends'
    geometry without any training:

        lam    d_eff            lam    d_eff
        0.001   57.1 (~ZCA)     1.0     4.2  (~wmw, 4.3)
        0.05    13.7            5.0     3.1  (~vicreg, 3.3)

    For reference the measured VICReg front sits only 23% away from pure
    per-channel standardization (`normalize`), with 22% off-diagonal energy --
    i.e. it barely whitens at all.

    cfg knobs:
      shrink_lam    lam above (default 1.0). 'lw' -> pick it by Ledoit-Wolf.
      shrink_space  'cov' (default) or 'corr' — WHICH matrix gets shrunk, and it
                    decides what the two limits are. This matters more than lam:

                      'cov'  : W = (Sigma + lam*m*I)^(-1/2)
                               lam -> 0    ZCA
                               lam -> inf  a GLOBAL SCALAR times I (measured
                                           diag-spread 1.0) — i.e. the 'mad'
                                           front, which is the worst performer
                                           here. This path never passes through
                                           'normalize'.
                      'corr' : W = (R + lam*I)^(-1/2) diag(1/std),  R the
                               correlation matrix.
                               lam -> 0    ZCA (of the correlation)
                               lam -> inf  'normalize' (up to a global scale)
                               So this is the family that actually interpolates
                               between the two fronts worth interpolating, and
                               the one to use when asking "how much decorrelation
                               does normalize want?".
    """
    from .data import Whitening as _W
    cfg = cfg or {}
    X = np.asarray(train_raw, dtype=np.float64)
    n, D = X.shape
    mu = X.mean(0)
    Xc = X - mu
    if str(cfg.get('shrink_space', 'cov')) == 'corr':
        std = Xc.std(0)
        std = np.clip(std, max(float(std.max()) * 1e-5, 1e-6), None)
        U = Xc / std                                  # standardised -> corr matrix
        R = U.T @ U / max(n - 1, 1)
        R = (R + R.T) / 2.0
        lam_cfg = cfg.get('shrink_lam', 1.0)
        if isinstance(lam_cfg, str) and lam_cfg.lower() in ('auto', 'deff', 'rank'):
            lam = _lam_for_target_deff(X, cfg)
        elif isinstance(lam_cfg, str):
            from sklearn.covariance import LedoitWolf
            dd = float(np.clip(LedoitWolf().fit(U).shrinkage_, 0.0, 1.0))
            lam = dd / max(1.0 - dd, 1e-6)
        else:
            lam = float(lam_cfg)
        ev, V = np.linalg.eigh(R + lam * np.eye(D))
        ev = np.clip(ev, max(float(ev[-1]) * 1e-12, 1e-12), None)
        # normalise the overall scale so lam does not silently rescale the space
        Wc = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T
        Wc = Wc * float(np.sqrt(1.0 + lam))           # -> identity as lam -> inf
        W = Wc / std[None, :]
        print(f'    [whiten shrink lam] lam: {lam:.4f}')
        return _W(mu.astype(np.float32), W.astype(np.float32))
    S = Xc.T @ Xc / max(n - 1, 1)
    S = (S + S.T) / 2.0
    m = float(np.trace(S) / D)
    lam_cfg = cfg.get('shrink_lam', 1.0)
    if isinstance(lam_cfg, str) and lam_cfg.lower() in ('auto', 'deff', 'rank'):
        return _shrink_auto(X, cfg)
    if isinstance(lam_cfg, str):                      # Ledoit-Wolf shrinkage
        from sklearn.covariance import LedoitWolf
        d = float(np.clip(LedoitWolf().fit(X).shrinkage_, 0.0, 1.0))
        lam = d / max(1.0 - d, 1e-6)                  # (1-d)S + d*m*I  ~  S + lam*m*I
    else:
        lam = float(lam_cfg)
    ev, V = np.linalg.eigh(S + lam * m * np.eye(D))
    ev = np.clip(ev, max(float(ev[-1]) * 1e-12, 1e-12), None)
    W = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T
    print(f'    [whiten shrink lam] lam: {lam:.4f}')
    return _W(mu.astype(np.float32), W.astype(np.float32))


# ---------------------------------------------------------------------------
# VICReg: a LEARNED linear front-end
# ---------------------------------------------------------------------------

def _vicreg_loss(za, zb, sim_coef=25.0, var_coef=25.0, cov_coef=1.0, eps=1e-4):
    """Standard VICReg loss: invariance + variance hinge + off-diagonal covariance."""
    import torch
    import torch.nn.functional as F
    n, d = za.shape
    inv = F.mse_loss(za, zb)
    var = (F.relu(1.0 - (za.var(0) + eps).sqrt()).mean()
           + F.relu(1.0 - (zb.var(0) + eps).sqrt()).mean())
    ca = (za - za.mean(0)).T @ (za - za.mean(0)) / max(n - 1, 1)
    cb = (zb - zb.mean(0)).T @ (zb - zb.mean(0)) / max(n - 1, 1)
    off = lambda M: M - torch.diag(torch.diag(M))       # zero the diagonal
    cov = (off(ca).pow(2).sum() + off(cb).pow(2).sum()) / d
    return sim_coef * inv + var_coef * var + cov_coef * cov


def fit_vicreg_whitening(train_raw, cfg=None, seed: int = 0):
    """Learn a LINEAR VICReg embedding and return it as a frozen Whitening.

    Bardes, Ponce & LeCun (ICLR 2022). The encoder is linear on purpose: the
    additive model y = theta*s + w must survive the front-end, and a linear map
    carries the signature exactly (s -> W s). Two augmented views of each pixel
    (sensor noise, illumination gain, band dropout) drive an invariance term,
    while variance + covariance terms push toward whitening.

    MEASURED: the fitted map ends up only ~23% away from plain per-band
    standardization, with 22% off-diagonal energy -- it barely whitens. The
    invariance term is what stops it: it penalises amplifying the directions the
    augmentations perturb, i.e. exactly the noise directions ZCA blows up. Its
    geometry is reproducible in closed form by `shrinkage_whitening` with
    shrink_space='corr', lam~2 -- which matched or beat it while needing no
    training.
    """
    import torch
    import torch.nn as nn
    cfg = cfg or {}
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    X = np.asarray(train_raw, dtype=np.float64)
    mu, std = X.mean(0), X.std(0) + 1e-8
    U = torch.tensor((X - mu) / std, dtype=torch.float32)
    n, D = U.shape
    enc = nn.Linear(D, D, bias=False)
    nn.init.eye_(enc.weight)                      # start at 'normalize'

    epochs = int(cfg.get('vicreg_epochs', 400))
    bs = min(int(cfg.get('batch_size', 512)), n)
    noise = float(cfg.get('vicreg_noise', 0.1))
    jitter = float(cfg.get('vicreg_jitter', 0.05))
    drop_p = float(cfg.get('vicreg_drop_p', 0.1))
    sim_c = float(cfg.get('vicreg_sim', 25.0))
    var_c = float(cfg.get('vicreg_var', 25.0))
    cov_c = float(cfg.get('vicreg_cov', 1.0))
    opt = torch.optim.Adam(enc.parameters(), lr=float(cfg.get('vicreg_lr', 1e-3)),
                           weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    def view(V):
        V = V + noise * torch.randn(V.shape, generator=gen)
        V = V * (1.0 + jitter * torch.randn(V.shape[0], 1, generator=gen))
        if drop_p > 0:
            V = V * (torch.rand(V.shape, generator=gen) > drop_p).float()
        return V

    enc.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = U[perm[i:i + bs]]
            if len(b) < 2:
                continue
            loss = _vicreg_loss(enc(view(b)), enc(view(b)), sim_c, var_c, cov_c)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
        sched.step()
    A = enc.weight.detach().cpu().numpy().astype(np.float64)
    # fold the standardisation in so the front-end acts on RAW x
    return Whitening(mu.astype(np.float32), (A / std[None, :]).astype(np.float32))


# ---------------------------------------------------------------------------
# The single dispatcher
# ---------------------------------------------------------------------------

def make_frontend(train_raw, cfg=None, seed: int = 0):
    """Fit the frozen front-end named by cfg['whiten_mode'] (default 'zca')."""
    cfg = cfg or {}
    X = np.asarray(train_raw, dtype=np.float32)
    mode = str(cfg.get('whiten_mode', 'zca')).lower()
    if mode == 'vicreg':
        return fit_vicreg_whitening(X, cfg, seed=seed)
    if mode == 'wmw':
        return wmw_whitening(X, cfg, seed=seed)
    if mode in ('shrink', 'shrinkage'):
        return shrinkage_whitening(X, cfg)
    if mode in ('mad', 'global_mad'):
        return global_mad_whitening(X)
    if mode in ('iqr', 'robust'):
        return robust_whitening_iqr(X)
    return Whitening.from_data(X, eig_floor=float(cfg.get('whiten_eig_floor', 0.0)),
                               mode=mode)          # zca | pca | cholesky | normalize


# ---------------------------------------------------------------------------
# shrink_lam: auto  —  pin the GEOMETRY, not the knob
# ---------------------------------------------------------------------------

def signal_rank(X) -> int:
    """#directions statistically distinguishable from noise (Marchenko-Pastur).

    Eigenvalues of the CORRELATION matrix above the MP bulk edge (1+sqrt(D/n))^2.
    For pure noise the whole spectrum sits under the edge, so this counts the
    directions that carry real structure. Measured: 3 on the Pavia IID pool for
    every n >= 200, 3 on the Pavia-4 spatial train box, 2 at n <= 100 (with 20
    samples you genuinely cannot resolve a third direction).
    """
    Z = np.asarray(X, dtype=np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    n, D = Z.shape
    ev = np.linalg.eigvalsh(Z.T @ Z / max(n - 1, 1))
    return max(int((ev > (1.0 + np.sqrt(D / max(n, 1))) ** 2).sum()), 1)


def _lam_for_target_deff(X, cfg=None) -> float:
    """The lam whose front-end has effective dimension == the target.

    WHY NOT LEDOIT-WOLF. LW answers "what shrinkage minimises the error of the
    COVARIANCE ESTIMATE", so its intensity -> 0 as n grows and lam -> 0, i.e. the
    front-end slides to ZCA. That is the right answer to the wrong question here:
    measured, more data does NOT make ZCA good for DART (n=4026: ZCA .828 vs
    normalize .885). The harm is structural -- whitening flattens the between-mode
    geometry however well it is estimated -- so any reliability-based criterion is
    wrong in the limit. Across n=20..4026 on the Pavia IID pool, LW's d_eff drifts
    4.6 -> 47.9, which silently changes the front-end underneath an n-sweep and
    makes the sweep uninterpretable.

    Pinning d_eff instead keeps the GEOMETRY fixed and lets lam absorb whatever
    n and the scene require:  lam* 8.36 -> 8.53 over n=200..4026 (2%), d_eff
    exactly at target, s_bar 0.441 -> 0.448. The same rule picks lam 8.5 for the
    IID pool and 14.4 for the spatial train box on its own.

    cfg: shrink_target_deff (default: signal_rank(X)), shrink_lam_bounds.
    """
    from .sigma import effective_dim
    import torch
    cfg = cfg or {}
    X = np.asarray(X, dtype=np.float32)
    target = cfg.get('shrink_target_deff')
    target = float(signal_rank(X) if target is None else target)
    lo, hi = cfg.get('shrink_lam_bounds', (1e-4, 1e4))

    def deff(lam):                                   # monotonically DECREASING in lam
        W = shrinkage_whitening(X, {**cfg, 'shrink_lam': float(lam),
                                    'shrink_space': cfg.get('shrink_space', 'corr')})
        return effective_dim(W(torch.tensor(X)).detach().cpu().numpy())

    # d_eff is bounded by what the spectrum allows: on weakly-correlated data it
    # stays ~D for EVERY lam, so an unreachable target would silently run the
    # bisection into a bound. Clamp to the achievable range and say so.
    d_hi, d_lo = deff(lo), deff(hi)                  # lam small -> large d_eff
    span = (min(d_lo, d_hi), max(d_lo, d_hi))
    if not (span[0] <= target <= span[1]):
        clamped = float(np.clip(target, *span))
        print(f"    [shrink auto] target d_eff={target:.2f} is outside the "
              f"achievable range [{span[0]:.2f}, {span[1]:.2f}] for this data "
              f"(lam cannot get there) -> using {clamped:.2f}", flush=True)
        target = clamped
    for _ in range(int(cfg.get('shrink_bisect_iters', 28))):
        mid = float(np.sqrt(lo * hi))
        if deff(mid) > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _shrink_auto(X, cfg):
    lam = _lam_for_target_deff(X, cfg)
    return shrinkage_whitening(X, {**cfg, 'shrink_lam': lam})
