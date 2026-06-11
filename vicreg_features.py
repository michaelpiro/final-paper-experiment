"""VICReg self-supervised preprocessing -> a linear latent front-end.

Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning" (ICLR 2022).

Why a LINEAR encoder here
-------------------------
The detector is built on the additive model y = theta*s + w with a KNOWN
target signature s, and the Rao / matched-filter statistics need s in the
same space as the data. A nonlinear feature map breaks that (s cannot be
carried through it). A *linear* map z = (x - mu) A^T preserves the additive
structure exactly: the signature transforms as s -> A s (handled by the
existing Whitening.transform_direction), so VICReg slots straight into the
frozen linear front-end the score nets already use.

What VICReg buys over plain ZCA
-------------------------------
Applied to the (linear) embedding z of two augmented views of each pixel:
  * covariance term  -> decorrelates the latent dims (like whitening),
  * variance term    -> unit variance per dim (like whitening),
  * invariance term  -> makes the projection robust to spectral
                        augmentations (sensor noise, illumination gain,
                        dead bands).
So this is a learned, noise-robust whitening rather than the closed-form
ZCA. It returns a frozen ``Whitening(mu, W)`` and is selected by
``whiten_mode: vicreg``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dsm_model import Whitening


def _vicreg_loss(za, zb, sim_coef=25.0, var_coef=25.0, cov_coef=1.0, eps=1e-4):
    """Standard VICReg loss on two embeddings za, zb of shape (n, d)."""
    n, d = za.shape
    # invariance: views of the same pixel should map to the same code
    inv = F.mse_loss(za, zb)
    # variance: hinge keeping each dim's std >= 1
    std_a = torch.sqrt(za.var(dim=0) + eps)
    std_b = torch.sqrt(zb.var(dim=0) + eps)
    var = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))
    # covariance: push off-diagonal covariances to zero (decorrelation)
    za_c = za - za.mean(dim=0)
    zb_c = zb - zb.mean(dim=0)
    cov_a = (za_c.T @ za_c) / max(n - 1, 1)
    cov_b = (zb_c.T @ zb_c) / max(n - 1, 1)
    off = lambda M: (M - torch.diag(torch.diag(M)))
    cov = (off(cov_a).pow(2).sum() + off(cov_b).pow(2).sum()) / d
    return sim_coef * inv + var_coef * var + cov_coef * cov, (inv, var, cov)


def _augment(U, noise, jitter, drop_p, gen):
    """Two augmented views of standardized spectra U (n, d).

    Spectral-domain invariances:
      * additive Gaussian sensor noise (std = ``noise``),
      * per-pixel multiplicative illumination gain (1 + jitter*N(0,1)),
      * random band dropout (fraction ``drop_p`` of bands zeroed).
    """
    def one():
        V = U + noise * torch.randn(U.shape, generator=gen)
        gain = 1.0 + jitter * torch.randn(U.shape[0], 1, generator=gen)
        V = V * gain
        if drop_p > 0:
            mask = (torch.rand(U.shape, generator=gen) > drop_p).float()
            V = V * mask
        return V
    return one(), one()


def fit_vicreg_whitening(train_raw: np.ndarray, cfg: dict,
                         seed: int = 0) -> Whitening:
    """Learn a linear VICReg embedding on the background and return it as a
    frozen ``Whitening(mu, W)`` (square D->D, so a drop-in for ZCA)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)

    X = np.asarray(train_raw, dtype=np.float64)
    mu = X.mean(0)
    std = X.std(0) + 1e-8
    U = torch.tensor((X - mu) / std, dtype=torch.float32)     # standardized
    n, D = U.shape

    # linear encoder z = U @ A^T  (square, bias-free; centering is in mu)
    enc = nn.Linear(D, D, bias=False)
    nn.init.eye_(enc.weight)                                  # start at identity

    epochs = int(cfg.get('vicreg_epochs', 400))
    bs = min(int(cfg.get('batch_size', 512)), n)
    lr = float(cfg.get('vicreg_lr', 1e-3))
    noise = float(cfg.get('vicreg_noise', 0.1))
    jitter = float(cfg.get('vicreg_jitter', 0.05))
    drop_p = float(cfg.get('vicreg_drop_p', 0.1))
    sim_c = float(cfg.get('vicreg_sim', 25.0))
    var_c = float(cfg.get('vicreg_var', 25.0))
    cov_c = float(cfg.get('vicreg_cov', 1.0))

    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    enc.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = U[perm[i:i + bs]]
            if len(b) < 2:
                continue
            va, vb = _augment(b, noise, jitter, drop_p, gen)
            loss, _ = _vicreg_loss(enc(va), enc(vb), sim_c, var_c, cov_c)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            opt.step()
        sched.step()

    A = enc.weight.detach().cpu().numpy().astype(np.float64)  # (D, D)
    # fold the standardization into the linear map so the front-end acts on
    # RAW x:  z = ((x - mu)/std) A^T = (x - mu) (A/std)^T
    Wmat = (A / std[None, :]).astype(np.float32)
    return Whitening(mu.astype(np.float32), Wmat)
