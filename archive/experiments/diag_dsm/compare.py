"""
Compare two DSM noise models on the IID single-class problem:

  (A) ISOTROPIC  DSM  — Σ_n = σ²I,  σ² = ρ·(1/d)·tr(Σ̂)        [what we use now]
  (B) DIAGONAL   DSM  — Σ_n = diag(σ_b²),  σ_b = sqrt(ρ·Var_b) [new, data-driven]

Both models:
  - see the SAME global_max-normalized data,
  - run in the SAME PCA-d latent space,
  - share architecture / lr / epochs / weight-init (identical seed),
  - differ ONLY in the noise covariance used during DSM training.

No other baselines are run.  Reports additive + replacement AUC.

Run:
    .venv/bin/python -u compare_diag_dsm.py
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from final_paper_experiments.data_utils import (
    load_and_normalize, compute_sigma_from_data, compute_sigma_diag_from_data,
)
from final_paper_experiments.baselines.detectors import dsm_additive, dsm_replacement
from dsm_model import ScoreNet, dsm_loss

# ----------------------------- config -------------------------------------
CFG = dict(
    dataset='data/pavia-u.mat',
    norm_mode='global_max',
    bkg_cls=2, target_cls=1,
    amplitude=0.15, target_fraction=0.10,
    test_size=2000,
    n_list=[50,100,200,500,1000,2000],
    latent_dim=5,
    rho=0.01,
    hidden=[64, 64], activation='silu',
    lr=1e-3, weight_decay=5 * 1e-5, batch=256, epochs=2000,
    seed=42,
)


def train(train_lat, sigma, weighted, cfg):
    """Train one DSM. sigma scalar (iso) or (d,) vector (diag)."""
    torch.manual_seed(cfg['seed'])  # identical init both models
    d = train_lat.shape[1]
    model = ScoreNet(d, list(cfg['hidden']), cfg['activation'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    X = torch.tensor(train_lat, dtype=torch.float32)
    N, bs = len(X), min(cfg['batch'], len(train_lat))
    for _ in range(cfg['epochs']):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            b = X[perm[i:i + bs]]
            loss = dsm_loss(model, b, sigma, weighted=weighted)
            opt.zero_grad();
            loss.backward();
            opt.step()
    model.eval()
    return model


def main():
    cfg = CFG
    np.random.seed(cfg['seed'])

    data, gt = load_and_normalize(cfg['dataset'], mode=cfg['norm_mode'])
    flat = data.reshape(-1, data.shape[-1])
    gt_flat = gt.reshape(-1)

    bkg = flat[gt_flat == cfg['bkg_cls']]
    tgt = flat[gt_flat == cfg['target_cls']]
    s_raw = tgt.mean(axis=0)
    print(f"norm={cfg['norm_mode']}  bkg={len(bkg)}  tgt={len(tgt)}  "
          f"||s_raw||={np.linalg.norm(s_raw):.4f}  d={cfg['latent_dim']}\n")

    # PCA fit on ALL pixels (matches the main pipeline)
    pca = PCA(n_components=cfg['latent_dim'], random_state=cfg['seed']).fit(flat)
    s_pca_add = (pca.components_ @ s_raw).astype(np.float32)  # additive: no centering
    s_pca_rep = pca.transform(s_raw[None]).flatten().astype(np.float32)  # replacement: centered

    rows = []
    for n in cfg['n_list']:
        rng = np.random.RandomState(cfg['seed'] + n)
        idx = rng.permutation(len(bkg))
        tr_raw = bkg[idx[:n]]
        te_raw = bkg[idx[n:n + cfg['test_size']]].copy()

        # plant additive targets in raw space
        n_pos = int(cfg['target_fraction'] * len(te_raw))
        pos = rng.choice(len(te_raw), n_pos, replace=False)
        labels = np.zeros(len(te_raw), dtype=int);
        labels[pos] = 1
        te_raw[pos] += cfg['amplitude'] * s_raw

        tr_pca = pca.transform(tr_raw).astype(np.float32)
        te_pca = pca.transform(te_raw).astype(np.float32)

        sigma_iso = compute_sigma_from_data(tr_pca, cfg['rho'])  # scalar
        sigma_diag = compute_sigma_diag_from_data(tr_pca, cfg['rho'])  # (d,)

        m_iso = train(tr_pca, sigma_iso, weighted=False, cfg=cfg)
        m_diag = train(tr_pca, sigma_diag, weighted=True, cfg=cfg)

        auc = lambda sc: roc_auc_score(labels, sc)
        a_iso = auc(dsm_additive(te_pca, tr_pca, m_iso, s_pca_add))
        a_diag = auc(dsm_additive(te_pca, tr_pca, m_diag, s_pca_add))
        r_iso = auc(dsm_replacement(te_pca, tr_pca, m_iso, s_pca_rep))
        r_diag = auc(dsm_replacement(te_pca, tr_pca, m_diag, s_pca_rep))

        rows.append((n, a_iso, a_diag, r_iso, r_diag))
        print(f"n={n:<5}  sigma_iso={sigma_iso:.4f}  "
              f"sigma_diag={np.array2string(sigma_diag, precision=3)}")
        print(f"         ADD : iso={a_iso:.3f}  diag={a_diag:.3f}   "
              f"(Δ={a_diag - a_iso:+.3f})")
        print(f"         REP : iso={r_iso:.3f}  diag={r_diag:.3f}   "
              f"(Δ={r_diag - r_iso:+.3f})\n", flush=True)

    print("=" * 60)
    print(f"{'n':>6} | {'ADD iso':>8} {'ADD diag':>9} | {'REP iso':>8} {'REP diag':>9}")
    print("-" * 60)
    for n, ai, ad, ri, rd in rows:
        print(f"{n:>6} | {ai:>8.3f} {ad:>9.3f} | {ri:>8.3f} {rd:>9.3f}")


if __name__ == '__main__':
    main()
