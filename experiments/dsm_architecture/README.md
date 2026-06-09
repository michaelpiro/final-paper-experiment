# DSM Architecture Study (placeholder)

**Status: to be (re)built.**

This experiment will compare DSM score-network *architectures* on the IID
real-pixel background task, all sharing the same frozen ZCA whitening front-end
(raw data in, whiten internally, data-space score out):

- **Gaussian-LMP** — sample mean/covariance Gaussian score (reference).
- **Affine-DSM-LMP** — single affine layer in whitened coordinates.
- **Bottleneck-DSM-LMP** — linear bottleneck (`D → r → D`) in whitened coordinates.
- (room for deeper / nonlinear cores)

Swept over `n_train`, scored at a fixed planting strength, reported as
P_det @ P_fa. Whitening is fit **from training data only** (relative eigenvalue
floor `λ_max·1e-3`), σ = √ρ in whitened space.

When built, this folder should follow the same self-contained layout as the
other experiments: `baselines.py` (or reuse), `pipeline`/`core.py`, a local
`run.py`, a `config.yaml`, and a `colab_*.ipynb`.
