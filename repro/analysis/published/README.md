# Published amplitude-sweep curves (camera-ready)

`amp_sweep_spatial.pdf` — the full spatial amplitude sweep referenced in the
paper: AUC vs additive amplitude (12 values, theta in [0.03, 0.95]), 5 seeds
with +-1 sigma bands, on Pavia University and San Diego I-II, for the proposed
detectors (DART/DART-CFAR/DARTS/DARTS-CFAR), the classical baselines
(AMF-global/AMF-local/GMM-Levin), the corrected LRao, and the four deep
detectors (THANTD, HTD-Net, TSTTD, OS-VAE).

Per-seed data:
- `theta_results_<scene>.json` — per-seed AUC for every detector and amplitude
  (LRao entries superseded by the definitive validation-early-stopping run:)
- `lrao_val2_metrics_<scene>.json` — the published LRao per-seed metrics
  (val early stopping, 5 seeded models; the LRao curves in the figure).

Regenerate from fresh runs with `repro.analysis.figures.amp_sweep_spatial()`.

The bundled figure was regenerated from the verified single-session rerun
(2026-07-30), which reproduced the published Table-1 rows bit-for-bit
(DARTS via the documented per-scene RNG protocol; OS-VAE within its known
scoring noise). DARTS retrained in isolation is statistically equivalent
but not bit-identical — the training order (DART then DARTS, single stream
on pavia4) is part of the protocol.
