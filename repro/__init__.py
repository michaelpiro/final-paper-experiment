"""repro — the clean, self-contained camera-ready experiment package.

One implementation, no vendored duplicates, no hardcoded hyperparameters:
every knob lives in repro/configs/*.yaml. Every trainer seeds torch/numpy
BEFORE model construction, so each (detector, seed) is a reproducible model.

Layout:
  core/       data loading, planting, whitening, metrics (verbatim ports of
              the paper pipeline's src/{data,models,detectors,metrics}.py)
  scenes/     spatial scene builders (Pavia scn4, San Diego I-II), fingerprint-asserted
  models/     one package per detector family: dart, darts, lrao, classical,
              deep/{thantd, htdnet, tsttd, osvae}
  protocols/  the two experiment runners: spatial.py, iid.py
  analysis/   paper figures + tables + verification against published numbers
"""
