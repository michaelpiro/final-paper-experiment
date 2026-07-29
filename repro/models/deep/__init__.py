"""Deep target-detector baselines (paper-faithful ports, validated against
their published results during the rebuttal): THANTD, HTD-Net, TSTTD (vendor
code kept verbatim inside its package), OS-VAE.

Uniform wrapper API per package:
    fit(tr, sig, cfg, seed, ckpt, device) -> state
    score(state, planted, sig, device)    -> (n,) scores
Every fit seeds torch/numpy BEFORE any model construction (the THANTD
seed-order bug found in the camera-ready code is fixed here).
Training budgets come from configs (spatial.yaml `deep:` section).
"""
from .htdnet import HTDNet    # noqa: F401
from .osvae import OSVAE      # noqa: F401
from .thantd import THANTD    # noqa: F401
from .tsttd import TSTTD      # noqa: F401

REGISTRY = {'THANTD': THANTD, 'HTDNet': HTDNet, 'TSTTD': TSTTD, 'OSVAE': OSVAE}
