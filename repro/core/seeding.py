"""Seeding contract: call seed_all(seed) BEFORE constructing any model.

The init must be a function of the seed alone — not of whatever the session
trained previously. (This exact bug was found and fixed in the camera-ready
LRao and THANTD trainers; every trainer in this package follows the contract.)
"""
import numpy as np
import torch


def seed_all(seed: int):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
