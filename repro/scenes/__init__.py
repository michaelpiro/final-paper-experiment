"""Scene builders for the spatial protocol. build(name, cfg) returns a dict
   {name, tr, tr_shape, te, te_shape, sig, data[, gt]} of raw float32 arrays,
   with the archive fingerprints asserted at build time.

  pavia4    — Pavia University scenario 4: train box side-cropped to the 4000
              budget, foreign bitumen signature scaled to the test-pixel norm.
  sandiego / sandiego2 — archive-exact half-open boxes from the region files,
              signature = GT-aircraft mean scaled to the MEDIAN test-pixel norm.
"""
import json
import os

import numpy as np

from . import pavia_protocol as PP
from . import sandiego_protocol as SDP

PAVIA_TRAIN_BOX = [85, 193, 207, 306]
PAVIA_TEST_BOX = [419, 508, 250, 334]


def build(name, cfg=None):
    cfg = cfg or {}
    if name == 'pavia4':
        data, gt = PP.get_pavia()
        tb = PP.side_crop(list(cfg.get('pavia_train_box', PAVIA_TRAIN_BOX)),
                          int(cfg.get('n_budget', PP.N_BUDGET)))
        test_box = list(cfg.get('pavia_test_box', PAVIA_TEST_BOX))
        tr = PP.crop(data, tb)
        te = PP.crop(data, test_box)
        sig = PP.foreign_signature(data, gt, te)
        scene = dict(name=name, tr=np.asarray(tr, np.float32),
                     tr_shape=(tb[1] - tb[0], tb[3] - tb[2]),
                     te=np.asarray(te, np.float32),
                     te_shape=(test_box[1] - test_box[0], test_box[3] - test_box[2]),
                     sig=np.asarray(sig, np.float32), data=data, gt=gt,
                     test_box=test_box)
        assert len(scene['tr']) == 4026 and abs(np.linalg.norm(sig) - 14357.9) < 1.0, \
            'pavia4 fingerprint mismatch'
    else:
        sc = SDP.build(name)                    # asserts its own fingerprints
        rj = json.load(open(SDP._find(f'{name}_regions.json')))
        tb = rj['train_box']
        scene = dict(name=name, tr=np.asarray(sc['tr'], np.float32),
                     tr_shape=(tb[1] - tb[0], tb[3] - tb[2]),
                     te=np.asarray(sc['te'], np.float32), te_shape=sc['te_shape'],
                     sig=np.asarray(sc['sig'], np.float32), data=sc['data'],
                     gt=sc['gt'], test_box=rj['test_box'])
    print(f"[{name}] train {len(scene['tr'])} px {scene['tr_shape']}  "
          f"test {len(scene['te'])} px {scene['te_shape']}  "
          f"||s||={np.linalg.norm(scene['sig']):.1f}", flush=True)
    return scene
