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

# A third box, spatially disjoint from BOTH the train and test boxes, used as the
# VALIDATION region. Chosen automatically by `find_val_box` unless a config gives
# `<scene>_val_box`.
#
# WHY A SEPARATE BOX. The calibrated trainers and ConfigurableLRao select their
# best epoch (and, with `detval`, sigma) on a held-out split. Drawing that split
# from INSIDE the train box makes it identically distributed with training, so it
# selects for fitting that box -- which is the opposite of what the spatial task
# needs. Measured on Pavia-4: the sigma optimum on a same-box hold-out is g=0.53,
# on the real test box it is g=1.69, a 3x error. A disjoint val box reproduces the
# train->test distribution shift, so selection sees the problem it is selecting for.

PAVIA_TRAIN_BOX = [85, 193, 207, 306]
PAVIA_TEST_BOX = [419, 508, 250, 334]

# ---------------------------------------------------------------------------
# pavia_mix — a second Pavia geometry in which a SCORE detector can actually beat
# the linear matched filter. The published pavia4 boxes cannot show that, and the
# reason is measurable:
#
#   * the pavia4 train box is 100% unlabelled and spectrally near-unimodal
#     (held-out GMM6-vs-Gaussian gain +2.6 nats, vs +6.7 for the IID class-mixture
#     pool), so the optimal score there IS the matched filter and there is nothing
#     for a non-linear score net to add;
#   * and the bitumen signature is an EASY direction for a linear filter, so even
#     with a multimodal box there is no headroom left (measured: multimodal boxes
#     alone lift DART-vs-AMF only from +0.001 to +0.022 AUC).
#
# Both have to change together. These boxes are multimodal (train gain +5.4) AND
# the target material (metal sheets, class 5) is a background mode the linear
# filter confuses — it is 10.8% of the train box and 2.9% of the test box, which
# is exactly the regime where modelling the background non-linearly pays. Measured
# (MLP[128], 4 net seeds x 3 planting seeds, DART-normalize minus AMF, AUC):
#     theta 0.075  +0.026 | 0.15  +0.052 | 0.30  +0.068
# and the front-end ordering matches the IID experiment (normalize > zca), which
# the pavia4 geometry inverts. Excluding the real class-5 pixels from scoring
# INCREASES the gap, so this is a detection result, not a mislabelling artefact.
PAVIA_MIX_TRAIN_BOX = [120, 184, 80, 143]     # 64x63 = 4032 px
PAVIA_MIX_TEST_BOX = [136, 225, 144, 229]     # 89x85 = 7565 px, same size as pavia4
PAVIA_MIX_CLS = 5                             # metal sheets — the planted target

# ---------------------------------------------------------------------------
# pavia_foreign — the CONTROL for pavia_mix. Same image, boxes that also span
# several materials, but the target material appears in NONE of them, as in the
# published pavia4. It exists to answer one question: how much of pavia_mix's
# advantage comes from the target being a background mode?
#
# Geometry: bitumen (class 7) occupies rows 296-377 only, so the whole bottom-
# left square below it is bitumen-free. All three boxes are drawn from
# [378,610,0,232] and carry meadows + asphalt + trees + bricks. Their class
# compositions match unusually well (train-vs-test L1 = 0.08, against 0.17 for
# pavia_mix and 0.20 for pavia4), so train->test transfer is as clean as this
# image allows.
#
# WHAT IT MEASURES (MLP[128], ~20k steps, 2 seeds x 3 planting seeds, DART minus
# AMF-global, AUC / Pd@.05):
#     theta 0.15   zca +0.027 / +0.051     normalize -0.056 / -0.052
#     theta 0.30   zca +0.004 / +0.008     normalize -0.047 / -0.210
# i.e. a score detector roughly TIES the matched filter here, and prefers the
# published zca front — whereas on pavia_mix, where the target material is in the
# training box, it wins by +0.23 AUC and prefers 'normalize'. Removing the target
# from training inverts the front-end preference and erases the gain; the swap
# ablation (training on pavia_mix's metal-free val box) reproduces that causally.
#
# Note the region is materially DIVERSE but not spectrally multimodal: k-means
# separation 0.41-0.43 vs 0.60-0.72 for the upper-middle of the image. Several
# named classes is not the same property as several spectral modes.
# TRAIN and VAL are SWAPPED relative to the triple the search returned, and the
# val region is now the TOP STRIP of the train region rather than a separate box:
# rows 418-436 validate, rows 436-507 train (side-cropped to the 4000 budget,
# which centres the crop and leaves a 6-row gap, so the two stay disjoint).
#
# BE AWARE WHAT THIS GIVES UP. An in-region hold-out is identically distributed
# with training, so best-epoch and sigma selection optimise for FITTING the train
# region rather than for TRANSFERRING off it -- the very thing find_val_box was
# written to avoid. Measured on Pavia-4: the sigma optimum is g=0.53 on a
# same-box hold-out against g=1.69 on the real test box, a 3x error. It also
# costs training pixels (4002 instead of 4030) and shrinks the monitor to 1530 px.
# Point `pavia_foreign_val_box` at a disjoint box in the config to get the other
# behaviour back without editing this file.
PAVIA_FOREIGN_TRAIN_BOX = [436, 507, 32, 117]   # -> side_crop -> [442,500,40,109], 4002 px
PAVIA_FOREIGN_TEST_BOX = [442, 531, 144, 229]   # 89x85 = 7565 px
PAVIA_FOREIGN_VAL_BOX = [418, 436, 32, 117]     # 18x85 = 1530 px, top strip of train
PAVIA_FOREIGN_CLS = 6                           # bare soil — absent from all three
# Classes 5 (metal), 6 (bare soil) and 7 (bitumen) are ALL absent from these
# boxes, so `pavia_foreign_cls` can be set to any of them in the config with no
# code change. Which one you pick decides what the pavia_mix comparison means:
#   cls 5  the SAME material pavia_mix plants -> a clean control in which the
#          only difference is whether that material is in the training box
#          (pavia_mix 10.8% train / 2.9% test) or absent from every box;
#   cls 6  bare soil, the target the IID experiment uses;
#   cls 7  bitumen, the target the published pavia4 uses.
# Currently 6, so the pavia_mix comparison also changes the target spectrum --
# set it to 5 if you want that confound removed.


# ---------------------------------------------------------------------------
# salinas — the AGRICULTURAL scene, wired on request. Boxes are pair 0 of the
# multimodality search, with the validation strip taken from pair 2's train box
# (the part not overlapping pair 0's).
#
#   train [96,160,120,183]  4032 px  stubble 31%, fallow_smooth 26%, unlab 24%,
#                                    fallow_plow 7%, fallow 6%, celery 5%
#   test  [0,89,72,157]     7565 px  stubble 33%, unlab 21%, fallow_smooth 19%,
#                                    celery 15%, fallow_plow 7%, vinyard 5%
#   val   [96,160,183,207]  1536 px  unlab 70%, celery 30%
#
# Train and test are genuinely well matched (class-composition L1 = 0.35, the
# best Salinas offers). The VAL strip is not: L1 = 1.27 against test, because
# the only part of pair 2's train box that does not overlap pair 0's is a 24-px
# column sitting on a field boundary, so it is 70% unlabelled plus one crop.
# Treat sigma/best-epoch selection on it with suspicion.
#
# TARGET is class 1 (brocoli green weeds 1) — absent from all three boxes, and
# the only signature for which AMF is not already saturated on this pair
# (AUC 0.981 at theta 0.15; every other admissible class is 0.99-1.00).
#
# READ BEFORE USING. Salinas is a LINEAR problem and DART is measured to lose on
# it: at theta 0.15 the matched filter saturates at AUC 1.000 on 14 of 16
# signatures, and once theta drops to 0.01-0.05 so AMF sits at 0.58-0.81, DART
# loses in all 30 cells tested (-0.03 to -0.19 AUC). Its fields are large,
# low-variance, well-separated clusters, so the matched filter is already near
# optimal. Use THETAS well below the Pavia values or every row reads 1.000.
SALINAS_TRAIN_BOX = [96, 160, 120, 183]     # 64x63 = 4032 px
SALINAS_TEST_BOX = [0, 89, 72, 157]         # 89x85 = 7565 px
SALINAS_VAL_BOX = [96, 160, 183, 207]       # 64x24 = 1536 px
SALINAS_CLS = 1                             # brocoli green weeds 1


def get_salinas():
    """(data HxWxD float64, gt HxW int) from the bundled Salinas .mat files.
    Uses the CORRECTED cube (204 bands; the 20 water-absorption bands removed)."""
    import scipy.io as sio
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = sio.loadmat(os.path.join(here, 'data', 'Salinas_corrected.mat'))
    g = sio.loadmat(os.path.join(here, 'data', 'Salinas_gt.mat'))
    cube = [v for k, v in d.items() if not k.startswith('__')][0]
    lab = [v for k, v in g.items() if not k.startswith('__')][0]
    return cube.astype(np.float64), lab.astype(int)


def _overlaps(a, b, margin=0):
    a = [a[0] - margin, a[1] + margin, a[2] - margin, a[3] + margin]
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _clip(box, H, W):
    return [max(box[0], 0), min(box[1], H), max(box[2], 0), min(box[3], W)]


def _trim_off(box, taken, H, W):
    """Clip `box` to the image, then trim it along whichever single edge removes
    the overlap with the least loss. Candidates are generated with a gap already,
    so a box that merely SHARES an edge is not treated as overlapping."""
    r0, r1, c0, c1 = _clip(box, H, W)
    for _ in range(len(taken) + 1):
        hit = next((t for t in taken if _overlaps([r0, r1, c0, c1], t)), None)
        if hit is None:
            break
        opts = []
        if r0 < hit[0] < r1: opts.append((r1 - hit[0], 'r1', hit[0]))
        if r0 < hit[1] < r1: opts.append((hit[1] - r0, 'r0', hit[1]))
        if c0 < hit[2] < c1: opts.append((c1 - hit[2], 'c1', hit[2]))
        if c0 < hit[3] < c1: opts.append((hit[3] - c0, 'c0', hit[3]))
        if not opts:
            return None                      # fully contained -> nothing to keep
        _, which, v = min(opts)
        r1, r0, c1, c0 = ((v, r0, c1, c0) if which == 'r1' else
                          (r1, v, c1, c0) if which == 'r0' else
                          (r1, r0, v, c0) if which == 'c1' else (r1, r0, c1, v))
    return [r0, r1, c0, c1] if (r1 - r0) > 4 and (c1 - c0) > 4 else None


def _composition(gt, box):
    sub = gt[box[0]:box[1], box[2]:box[3]]
    ids, cnt = np.unique(sub, return_counts=True)
    tot = max(cnt.sum(), 1)
    return {int(i): c / tot for i, c in zip(ids, cnt)}


def find_val_box(gt, train_box, test_box, forbid_cls=(), min_px=400, margin=2):
    """The VALIDATION region: a box that CONTINUES the test box (preferred) or
    the train box, sharing an edge with it.

    Rationale, learned the hard way. A validation set drawn from INSIDE the train
    box is identically distributed with training, so it selects for fitting that
    box rather than for transferring off it (measured on Pavia-4: sigma optimum
    g=0.53 in-box vs g=1.69 on the test box). But an arbitrary disjoint box is no
    better -- the first automatic choice here landed next to the TRAIN box with
    classes {0,1,2,4,5,9} against the test box's {0,1,4}, and its AUC ranking was
    flat and uninformative.

    What the validation region has to do is STAND IN FOR THE TEST REGION, so we
    take an adjacent continuation of it and score candidates by how closely their
    class composition matches the test box. On Pavia-4 that selects
    [330,419,250,334] -- the same size as the test box, immediately above it, and
    classes {0,1,4}, exactly the test composition. Boxes are clipped to the image
    and pulled back off any overlap, which is what makes it work on the 100x100
    San Diego scenes where train+test nearly tile the frame.
    """
    H, W = gt.shape
    ref_comp = _composition(gt, test_box)
    taken = [train_box, test_box]
    cands = []
    for ref, pref in ((test_box, 0), (train_box, 1)):     # prefer continuing TEST
        r0, r1, c0, c1 = ref
        h, w = r1 - r0, c1 - c0
        g = int(margin)
        for side, box in (('below', [r1 + g, r1 + g + h, c0, c1]),
                          ('above', [r0 - g - h, r0 - g, c0, c1]),
                          ('right', [r0, r1, c1 + g, c1 + g + w]),
                          ('left',  [r0, r1, c0 - g - w, c0 - g])):
            b = _trim_off(box, taken, H, W)
            if b is None:
                continue
            npx = (b[1] - b[0]) * (b[3] - b[2])
            if npx < min_px:
                continue
            # Real target pixels must not sit in the validation region (they would
            # be unlabelled positives among the "background"). On the San Diego
            # scenes the train/test boxes were placed to AVOID the aircraft, so
            # every leftover area contains them -- rejecting the candidate outright
            # leaves nothing. Trim around their bounding box instead.
            sub = gt[b[0]:b[1], b[2]:b[3]]
            if any((sub == f).any() for f in forbid_cls):
                mask = np.isin(gt, list(forbid_cls))
                rr, cc = np.where(mask)
                if len(rr):
                    b = _trim_off(b, [[rr.min(), rr.max() + 1,
                                       cc.min(), cc.max() + 1]], H, W)
                if b is None:
                    continue
                sub = gt[b[0]:b[1], b[2]:b[3]]
                if any((sub == f).any() for f in forbid_cls):
                    continue
                npx = (b[1] - b[0]) * (b[3] - b[2])
                if npx < min_px:
                    continue
            comp = _composition(gt, b)
            # L1 distance between class histograms (0 = identical composition)
            keys = set(comp) | set(ref_comp)
            dist = sum(abs(comp.get(k, 0.0) - ref_comp.get(k, 0.0)) for k in keys)
            cands.append((pref, dist, -npx, b, side, 'test' if pref == 0 else 'train'))
    if not cands:
        return None
    cands.sort()
    best = cands[0]
    print(f"    [val box] continues {best[5]} {best[4]}: {best[3]} "
          f"({(best[3][1]-best[3][0])*(best[3][3]-best[3][2])}px, "
          f"class-composition L1 vs test = {best[1]:.3f})", flush=True)
    return [int(v) for v in best[3]]


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
        vb = cfg.get('pavia_val_box') or find_val_box(
            np.asarray(gt, int), tb, test_box, forbid_cls=(PP.FOREIGN_CLS,))
        if vb:
            scene['val'] = np.asarray(PP.crop(data, vb), np.float32)
            scene['val_shape'] = (vb[1] - vb[0], vb[3] - vb[2])
            scene['val_box'] = vb
    elif name == 'salinas':
        data, gt = get_salinas()
        tb = PP.side_crop(list(cfg.get('salinas_train_box', SALINAS_TRAIN_BOX)),
                          int(cfg.get('n_budget', PP.N_BUDGET)))
        test_box = list(cfg.get('salinas_test_box', SALINAS_TEST_BOX))
        cls = int(cfg.get('salinas_cls', SALINAS_CLS))
        tr = PP.crop(data, tb)
        te = PP.crop(data, test_box)
        sig = PP.foreign_signature(data, gt, te, cls=cls)
        scene = dict(name=name, tr=np.asarray(tr, np.float32),
                     tr_shape=(tb[1] - tb[0], tb[3] - tb[2]),
                     te=np.asarray(te, np.float32),
                     te_shape=(test_box[1] - test_box[0], test_box[3] - test_box[2]),
                     sig=np.asarray(sig, np.float32), data=data, gt=gt,
                     test_box=test_box)
        assert len(scene['tr']) == 4032 and len(scene['te']) == 7565, \
            'salinas fingerprint mismatch'
        vb = list(cfg.get('salinas_val_box', SALINAS_VAL_BOX))
        for _b in (tb, test_box, vb):
            assert not (gt[_b[0]:_b[1], _b[2]:_b[3]] == cls).any(), \
                'salinas: the target class must not appear in any box'
        if vb:
            scene['val'] = np.asarray(PP.crop(data, vb), np.float32)
            scene['val_shape'] = (vb[1] - vb[0], vb[3] - vb[2])
            scene['val_box'] = vb
    elif name == 'pavia_foreign':
        # Same recipe as pavia_mix; the val box is PINNED (chosen with the train
        # and test boxes as one matched triple) rather than derived, so
        # find_val_box is not consulted. See PAVIA_FOREIGN_* above.
        data, gt = PP.get_pavia()
        tb = PP.side_crop(list(cfg.get('pavia_foreign_train_box',
                                       PAVIA_FOREIGN_TRAIN_BOX)),
                          int(cfg.get('n_budget', PP.N_BUDGET)))
        test_box = list(cfg.get('pavia_foreign_test_box', PAVIA_FOREIGN_TEST_BOX))
        cls = int(cfg.get('pavia_foreign_cls', PAVIA_FOREIGN_CLS))
        tr = PP.crop(data, tb)
        te = PP.crop(data, test_box)
        sig = PP.foreign_signature(data, gt, te, cls=cls)
        scene = dict(name=name, tr=np.asarray(tr, np.float32),
                     tr_shape=(tb[1] - tb[0], tb[3] - tb[2]),
                     te=np.asarray(te, np.float32),
                     te_shape=(test_box[1] - test_box[0], test_box[3] - test_box[2]),
                     sig=np.asarray(sig, np.float32), data=data, gt=gt,
                     test_box=test_box)
        assert len(scene['tr']) == 4002 and len(scene['te']) == 7565, \
            'pavia_foreign fingerprint mismatch'
        for _b in (tb, test_box):
            assert not (gt[_b[0]:_b[1], _b[2]:_b[3]] == cls).any(), \
                'pavia_foreign: the target class must not appear in any box'
        vb = list(cfg.get('pavia_foreign_val_box', PAVIA_FOREIGN_VAL_BOX))
        if vb:
            assert not (gt[vb[0]:vb[1], vb[2]:vb[3]] == cls).any(), \
                'pavia_foreign: the target class must not appear in the val box'
            scene['val'] = np.asarray(PP.crop(data, vb), np.float32)
            scene['val_shape'] = (vb[1] - vb[0], vb[3] - vb[2])
            scene['val_box'] = vb
    elif name == 'pavia_mix':
        # Same image and same recipe as pavia4 — only the two boxes and the
        # target material differ. See PAVIA_MIX_* above for why.
        data, gt = PP.get_pavia()
        tb = PP.side_crop(list(cfg.get('pavia_mix_train_box', PAVIA_MIX_TRAIN_BOX)),
                          int(cfg.get('n_budget', PP.N_BUDGET)))
        test_box = list(cfg.get('pavia_mix_test_box', PAVIA_MIX_TEST_BOX))
        cls = int(cfg.get('pavia_mix_cls', PAVIA_MIX_CLS))
        tr = PP.crop(data, tb)
        te = PP.crop(data, test_box)
        sig = PP.foreign_signature(data, gt, te, cls=cls)
        scene = dict(name=name, tr=np.asarray(tr, np.float32),
                     tr_shape=(tb[1] - tb[0], tb[3] - tb[2]),
                     te=np.asarray(te, np.float32),
                     te_shape=(test_box[1] - test_box[0], test_box[3] - test_box[2]),
                     sig=np.asarray(sig, np.float32), data=data, gt=gt,
                     test_box=test_box)
        assert len(scene['tr']) == 4032 and len(scene['te']) == 7565, \
            'pavia_mix fingerprint mismatch'
        vb = cfg.get('pavia_mix_val_box') or find_val_box(
            np.asarray(gt, int), tb, test_box, forbid_cls=(cls,))
        if vb:
            scene['val'] = np.asarray(PP.crop(data, vb), np.float32)
            scene['val_shape'] = (vb[1] - vb[0], vb[3] - vb[2])
            scene['val_box'] = vb
    else:
        sc = SDP.build(name)                    # asserts its own fingerprints
        rj = json.load(open(SDP._find(f'{name}_regions.json')))
        tb = rj['train_box']
        scene = dict(name=name, tr=np.asarray(sc['tr'], np.float32),
                     tr_shape=(tb[1] - tb[0], tb[3] - tb[2]),
                     te=np.asarray(sc['te'], np.float32), te_shape=sc['te_shape'],
                     sig=np.asarray(sc['sig'], np.float32), data=sc['data'],
                     gt=sc['gt'], test_box=rj['test_box'])
        vb = cfg.get(f'{name}_val_box') or rj.get('val_box') or find_val_box(
            np.asarray(sc['gt'], int), tb, rj['test_box'], forbid_cls=(1,))
        if vb:
            d3 = sc['data']
            scene['val'] = np.asarray(
                d3[vb[0]:vb[1], vb[2]:vb[3]].reshape(-1, d3.shape[-1]), np.float32)
            scene['val_shape'] = (vb[1] - vb[0], vb[3] - vb[2])
            scene['val_box'] = vb
    _v = (f"  val {len(scene['val'])} px {scene['val_shape']} @{scene['val_box']}"
          if 'val' in scene else "  val <none found>")
    print(f"[{name}] train {len(scene['tr'])} px {scene['tr_shape']}  "
          f"test {len(scene['te'])} px {scene['te_shape']}{_v}  "
          f"||s||={np.linalg.norm(scene['sig']):.1f}", flush=True)
    return scene
