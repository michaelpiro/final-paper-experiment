"""
Interactive box picker for the CF-Attention spatial experiment.

Opens a window showing the Pavia-U false-color image (also dimming pixels of
the target class so you can SEE where the trees are — including the unlabeled
ones the GT mask missed).  You then drag TWO rectangles:

  1.  TRAIN  (green)
  2.  TEST   (red)

Both must avoid the highlighted target-class pixels.  When you press Enter,
the coordinates are written into cfattn.yaml (under train_box / test_box).

Usage:
    .venv/bin/python pick_boxes.py
    .venv/bin/python pick_boxes.py --config cfattn.yaml          # default
    .venv/bin/python pick_boxes.py --target-cls 1                # override

Controls:
    drag           draw a rectangle
    `c` or right   clear the rectangle currently being drawn
    `u`            undo (clear ALL drawn rectangles, start over)
    Enter / `s`    save the two rectangles and exit
    Esc / `q`      quit without saving
"""

import argparse
import os
import sys

import numpy as np
import scipy.io
import yaml
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cfattn.yaml'),
                   help='YAML to update with train_box / test_box')
    p.add_argument('--target-cls', type=int, default=None,
                   help='Override target class (default: read from yaml)')
    args = p.parse_args()

    # ----- load config -----
    cfg_path = os.path.abspath(args.config)
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}"); sys.exit(1)
    cfg = yaml.safe_load(open(cfg_path))
    target_cls = args.target_cls if args.target_cls is not None else cfg.get('target_cls', None)
    dataset = cfg['dataset']
    if not os.path.isabs(dataset):
        dataset = os.path.join(os.path.dirname(cfg_path), dataset)

    # ----- load image + GT -----
    mat = scipy.io.loadmat(dataset)
    data = mat['data'].astype(np.float32)        # (H, W, 103)
    gt   = mat['map'].astype(int)                # (H, W)
    H, W, B = data.shape
    print(f"Loaded {dataset}: {H}x{W}x{B}, target_cls={target_cls}")

    # ----- false-color RGB (bands ~ R/G/B for ROSIS) -----
    bands = (60, 30, 10)
    rgb = data[..., list(bands)]
    lo  = np.percentile(rgb, 2.0, axis=(0, 1), keepdims=True)
    hi  = np.percentile(rgb, 98.0, axis=(0, 1), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)

    # ----- target-class overlay (RED tint, only if target_cls is set) -----
    overlay = rgb.copy()
    if target_cls is not None:
        tgt_mask = (gt == target_cls)
        overlay[tgt_mask] = np.array([1.0, 0.1, 0.1])
        n_labeled_tgt = int(tgt_mask.sum())
        print(f"Labeled target-class ({target_cls}) pixels: {n_labeled_tgt}")
        print("These appear in BRIGHT RED — avoid them when drawing boxes.")
    else:
        n_labeled_tgt = 0
        print("No target_cls set — all classes shown as-is. Pick boxes freely.")

    # ----- interactive selectors -----
    for backend in ('MacOSX', 'Qt5Agg', 'Qt4Agg', 'WXAgg'):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(overlay)
    ax.set_title(
        (f"target cls {target_cls}: {n_labeled_tgt} labeled (red).  " if target_cls is not None else "") +
        "1) drag TRAIN  2) drag TEST.  Enter=save  u=undo  q=quit",
        fontsize=10)
    ax.set_xlabel(f"col (0..{W-1})"); ax.set_ylabel(f"row (0..{H-1})")

    boxes = []   # list of (r0, r1, c0, c1)
    patches = []
    LABELS = ['TRAIN (green)', 'TEST (red)']
    COLORS = ['lime', 'red']

    def on_select(eclick, erelease):
        if len(boxes) >= 2:
            print("Already have train+test. Press u to undo, or Enter to save.")
            return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        c0, c1 = int(round(x0)), int(round(x1))
        r0, r1 = int(round(y0)), int(round(y1))
        c0 = max(0, c0); c1 = min(W, c1); r0 = max(0, r0); r1 = min(H, r1)
        if r1 - r0 < 5 or c1 - c0 < 5:
            print(f"Rectangle too small ({r1-r0}x{c1-c0}); ignoring.")
            return

        # check target-class pixels inside (only if target_cls is set)
        n_tgt_inside = int(np.sum(gt[r0:r1, c0:c1] == target_cls)) if target_cls is not None else 0
        CLS_NAMES = {0:'unlabeled', 1:'asphalt', 2:'meadows', 3:'gravel',
                     4:'trees', 5:'metal_sheets', 6:'bare_soil', 7:'bitumen',
                     8:'bricks', 9:'shadows'}
        cls, cnt = np.unique(gt[r0:r1, c0:c1], return_counts=True)
        comp = ", ".join(f"{CLS_NAMES.get(int(c), f'cls{c}')}={int(n)}"
                        for c, n in zip(cls, cnt))

        idx = len(boxes)
        boxes.append((r0, r1, c0, c1))
        rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                             linewidth=2, edgecolor=COLORS[idx],
                             facecolor='none')
        ax.add_patch(rect); patches.append(rect)
        ax.text(c0 + 3, r0 + 12, LABELS[idx], color=COLORS[idx],
                fontsize=10, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

        warn = "" if n_tgt_inside == 0 else f"  WARNING: {n_tgt_inside} target px inside!"
        print(f"{LABELS[idx]}: rows [{r0}:{r1}] cols [{c0}:{c1}] "
              f"({(r1-r0)*(c1-c0)} px)  {comp}{warn}")
        fig.canvas.draw()

    def on_key(event):
        if event.key in ('u', 'r'):
            print("Undo: cleared all rectangles.")
            for p in patches: p.remove()
            patches.clear(); boxes.clear()
            for t in list(ax.texts): t.remove()
            fig.canvas.draw()
        elif event.key in ('enter', 's'):
            if len(boxes) != 2:
                print(f"Need exactly 2 rectangles (have {len(boxes)}).")
                return
            cfg['train_box'] = list(boxes[0])
            cfg['test_box']  = list(boxes[1])
            yaml.dump(cfg, open(cfg_path, 'w'),
                      sort_keys=False, default_flow_style=False)
            print(f"\nSaved to {cfg_path}:")
            print(f"  train_box: {cfg['train_box']}")
            print(f"  test_box:  {cfg['test_box']}")
            plt.close(fig)
        elif event.key in ('escape', 'q'):
            print("Quit without saving.")
            plt.close(fig)

    selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1],                                # left click only
        minspanx=5, minspany=5, spancoords='pixels',
        interactive=False)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\nDraw TRAIN first, then TEST. Press Enter when done.\n")
    plt.show()


if __name__ == '__main__':
    main()
