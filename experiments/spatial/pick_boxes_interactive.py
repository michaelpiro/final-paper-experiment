"""
pick_boxes_interactive.py — Interactive tool to select 4 (train, test) box pairs.

Usage (run LOCALLY, not on Colab):
    .venv/bin/python experiments/spatial/pick_boxes_interactive.py

Displays the Pavia-U false-color image (bands 60/30/10) alongside the GT label
colormap. For each of 4 pairs:
  1. "Draw TRAIN box #{i}: click + drag"
  2. "Draw TEST  box #{i}: click + drag"

Shows class breakdown text after each box. Saves to:
    experiments/spatial/manual_boxes.json

Keys:
    u / r     — undo last box
    Enter / s — accept current pair and move to next
    q / Esc   — quit without saving
"""

import argparse, json, os, sys

import numpy as np
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector

_EXP  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_EXP))

CLS_NAMES = {
    0: 'unlabeled', 1: 'asphalt', 2: 'meadows', 3: 'gravel',
    4: 'trees',     5: 'metal_sheets', 6: 'bare_soil', 7: 'bitumen',
    8: 'bricks',    9: 'shadows',
}
CLS_COLORS = {
    0: '#000000', 1: '#808080', 2: '#00ff00', 3: '#d2691e',
    4: '#006400', 5: '#add8e6', 6: '#a52a2a', 7: '#800080',
    8: '#ff4500', 9: '#00008b',
}

N_PAIRS = 4


def _false_color(data, bands=(60, 30, 10)):
    rgb = data[..., list(bands)].astype(np.float32)
    lo  = np.percentile(rgb, 2,  axis=(0, 1), keepdims=True)
    hi  = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)


def _gt_colormap(gt):
    H, W = gt.shape
    img  = np.zeros((H, W, 3), dtype=np.float32)
    from matplotlib.colors import to_rgb
    for cid, hex_col in CLS_COLORS.items():
        mask = (gt == cid)
        img[mask] = to_rgb(hex_col)
    return img


def _box_stats(gt, box):
    r0, r1, c0, c1 = box
    patch = gt[r0:r1, c0:c1].ravel()
    cls_ids, cnts = np.unique(patch, return_counts=True)
    total = int(cnts.sum())
    stats = {CLS_NAMES.get(int(c), f'cls{c}'): int(n)
             for c, n in zip(cls_ids, cnts)}
    dom = CLS_NAMES.get(int(cls_ids[cnts.argmax()]), f'cls?')
    return stats, total, dom


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default=os.path.join(_ROOT, 'data', 'pavia-u.mat'))
    p.add_argument('--out', default=os.path.join(_EXP, 'manual_boxes.json'))
    p.add_argument('--n-pairs', type=int, default=N_PAIRS)
    args = p.parse_args()

    if not os.path.exists(args.dataset):
        sys.exit(f"Dataset not found: {args.dataset}")

    mat  = scipy.io.loadmat(args.dataset)
    data = mat['data'].astype(np.float32)
    gt   = mat['map'].astype(int)
    H, W, B = data.shape
    print(f"Loaded {args.dataset}: {H}×{W}×{B}")

    false_color = _false_color(data)
    gt_img      = _gt_colormap(gt)

    # Try to use an interactive backend
    for bk in ('MacOSX', 'Qt5Agg', 'Qt4Agg', 'TkAgg', 'WXAgg'):
        try:
            matplotlib.use(bk)
            break
        except Exception:
            continue

    saved_pairs = []   # list of dicts with train_box, test_box, stats

    for pair_idx in range(args.n_pairs):
        for role_idx, role in enumerate(('TRAIN', 'TEST')):
            color = 'lime' if role == 'TRAIN' else 'red'
            print(f"\n── Pair {pair_idx+1}/{args.n_pairs}: draw {role} box ──")
            print(f"   Click + drag on the image.  Enter=accept  u=undo  q=quit")

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            ax_fc, ax_gt = axes
            ax_fc.imshow(false_color)
            ax_fc.set_title(f'False color   |  Pair {pair_idx+1}, {role} box', fontsize=11)
            ax_fc.set_xlabel(f'col (0..{W-1})'); ax_fc.set_ylabel(f'row (0..{H-1})')

            ax_gt.imshow(gt_img)
            ax_gt.set_title('Ground truth classes', fontsize=11)
            legend_handles = [
                mpatches.Patch(color=CLS_COLORS[c], label=f'{c}: {CLS_NAMES[c]}')
                for c in sorted(CLS_COLORS)
            ]
            ax_gt.legend(handles=legend_handles, loc='lower right',
                         fontsize=7, framealpha=0.7)

            # Overlay already-saved boxes on both panels
            for prev_idx, pair_data in enumerate(saved_pairs):
                for prev_role, prev_color in [('train_box', 'lime'), ('test_box', 'red')]:
                    if prev_role not in pair_data:
                        continue
                    b = pair_data[prev_role]
                    r0, r1, c0, c1 = b
                    for a in (ax_fc, ax_gt):
                        rect = mpatches.Rectangle((c0, r0), c1-c0, r1-r0,
                                                  lw=1.5, edgecolor=prev_color,
                                                  facecolor='none', alpha=0.5,
                                                  linestyle='--')
                        a.add_patch(rect)
                        a.text(c0+2, r0+10, f'P{prev_idx+1}',
                               color=prev_color, fontsize=8, alpha=0.7)

            # Overlay current pair's already-drawn box (if role==TEST)
            state = {'box': None, 'rect_fc': None, 'rect_gt': None,
                     'accepted': False, 'abort': False}
            if role_idx == 1 and pair_idx < len(saved_pairs) and 'train_box' in saved_pairs[pair_idx]:
                b = saved_pairs[pair_idx]['train_box']
                r0, r1, c0, c1 = b
                for a, ec in [(ax_fc, 'lime'), (ax_gt, 'lime')]:
                    rect = mpatches.Rectangle((c0, r0), c1-c0, r1-r0,
                                              lw=2, edgecolor=ec, facecolor='none')
                    a.add_patch(rect)
                    a.text(c0+2, r0+10, f'P{pair_idx+1} TRAIN',
                           color='lime', fontsize=9)

            fig.suptitle(
                f'Pair {pair_idx+1}/{args.n_pairs} — Draw {role} box  '
                f'[u=undo  Enter/s=accept  q=quit]',
                fontsize=13, fontweight='bold',
                color='lime' if role == 'TRAIN' else 'tomato'
            )
            fig.tight_layout()

            info_text = fig.text(0.5, 0.01, 'No box selected yet.',
                                  ha='center', fontsize=10, color='white',
                                  bbox=dict(facecolor='#222', alpha=0.8))

            def on_select(eclick, erelease):
                x0, x1 = sorted([eclick.xdata, erelease.xdata])
                y0, y1 = sorted([eclick.ydata, erelease.ydata])
                c0_, c1_ = max(0, int(round(x0))), min(W, int(round(x1)))
                r0_, r1_ = max(0, int(round(y0))), min(H, int(round(y1)))
                if r1_ - r0_ < 5 or c1_ - c0_ < 5:
                    print(f"   Box too small ({r1_-r0_}×{c1_-c0_}); try again.")
                    return
                state['box'] = (r0_, r1_, c0_, c1_)
                # Remove old rectangles
                if state['rect_fc']: state['rect_fc'].remove()
                if state['rect_gt']: state['rect_gt'].remove()
                state['rect_fc'] = mpatches.Rectangle(
                    (c0_, r0_), c1_-c0_, r1_-r0_,
                    lw=2, edgecolor=color, facecolor='none')
                state['rect_gt'] = mpatches.Rectangle(
                    (c0_, r0_), c1_-c0_, r1_-r0_,
                    lw=2, edgecolor=color, facecolor='none')
                ax_fc.add_patch(state['rect_fc'])
                ax_gt.add_patch(state['rect_gt'])
                stats, total, dom = _box_stats(gt, (r0_, r1_, c0_, c1_))
                comp_str = '  '.join(f"{k}={v}" for k, v in sorted(stats.items(),
                                                                      key=lambda x: -x[1])[:5])
                info_text.set_text(
                    f"[{role}] rows [{r0_}:{r1_}] cols [{c0_}:{c1_}]  "
                    f"{total} px  dominant={dom}  |  {comp_str}"
                )
                print(f"   [{role}] rows[{r0_}:{r1_}] cols[{c0_}:{c1_}]  "
                      f"{total} px  dominant={dom}")
                print(f"   Classes: {comp_str}")
                fig.canvas.draw()

            def on_key(event):
                if event.key in ('u', 'r'):
                    if state['rect_fc']: state['rect_fc'].remove(); state['rect_fc'] = None
                    if state['rect_gt']: state['rect_gt'].remove(); state['rect_gt'] = None
                    state['box'] = None
                    info_text.set_text('Box cleared — redraw.')
                    fig.canvas.draw()
                    print("   Cleared box.")
                elif event.key in ('enter', 's'):
                    if state['box'] is None:
                        print("   No box drawn yet. Click + drag first.")
                        return
                    state['accepted'] = True
                    plt.close(fig)
                elif event.key in ('escape', 'q'):
                    state['abort'] = True
                    plt.close(fig)

            sel = RectangleSelector(
                ax_fc, on_select, useblit=True,
                button=[1], minspanx=5, minspany=5, spancoords='pixels',
                interactive=False)
            fig.canvas.mpl_connect('key_press_event', on_key)

            plt.show()

            if state['abort']:
                print("\nAborted. No file saved.")
                sys.exit(0)

            if not state['accepted'] or state['box'] is None:
                print("Window closed without accepting. Retrying this box.")
                # retry same box (loop will redo)
                # to avoid infinite loop, just exit
                print("Exiting without saving.")
                sys.exit(1)

            box_coords = state['box']
            stats, total, dom = _box_stats(gt, box_coords)

            if role_idx == 0:
                # Starting a new pair
                saved_pairs.append({
                    'train_box': list(box_coords),
                    'train_stats': stats,
                    'train_total': total,
                    'train_dominant': dom,
                })
            else:
                # Completing the pair
                saved_pairs[pair_idx]['test_box']        = list(box_coords)
                saved_pairs[pair_idx]['test_stats']      = stats
                saved_pairs[pair_idx]['test_total']      = total
                saved_pairs[pair_idx]['test_dominant']   = dom

            print(f"   ✓ Pair {pair_idx+1} {role}: rows{list(box_coords[:2])} "
                  f"cols{list(box_coords[2:])}  {total} px  dominant={dom}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"All {args.n_pairs} pairs selected:")
    for i, pair in enumerate(saved_pairs):
        print(f"  Pair {i+1}:")
        print(f"    TRAIN: {pair['train_box']}  "
              f"{pair['train_total']} px  dom={pair['train_dominant']}")
        print(f"    TEST:  {pair['test_box']}  "
              f"{pair['test_total']} px  dom={pair['test_dominant']}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(saved_pairs, f, indent=2)
    print(f"\nSaved to: {args.out}")
    print("You can now run:  python run_colab.py --config colab.yaml")


if __name__ == '__main__':
    main()
