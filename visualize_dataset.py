"""
Helper for visualizing hyperspectral datasets and interactively selecting polygons.

Usage as a script:
    python visualize_dataset.py --dataset real_datasets/Sandiego.mat
"""

import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


def load_dataset(path: str):
    """Load a .mat hyperspectral dataset. Returns (data H×W×B normalized to [0,1], gt_map H×W)."""
    mat = scipy.io.loadmat(path)
    data = mat["data"].astype(np.float64)
    lo, hi = data.min(), data.max()
    if hi > lo:
        data = (data - lo) / (hi - lo)
    gt_map = mat.get("map", np.zeros(data.shape[:2]))
    return data, gt_map


def false_color(data: np.ndarray, bands=(50, 30, 10)) -> np.ndarray:
    """
    Build an RGB false-color image from three spectral bands.
    Applies 2nd–98th percentile stretch per channel to [0, 1].
    data: H×W×B
    Returns: H×W×3 float in [0,1]
    """
    H, W, B = data.shape
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    for i, b in enumerate(bands):
        b = min(b, B - 1)
        ch = data[:, :, b].astype(np.float64)
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi > lo:
            ch = (ch - lo) / (hi - lo)
        ch = np.clip(ch, 0, 1)
        rgb[:, :, i] = ch
    return rgb


def select_polygon(image: np.ndarray, title: str = "Draw polygon, then close window") -> np.ndarray:
    """
    Show the image and let the user draw a polygon with PolygonSelector.
    Returns a boolean mask (H×W) of pixels inside the polygon.
    Left-click to add vertices, right-click to finish.
    """
    H, W = image.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(title, fontsize=10)

    verts_holder = [None]

    def on_select(verts):
        verts_holder[0] = verts

    selector = PolygonSelector(ax, on_select, useblit=True,
                               props=dict(color='yellow', linewidth=2))

    print(f"[Polygon selector] {title}")
    print("  Click to add vertices. Press Enter or close window when done.")
    plt.tight_layout()
    plt.show(block=True)

    if verts_holder[0] is None:
        print("  Warning: no polygon was drawn, returning empty mask.")
        return np.zeros((H, W), dtype=bool)

    verts = np.array(verts_holder[0])
    path = Path(verts)

    # Build pixel coordinate grid (col, row) = (x, y)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.stack([cols.ravel(), rows.ravel()], axis=1).astype(float)
    mask = path.contains_points(coords).reshape(H, W)
    return mask


def get_pixels_in_polygon(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract all pixels inside mask from a H×W×B hyperspectral image.
    Returns (N, B) array where N = number of True pixels in mask.
    """
    return data[mask]  # boolean indexing gives (N, B)


def overlay_polygons(image: np.ndarray, masks: list, labels: list = None,
                     colors=None, save_path: str = None):
    """
    Draw polygon outlines on the false-color image and optionally save.
    masks: list of (H×W) bool arrays
    """
    if colors is None:
        colors = ['yellow', 'red', 'cyan', 'lime']
    if labels is None:
        labels = [f"Region {i+1}" for i in range(len(masks))]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    from matplotlib.patches import Patch
    legend_handles = []
    for mask, label, color in zip(masks, labels, colors):
        # Draw contour of the mask
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=2)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=label))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
    ax.set_title("False-color image with selected regions")
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show(block=True)
    plt.close(fig)


# --- Script entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hyperspectral dataset")
    parser.add_argument("--dataset", default="real_datasets/Sandiego.mat",
                        help="Path to .mat file")
    parser.add_argument("--bands", nargs=3, type=int, default=[50, 30, 10],
                        help="Three band indices for false-color (R G B)")
    args = parser.parse_args()

    data, gt_map = load_dataset(args.dataset)
    print(f"Loaded {args.dataset}: shape={data.shape}")

    rgb = false_color(data, bands=tuple(args.bands))

    print("\nStep 1: Draw background polygon")
    bkg_mask = select_polygon(rgb, title="Background region — draw polygon, close when done")
    print(f"  Background pixels selected: {bkg_mask.sum()}")

    print("\nStep 2: Draw target polygon")
    tgt_mask = select_polygon(rgb, title="Target region — draw polygon, close when done")
    print(f"  Target pixels selected: {tgt_mask.sum()}")

    overlay_polygons(rgb, [bkg_mask, tgt_mask],
                     labels=["Background", "Target"],
                     colors=["yellow", "red"])
