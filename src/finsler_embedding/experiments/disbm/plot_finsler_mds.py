"""
Finsler MDS embedding of a DiSBM, as drawn in the Finsler MDS paper: the nodes live in a canonical
Randers space with one dimension more than the data, whose vertical axis carries the asymmetry
(going up costs more than going down). Nodes are coloured by community; the three axes share the
same scale, so the height is directly comparable with the spread of the communities. One file per
value of p - q, in the same style as plot_direction.

    python -m finsler_embedding.experiments.disbm.plot_finsler_mds [--p-minus-q 0 0.4 0.8] [--out disbm_finsler_mds]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox

from finsler_embedding.experiments.reproducibility import seed_everything

from .disbm import graph_info
from .finsler_mds import finsler_mds_embedding

INK = "#0b0b0b"


def embed(delta, r, K, N, alpha, embedding_dim, seed):
    """Finsler MDS coordinates (N, embedding_dim + 1) and community labels (N,) of one DiSBM sample."""
    seed_everything(seed)
    p = (1 - r + delta) / 2
    _, _, labels, adjacency = graph_info(p=p, q=1 - p - r, r=r, K=K, N=N)
    fit = finsler_mds_embedding(adjacency, embedding_dim, alpha, seed=seed)
    coordinates = fit.coordinates.numpy()
    return coordinates - coordinates.mean(0), labels.numpy()


def community_colors(labels, K):
    """One colour per community: cyclic hues (the communities form a cycle), softened to stay readable on white."""
    return hsv_to_rgb(np.stack([labels / K, np.full(len(labels), 0.6), np.full(len(labels), 0.85)], axis=1))


def content_extent(fig):
    """Pixel extent (left, right, top, bottom, top-down rows) of everything drawn on the figure."""
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    rows, cols = np.nonzero((image < 250).any(axis=2))
    return cols.min(), cols.max() + 1, rows.min(), rows.max() + 1


def panel(coordinates, labels, K, delta=None, elev=18, azim=-62, *, colors=None, size=4, figsize=(3.2, 2.4)):
    """3D scatter, last axis vertical, one scale on the three axes (the box is cut to the data).

    ``delta`` is the value of p - q written above the cloud; nothing is written if it is None. The points
    are colored by community unless ``colors`` is given (for data without communities, pass ``colors``).
    The arrow omega beside the cloud is the vertical axis of the view, the direction of the drift.
    Returns the figure and the box (inches) that crops it to its content.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", proj_type="ortho")
    half = 1.05 * np.abs(coordinates[:, :2]).max()
    half_z = max(0.2 * half, 1.2 * np.abs(coordinates[:, 2]).max())
    if colors is None:
        colors = community_colors(labels, K)
    ax.scatter(*coordinates.T, c=colors, s=size, linewidths=0, depthshade=False)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half_z, half_z)
    ax.set_box_aspect((half, half, half_z), zoom=1.1)  # same data-to-box scale on the three axes
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    left, right, top, bottom = content_extent(fig)
    width, height = fig.get_size_inches() * fig.dpi
    # Direction of the drift omega, beside the cloud: the vertical axis of the view (length exaggerated)
    x = (left - 0.12 * fig.dpi) / width
    middle = 1 - 0.5 * (top + bottom) / height
    length = 0.5 * fig.dpi / height
    fig.add_artist(FancyArrowPatch(
        (x, middle - length / 2), (x, middle + length / 2), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=10, color=INK, linewidth=1.2, shrinkA=0, shrinkB=0,
    ))
    fig.text(x, middle + length / 2 + 0.02, r"$\omega$", color=INK, fontsize=10, ha="center", va="bottom")
    if delta is not None:
        fig.text(0.5 * (left + right) / width, 1 - (top - 0.08 * fig.dpi) / height, rf"$p - q = {delta:g}$",
                 color=INK, fontsize=9, ha="center", va="bottom")

    left, right, top, bottom = content_extent(fig)
    pad = 0.03 * fig.dpi
    crop = Bbox.from_extents(
        (left - pad) / fig.dpi, (height - bottom - pad) / fig.dpi, (right + pad) / fig.dpi, (height - top + pad) / fig.dpi
    )
    return fig, crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-minus-q", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Strength of the canonical drift.")
    parser.add_argument("--embedding-dim", type=int, default=2, help="Dimension of the data; the embedding has one more.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", default="pdf", help="Figure format (pdf, png, ...).")
    parser.add_argument("--out", default="disbm_finsler_mds", help="Prefix: writes <out>_pq<p-q>.<format>.")
    args = parser.parse_args()
    if args.embedding_dim != 2:
        parser.error("the plot is a 3D view: use --embedding-dim 2")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    for delta in args.p_minus_q:
        coordinates, labels = embed(delta, args.r, args.K, args.N, args.alpha, args.embedding_dim, args.seed)
        height = coordinates[:, -1]
        print(
            f"p - q = {delta:+.2f}: spread of the heights {height.std():.3f} "
            f"(of the first two axes {coordinates[:, :2].std():.3f})",
            flush=True,
        )
        path = f"{args.out}_pq{delta:g}.{args.format}"
        fig, crop = panel(coordinates, labels, args.K, delta)
        fig.savefig(path, bbox_inches=crop, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
