"""
The proposed moment-Randers embedding and the Finsler MDS baseline on the same DiSBM samples.
Three separate figures, to be placed side by side in a document:

    <out>_proposed.pdf     2D spectral embedding, with the mean drift of every community;
    <out>_finsler_mds.pdf  Finsler MDS: canonical Randers space with one more dimension, the vertical axis
                           is the drift axis (same scale on the three axes), as in the paper: omega points
                           up, going up costs more than going down;
    <out>_cosine.pdf       signed cosine between the node drifts and the forward direction of the cycle of
                           communities, as a function of p - q: median over the nodes of all seeds,
                           band = 10-90% range.

The sizes are those of a 5.5 in text column (ICLR): the figures are placed at scale one, 1.4 in, 2.0 in and
2.05 in wide. The two embeddings are drawn at the value of p - q marked in the cosine figure. The coordinates are not
rescaled; the arrows of the proposed figure are the drifts at their own length, in the units of the
coordinates, while the arrow omega of the Finsler MDS figure only shows a direction.

    python -m finsler_embedding.experiments.disbm.plot_comparison [--p-minus-q 0.8] [--out disbm_comparison]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

from finsler_embedding.core.inputs import from_graph
from finsler_embedding.experiments.reproducibility import seed_everything

from .disbm import DiSBMConfig, graph_info
from .embedding import EmbeddingConfig, run_sweep
from .finsler_mds import finsler_mds_embedding
from .plot_direction import GRID, INK, MUTED
from .plot_finsler_mds import community_colors, content_extent, panel

# Categorical slots 1 and 2 of the reference palette (validated together)
PROPOSED, MDS = "#2a78d6", "#eb6834"


def save(fig, path, pad=0.03):
    """Save the figure cropped to what is drawn on it, with ``pad`` inches around."""
    left, right, top, bottom = content_extent(fig)
    height = fig.get_size_inches()[1] * fig.dpi
    pad *= fig.dpi
    fig.savefig(
        path,
        bbox_inches=Bbox.from_extents(
            (left - pad) / fig.dpi, (height - bottom - pad) / fig.dpi, (right + pad) / fig.dpi, (height - top + pad) / fig.dpi
        ),
        dpi=300,
    )
    print(f"Saved {path}")


def ours_figure(adjacency, labels, K):
    """Spectral embedding of the nodes in R^2, with the drift of each community drawn at its own length.

    The arrow of a community is the mean over its nodes of ``result.drift`` (the vector used for the cosine),
    in the units of the embedding coordinates, neither normalised nor rescaled.
    """
    result = from_graph(adjacency, 1.0, theta=1.0, n_components=2)
    coordinates = result.coordinates.numpy()
    coordinates = coordinates - coordinates.mean(0)
    means = np.stack([coordinates[labels == k].mean(0) for k in range(K)])
    drift = np.stack([result.drift[labels == k].mean(0).numpy() for k in range(K)])

    fig = plt.figure(figsize=(1.45, 1.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(*coordinates.T, c=community_colors(labels, K), s=4, linewidths=0)
    ax.quiver(*means.T, *drift.T, angles="xy", scale_units="xy", scale=1, color=INK,
              width=0.008, headwidth=4.5, headlength=5, headaxislength=4.5, zorder=3)
    half = 1.05 * max(np.abs(coordinates).max(), np.abs(means + drift).max())
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, {"ring radius": np.linalg.norm(means, axis=1).mean(), "drift length": np.linalg.norm(drift, axis=1).mean()}


def mds_figure(adjacency, labels, K, alpha, seed):
    """Finsler MDS embedding in R^3, as drawn in the Finsler MDS paper (see ``plot_finsler_mds.panel``).

    The vertical axis is the drift axis, and the three axes share one scale: nothing is rescaled in the
    embedding. The fitted metric is the canonical Randers metric |v| + alpha v_last, with a constant drift
    omega along the vertical axis, pointing up (going up costs more than going down). The arrow omega
    beside the cloud only shows the direction of the drift, its length is not to scale.
    """
    coordinates = finsler_mds_embedding(adjacency, 2, alpha, seed=seed).coordinates.numpy()
    coordinates = coordinates - coordinates.mean(0)
    fig, _ = panel(coordinates, labels, K)
    means = np.stack([coordinates[labels == k].mean(0) for k in range(K)])
    return fig, {"ring radius": np.linalg.norm(means[:, :2], axis=1).mean(), "alpha": alpha,
                 "spread of the heights": coordinates[:, 2].std(), "spread in the plane": coordinates[:, :2].std()}


def cosine_figure(x, stats, delta):
    """Signed cosine with the forward direction versus p - q, one line per method."""
    fig, ax = plt.subplots(figsize=(2.1, 1.5), layout="constrained")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axvline(delta, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    for name, color, q in stats:
        ax.fill_between(x, q[:, 0], q[:, 2], color=color, alpha=0.18, linewidth=0)
        ax.plot(x, q[:, 1], color=color, linewidth=1.2, marker="o", markersize=2.8, markeredgecolor="white",
                markeredgewidth=0.5, solid_capstyle="round", solid_joinstyle="round", label=name, zorder=3)
    ax.set_xlabel(r"$p - q$", color=INK, fontsize=8)
    ax.set_ylabel("signed cosine", color=INK, fontsize=8)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7)
    # Frameless legend in the empty upper-left corner (text in ink, the line key carries the colour)
    ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=INK, handlelength=1.6, borderaxespad=0.2)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-minus-q", type=float, default=0.8, help="Asymmetry at which the embeddings are drawn.")
    parser.add_argument("--seeds", type=int, default=5, help="Graph samples per value of p - q in the cosine figure.")
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Finsler MDS: strength of the canonical drift.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the sample drawn in the two embeddings.")
    parser.add_argument("--format", default="pdf", help="Figure format (pdf, png, ...).")
    parser.add_argument("--out", default="disbm_comparison", help="Prefix: writes <out>_proposed, _finsler_mds, _cosine.")
    args = parser.parse_args()
    K = args.K

    # The embeddings: one sample, embedded by both methods
    seed_everything(args.seed)
    p = (1 - args.r + args.p_minus_q) / 2
    _, _, labels, adjacency = graph_info(p=p, q=1 - p - args.r, r=args.r, K=K, N=args.N)
    labels = labels.numpy()

    # The cosine: sweep of p - q, same graphs for both methods
    # Same grid as plot_direction: regular, refined near p = q
    x = np.unique(np.round(np.concatenate([np.arange(-0.8, 0.81, 0.1), [-0.05, -0.02, 0.02, 0.05]]), 2)) + 0.0
    model = DiSBMConfig(n=args.N, communities=K, r=args.r)
    stats = []
    for name, color, method in (("Ours", PROPOSED, "spectral"), ("Finsler MDS", MDS, "finsler_mds")):
        print(f"cosine sweep: {name}", flush=True)
        config = EmbeddingConfig(method=method, mds_alpha=args.alpha)
        cosine = run_sweep(x, model, config, args.seeds)["cos"]
        for delta, quantiles in zip(x, cosine):
            print(f"  p - q = {delta:+.2f}: median cos {quantiles[1]:+.3f} (10-90%: {quantiles[0]:+.3f}, {quantiles[2]:+.3f})", flush=True)
        stats.append((name, color, cosine))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    figures = {
        "proposed": ours_figure(adjacency, labels, K),
        "finsler_mds": mds_figure(adjacency, labels, K, args.alpha, args.seed),
    }
    for name, (fig, scales) in figures.items():
        print(f"{name}: " + ", ".join(f"{key} {value:.3f}" for key, value in scales.items()), flush=True)  # embedding units, as drawn
        save(fig, f"{args.out}_{name}.{args.format}")
    save(cosine_figure(x, stats, args.p_minus_q), f"{args.out}_cosine.{args.format}")


if __name__ == "__main__":
    main()
