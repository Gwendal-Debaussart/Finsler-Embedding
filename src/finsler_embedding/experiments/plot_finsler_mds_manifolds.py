"""Finsler MDS embeddings of the Swiss roll and Matsumoto datasets, in the style of the Finsler MDS paper.

The nodes live in the canonical Randers space of dimension 3 (2 + 1), whose vertical axis carries the
asymmetry (going up costs more than going down). The three axes share one scale and nothing is
rescaled, the arrow ``omega`` beside the cloud only shows the direction of the constant drift. The
Swiss roll is colored by its angle, the Matsumoto surface by its height.

    python -m finsler_embedding.experiments.plot_finsler_mds_manifolds [--format pdf] [--out finsler_mds]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .disbm.plot_comparison import save
from .disbm.plot_finsler_mds import panel
from .finsler_mds_manifolds import run

# Thesis palette, as in the figures of the experiments
TWO_TONE = LinearSegmentedColormap.from_list("two_tone", ["#072AC8", "#9A44C5", "#ff459c", "#F96C39"])

# Canvas (inches) and elevation of the view of each dataset. The Matsumoto surface is nearly flat in
# the embedding (its height axis is short), so it is seen from higher up.
VIEWS = {"swiss_roll": ((2.4, 3.0), 18), "matsumoto": ((3.2, 2.4), 32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.9, help="Swiss roll: strength of the drift.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Strength of the canonical drift.")
    parser.add_argument("--k", type=int, default=10, help="Neighbors of the kNN graph.")
    parser.add_argument("--datasets", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    parser.add_argument("--format", default="pdf", help="Figure format (pdf, png, ...).")
    parser.add_argument("--out", default="finsler_mds", help="Prefix: writes <out>_swiss_roll and <out>_matsumoto.")
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    for dataset in args.datasets:
        figsize, elev = VIEWS[dataset]
        rows, data, points = run(dataset, args.n, args.beta, args.seed, args.alpha, args.k, graph="knn")  # Finsler MDS uses the k-NN graph
        coordinates = points.numpy() - points.numpy().mean(0)
        values = np.asarray(data["labels"]) if dataset == "swiss_roll" else data["X_graph"][:, 2].numpy()
        colors = TWO_TONE(Normalize(values.min(), values.max())(values))
        mds = next(row for row in rows if row["method"] == "finsler_mds")
        print(f"{dataset}: spread of the heights {mds['height_std']:.2f}, in the plane {mds['plane_std']:.2f}", flush=True)
        fig, _ = panel(coordinates, None, 0, elev=elev, colors=colors, size=3, figsize=figsize)
        save(fig, f"{args.out}_{dataset}.{args.format}")


if __name__ == "__main__":
    main()
