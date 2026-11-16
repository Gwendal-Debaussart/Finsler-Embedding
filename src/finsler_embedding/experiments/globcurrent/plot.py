"""Figures of the GlobCurrent experiment: the currents, the proposed embedding, Finsler MDS.

Three separate figures, to be placed side by side (each 1.8 in wide, a third of a 5.5 in text column):

    <out>_currents.pdf     the surface currents on the map (km), arrows at the current ``w`` of the data;
    <out>_proposed.pdf     the 2D Isomap embedding with the estimated centroid ``hat c = sqrt(kappa) V`` (the estimate
                           of ``J w``), drawn with the same arrow scale as the data;
    <out>_finsler_mds.pdf  Finsler MDS, canonical Randers space of dimension 3 (same style as the DiSBM and the
                           manifolds): the vertical axis is the drift axis, omega points up, the three axes share one
                           scale and nothing is rescaled.

The nodes are coloured by the speed of the current. The embeddings are drawn as the methods return them, up to a
rigid motion (rotation or reflection, no rescaling) that puts the Isomap coordinates in the orientation of the
map; the same motion is applied to the arrows. The arrows of the two vector panels share one global scale
(``--arrow-km``: the length in km of an arrow of norm one, in units of the ship speed): lengths are proportional
to the norm of the vectors, nothing is normalised.

    python -m finsler_embedding.experiments.globcurrent.plot [--beta 0.9] [--out results/globcurrent/figure]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from scipy.linalg import orthogonal_procrustes

from finsler_embedding.core.embeddings import isomap_embedding
from finsler_embedding.core.operators import randers_approximation
from finsler_embedding.experiments.disbm.plot_comparison import save
from finsler_embedding.experiments.disbm.plot_finsler_mds import INK, panel
from finsler_embedding.experiments.finsler_mds_manifolds import finsler_mds_points, path_distances
from finsler_embedding.experiments.legacy.evaluation import build_graph
from finsler_embedding.experiments.plot_finsler_mds_manifolds import TWO_TONE

from .data import EARTH_RADIUS_KM, make_globcurrent


def map_coordinates(lon: torch.Tensor, lat: torch.Tensor) -> np.ndarray:
    """Plane coordinates (N, 2), in km, of the nodes: equirectangular projection at the mean latitude (display only)."""
    lon, lat = lon.numpy().astype(np.float64), lat.numpy().astype(np.float64)
    return EARTH_RADIUS_KM * np.stack([np.cos(lat.mean()) * (lon - lon.mean()), lat - lat.mean()], axis=1)


def vector_panel(points, arrows, colors, anchor, arrow_km, figsize=(1.8, 1.1)):
    """Scatter of the nodes (km) coloured by speed, with the arrows (N, 2) drawn at ``anchor`` (boolean mask)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(*points.T, c=colors, s=1.1, linewidths=0, zorder=1)
    ax.quiver(*points[anchor].T, *(arrow_km * arrows[anchor]).T, angles="xy", scale_units="xy", scale=1, color=INK,
              width=0.0055, headwidth=4.2, headlength=4.8, headaxislength=4.2, zorder=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beta", type=float, default=0.9, help="Strength: the strongest current over the ship speed.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Finsler MDS: strength of the canonical drift.")
    parser.add_argument("--radius", type=float, default=150.0, help="Radius graph of the proposed method (km).")
    parser.add_argument("--k", type=int, default=10, help="Neighbors of the kNN graph of Finsler MDS.")
    parser.add_argument("--stride", type=int, default=4, help="One arrow every <stride> nodes of the grid, in both directions.")
    parser.add_argument("--arrow-km", type=float, default=120.0, help="Length (km) of an arrow of norm one.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", default="20020101")
    parser.add_argument("--format", default="pdf", help="Figure format (pdf, png, ...).")
    parser.add_argument("--out", default="globcurrent", help="Prefix: writes <out>_currents, _proposed, _finsler_mds.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = make_globcurrent(10**6, args.beta, date=args.date)  # every ocean node of the grid
    X, P, w = data["X_graph"], data["P"], data["w"]
    n = X.shape[0]
    speed = np.asarray(data["labels"])
    colors = TWO_TONE(Normalize(0, speed.max())(speed))
    chart = map_coordinates(data["lon"], data["lat"])
    ix = np.rint((np.rad2deg(data["lon"].numpy()) - 14.875) / 0.25).astype(int)  # indices on the 0.25 degree grid
    iy = np.rint((np.rad2deg(data["lat"].numpy()) + 40.125) / 0.25).astype(int)
    anchor = (ix % args.stride == 0) & (iy % args.stride == 0)
    print(f"{n} nodes, ship speed {data['ship_speed']:.2f} m/s, {anchor.sum()} arrows", flush=True)

    # Proposed method: 2D Isomap embedding, drift and centroid from the graph operators
    W, edges, lengths, eps = build_graph(X, data["F"], "radius", args.k, 1.0, 0.5, 2, 3.0, args.radius, "cpu")
    Y = isomap_embedding(X.detach(), n_components=2, n_neighbors=10)
    result = randers_approximation(W, eps, X_low=Y, m=2)
    rotation = orthogonal_procrustes(Y.numpy() - Y.numpy().mean(0), chart - chart.mean(0))[0]  # rigid motion
    embedded = (Y.numpy() - Y.numpy().mean(0)) @ rotation
    centroid = result.centroid.numpy() @ rotation  # estimate of J w

    # Finsler MDS: directed path distances on the kNN graph, canonical Randers space of dimension 3
    _, knn_edges, knn_lengths, _ = build_graph(X, data["F"], "knn", args.k, 1.0, 0.5, 2, 3.0, None, "cpu")
    mds = finsler_mds_points(path_distances(knn_edges, knn_lengths, n), args.alpha, args.seed).numpy()
    mds = mds - mds.mean(0)
    # Rotation about the drift axis (an isometry of the canonical Randers space): the long axis of the cloud faces the camera
    _, _, axes = np.linalg.svd(mds[:, :2], full_matrices=False)
    mds[:, :2] = mds[:, :2] @ axes.T
    print(f"Finsler MDS: spread of the heights {mds[:, 2].std():.1f} km, in the plane {mds[:, :2].std():.1f} km", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9})
    save(vector_panel(chart, w.numpy(), colors, anchor, args.arrow_km), f"{args.out}_currents.{args.format}")
    save(vector_panel(embedded, centroid, colors, anchor, args.arrow_km), f"{args.out}_proposed.{args.format}")
    fig, _ = panel(mds, None, 0, elev=40, azim=-90, colors=colors, size=1.1, figsize=(1.8, 1.4))
    save(fig, f"{args.out}_finsler_mds.{args.format}")


if __name__ == "__main__":
    main()
