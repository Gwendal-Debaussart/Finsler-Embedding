"""Baselines on the manifold datasets (Swiss roll, Matsumoto): Finsler MDS and Perrault-Joncas & Meila.

The methods are given the directed weighted graphs whose edge lengths are the Finsler distances between
neighbors (as in ``finsler_embedding.evaluate``). Ours and Perrault-Joncas & Meila use the radius graph of the
quantitative Swiss roll evaluation, where the kernel is not truncated unevenly. Finsler MDS needs a finite
dissimilarity between every pair of samples, and the radius graph is not strongly connected under a strong
drift, so it uses the directed k-NN graph, which is.

* Moment-Randers (ours): 2D Isomap embedding of the points, drift estimated from the graph operators.
* Perrault-Joncas and Meila (2011): vector field from the antisymmetric part of the affinity matrix,
  ``(H_aa - H_ss) Y / 2``, in the coordinates of the same Isomap embedding. It has a direction but no
  calibrated magnitude, and no metric.
* Finsler MDS (Dages et al., 2025): the dissimilarity between two samples is the length of the
  shortest directed path in the k-NN graph, embedded in the canonical Randers space of dimension 3
  (2 + 1) whose metric is ``|v| + alpha v_3``, with a constant drift along the last axis.

Directional accuracy is the signed cosine between the recovered drift and the true one, ``C J c``,
mapped by the Jacobian ``J`` of the embedding (as in the paper, Table 1). The embedding of Finsler MDS
has one more dimension than the manifold, so its constant drift is compared through its
*tangential* part, that is, projected on the image of ``J``. This is the drift of the Randers metric
induced on the embedded manifold. The plain cosine, without projection, is also reported.

The strength of the asymmetry is the quantity ``||c||^2_{g_BL}`` of the unit ball of the metric, which does
not depend on the embedding. The strength of Finsler MDS is the one of its induced Randers metric on the
tangent plane, which has the Euclidean norm and the drift ``alpha ||B^T e_last||`` (``B`` is a basis of
the image of ``J``). The method of Perrault-Joncas and Meila has no metric, hence no strength.

    python -m finsler_embedding.experiments.finsler_mds_manifolds [--seeds 3] [--out results/finsler_mds]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from finsler_embedding.core.embeddings import isomap_embedding
from finsler_embedding.core.operators import randers_approximation
from finsler_embedding.experiments.baselines import perrault_joncas_meila
from finsler_embedding.experiments.baselines.finsler_mds import smacof
from finsler_embedding.experiments.legacy.evaluation import (
    DATASETS,
    build_graph,
    evaluate_embedding,
    local_jacobian,
)

# Euclidean radius from which the radius graph is built (it is enlarged when the kernel would be truncated)
DEFAULT_RADIUS = {"swiss_roll": 5.0, "matsumoto": 1.0}


def path_distances(edges: torch.Tensor, lengths: torch.Tensor, n: int) -> np.ndarray:
    """Directed shortest-path distances of a weighted directed graph (N, N).

    Raises
    ------
    ValueError
        If the graph is not strongly connected, since Finsler MDS needs a finite dissimilarity
        between every pair of samples.
    """
    graph = csr_matrix((lengths.numpy().astype(float), (edges[:, 0].numpy(), edges[:, 1].numpy())), shape=(n, n))
    if connected_components(graph, directed=True, connection="strong")[0] > 1:
        raise ValueError("the directed graph is not strongly connected")
    return dijkstra(graph, directed=True)


def finsler_mds_points(distances: np.ndarray, alpha: float, seed: int, n_components: int = 3) -> torch.Tensor:
    """Finsler MDS coordinates (N, n_components), the last axis is the drift axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coordinates, _ = smacof(
            distances, randers_w_alpha=alpha, n_components=n_components, n_init=1, max_iter=300, random_state=seed
        )
    return torch.as_tensor(coordinates, dtype=torch.float32)


def tangent_basis(jacobian: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (N, l, m) of the image of each Jacobian (N, l, m)."""
    return torch.linalg.qr(jacobian)[0]


def mds_directional_cosines(jacobian: torch.Tensor, centroid: torch.Tensor) -> dict[str, torch.Tensor]:
    """Signed cosines (N,) between the constant drift of Finsler MDS and the true drift ``J c``.

    The drift is ``-e_last``, towards cheaper travel as the drift of our method. ``tangential`` is the
    cosine of its projection on the image of the Jacobian, ``direct`` the cosine without projection.
    """
    true_drift = torch.einsum("nlm,nm->nl", jacobian, centroid)
    drift = torch.zeros_like(true_drift)
    drift[:, -1] = -1.0
    basis = tangent_basis(jacobian)
    tangential = torch.einsum("nlk,nmk,nm->nl", basis, basis, drift)
    cosine = torch.nn.functional.cosine_similarity
    return {"direct": cosine(drift, true_drift, dim=1), "tangential": cosine(tangential, true_drift, dim=1)}


def randers_strength(drift_norm: torch.Tensor, n_theta: int = 720) -> torch.Tensor:
    """``||c||^2_{g_BL}`` of the 2D Randers metric ``|v| + s u^T v`` with ``s = drift_norm`` (N,) and ``|u| = 1``.

    Same quadrature as ``empirical_unit_ball_info``: the radius of the unit ball is ``1 / (1 + s cos(theta))``,
    ``c = int r^3 u / (3 |B|)`` and ``g_BL^{-1} = int r^4 u u^T / |B|``.
    """
    theta = torch.linspace(0, 2 * torch.pi, n_theta + 1)[:-1]
    u = torch.stack([theta.cos(), theta.sin()], dim=1)
    r = 1.0 / (1.0 + drift_norm[:, None] * theta.cos()[None])
    d_theta = 2 * torch.pi / n_theta
    area = 0.5 * (r**2).sum(dim=1) * d_theta
    c = torch.einsum("nt,ti->ni", r**3, u) * d_theta / (3 * area[:, None])
    g_bl_inv = torch.einsum("nt,ti,tj->nij", r**4, u, u) * d_theta / area[:, None, None]
    return torch.einsum("ni,nij,nj->n", c, torch.linalg.inv(g_bl_inv), c)


def mds_strength(jacobian: torch.Tensor, alpha: float) -> torch.Tensor:
    """``||c||^2_{g_BL}`` (N,) of the Randers metric induced by Finsler MDS on the embedded manifold."""
    basis = tangent_basis(jacobian)
    tangential_drift = alpha * basis[:, -1, :].norm(dim=1)  # alpha ||B^T e_last||
    return randers_strength(tangential_drift)


def run(dataset: str, n: int, beta: float, seed: int, alpha: float, k: int, graph: str = "radius", radius: float | None = None):
    """One sampled dataset, the three methods. Returns the rows of results, the data and the Finsler MDS points."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = DATASETS[dataset](n, beta)
    if graph == "radius" and radius is None:
        radius = DEFAULT_RADIUS[dataset]
    W, edges, lengths, eps = build_graph(data["X_graph"], data["F"], graph, k, 1.0, 0.5, 2, 3.0, radius, "cpu")
    truth = data["truth"]
    cosine = torch.nn.functional.cosine_similarity
    c_norm_true = torch.einsum("ni,nij,nj->n", truth["c"], torch.linalg.inv(truth["g_bl_inv"]), truth["c"])

    # Ours: Isomap embedding in R^2 and drift from the graph operators
    Y = isomap_embedding(data["X_graph"].detach(), n_components=2, n_neighbors=10)
    J = local_jacobian(data["X_chart"].detach(), Y, data["P"])
    result = randers_approximation(W, eps, X_low=Y, m=2)
    metrics = evaluate_embedding(result, Y, J, truth, 2, {})
    true_drift = torch.einsum("nlm,nm->nl", J, truth["c"])
    cos_ours = cosine(result.drift, true_drift, dim=1)
    ours = {
        "method": "ours",
        "cos_V": metrics["cos_V_all"],
        "cos_V_q10": float(cos_ours.quantile(0.1)),
        "relerr_c_norm": metrics["relerr_c_norm_all"],
        "admissible": metrics["admissible"],
    }

    # Perrault-Joncas and Meila: vector field of the antisymmetric part, in the coordinates of the same embedding
    field = torch.as_tensor(perrault_joncas_meila.vector_field(W.numpy(), Y.numpy()), dtype=torch.float32)
    # The normalization divides by the out-degrees, so the field is undefined at the nodes that point to a node
    # without out-edge (the kernel is truncated), which are left out
    finite = torch.isfinite(field).all(dim=1)
    field, drift, direction = field[finite], result.drift[finite], true_drift[finite]
    cos_pjm = cosine(field, direction, dim=1)
    scale = (field * drift).sum(dim=1) / (drift**2).sum(dim=1)  # field = scale * our drift
    pjm = {
        "method": "perrault_joncas_meila",
        "cos_V": float(cos_pjm.median()),
        "cos_V_q10": float(cos_pjm.quantile(0.1)),
        "cos_with_ours": float(cosine(field, drift, dim=1).median()),
        "scale_to_ours_q10": float(scale.quantile(0.1)),
        "scale_to_ours_median": float(scale.median()),
        "scale_to_ours_q90": float(scale.quantile(0.9)),
        "defined": float(finite.float().mean()),
    }

    # Finsler MDS: directed path distances, canonical Randers space of dimension 3. It needs a finite distance
    # between every pair, so it gets the k-NN graph (strongly connected) rather than the truncated radius graph
    if graph != "knn":
        _, edges, lengths, _ = build_graph(data["X_graph"], data["F"], "knn", k, 1.0, 0.5, 2, 3.0, None, "cpu")
    distances = path_distances(edges, lengths, n)
    points = finsler_mds_points(distances, alpha, seed)
    J3 = local_jacobian(data["X_chart"].detach(), points, data["P"])
    cosines = mds_directional_cosines(J3, truth["c"])
    strength = mds_strength(J3, alpha)
    mds = {
        "method": "finsler_mds",
        "cos_V": float(cosines["tangential"].median()),
        "cos_V_q10": float(cosines["tangential"].quantile(0.1)),
        "cos_V_direct": float(cosines["direct"].median()),
        "relerr_c_norm": float(((strength - c_norm_true).abs() / c_norm_true).median()),
        "height_std": float(points[:, -1].std()),
        "plane_std": float(points[:, :-1].std()),
    }
    if dataset == "matsumoto":  # the drift axis of Finsler MDS against the height map of the surface
        mds["height_corr"] = float(np.corrcoef(points[:, -1].numpy(), data["X_graph"][:, 2].numpy())[0, 1])
    base = {"dataset": dataset, "N": n, "beta": beta, "seed": seed, "alpha": alpha, "k": k, "graph": graph}
    return [base | row for row in (ours, pjm, mds)], data, points


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["swiss_roll", "matsumoto"], choices=["swiss_roll", "matsumoto"])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.9], help="Swiss roll drift strengths (Matsumoto has none).")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Finsler MDS: strength of the canonical drift.")
    parser.add_argument("--graph", choices=["radius", "knn"], default="radius")
    parser.add_argument("--radius", type=float, default=None, help="Radius graph: Euclidean radius (default by dataset).")
    parser.add_argument("--k", type=int, default=10, help="Neighbors of the kNN graph.")
    parser.add_argument("--out", type=Path, default=Path("results/finsler_mds"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        for beta in args.betas if dataset == "swiss_roll" else [None]:  # the Matsumoto metric has no strength parameter
            for seed in range(args.seeds):
                new_rows, _, _ = run(dataset, args.n, 0.5 if beta is None else beta, seed, args.alpha, args.k, args.graph, args.radius)
                for row in new_rows:
                    row["beta"] = beta
                rows += new_rows
                print(
                    f"{dataset} beta={beta} seed={seed}: "
                    + ", ".join(f"{r['method']} cos {r['cos_V']:+.3f}" for r in new_rows),
                    flush=True,
                )
    df = pd.DataFrame(rows)
    df["beta"] = df["beta"].fillna(-1)  # keeps the Matsumoto rows in the groupby
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "manifolds.csv", index=False)
    print(f"\nSaved {len(df)} rows to {args.out / 'manifolds.csv'}\nMedian over seeds:")
    metrics = [
        "cos_V", "cos_V_q10", "cos_V_direct", "relerr_c_norm", "cos_with_ours", "scale_to_ours_q10",
        "scale_to_ours_median", "scale_to_ours_q90", "defined", "admissible", "height_corr", "height_std", "plane_std",
    ]
    print(df.groupby(["dataset", "beta", "method"])[[m for m in metrics if m in df]].median().round(3).to_string())


if __name__ == "__main__":
    main()
