"""Finsler MDS baseline on sampled DiSBM graphs.

Finsler MDS embeds *dissimilarities* in a canonical Randers space: a flat space
with metric ``F(v) = |v| + alpha * v_last``, that is, a constant drift along the
last axis. As in the Finsler MDS paper, the space has one dimension more than
the data, which carries the asymmetry: a point is higher when it is costlier to
reach. The dissimilarities of a graph are its directed shortest paths (Dijkstra),
here with unit edge lengths since a DiSBM is unweighted.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from finsler_embedding.experiments.baselines.finsler_mds import smacof


@dataclass
class FinslerMDSEmbedding:
    """Finsler MDS embedding of one graph.

    Attributes
    ----------
    coordinates : torch.Tensor (N, embedding_dim + 1)
        Node coordinates. The last axis is the drift axis: going up costs
        ``alpha`` more than going down.
    alpha : float
        Strength of the canonical drift.
    stress : float
        Raw directed stress of the embedding.
    """

    coordinates: torch.Tensor
    alpha: float
    stress: float


def hop_distances(adjacency: torch.Tensor) -> np.ndarray:
    """Directed shortest-path distances of an unweighted graph.

    Parameters
    ----------
    adjacency : torch.Tensor (N, N)
        Directed adjacency matrix; a nonzero ``(i, j)`` entry is an edge from
        ``i`` to ``j``. Self-loops are irrelevant.

    Returns
    -------
    numpy.ndarray (N, N)
        Number of edges of the shortest directed path from ``i`` to ``j``.

    Raises
    ------
    ValueError
        If the graph is not strongly connected, since Finsler MDS needs a
        finite dissimilarity between every pair of nodes.
    """
    graph = csr_matrix(adjacency.detach().cpu().numpy() != 0)
    distances = dijkstra(graph, directed=True, unweighted=True)
    if not np.isfinite(distances).all():
        raise ValueError("the directed graph is not strongly connected")
    return distances


def canonical_c_norm(alpha: float, dim: int) -> float:
    """``||c||_{g_BL}`` of the canonical Randers metric ``|v| + alpha v_last`` in R^dim.

    The centroid norm of this metric does not depend on the direction of the
    drift: ``||c||^2_{g_BL} = alpha^2 / (1 + (dim + 2) alpha^2)``.
    """
    return alpha / (1.0 + (dim + 2) * alpha**2) ** 0.5


def finsler_mds_embedding(
    adjacency: torch.Tensor,
    embedding_dim: int = 2,
    alpha: float = 0.5,
    *,
    seed: int | None = None,
    n_init: int = 1,
    max_iter: int = 300,
) -> FinslerMDSEmbedding:
    """Embed a directed graph in the canonical Randers space of dimension ``embedding_dim + 1``.

    Parameters
    ----------
    adjacency : torch.Tensor (N, N)
        Directed adjacency matrix of a strongly connected graph.
    embedding_dim : int, default=2
        Dimension of the data; the embedding has one more for the drift.
    alpha : float, default=0.5
        Strength of the canonical drift, ``0 <= alpha < 1``. The default is the
        value used in the Finsler MDS paper.
    seed : int or None
        Seed of the random initial configuration.
    n_init : int, default=1
        Number of random initializations (the lowest stress is kept).
    max_iter : int, default=300
        SMACOF iterations per initialization.
    """
    coordinates, stress = smacof(
        hop_distances(adjacency),
        randers_w_alpha=alpha,
        n_components=embedding_dim + 1,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    return FinslerMDSEmbedding(
        coordinates=torch.as_tensor(coordinates, dtype=torch.float32),
        alpha=alpha,
        stress=float(stress),
    )
