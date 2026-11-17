import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from finsler_embedding.experiments.graph import build_data_radius


def drifting_plane(drift: float):
    """Randers metric ``|v| + drift v_x`` of the plane: going along ``-x`` is ``(1 - drift) / (1 + drift)`` times cheaper."""
    return lambda x, v: v.norm(dim=-1) + drift * v[:, 0]


def strong_components(edges: torch.Tensor, n: int) -> int:
    e = edges.numpy()
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(graph, directed=True, connection="strong")[0]


def test_radius_graph_keeps_both_directions_of_a_neighbor_pair():
    n = 150
    X = torch.rand(n, 2, generator=torch.Generator().manual_seed(0))
    metric = drifting_plane(0.95)
    edges_cut, _, eps = build_data_radius(X, 0.3, metric, truncate="edge")
    pairs, lengths, _ = build_data_radius(X, 0.3, metric, truncate="pair")
    everything, _, _ = build_data_radius(X, 0.3, metric, truncate="none")

    # The kernel cut drops the expensive direction, which leaves nodes that cannot be reached against the drift
    assert strong_components(edges_cut, n) > 1
    # Both directions of a pair kept by the cut make the graph strongly connected, whatever the drift
    assert strong_components(pairs, n) == 1
    keys = set((pairs[:, 0] * n + pairs[:, 1]).tolist())
    assert all(int(j) * n + int(i) in keys for i, j in pairs)
    # They are the pairs of the cut, and a subset of the Euclidean radius graph
    cut_keys = set((edges_cut[:, 0] * n + edges_cut[:, 1]).tolist())
    assert cut_keys <= keys
    assert len(pairs) < len(everything)
    assert len(pairs) <= 2 * len(edges_cut)
