"""Directed stochastic block model generation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiSBMConfig:
    """Configuration for sampling a cyclic directed stochastic block model.

    Parameters
    ----------
    n : int, default=1000
        Number of graph nodes.
    communities : int, default=15
        Number of cyclic communities.
    r : float, default=0.1
        Within-community edge probability.
    """

    n: int = 1000
    communities: int = 15
    r: float = 0.1


def construct_Q_matrix(
    K: int, p: float, q: float, r: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Construct the cyclic DiSBM community probability matrix.

    Parameters
    ----------
    K : int
        Number of communities.
    p : float
        Probability of an edge from community ``i`` to ``i + 1``.
    q : float
        Probability of an edge from community ``i`` to ``i - 1``.
    r : float
        Probability of an edge within the same community.
    dtype : torch.dtype, default=torch.float32
        Data type of the returned matrix.

    Returns
    -------
    torch.Tensor (K, K)
        Community-to-community edge probabilities.
    """
    if K < 3:
        raise ValueError("K must be >= 3")
    if min(p, q, r) < 0 or abs(p + q + r - 1) >= 1e-8:
        raise ValueError("p, q, and r must be nonnegative and sum to one")

    probabilities = torch.zeros((K, K), dtype=dtype)
    indices = torch.arange(K)
    probabilities[indices, (indices + 1) % K] = p
    probabilities[indices, (indices - 1) % K] = q
    probabilities[indices, indices] = r
    return probabilities


def construct_graph_from_Q(
    N: int, Q: torch.Tensor, community_labels: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a directed adjacency matrix from community probabilities.

    Parameters
    ----------
    N : int
        Number of graph nodes.
    Q : torch.Tensor (K, K)
        Community-to-community edge probabilities.
    community_labels : torch.Tensor (N,), optional
        Node community labels. If omitted, labels are sampled uniformly.

    Returns
    -------
    adjacency : torch.Tensor (N, N)
        Sampled directed adjacency matrix.
    community_labels : torch.Tensor (N,)
        Community label for every node.
    """
    K = Q.shape[0]
    if K < 3 or N <= 0:
        raise ValueError("K must be >= 3 and N must be positive")
    if community_labels is None:
        community_labels = torch.randint(0, K, (N,))
    elif (
        community_labels.shape != (N,)
        or community_labels.min() < 0
        or community_labels.max() >= K
    ):
        raise ValueError("community_labels must have shape (N,) and lie in [0, K)")
    return (
        torch.bernoulli(Q[community_labels][:, community_labels]),
        community_labels,
    )


def graph_info(
    p: float,
    q: float,
    r: float,
    K: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a cyclic DiSBM and return its edge-list representation.

    Parameters
    ----------
    p : float
        Probability of an edge from community ``i`` to ``i + 1``.
    q : float
        Probability of an edge from community ``i`` to ``i - 1``.
    r : float
        Probability of an edge within the same community.
    K : int
        Number of communities.
    N : int
        Number of graph nodes.

    Returns
    -------
    edges : torch.Tensor (M, 2)
        Directed source and destination node indices.
    edge_weights : torch.Tensor (M,)
        Edge weights corresponding to ``edges``.
    community_labels : torch.Tensor (N,)
        Community label for each node.
    adjacency : torch.Tensor (N, N)
        Sampled directed adjacency matrix.
    """
    Q = construct_Q_matrix(K, p, q, r)
    adjacency, labels = construct_graph_from_Q(N, Q)
    edges = torch.nonzero(adjacency)
    edge_weights = adjacency[edges[:, 0], edges[:, 1]]
    return edges, edge_weights, labels, adjacency
