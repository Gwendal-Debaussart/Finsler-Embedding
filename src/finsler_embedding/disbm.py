from networkx import edges
import torch


def construct_Q_matrix(K, p, q, r, dtype=torch.float32):
    assert K >= 3, "K must be >= 3"
    assert p >= 0 and q >= 0 and r >= 0, "Probabilities must be non-negative"

    assert abs(p + q + r - 1) < 1e-8, f"p + q + r = {p + q + r}, expected 1"

    Q = torch.zeros((K, K), dtype=dtype)
    indices = torch.arange(K)
    Q[indices, (indices + 1) % K] = p
    Q[indices, (indices - 1) % K] = q
    Q[indices, indices] = r

    return Q


def construct_graph_from_Q(N: int, Q: torch.Tensor, community_labels=None):
    K = Q.shape[0]
    assert K >= 3, "K must be >= 3"
    assert N > 0, "N must be > 0"

    if community_labels is None:
        community_labels = torch.randint(0, K, (N,))
    else:
        assert len(community_labels) == N, "Length of community_labels must be N"
        assert community_labels.max() < K, "Community labels must be in range [0, K-1]"

    A = torch.zeros((N, N), dtype=torch.float32)
    A = torch.bernoulli(Q[community_labels][:, community_labels])

    return A, community_labels


def graph_info(
    p: float,
    q: float,
    r: float,
    K: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples a graph from a directed stochastic block model (DSBM) with
    K communities and N nodes, using the specified probabilities
    p, q, and r for edges between communities.

    Parameters:
    ----------
    p : float
        Probability of an edge from community i to community (i+1) mod K.
    q : float
        Probability of an edge from community i to community (i-1) mod K.
    r : float
        Probability of an edge from community i to itself.
    K : int
        Number of communities (must be >= 3).
    N : int
        Total number of nodes in the graph.

    Returns:
    -------
    edges : torch.Tensor (M, 2)
        Tensor containing the edges of the graph, where M is the number of edges.
    dst_edges : torch.Tensor (M,)
        Tensor containing the weights of the edges corresponding to the edges tensor.
    community_labels : torch.Tensor (N,)
        Tensor containing the community labels for each node.
    A : torch.Tensor (N, N)
        Tensor containing the adjacency matrix of the graph.
    """

    Q = construct_Q_matrix(K, p, q, r)
    A, community_labels = construct_graph_from_Q(N, Q)
    edges = torch.nonzero(A)
    dst_edges = A[edges[:, 0], edges[:, 1]]
    return edges, dst_edges, community_labels, A
