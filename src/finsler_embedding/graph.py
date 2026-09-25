import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from geodesic_toolbox import RandersMetrics, GEORCEFinsler

from finsler_embedding.operators import apply_gaussian_kernel


def symmetrize_edges(edges: torch.Tensor) -> torch.Tensor:
    """
    Symmetrizes the edges of a graph represented by a tensor of shape (M, 2).
    So that if (i, j) is an edge, then (j, i) is also an edge.

    Parameters:
    ----------
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.

    Returns:
    -------
    sym_edges : torch.Tensor (M', 2)
        The symmetrized edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    """
    sym_edges = torch.cat([edges, edges.flip(1)], dim=0)
    sym_edges = torch.unique(sym_edges, dim=0)
    return sym_edges


def get_knn_graph(
    X: torch.Tensor, n_neighbors: int, device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Computes the k-nearest neighbors graph for the given data points.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        The input points.
    n_neighbors : int
        The number of neighbors to consider for each point.
    device : torch.device
        The device to which the output tensor should be moved.

    Returns:
    -------
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    """
    if n_neighbors == -1:
        # Fully connected graph
        N = X.shape[0]
        edges = torch.combinations(torch.arange(N, device=device), r=2)
        return edges

    graph = kneighbors_graph(
        X.detach().cpu().numpy(), n_neighbors=n_neighbors, mode="distance", include_self=False
    )
    # Retrieve al the pairs of edges from the graph
    edges = np.array(graph.nonzero()).T
    edges = torch.from_numpy(edges).to(device)
    edges = symmetrize_edges(edges)
    return edges


def compute_epsilon_rate(N: int, m: int) -> float:
    """
    Computes the epsilon value based on the number of points N and the dimension m.
    This is a heuristic to set the scale of the kernel based on the data.
    It usually sucks and give waaaay too low values of epsilon.

    Parameters:
    ----------
    N : int
        The number of data points.
    m : int
        The dimension of the data.

    Returns:
    -------
    epsilon : float
        The computed epsilon value.
    """
    return (np.log(N) / N) ** (1 / (m + 4))


def compute_epsilon_nn(X: torch.Tensor) -> float:
    """
    Compute the bandwidth parameter as the min max distance between points in the dataset X.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        The input points.

    Returns:
    -------
    epsilon : float
        The computed bandwidth parameter.
    """
    neighbor_rank = 2
    eps = 1e-8
    K = X.shape[0]
    dst_centroids = torch.cdist(X, X, p=2)  # (K,K)
    dst_centroids.fill_diagonal_(float("inf"))
    sorted_dst, _ = torch.sort(dst_centroids, dim=1)
    rank = min(max(neighbor_rank - 1, 0), max(K - 2, 0))  # (K,)
    nn_dist = sorted_dst[:, rank]  # (K,)
    nn_scale2 = nn_dist.pow(2).clamp_min(eps)
    return nn_scale2.mean().item()


def compute_epsilon_empirical(dst_edges: torch.Tensor, scale: float = 1.0) -> float:
    """
    Compute the bandwidth parameter as the standard deviation of the distances in dst_edges.

    Parameters:
    ----------
    dst_edges : torch.Tensor (M,)
        The distances between points in the dataset.
    scale : float
        A scaling factor to adjust the computed bandwidth. (default is 1.0)

    Returns:
    -------
    epsilon : float
        The computed bandwidth parameter.
    """
    return dst_edges.std().item() * scale


def distance_matrix_straight_line(
    X: torch.Tensor,
    edges: torch.Tensor,
    randers: RandersMetrics,
    batch_size: int = 100,
    num_quad_points: int = 10,
    pbar: bool = True,
) -> torch.Tensor:
    """
    Compute Randers distances along straight-line segments for graph edges.

    The integral
        d_F(x_i, x_j) = ∫_0^1 F(x_i + t(x_j-x_i), x_j-x_i) dt
    is evaluated using composite trapezoidal quadrature on [0, 1].

    For a Randers metric F(x, v) = alpha(x, v) + beta(x, v), the segments i -> j and j -> i go
    through the same points with opposite velocities, and F(x, -v) = alpha(x, v) - beta(x, v):
    both directions are computed from a single evaluation of alpha and beta.
    """
    t = torch.linspace(0.0, 1.0, num_quad_points, device=X.device, dtype=X.dtype)
    if isinstance(randers, RandersMetrics):
        return _randers_straight_line_pairs(X, edges, randers, t, batch_size, pbar)

    dst_mat = torch.zeros(edges.shape[0], device=X.device, dtype=X.dtype)

    if pbar:
        pbar_ = tqdm(range(0, edges.shape[0], batch_size), desc="Computing Randers distances")
    else:
        pbar_ = range(0, edges.shape[0], batch_size)

    for start in pbar_:
        batch_edges = edges[start : start + batch_size]
        x0 = X[batch_edges[:, 0]]
        x1 = X[batch_edges[:, 1]]
        dx = x1 - x0
        x = x0[:, None, :] + t[None, :, None] * dx[:, None, :]
        v = dx[:, None, :].expand(-1, num_quad_points, -1)
        F = randers(x.reshape(-1, X.shape[-1]), v.reshape(-1, X.shape[-1]))
        F = F.reshape(x.shape[0], num_quad_points)
        dst = torch.trapezoid(F, t, dim=1)

        dst_mat[start : start + batch_edges.shape[0]] = dst

    return dst_mat


def _randers_straight_line_pairs(X, edges, randers, t, batch_size, pbar):
    """Straight-line Randers distances of the edges, computing each pair {i, j} once for both directions."""
    N, D = X.shape
    n_quad = t.shape[0]
    # One segment per unordered pair, oriented from the smaller to the larger index
    lo, hi = edges.min(dim=1).values, edges.max(dim=1).values
    pair_key, inverse = torch.unique(lo * N + hi, return_inverse=True)
    pairs = torch.stack([pair_key // N, pair_key % N], dim=1)
    d_fwd = torch.zeros(pairs.shape[0], device=X.device, dtype=X.dtype)
    d_bwd = torch.zeros_like(d_fwd)

    batch_size = batch_size * 2  # each pair gives two edges
    starts = range(0, pairs.shape[0], batch_size)
    for start in tqdm(starts, desc="Computing Randers distances") if pbar else starts:
        p = pairs[start : start + batch_size]
        x0, x1 = X[p[:, 0]], X[p[:, 1]]
        dx = x1 - x0
        x = (x0[:, None, :] + t[None, :, None] * dx[:, None, :]).reshape(-1, D)
        v = dx[:, None, :].expand(-1, n_quad, -1).reshape(-1, D)
        alpha = randers.base_cometric.metric(x, v).reshape(-1, n_quad)
        beta = randers.beta_form(x, v).reshape(-1, n_quad)
        d_fwd[start : start + p.shape[0]] = torch.trapezoid(alpha + beta, t, dim=1)
        d_bwd[start : start + p.shape[0]] = torch.trapezoid(alpha - beta, t, dim=1)

    # Edge i -> j is the forward direction of its pair if i < j, the backward one otherwise
    return torch.where(edges[:, 0] < edges[:, 1], d_fwd[inverse], d_bwd[inverse])


def georce_distance_matrix(
    X: torch.Tensor,
    edges: torch.Tensor,
    randers: RandersMetrics,
    batch_size: int = 100,
    pbar: bool = True,
) -> torch.Tensor:
    """
    Compute the geodesic distance matrix for a set of edges and a Randers metric.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        The input points.
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    randers : RandersMetrics
        The Randers metric object.
    batch_size : int
        The batch size for computing distances. (default is 100)

    Returns:
    -------
    dst_mat : torch.Tensor (N, N)
        The constructed distance matrix.
    """
    dst_mat = torch.zeros(edges.shape[0], device=X.device, dtype=X.dtype)
    solver = GEORCEFinsler(finsler=randers, T=25, max_iter=20)
    if pbar:
        pbar_ = tqdm(range(0, edges.shape[0], batch_size), desc="Computing Randers distances")
    else:
        pbar_ = range(0, edges.shape[0], batch_size)
    for start in pbar_:
        batch_edges = edges[start : start + batch_size]
        x0 = X[batch_edges[:, 0]]
        x1 = X[batch_edges[:, 1]]
        dst = solver(x0, x1)
        dst_mat[start : start + batch_edges.shape[0]] = dst
    return dst_mat


def construct_distance_matrix(
    X: torch.Tensor,
    edges: torch.Tensor,
    randers: RandersMetrics,
    batch_size: int = 100,
    use_approx: bool = True,
    pbar: bool = True,
) -> torch.Tensor:
    """Construct a distance matrix from a set of edges and a Randers metric.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        The input points.
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    randers : RandersMetrics
        The Randers metric object.
    batch_size : int
        The batch size for computing distances. (default is 100)
    use_approx : bool
        Whether to use the straight-line approximation for distance computation. If False, the geodesic distance is computed. (default is True)
    Returns:
    -------
    dst_mat : torch.Tensor (N, N)
        The constructed distance matrix.
    """
    if use_approx:
        dst_mat = distance_matrix_straight_line(
            X, edges, randers, batch_size=batch_size, pbar=pbar
        )
    else:
        dst_mat = georce_distance_matrix(X, edges, randers, batch_size=batch_size, pbar=pbar)
    return dst_mat


def build_data_knn(
    X_graph: torch.Tensor,
    k: int,
    randers: RandersMetrics,
    eps_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Builds a k-nearest neighbors graph and computes the distance matrix and epsilon value.

    Parameters:
    ----------
    X_graph : torch.Tensor (N, D)
        The input points.
    k : int
        The number of neighbors to consider for each point.
    randers : RandersMetrics
        The Randers metric to use for distance computation.
    eps_scale : float
        The scale factor for computing epsilon.

    Returns:
    -------
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    dst : torch.Tensor (M,)
        The distance matrix corresponding to the edges.
    eps : float
        The computed epsilon value based on the distance matrix and scale factor.
    """
    edges = get_knn_graph(X_graph, n_neighbors=k, device="cpu")
    dst = construct_distance_matrix(X_graph, edges, randers, use_approx=True, pbar=True)
    eps = compute_epsilon_empirical(dst, scale=eps_scale)
    return edges, dst, eps


def build_data_radius(
    X_graph: torch.Tensor,
    radius: float,
    randers: RandersMetrics,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Builds a radius graph and computes the distance matrix and epsilon value.

    Parameters:
    ----------
    X_graph : torch.Tensor (N, D)
        The input points.
    radius : float
        The radius to consider for each point.
    randers : RandersMetrics
        The Randers metric to use for distance computation.

    Returns:
    -------
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    dst : torch.Tensor (M,)
        The distance matrix corresponding to the edges.
    eps : float
        The computed epsilon value based on the distance matrix and scale factor.
    """
    A = radius_neighbors_graph(X_graph.detach().numpy(), radius=radius, include_self=False)
    edges = symmetrize_edges(torch.from_numpy(np.array(A.nonzero()).T))
    dst = construct_distance_matrix(X_graph, edges, randers, use_approx=True, pbar=True)
    delta_norm = (X_graph[edges[:, 1]] - X_graph[edges[:, 0]]).norm(dim=1)
    rho = (dst.detach() / delta_norm).min().item()
    eps = rho * radius / 3
    return edges, dst, eps


def build_graph(
    X_graph: torch.Tensor,
    F: RandersMetrics,
    graph_type: str,
    k: int,
    eps_scale: float,
    eps0: float,
    m: int,
    cut: float = 3.0,
    radius: float = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Constructs the graph data and the directed gaussian kernel W from the low-dimensional
    embedding X_graph and the Randers metric F.
    The graph type ("knn" or "radius") determines the connectivity.

    If graph_type is "knn", the graph uses k-nearest neighbors. Epsilon is computed based on
    the distance matrix and `eps_scale`.

    If graph_type is "radius":
        - If `radius` is None, the graph uses a dynamically calculated radius based on
            `eps0` and `compute_epsilon_rate`.
        - If `radius` is provided, the graph uses this fixed Euclidean radius.

    The distance matrix is computed using straight line approximation.

    Notes on Radius Graph Construction:
    - For the general radius graph, edges connect pairs with Euclidean distance < `cut * eps`.
        The kernel is negligible past d_F = 3 eps. For Randers metrics, this vanishing
        point is related to the Euclidean distance 3 eps / (1 - ||b||). Using
        `cut >= 3 / (1 - ||b||)` helps prevent kernel truncation.
    - For a fixed Euclidean radius, edges connect pairs with distance < `radius`.
      Epsilon is set such that $d_F < 3 text{eps}$ implies the pair is within the radius,
      specifically $text{eps} = rho cdot text{radius} / 3$, where $rho = min(d_F / |x_j - x_i|)$
      over the edges.

    Parameters:
    ----------
    X_graph : torch.Tensor (N, D)
        The input points.
    F : RandersMetrics
        The Randers metric to use for distance computation.
    graph_type : str
        The type of graph to construct ("knn" or "radius").
    k : int
        The number of neighbors to consider for each point (used if graph_type is "knn").
    eps_scale : float
        The scale factor for computing epsilon (used if graph_type is "knn").
    eps0 : float
        The base epsilon value (used if graph_type is "radius" and radius is None).
    m : int
        The dimension of the data (used if graph_type is "radius" and radius is None).
    cut : float
        The multiplier for the radius when graph_type is "radius" and radius is None. Default is 3.0.
    radius : float
        The fixed Euclidean radius to use for graph construction (used if graph_type is "radius"). Default is None.

    Returns:
    -------
    W : torch.Tensor (N, N)
        The directed gaussian kernel matrix.
    edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    dst : torch.Tensor (M,)
        The distance matrix corresponding to the edges.
    eps : float
        The computed epsilon value based on the distance matrix and scale factor.
    """
    VALID_GRAPHS = ["knn", "radius"]
    if graph_type not in VALID_GRAPHS:
        raise ValueError(f"Unknown graph {graph_type!r}, must be one of {VALID_GRAPHS!r}")
    N = X_graph.shape[0]
    if graph_type == "knn":
        edges, dst, eps = build_data_knn(X_graph, k, F, eps_scale)
    elif graph_type == "radius":
        if radius is None:
            eps = eps0 * compute_epsilon_rate(N, m)
            radius = cut * eps
            edges, dst, _ = build_data_radius(X_graph, radius, F)
        else:
            edges, dst, eps = build_data_radius(X_graph, radius, F)
    W = apply_gaussian_kernel(edges, dst.detach(), eps, N)
    return W, edges, dst.detach(), eps
