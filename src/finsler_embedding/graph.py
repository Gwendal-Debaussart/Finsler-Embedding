import torch
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

from geodesic_toolbox import RandersMetrics, GEORCEFinsler


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
    return (np.log(N) / N) ** (1 / m + 4)


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
    """
    dst_mat = torch.zeros(edges.shape[0], device=X.device, dtype=X.dtype)

    t = torch.linspace(0.0, 1.0, num_quad_points, device=X.device, dtype=X.dtype)

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
