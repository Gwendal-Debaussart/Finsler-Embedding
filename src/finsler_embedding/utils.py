import logging

import torch
from sklearn.neighbors import kneighbors_graph
from geodesic_toolbox import *
from scipy.special import factorial
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_bounds(embeddings: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Compute the bounds of the embeddings.

    Parameters:
    ----------
    embeddings : torch.Tensor (n_points, 2)
        The embeddings of the points.
    margin : float
        Margin scaling factor to add to the bounds. (default is 0)
        This is useful to avoid points being too close to the edges of the plot.

    Returns:
    -------
    torch.Tensor (4,)
        [min_x, max_x, min_y, max_y], the bounds of the embeddings.
    """
    min_x, max_x = embeddings[:, 0].min(), embeddings[:, 0].max()
    min_y, max_y = embeddings[:, 1].min(), embeddings[:, 1].max()
    # Add margin to the bounds
    min_x -= margin * (max_x - min_x)
    max_x += margin * (max_x - min_x)
    min_y -= margin * (max_y - min_y)
    max_y += margin * (max_y - min_y)
    # Ensure the bounds are in the correct order
    min_x, max_x = min(min_x, max_x), max(min_x, max_x)
    min_y, max_y = min(min_y, max_y), max(min_y, max_y)
    # Create a tensor with the bounds
    bounds = [min_x, max_x, min_y, max_y]
    bounds = torch.tensor(bounds)
    return bounds


def find_espilon(dst_mat: torch.Tensor) -> float:
    """
    Find epsilon the standard deviation of the non-zero values of the upper triangular part of the distance matrix, excluding the diagonal.

    Parameters:
    -----------
    dst_mat : torch.Tensor (N, N)
        The distance matrix.

    Returns:
    --------
    epsilon : float
        The standard deviation of the non-zero values of the upper triangular part of the distance matrix, excluding the diagonal.
    """
    upper_triangular = dst_mat[torch.triu(torch.ones(dst_mat.shape), diagonal=1) == 1]
    # Filter zero values
    upper_triangular = upper_triangular[upper_triangular > 0]
    epsilon = upper_triangular.std().item()
    return epsilon


def get_euclidean_knn_dst_matrix(z, n_neighbors=8):
    """
    Computes the k-nearest neighbors distance matrix for the given data points.

    Parameters:
    -----------
    z : torch.Tensor (N, D)
      The data points
    n_neighbors : int
      The number of neighbors to consider for each point

    Returns:
    --------
    dst_mat : torch.Tensor (N, N)
      The distance matrix
    """
    dst_mat = kneighbors_graph(z, n_neighbors=n_neighbors, mode="distance", include_self=False)
    dst_mat = dst_mat.toarray()  # Convert sparse matrix to dense array for visualization
    dst_mat = torch.tensor(dst_mat, dtype=torch.float32)
    return dst_mat


def get_K(dst_mat, epsilon):
    """
    Computes the kernel matrix K using the distance matrix and epsilon.
    Parameters:
    -----------
    dst_mat : torch.Tensor (N, N)
        The distance matrix
    epsilon : float
        The standard deviation used in the kernel computation

    Returns:
    --------
    K : torch.Tensor (N, N)
        The kernel matrix
    """
    K = torch.exp(-dst_mat / epsilon)
    return K


def local_poly_gradient(
    x: torch.Tensor, X: torch.Tensor, f0: torch.Tensor, F: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched local quadratic polynomial reconstruction. We assume that the local polynomial is of the form
    f(x + dx) = f(x) + a^T dx + 1/2 dx^T H dx,
    where a is the gradient and H is the Hessian.
    We want to solve for a and H given the values of f at the center point x and its neighbors X.
    We use least-squares to solve for the coefficients of the polynomial, and then extract the gradient from the coefficients.

    Parameters
    ----------
    x : torch.Tensor (B, D)
        Center points.
    X : torch.Tensor (B, M, D)
        Neighbor points for each center.
    f0 : torch.Tensor (B, N)
        Values of N functions at each center.
    F : torch.Tensor (B, M, N)
        Values of N functions at each neighbor.

    Returns
    -------
    grad : torch.Tensor (B, N, D)
        grad[b, k, d] = ∂f_k / ∂x_d at center b.
    hess : torch.Tensor (B, N, D, D)
        hess[b, k, d1, d2] = ∂²f_k / ∂x_d1 ∂x_d2 at center b.
    """

    B, M, D = X.shape

    assert x.shape == (B, D)
    assert f0.ndim == 2
    assert F.shape == (B, M, f0.shape[1])

    # Shift coordinates
    dx = X - x[:, None, :]  # (B, M, D)

    ij = torch.triu_indices(
        D,
        D,
        device=X.device,
    )
    quadratic = dx[:, :, ij[0]] * dx[:, :, ij[1]]  # x^i x^j

    # For diagonal terms: 1/2 H_jj dx_j²
    # For off-diagonal terms: H_jk dx_j dx_k
    diagonal = ij[0] == ij[1]
    quadratic = torch.where(
        diagonal[None, None, :],
        0.5 * quadratic,
        quadratic,
    )

    # Design matrix
    A = torch.cat(
        [dx, quadratic],
        dim=-1,
    )  # (B, M, D + D(D+1)/2)

    rhs = F - f0[:, None, :]  # (B, M, N)
    if rhs.is_complex():
        A = A.to(rhs.dtype)
    sol = torch.linalg.lstsq(A, rhs).solution
    grad = sol[:, :D, :].transpose(1, 2)  # (B, N, D)
    hess = torch.zeros(B, f0.shape[1], D, D, device=X.device, dtype=rhs.dtype)
    hess[:, :, ij[0], ij[1]] = sol[:, D:, :].transpose(1, 2)
    hess[:, :, ij[1], ij[0]] = hess[:, :, ij[0], ij[1]]  # Symmetrize the Hess

    return grad, hess


@torch.no_grad()
def build_grad_phi(
    X: torch.Tensor, valid_edges: torch.Tensor, eig_vecs: torch.Tensor
) -> torch.Tensor:
    """
    Computes the gradient of the eigenvectors with respect to the input points X using local polynomial reconstruction.
    We have that grad_phi[i,k,d] = ∂phi_k / ∂x_d at point i.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        The input points.
    valid_edges : torch.Tensor (M, 2)
        The edges of the graph, where each edge is represented by a pair of indices (i, j) indicating that point j is a neighbor of point i.
    eig_vecs : torch.Tensor (N, K)
        The eigenvectors of the graph Laplacian.

    Returns:
    -------
    grad_phi : torch.Tensor (N, K, D)
        The gradient of the eigenvectors with respect to the input points X.
    """
    N, D = X.shape
    K = eig_vecs.shape[1]
    # Build the operateur \nabla phi
    grad_phi = torch.zeros(N, K, X.shape[1], device=X.device, dtype=eig_vecs.dtype)
    for i in tqdm(range(N)):
        # Cant't really batch it because not always the same
        # number of neighbors for each point. So we have to do it one by one.
        x_i = X[i]
        neighbors = valid_edges[valid_edges[:, 0] == i][:, 1]
        x_i_neighbors = X[neighbors]
        f0 = eig_vecs[i, :]
        F = eig_vecs[neighbors, :]
        grad_phi_i_, hess_phi_i_ = local_poly_gradient(
            x=x_i.unsqueeze(0),
            X=x_i_neighbors.unsqueeze(0),
            f0=f0.unsqueeze(0),
            F=F.unsqueeze(0),
        )  # (1, N, D), (1, N, D, D)
        grad_phi[i, :, :] = grad_phi_i_[0]
    return grad_phi


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


def get_knn_graph(X: torch.Tensor, n_neighbors: int, device: torch.device) -> torch.Tensor:
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


def distance_matrix_straight_line(
    X: torch.Tensor,
    edges: torch.Tensor,
    randers: RandersMetrics,
    batch_size: int = 100,
    num_quad_points: int = 10,
) -> torch.Tensor:
    """
    Compute Randers distances along straight-line segments for graph edges.

    The integral
        d_F(x_i, x_j) = ∫_0^1 F(x_i + t(x_j-x_i), x_j-x_i) dt
    is evaluated using composite trapezoidal quadrature on [0, 1].
    """
    dst_mat = torch.zeros(edges.shape[0], device=X.device, dtype=X.dtype)

    t = torch.linspace(0.0, 1.0, num_quad_points, device=X.device, dtype=X.dtype)

    for start in tqdm(
        range(0, edges.shape[0], batch_size), desc="Computing Randers distances"
    ):
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
    for start in tqdm(
        range(0, edges.shape[0], batch_size), desc="Computing Randers distances"
    ):
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
        dst_mat = distance_matrix_straight_line(X, edges, randers, batch_size=batch_size)
    else:
        dst_mat = georce_distance_matrix(X, edges, randers, batch_size=batch_size)
    return dst_mat


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


def laplacian_kernel(
    edges: torch.Tensor, dst_edges: torch.Tensor, eps: float, N: int, m: int = 2
):
    dst_mat = torch.zeros((N, N), device=dst_edges.device, dtype=dst_edges.dtype)
    dst_mat[edges[:, 0], edges[:, 1]] = dst_edges

    W = torch.zeros_like(dst_mat)
    W[edges[:, 0], edges[:, 1]] = torch.exp(-dst_mat[edges[:, 0], edges[:, 1]] / eps)
    W.fill_diagonal_(0)

    mu_0 = factorial(m - 1)
    mu_1 = factorial(m)
    return W, mu_0, mu_1


def gaussian_kernel(edges, dst_edges, eps: float, N: int, m: int = 2):
    dst_mat = torch.zeros((N, N), device=dst_edges.device, dtype=dst_edges.dtype)
    dst_mat[edges[:, 0], edges[:, 1]] = dst_edges

    W = torch.zeros_like(dst_mat)
    W[edges[:, 0], edges[:, 1]] = torch.exp(-dst_mat[edges[:, 0], edges[:, 1]] ** 2 / eps**2)
    W.fill_diagonal_(0)

    mu_0 = 1 / 2 * torch.lgamma(torch.tensor(m) / 2).exp()
    mu_1 = 1 / 2 * torch.lgamma(torch.tensor(m + 1) / 2).exp()

    return W, mu_0, mu_1


def construct_operators(W: torch.Tensor, eps: float, theta: int = 1):
    D = torch.sum(W, dim=1)
    D_prime = torch.sum(W, dim=0)
    Q = (D + D_prime) / 2
    Q_theta = Q**theta
    Q_inv_theta = 1 / Q_theta
    Q_theta = torch.diag_embed(Q_theta)
    Q_inv_theta = torch.diag_embed(Q_inv_theta)

    W_theta = Q_inv_theta @ W @ Q_inv_theta
    W_theta_s = (W_theta + W_theta.T) / 2
    W_theta_a = (W_theta - W_theta.T) / 2
    D_theta_s = torch.sum(W_theta_s, dim=1)
    D_theta_a = torch.sum(W_theta_a, dim=1)
    D_theta_a = torch.diag_embed(D_theta_a)
    D_theta_s_inv = 1 / D_theta_s

    Id = torch.eye(W_theta.shape[0]).to(W_theta.device).to(W_theta.dtype)
    P_theta_s = D_theta_s_inv[..., None] * W_theta_s - Id
    P_theta_a = D_theta_s_inv[..., None] * (W_theta_a - D_theta_a)

    L_theta_s = 1 / eps**2 * P_theta_s
    L_theta_a = 1 / eps * P_theta_a
    return L_theta_s, L_theta_a


##############################
# Helpers
##############################


def b_to_v(b_val, A_inv_val):
    """
    Transform the vector b to v using the cometric A_inv.

    Parameters:
    ----------
    b_val : torch.Tensor (B, d)
        The vector b in the tangent space.
    A_inv_val : torch.Tensor (B, d, d) or (B, d)
        The inverse metric tensor A_inv at the point, which can be either a full matrix or a diagonal representation.

    Returns:
    -------
    v_val : torch.Tensor (B, d)
        The vector v in the tangent space, computed from b and A_inv.
    """
    if A_inv_val.ndim == 3:  # Full matrix case
        Ab = torch.einsum("bij,bj->bi", A_inv_val, b_val)
    elif A_inv_val.ndim == 2:  # Diagonal case
        Ab = A_inv_val * b_val
    else:
        raise ValueError("A_inv_val must be either a 2D or 3D tensor.")

    b_norm_sqr = torch.einsum("bi,bi->b", b_val, Ab)
    denom = 1 - b_norm_sqr
    v_val = Ab / denom.unsqueeze(-1)
    return v_val


def v_to_b(v_val, A_val):
    """
    Revert the transformation from v to b using the cometric A.

    Parameters:
    ----------
    v_val : torch.Tensor (B, d)
        The vector v in the tangent space.
    A_val : torch.Tensor (B, d, d) or (B, d)
        The metric tensor A at the point, which can be either a full matrix or a diagonal representation.

    Returns:
    -------
    b_val : torch.Tensor (B, d)
        The vector b in the tangent space, computed from v and A.
    """
    if A_val.ndim == 3:  # Full matrix case
        Av = torch.einsum("bij,bj->bi", A_val, v_val)
    elif A_val.ndim == 2:  # Diagonal case
        Av = A_val * v_val
    else:
        raise ValueError("A_val must be either a 2D or 3D tensor.")

    v_norm_sqr = torch.einsum("bi,bi->b", v_val, Av)
    denom = 1 + (1 + 4 * v_norm_sqr).sqrt()
    b_val = 2 * Av / denom.unsqueeze(-1)
    return b_val


def plot_mf_and_omega(
    base_cometric: CoMetric, X: torch.Tensor, randers_metric: RandersMetrics, bounds: tuple
) -> tuple:
    mf = get_mf_image(base_cometric, X, bounds)
    omega_X = randers_metric.omega(X) * randers_metric.beta

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im = axes[0].imshow(mf.cpu().numpy(), cmap="seismic", origin="lower", extent=bounds)
    t_cbar = axes[0].scatter(
        X[:, 0].detach().cpu(), X[:, 1].detach().cpu(), s=10, edgecolor="k", linewidth=0.5
    )
    axes[0].set_xlim(bounds[0], bounds[1])
    axes[0].set_ylim(bounds[2], bounds[3])
    axes[0].set_title("Density of the Dataset")
    axes[1].scatter(X[:, 0].detach().cpu(), X[:, 1].detach().cpu(), s=10)
    skip = 20  # Adjust this value to change the density of the quiver plot
    axes[1].quiver(
        X[::skip, 0].detach().cpu(),
        X[::skip, 1].detach().cpu(),
        omega_X[::skip, 0].detach().cpu(),
        omega_X[::skip, 1].detach().cpu(),
        color="red",
        scale=1,
        angles="xy",
        scale_units="xy",
    )
    axes[1].set_title("Dataset with Omega vector field")
    for ax in axes:
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(t_cbar, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, axes


def plot_side_by_side(
    X, b_true, b_hat, scale_b: float = 5.0, scale_bhat: float = 5.0, skip: int = 2
):
    """
    Plots the true vector field, the estimated vector field, and the error between them side by side.

    Parameters:
    ----------
    X : torch.Tensor (N, 2)
        The input points in 2D space.
    b_true : torch.Tensor (N, 2)
        The true vector field values at the input points.
    b_hat : torch.Tensor (N, 2)
        The estimated vector field values at the input points.
    scale_bhat : float
        The scale factor for the quiver plot of the estimated vector field. Default is 5.0.
    skip : int
        The step size for plotting the quiver arrows. Default is 2, meaning every second point will be plotted to reduce clutter.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].quiver(
        X[::skip, 0].detach().cpu(),
        X[::skip, 1].detach().cpu(),
        b_true[::skip, 0].detach().cpu(),
        b_true[::skip, 1].detach().cpu(),
        color="blue",
        scale=scale_b,
        angles="xy",
        scale_units="xy",
    )
    axes[0].set_title("True b vector field")
    axes[1].quiver(
        X[::skip, 0].detach().cpu(),
        X[::skip, 1].detach().cpu(),
        b_hat[::skip, 0].detach().cpu(),
        b_hat[::skip, 1].detach().cpu(),
        color="red",
        scale=scale_bhat,
        angles="xy",
        scale_units="xy",
    )
    axes[1].set_title("Estimated b vector field")
    axes[2].quiver(
        X[::skip, 0].detach().cpu(),
        X[::skip, 1].detach().cpu(),
        (b_true - b_hat)[::skip, 0].detach().cpu(),
        (b_true - b_hat)[::skip, 1].detach().cpu(),
        color="green",
        scale=5,
        angles="xy",
        scale_units="xy",
    )
    axes[2].set_title("Error in b vector field")
    plt.tight_layout()
    return fig, axes


##############################
# Solver analytical
##############################
def solve_lstsq(
    f_grad_values: torch.Tensor, Lf_values: torch.Tensor, c_km: float
) -> torch.Tensor:
    """
    Solves the separable least squares problem:
        min_v sum_{k,i} |Lf_k(x_i) - c_km <v(x_i), grad f_k(x_i)>|^2
    independently at each spatial point x_i.

    Parameters:
    ----------
    f_grad_values : torch.Tensor (K, N, d)
        The gradients of the test functions, where K is the number of test functions, N is the number of data points, and d is the dimension of the space.
    Lf_values : torch.Tensor (K, N)
        The values of the operator applied to the test functions, where K is the number of test functions and N is the number of data points.
    c_km : float
        A scaling constant for the least squares problem.

    Returns:
    -------
    v_est : torch.Tensor (N, d)
        The estimated vector field v at each data point, where N is the number of data points and d is the dimension of the space.
    """
    K, N, d = f_grad_values.shape

    if Lf_values.shape != (K, N):
        raise ValueError(
            f"Expected Lf_values to have shape {(K, N)}, " f"got {Lf_values.shape}"
        )

    # Rearrange to have one least-squares system A[i] for every x_i.
    A = c_km * rearrange(f_grad_values, "k n d -> n k d")
    b = rearrange(
        Lf_values,
        "k n -> n k 1",
    )

    v_est = torch.linalg.lstsq(A, b).solution
    v_est = rearrange(v_est, "n d 1 -> n d")

    return v_est


##############################
# Solver deep
##############################


class V_estimator(torch.nn.Module):
    def __init__(self, hidden_dims: list[int], dim: int = 2):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.dim = dim
        layers = []
        input_dim = dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.SiLU())
            input_dim = h_dim
        layers.append(torch.nn.Linear(input_dim, dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OmegaFromV(torch.nn.Module):
    def __init__(self, v_estimator: V_estimator, cometric: CoMetric):
        super().__init__()
        self.v_estimator = v_estimator
        self.cometric = cometric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_x = self.v_estimator(x)
        A_x = self.cometric.metric_tensor(x)
        return v_to_b(v_x, A_x)


def learn_v(
    X: torch.Tensor,
    f_values: torch.Tensor,
    Lf_values: torch.Tensor,
    f_grad_values: torch.Tensor,
    c_km: float,
    v_model: V_estimator,
    device: torch.device = torch.device("cpu"),
    b_size: int = 128,
    n_epochs: int = 1000,
) -> list[float]:
    optim = torch.optim.Adam(v_model.parameters(), lr=1e-3)

    loss_list = []
    pbar = tqdm(range(n_epochs), desc="Learning v")
    for epoch in pbar:
        idx_i = torch.randint(0, X.shape[0], (b_size,), device=device)
        if epoch == 0:
            idx_k = torch.arange(0, f_values.shape[0] - 1, device=device)
        else:
            idx_k = torch.randint(0, f_values.shape[0], (b_size,), device=device)
        X_i = X[idx_i].requires_grad_(True)  # (B, D)

        lf_i = Lf_values[idx_k, :][:, idx_i].detach()  # (K, B)
        grad_f_i = f_grad_values[idx_k, :][:, idx_i, :].detach()  # (K, B, D)
        v_x = v_model(X_i)  # (B, D)
        dot_product = c_km * torch.einsum("kbd,bd->kb", grad_f_i, v_x)  # (K, B)
        loss = torch.nn.functional.mse_loss(dot_product, lf_i.detach())  # (K, B)
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_list.append(loss.item())

        pbar.set_description(f"Loss: {loss.item():.4e}")
    return loss_list
