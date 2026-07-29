import torch
from sklearn.neighbors import kneighbors_graph
from geodesic_toolbox import *
from tqdm import tqdm


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
    epsilon = upper_triangular.std()
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
    dst_mat = kneighbors_graph(z, n_neighbors=n_neighbors, mode='distance', include_self=False)
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


def loss_function_omega(
    z_low: torch.Tensor,
    beta_hat: torch.Tensor,
    omega_hat: torch.nn.Module,
    edges: torch.Tensor,
    criterion: torch.nn.Module = torch.nn.MSELoss(),
) -> torch.Tensor:
    """
    Compares the values of beta_hat with the inner product of omega_hat(z_i) and the difference z_j - z_i for all edges (i,j)
    using the provided loss criterion.

    Parameters:
    -----------
    z_low : torch.Tensor (N, D)
        The low-dimensional embeddings of the data points.
    beta_hat : torch.Tensor (N, N)
        The estimated beta values for each pair of points.
    omega_hat : torch.nn.Module
        A function that takes a tensor of shape (N, D) and returns a tensor of shape (N, D), representing the estimated omega values for each point.
    edges : torch.Tensor (M, 2)
        The indices of the edges in the graph.
    criterion : torch.nn.Module
        The loss function to use for computing the loss.

    Returns:
    --------
    loss : torch.Tensor
        The computed loss value.
    """
    beta_values = beta_hat[edges[:, 0], edges[:, 1]]  # (M,)

    z_i = z_low[edges[:, 0]]
    z_j = z_low[edges[:, 1]]
    omega_hat_i = omega_hat(z_i)  # (M, D)
    dz = z_j - z_i  # (M, D)

    logits = torch.einsum('bi,bi->b', omega_hat_i, dz)  # (M,)
    loss = criterion(logits, beta_values)
    return loss


class OmegaModel(torch.nn.Module):
    """Simple MLP based model to learn the omega vector field from the low-dimensional embeddings."""

    def __init__(self, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        layers = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.GELU())
            input_dim = h_dim
        layers.append(torch.nn.Linear(input_dim, latent_dim))
        # layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


def learn_omega(
    z_low: torch.Tensor,
    beta_hat: torch.Tensor,
    edges: torch.Tensor,
    omega_hat: torch.nn.Module,
    lr: float = 1e-2,
    n_iter: int = 1000,
) -> list[float]:
    """
    Function to learn the omega_hat model by minimizing the loss between beta_hat and the inner product of omega_hat(z_i) and (z_j - z_i) for all edges (i,j).

    Parameters:
    -----------
    z_low : torch.Tensor (N, D)
        The low-dimensional embeddings of the data points.
    beta_hat : torch.Tensor (N, N)
        The estimated beta values for each pair of points.
    edges : torch.Tensor (M, 2)
        The indices of the edges in the graph.
    omega_hat : torch.nn.Module
        A function that takes a tensor of shape (N, D) and returns a tensor of shape (N, D), representing the estimated omega values for each point.
    lr : float
        The learning rate for the optimizer.
    n_iter : int
        The number of iterations for the optimization process.

    Returns:
    --------
    losses : list[float]
        A list containing the loss values at each iteration of the optimization process.
    """
    loss_0 = loss_function_omega(z_low, beta_hat, omega_hat, edges)

    optim = torch.optim.Adam(omega_hat.parameters(), lr=lr)

    losses = [loss_0.item()]
    pbar = tqdm(range(n_iter), desc="Learning omega_hat")
    for i in pbar:
        optim.zero_grad()
        loss = loss_function_omega(z_low, beta_hat, omega_hat, edges)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())
    return losses


class CometricHat(CoMetric):
    """Simple MLP based model to learn the cometric matrix from the low-dimensional embeddings."""

    def __init__(self, latent_dim: int, hidden_dims: list[int]):
        super().__init__(is_diag=True)
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Assume a diagonal cometric matrix
        layers = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.GELU())
            input_dim = h_dim
        layers.append(torch.nn.Linear(input_dim, latent_dim))
        layers.append(torch.nn.Softplus())  # Ensure positivity
        self.model = torch.nn.Sequential(*layers)

    def metric_tensor(self, z):
        # Return the diagonal of the cometric matrix
        diag = self.model(z)
        return diag

    def forward(self, z):
        return 1 / self.metric_tensor(z)  # Return the diagonal of the metric matrix


# class CometricHat(CoMetric):
# Same as above but we work in the log space to ensure positivity of the cometric matrix
#     def __init__(self, latent_dim:int,hidden_dims:list[int]):
#         super().__init__(is_diag=True)
#         self.latent_dim = latent_dim
#         self.hidden_dims = hidden_dims

#         # Assume a diagonal cometric matrix
#         # The be PSD we work in the log space
#         layers = []
#         input_dim = latent_dim
#         for h_dim in hidden_dims:
#             layers.append(torch.nn.Linear(input_dim, h_dim))
#             layers.append(torch.nn.GELU())
#             input_dim = h_dim
#         # Assume
#         layers.append(torch.nn.Linear(input_dim, latent_dim))
#         self.model = torch.nn.Sequential(*layers)

#     def metric_tensor(self, z):
#         # Return the diagonal of the cometric matrix
#         diag = self.model(z)
#         diag = torch.exp(diag)  # Ensure positivity
#         return diag

#     def forward(self, z):
#         return 1 / self.metric_tensor(z)  # Return the diagonal of the metric matrix


def loss_function_alpha(
    z_low: torch.Tensor,
    alpha_hat: torch.Tensor,
    alpha_model: CoMetric,
    edges: torch.Tensor,
    criterion: torch.nn.Module = torch.nn.MSELoss(),
) -> torch.Tensor:
    """
    Compares the values of alpha_hat with the Riemannian norm of the difference z_j - z_i under the cometric defined by alpha_model for all edges (i,j)

    Parameters:
    -----------
    z_low : torch.Tensor (N, D)
        The low-dimensional embeddings of the data points.
    alpha_hat : torch.Tensor (N, N)
        The estimated alpha values for each pair of points.
    alpha_model : CoMetric
        A function that takes a tensor of shape (N, D) and returns a tensor of shape (N, D), representing the estimated cometric values for each point.
    edges : torch.Tensor (M, 2)
        The indices of the edges in the graph.
    criterion : torch.nn.Module
        The loss function to use for computing the loss.

    Returns:
    --------
    loss : torch.Tensor
        The computed loss value.
    """
    z_i = z_low[edges[:, 0]]
    z_j = z_low[edges[:, 1]]
    alpha_values = alpha_hat[edges[:, 0], edges[:, 1]]  # (M,)

    dz = z_j - z_i
    alpha_values_hat = alpha_model.metric(z_i, dz)  # (M,)
    return criterion(alpha_values_hat, alpha_values)


def learn_alpha(
    z_low: torch.Tensor,
    alpha_hat: torch.Tensor,
    edges: torch.Tensor,
    alpha_model: torch.nn.Module,
    lr: float = 1e-2,
    n_iter: int = 1000,
) -> list[float]:
    """
    Function to learn the alpha_model by minimizing the loss between alpha_hat and the Riemannian norm of the difference z_j - z_i under the cometric defined by alpha_model for all edges (i,j).

    Parameters:
    -----------
    z_low : torch.Tensor (N, D)
        The low-dimensional embeddings of the data points.
    alpha_hat : torch.Tensor (N, N)
        The estimated alpha values for each pair of points.
    edges : torch.Tensor (M, 2)
        The indices of the edges in the graph.
    alpha_model : torch.nn.Module
        A function that takes a tensor of shape (N, D) and returns a tensor of shape (N, D), representing the estimated cometric values for each point.
    lr : float
        The learning rate for the optimizer.
    n_iter : int
        The number of iterations for the optimization process.

    Returns:
    --------
    losses : list[float]
        A list containing the loss values at each iteration of the optimization process.
    """
    loss_0 = loss_function_alpha(z_low, alpha_hat, alpha_model, edges)

    optim = torch.optim.Adam(alpha_model.parameters(), lr=lr)

    losses = [loss_0.item()]
    pbar = tqdm(range(n_iter), desc="Learning alpha_hat")
    for i in pbar:
        optim.zero_grad()
        loss = loss_function_alpha(z_low, alpha_hat, alpha_model, edges)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())
    return losses
