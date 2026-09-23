import numpy as np
import torch
import matplotlib.pyplot as plt
from geodesic_toolbox import *
from functools import partial
from sklearn.datasets import make_swiss_roll
from finsler_embedding.utils import *
from scipy.special import factorial
from einops import rearrange

device = "cpu"


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


def get_x_plt(resolution: int, bounds: tuple = (-2, 2, -2, 2)):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    xx, yy = np.meshgrid(x, y)
    X_plt = np.stack([xx.flatten(), yy.flatten()], axis=1)
    return torch.from_numpy(X_plt).float().to(device)

def sample_uniform(n_samples: int, bounds: tuple = (-2, 2, -2, 2)):
    x = np.random.uniform(bounds[0], bounds[1], n_samples)
    y = np.random.uniform(bounds[2], bounds[3], n_samples)
    samples = np.stack([x, y], axis=1)
    return torch.from_numpy(samples).float().to(device)


class RoundOmega(torch.nn.Module):
    def __init__(self, cometric: CoMetric, freq: float = 1.0):
        super().__init__()
        self.cometric = cometric
        self.freq = freq

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = z.clone()
        omega[:, 0] = -self.freq * z[:, 1]
        omega[:, 1] = self.freq * z[:, 0]
        norm_omega = self.cometric.cometric(z, omega)
        omega = omega / (norm_omega.unsqueeze(1) + 1e-8)  # Avoid division by zero
        return omega

N = 3000
X = sample_uniform(N, bounds=(-2, 2, -2, 2))
X = X.to(torch.float).to(device)
bounds = get_bounds(X, margin=0.2)

base_cometric = IdentityCoMetric().to(device)

omega_true = RoundOmega(base_cometric, freq=1.0)

randers_metric = RandersMetrics(
    base_cometric=base_cometric,
    omega=omega_true,
    beta=0.5
)

solver_f = SolverGraphFinsler(finsler_metric=randers_metric,
                            data=X,
                            n_neighbors=8,
                            )


THETA = 1
dst_mat = solver_f.W
valid_edges_plt = dst_mat > 0
valid_edges = valid_edges_plt.nonzero(as_tuple=False)

epsilon = find_espilon(dst_mat)*2
W = torch.exp(-dst_mat / epsilon) # (N, N)
mu_0 = factorial(2-1)
mu_1 = factorial(2)
# W = 1/epsilon**2 * torch.exp(-dst_mat**2 / epsilon**2)
# mu0 = 1/2 * torch.lgamma(torch.tensor(2.0)/2).exp()
# mu1 = torch.lgamma(torch.tensor(3.0)/2).exp()

D = torch.diag_embed(torch.sum(W, dim=1)) 
D_prime = torch.diag_embed(torch.sum(W, dim=0))
Q = (D + D_prime) / 2
Q_theta = torch.matrix_power(Q, THETA)
W_theta = torch.inverse(Q_theta) @ W @ torch.inverse(Q_theta)
D_theta = torch.diag_embed(torch.sum(W_theta, dim=1))
D_theta_prime = torch.diag_embed(torch.sum(W_theta, dim=0))
D_theta_s = (D_theta + D_theta_prime) / 2
D_theta_a = (D_theta - D_theta_prime) / 2
W_theta_s = (W_theta + W_theta.T) / 2
W_theta_a = (W_theta - W_theta.T) / 2

Id = torch.eye(W_theta.shape[0]).to(W_theta.device).to(W_theta.dtype)
P_theta_s = torch.inverse(D_theta_s) @ W_theta_s - Id
P_theta_a = torch.inverse(D_theta_s) @ (W_theta_a - D_theta_a)

L_theta_s = 1 / epsilon**2 * P_theta_s
L_theta_a = 1 / epsilon * P_theta_a



def gaussian(x, center, sigma):
    """
    A Gaussian function centered at the given point.

    Parameters:
    ----------
    x : torch.Tensor (B, D)
        The input points in D-dimensional space.
    center : torch.Tensor (D,)
        The center of the Gaussian function.
    sigma : float
        The standard deviation of the Gaussian function.
    """
    distance_squared = torch.sum((x - center) ** 2, dim=-1)
    return torch.exp(-distance_squared / (2 * sigma ** 2))


K_ = X.shape[0]  # Number of test functions, one for each point in X
radius_list = [0.01, 0.1, 0.2, 0.5]
radius_list = [0.1]
K = len(radius_list)*K_

f_values = []
Lf_values = []
f_grad_values = []

for radius in radius_list:
    f_values_radius = torch.zeros((K_, X.shape[0]))
    Lf_values_radius = torch.zeros((K_, X.shape[0]))
    f_grad_values_radius = torch.zeros((K_, X.shape[0], X.shape[1]))
    X = X.requires_grad_()
    for i in range(K_):
        center = X[i].detach()  # (D,)
        f_ = gaussian(X, center=center.detach(), sigma=radius)  # (N,)
        f_grad_ = torch.autograd.grad(f_.sum(), X, create_graph=True)[0]  # (N, D)
        
        f_values_radius[i] = f_
        f_grad_values_radius[i] = f_grad_
        Lf_values_radius[i] = L_theta_a @ f_  # (N,)

    f_values.append(f_values_radius.detach())
    Lf_values.append(Lf_values_radius.detach())
    f_grad_values.append(f_grad_values_radius.detach())

f_values = torch.cat(f_values, dim=0)  # (K, N)
Lf_values = torch.cat(Lf_values, dim=0)  # (K, N)
f_grad_values = torch.cat(f_grad_values, dim=0)  # (K, N, D)

print(f"shapes : {f_values.shape}, {Lf_values.shape}, {f_grad_values.shape}")

# Check no problem with gradients
assert torch.all(torch.isfinite(f_grad_values)), "f_grad_values contains NaN or Inf"
assert torch.all(torch.isfinite(Lf_values)), "Lf_values contains NaN or Inf"


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


m = X.shape[1]
c_km = - mu_1 / mu_0 * (m+1)/m

b_true = randers_metric.omega(X) * randers_metric.beta
v_true = b_to_v(b_true, base_cometric.cometric_tensor(X))
true_rhs = c_km * torch.einsum('n d, k n d -> k n', v_true, f_grad_values)

delta_rhs = true_rhs - Lf_values
delta_rhs_norm = torch.norm(delta_rhs, dim=1)
print(delta_rhs_norm.mean(), delta_rhs_norm.max(), delta_rhs_norm.min())
