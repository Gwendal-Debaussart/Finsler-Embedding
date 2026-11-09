import logging

import torch
from .core.jacobians import local_poly_gradient
from sklearn.neighbors import kneighbors_graph
from geodesic_toolbox import *
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


def get_grid_pts(bounds: torch.Tensor, n_pts: int) -> torch.Tensor:
    """
    Generate a grid of points within the specified bounds.

    Parameters:
    ----------
    bounds : torch.Tensor (4,)
        [min_x, max_x, min_y, max_y], the bounds of the grid.
    n_pts : int
        The number of points along each axis.

    Returns:
    -------
    torch.Tensor (n_pts*n_pts, 2)
        The grid points within the specified bounds.
    """
    min_x, max_x, min_y, max_y = bounds
    x = torch.linspace(min_x, max_x, n_pts)
    y = torch.linspace(min_y, max_y, n_pts)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return grid_points


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
