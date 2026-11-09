"""Optional Jacobian estimators used by the public API."""

from collections.abc import Callable

import torch
from sklearn.neighbors import NearestNeighbors


def local_poly_gradient(
    x: torch.Tensor, X: torch.Tensor, f0: torch.Tensor, F: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct local gradients and Hessians with a quadratic polynomial.

    Parameters
    ----------
    x : torch.Tensor (B, D)
        Center points.
    X : torch.Tensor (B, M, D)
        Neighbor points for each center.
    f0 : torch.Tensor (B, K)
        Values of the functions at each center.
    F : torch.Tensor (B, M, K)
        Values of the functions at each neighbor.

    Returns
    -------
    grad : torch.Tensor (B, K, D)
        Reconstructed first derivatives at each center.
    hess : torch.Tensor (B, K, D, D)
        Reconstructed symmetric Hessians at each center.
    """
    B, M, D = X.shape
    if x.shape != (B, D) or f0.ndim != 2 or F.shape != (B, M, f0.shape[1]):
        raise ValueError("local polynomial inputs have incompatible shapes")

    dx = X - x[:, None, :]
    ij = torch.triu_indices(D, D, device=X.device)
    quadratic = dx[:, :, ij[0]] * dx[:, :, ij[1]]
    diagonal = ij[0] == ij[1]
    quadratic = torch.where(diagonal[None, None, :], 0.5 * quadratic, quadratic)
    design = torch.cat([dx, quadratic], dim=-1)
    rhs = F - f0[:, None, :]
    if rhs.is_complex():
        design = design.to(rhs.dtype)
    solution = torch.linalg.lstsq(design, rhs).solution
    grad = solution[:, :D, :].transpose(1, 2)
    hess = torch.zeros(B, f0.shape[1], D, D, device=X.device, dtype=rhs.dtype)
    hess[:, :, ij[0], ij[1]] = solution[:, D:, :].transpose(1, 2)
    hess[:, :, ij[1], ij[0]] = hess[:, :, ij[0], ij[1]]
    return grad, hess


def local_least_squares(
    X: torch.Tensor, Y: torch.Tensor, intrinsic_dim: int, k: int = 15, **_: object
) -> torch.Tensor:
    """Estimate embedding Jacobians with local linear least squares.

    Parameters
    ----------
    X : torch.Tensor (N, D)
        Input coordinates used to find local neighborhoods.
    Y : torch.Tensor (N, L)
        Embedded coordinates whose derivatives are estimated.
    intrinsic_dim : int
        Number of input tangent coordinates to retain.
    k : int, default=15
        Number of neighbors used for each local least-squares problem.
    **_ : object
        Additional unused estimator parameters.

    Returns
    -------
    torch.Tensor (N, L, intrinsic_dim)
        Jacobian of the embedding at each input sample.
    """
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be two-dimensional tensors with equal sample counts")
    n = X.shape[0]
    if not 1 <= intrinsic_dim <= X.shape[1] or not 1 <= k < n:
        raise ValueError("intrinsic_dim and k are incompatible with X")
    indices = torch.as_tensor(
        NearestNeighbors(n_neighbors=k + 1).fit(X.detach().cpu().numpy())
        .kneighbors(return_distance=False)[:, 1:],
        device=X.device,
    )
    dx = X[indices] - X[:, None]
    dy = Y[indices] - Y[:, None]
    solution = torch.linalg.lstsq(dx, dy).solution
    return solution.transpose(1, 2)[..., :intrinsic_dim]


def polynomial(X: torch.Tensor, Y: torch.Tensor, intrinsic_dim: int, **params: object) -> torch.Tensor:
    """Estimate Jacobians with local polynomial reconstruction.

    Parameters
    ----------
    X : torch.Tensor (N, D)
        Input coordinates used to find local neighborhoods.
    Y : torch.Tensor (N, L)
        Embedded coordinates whose derivatives are estimated.
    intrinsic_dim : int
        Number of input tangent coordinates to retain.
    **params : object
        Estimator parameters. ``k`` optionally sets the neighborhood size.

    Returns
    -------
    torch.Tensor (N, L, intrinsic_dim)
        Jacobian of the embedding at each input sample.
    """
    k = int(params.get("k", max(2 * intrinsic_dim + 1, 6)))
    if k >= X.shape[0]:
        raise ValueError("polynomial Jacobian needs fewer neighbors than samples")
    indices = torch.as_tensor(
        NearestNeighbors(n_neighbors=k + 1).fit(X.detach().cpu().numpy())
        .kneighbors(return_distance=False)[:, 1:],
        device=X.device,
    )
    gradients, _ = local_poly_gradient(X, X[indices], Y, Y[indices])
    return gradients[..., :intrinsic_dim]


def estimate_jacobian(
    method: None | str | Callable[..., torch.Tensor],
    X: torch.Tensor,
    Y: torch.Tensor,
    intrinsic_dim: int,
    params: dict | None = None,
) -> torch.Tensor | None:
    """Dispatch an optional Jacobian estimator and validate its result.

    Parameters
    ----------
    method : str, callable, or None
        Estimator name, estimator callable, or ``None`` to skip estimation.
    X : torch.Tensor (N, D)
        Input coordinates.
    Y : torch.Tensor (N, L)
        Embedded coordinates.
    intrinsic_dim : int
        Number of Jacobian input dimensions.
    params : dict or None, optional
        Parameters forwarded to the estimator.

    Returns
    -------
    torch.Tensor or None (N, L, intrinsic_dim)
        Estimated Jacobians, or ``None`` when ``method`` is ``None``.
    """
    if method is None:
        return None
    if isinstance(method, str):
        if method == "local_least_squares":
            function = local_least_squares
        elif method == "polynomial":
            function = polynomial
        else:
            raise ValueError("jacobian must be None, 'local_least_squares', 'polynomial', or callable")
    elif callable(method):
        function = method
    else:
        raise TypeError("jacobian must be None, a supported string, or callable")
    result = function(X, Y, intrinsic_dim=intrinsic_dim, **(params or {}))
    expected = (X.shape[0], Y.shape[1], intrinsic_dim)
    if not isinstance(result, torch.Tensor) or result.shape != expected:
        raise ValueError(f"Jacobian estimator must return a tensor of shape {expected}")
    return result
