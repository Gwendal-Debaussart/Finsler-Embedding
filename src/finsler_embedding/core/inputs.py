"""Validated input-specific entry points for Finsler embeddings."""

from collections.abc import Callable

import torch

from .embeddings import EmbeddingInput, select_embedding
from .jacobians import estimate_jacobian
from .kernels import evaluate_kernel, validate_bandwidth, validate_kernel
from .operators import get_Q_inv_theta, get_W_thetas
from .randers import reconstruct_randers


def _validate_matrix(values: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a square, finite, nonnegative matrix.

    Parameters
    ----------
    values : torch.Tensor (N, N)
        Matrix to validate.
    name : str
        Name used in validation error messages.

    Returns
    -------
    torch.Tensor (N, N)
        The validated input matrix.

    Raises
    ------
    ValueError
        If ``values`` is not a square, finite, nonnegative matrix.
    """
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional torch.Tensor")
    if values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be square")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must contain finite, nonnegative values")
    return values


def _run(
    distances: torch.Tensor,
    *,
    points: torch.Tensor | None,
    bandwidth: float,
    kernel: str,
    theta: float,
    n_components: int,
    embedding: str | Callable[..., torch.Tensor],
    embedding_input: EmbeddingInput,
    jacobian: None | str | Callable[..., torch.Tensor],
    jacobian_params: dict | None,
    embedding_params: dict | None,
    edge_mask: torch.Tensor | None = None,
):
    """Build the shared Finsler embedding pipeline.

    Parameters
    ----------
    distances : torch.Tensor (N, N)
        Directed pairwise distances or edge lengths.
    points : torch.Tensor or None (N, D)
        Optional point-cloud coordinates used by point-based embeddings and
        Jacobian estimators.
    bandwidth : float
        Positive kernel bandwidth.
    kernel : str
        Kernel name, either ``"gaussian"`` or ``"exponential"``.
    theta : float
        Degree-normalization exponent in the interval ``[0, 1]``.
    n_components : int
        Number of coordinates in the output embedding.
    embedding : str or callable
        Embedding method. Supported names are ``"spectral"`` and
        ``"isomap"``; a callable may also be supplied.
    embedding_input : {"kernel", "points"}
        Input passed to a custom callable. Named embeddings retain their
        existing input behavior.
    jacobian : str, callable, or None
        Optional Jacobian estimator.
    jacobian_params : dict or None
        Parameters forwarded to the Jacobian estimator.
    embedding_params : dict or None
        Parameters forwarded to the embedding method.
    edge_mask : torch.Tensor (N, N) or None
        Optional mask identifying present graph edges.

    Returns
    -------
    RandersResult
        Typed result containing the kernel, operators, embedding, moments, and
        moment-Randers reconstruction.
    """
    validate_bandwidth(bandwidth)
    validate_kernel(kernel)
    if not torch.isfinite(torch.as_tensor(theta)) or not 0 <= theta <= 1:
        raise ValueError("theta must be finite and in the interval [0, 1]")
    if points is not None and (points.ndim != 2 or points.shape[0] != distances.shape[0]):
        raise ValueError("points must have shape (N, D)")
    directed_kernel = evaluate_kernel(distances, bandwidth, kernel)
    if edge_mask is not None:
        directed_kernel = directed_kernel.masked_fill(~edge_mask, 0)
    # Exclude self-interactions from graph degrees and operators.
    directed_kernel.fill_diagonal_(0)
    q_inv_theta = get_Q_inv_theta(directed_kernel, theta=theta)
    symmetric_kernel, _ = get_W_thetas(directed_kernel, q_inv_theta)
    coordinates = select_embedding(
        embedding,
        symmetric_kernel=symmetric_kernel,
        points=points,
        n_components=n_components,
        embedding_input=embedding_input,
        embedding_params=embedding_params,
    )
    jac = estimate_jacobian(
        jacobian, points if points is not None else distances, coordinates,
        n_components, jacobian_params,
    )
    return reconstruct_randers(
        directed_kernel, bandwidth, coordinates, n_components, kernel, theta, jac
    )


def from_distances(
    distances: torch.Tensor,
    bandwidth: float,
    *,
    kernel: str = "gaussian",
    theta: float = 1.0,
    n_components: int = 2,
    embedding: str | Callable[..., torch.Tensor] = "spectral",
    embedding_input: EmbeddingInput = "kernel",
    jacobian: None | str | Callable[..., torch.Tensor] = None,
    jacobian_params: dict | None = None,
    embedding_params: dict | None = None,
):
    """Construct a Finsler embedding from directed pairwise distances.

    Parameters
    ----------
    distances : torch.Tensor (N, N)
        Directed pairwise distances. Entry ``(i, j)`` is the distance from
        sample ``i`` to sample ``j``.
    bandwidth : float
        Positive bandwidth used to evaluate the directed kernel.
    kernel : str, default="gaussian"
        Kernel name, either ``"gaussian"`` or ``"exponential"``.
    theta : float, default=1.0
        Degree-normalization exponent in the interval ``[0, 1]``.
    n_components : int, default=2
        Number of coordinates in the embedding.
    embedding : str or callable, default="spectral"
        Embedding method. Supported names are ``"spectral"`` and
        ``"isomap"``; a callable may also be supplied.
    embedding_input : {"kernel", "points"}, default="kernel"
        Input passed to a custom callable. Selecting ``"points"`` raises a
        ValueError because this entry point has no point-cloud coordinates.
    jacobian : str, callable, or None, default=None
        Optional Jacobian estimator. Supported names are
        ``"local_least_squares"`` and ``"polynomial"``.
    jacobian_params : dict or None, default=None
        Parameters forwarded to the Jacobian estimator.
    embedding_params : dict or None, default=None
        Parameters forwarded to the embedding method.

    Returns
    -------
    RandersResult
        Typed embedding and moment-Randers reconstruction result.
    """
    return _run(
        _validate_matrix(distances, "distances"), points=None, bandwidth=bandwidth,
        kernel=kernel, theta=theta, n_components=n_components, embedding=embedding,
        embedding_input=embedding_input,
        jacobian=jacobian, jacobian_params=jacobian_params,
        embedding_params=embedding_params,
    )


def from_points(
    points: torch.Tensor,
    *,
    distances: torch.Tensor,
    bandwidth: float,
    kernel: str = "gaussian",
    theta: float = 1.0,
    n_components: int = 2,
    embedding: str | Callable[..., torch.Tensor] = "spectral",
    embedding_input: EmbeddingInput = "kernel",
    jacobian: None | str | Callable[..., torch.Tensor] = None,
    jacobian_params: dict | None = None,
    embedding_params: dict | None = None,
):
    """Construct a Finsler embedding from a point cloud.

    Parameters
    ----------
    points : torch.Tensor (N, D)
        Input point-cloud coordinates.
    distances : torch.Tensor (N, N)
        Directed pairwise distances. This argument is required because a
        symmetric distance computed from ``points`` contains no directional
        information.
    bandwidth : float
        Positive bandwidth used to evaluate the directed kernel.
    kernel : str, default="gaussian"
        Kernel name, either ``"gaussian"`` or ``"exponential"``.
    theta : float, default=1.0
        Degree-normalization exponent in the interval ``[0, 1]``.
    n_components : int, default=2
        Number of coordinates in the embedding.
    embedding : str or callable, default="spectral"
        Embedding method. Supported names are ``"spectral"`` and
        ``"isomap"``; a callable may also be supplied.
    embedding_input : {"kernel", "points"}, default="kernel"
        Input passed to a custom callable. ``"kernel"`` preserves the
        existing behavior; ``"points"`` passes the input point cloud.
    jacobian : str, callable, or None, default=None
        Optional Jacobian estimator. Supported names are
        ``"local_least_squares"`` and ``"polynomial"``.
    jacobian_params : dict or None, default=None
        Parameters forwarded to the Jacobian estimator.
    embedding_params : dict or None, default=None
        Parameters forwarded to the embedding method.

    Returns
    -------
    RandersResult
        Typed embedding and moment-Randers reconstruction result.
    """
    if points.ndim != 2 or not torch.isfinite(points).all():
        raise ValueError("points must be a finite two-dimensional tensor")
    if distances.shape[0] != points.shape[0]:
        raise ValueError("distances and points must have the same number of samples")
    return _run(
        _validate_matrix(distances, "distances"), points=points, bandwidth=bandwidth,
        kernel=kernel, theta=theta, n_components=n_components, embedding=embedding,
        embedding_input=embedding_input,
        jacobian=jacobian, jacobian_params=jacobian_params,
        embedding_params=embedding_params,
    )


def from_graph(
    adjacency: torch.Tensor,
    bandwidth: float,
    *,
    kernel: str = "gaussian",
    theta: float = 1.0,
    n_components: int = 2,
    embedding: str | Callable[..., torch.Tensor] = "spectral",
    embedding_input: EmbeddingInput = "kernel",
    jacobian: None | str | Callable[..., torch.Tensor] = None,
    jacobian_params: dict | None = None,
    embedding_params: dict | None = None,
):
    """Construct an embedding from directed graph edge lengths.

    Parameters
    ----------
    adjacency : torch.Tensor (N, N)
        Matrix of directed edge lengths. Entry ``(i, j)`` is the edge length
        from node ``i`` to node ``j``; zero denotes a missing edge.
    bandwidth : float
        Positive bandwidth used to evaluate the directed kernel.
    kernel : str, default="gaussian"
        Kernel name, either ``"gaussian"`` or ``"exponential"``.
    theta : float, default=1.0
        Degree-normalization exponent in the interval ``[0, 1]``.
    n_components : int, default=2
        Number of coordinates in the embedding.
    embedding : str or callable, default="spectral"
        Embedding method. Supported names are ``"spectral"`` and
        ``"isomap"``; a callable may also be supplied.
    embedding_input : {"kernel", "points"}, default="kernel"
        Input passed to a custom callable. Selecting ``"points"`` raises a
        ValueError because this entry point has no point-cloud coordinates.
    jacobian : str, callable, or None, default=None
        Optional Jacobian estimator. It is usually omitted because graph
        inputs do not provide point coordinates.
    jacobian_params : dict or None, default=None
        Parameters forwarded to the Jacobian estimator.
    embedding_params : dict or None, default=None
        Parameters forwarded to the embedding method.

    Returns
    -------
    RandersResult
        Typed embedding and moment-Randers reconstruction result.
    """
    adjacency = _validate_matrix(adjacency, "adjacency")
    return _run(
        adjacency, points=None, bandwidth=bandwidth,
        kernel=kernel, theta=theta, n_components=n_components, embedding=embedding,
        embedding_input=embedding_input,
        jacobian=jacobian, jacobian_params=jacobian_params,
        embedding_params=embedding_params, edge_mask=adjacency != 0,
    )
