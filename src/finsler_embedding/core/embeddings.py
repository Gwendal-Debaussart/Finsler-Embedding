"""Embedding selection for the Finsler construction pipeline."""

from collections.abc import Callable

import torch
from sklearn.manifold import Isomap

from .operators import leading_eigenvectors


def _validate_components(n: int, n_components: int) -> None:
    """Validate an embedding component count.

    Parameters
    ----------
    n : int
        Number of samples.
    n_components : int
        Requested number of embedding coordinates.

    Returns
    -------
    None
        This function returns after successful validation.
    """
    if not isinstance(n_components, int) or not 1 <= n_components < n:
        raise ValueError("n_components must be an integer in [1, n)")


def spectral_embedding(symmetric_kernel: torch.Tensor, n_components: int) -> torch.Tensor:
    """Compute a spectral embedding from a symmetric kernel.

    Parameters
    ----------
    symmetric_kernel : torch.Tensor (N, N)
        Symmetric, degree-normalized kernel ``W_theta^s`` used to construct
        ``L_theta^s``. The leading nontrivial eigenvectors of this operator
        are the embedding proposed in the paper.
    n_components : int
        Number of nonconstant embedding coordinates.

    Returns
    -------
    torch.Tensor (N, n_components)
        Leading nontrivial spectral coordinates.
    """
    _validate_components(symmetric_kernel.shape[0], n_components)
    return leading_eigenvectors(symmetric_kernel, n_components)


def isomap_embedding(
    points: torch.Tensor, n_components: int, n_neighbors: int = 10
) -> torch.Tensor:
    """Compute an Isomap embedding from point-cloud coordinates.

    Parameters
    ----------
    points : torch.Tensor (N, D)
        Input point-cloud coordinates.
    n_components : int
        Number of embedding coordinates.
    n_neighbors : int, default=10
        Number of neighbors used to construct the Isomap graph.

    Returns
    -------
    torch.Tensor (N, n_components)
        Isomap coordinates on the input tensor's device and with its dtype.
    """
    _validate_components(points.shape[0], n_components)
    if not isinstance(n_neighbors, int) or not 1 <= n_neighbors < points.shape[0]:
        raise ValueError("n_neighbors must be in [1, n)")
    result = Isomap(n_components=n_components, n_neighbors=n_neighbors).fit_transform(
        points.detach().cpu().numpy()
    )
    return torch.as_tensor(result, dtype=points.dtype, device=points.device)


def select_embedding(
    embedding: str | Callable[..., torch.Tensor],
    *,
    symmetric_kernel: torch.Tensor,
    points: torch.Tensor | None,
    n_components: int,
    embedding_params: dict | None = None,
) -> torch.Tensor:
    """Select and compute an embedding using a named method or callable.

    Parameters
    ----------
    embedding : str or callable
        ``"spectral"``, ``"isomap"``, or a callable returning coordinates.
        If callable its interface must accept a symmetric kernel and ``n_components`` as keyword arguments.
    symmetric_kernel : torch.Tensor (N, N)
        Symmetric kernel used by spectral and custom embeddings.
    points : torch.Tensor or None (N, D)
        Point-cloud coordinates required by Isomap.
    n_components : int
        Number of embedding coordinates.
    embedding_params : dict or None, optional
        Additional keyword arguments passed to the selected method.

    Returns
    -------
    torch.Tensor (N, n_components)
        Validated embedding coordinates.
    """
    params = embedding_params or {}
    if callable(embedding):
        coordinates = embedding(symmetric_kernel, n_components=n_components, **params)
    elif embedding == "spectral":
        coordinates = spectral_embedding(symmetric_kernel, n_components)
    elif embedding == "isomap":
        if points is None:
            raise ValueError("the isomap embedding requires point-cloud coordinates")
        coordinates = isomap_embedding(points, n_components, **params)
    else:
        raise ValueError(
            f"embedding must be 'spectral', 'isomap', or a callable, got {embedding!r}"
        )
    if not isinstance(coordinates, torch.Tensor) or coordinates.shape != (
        symmetric_kernel.shape[0],
        n_components,
    ):
        raise ValueError("embedding must return a tensor of shape (N, n_components)")
    return coordinates
