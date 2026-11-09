"""Directed kernels and their asymptotic constants."""

from math import gamma

import torch

KERNELS = ("gaussian", "exponential")


def _edge_kernel(
    edges: torch.Tensor,
    distances: torch.Tensor,
    bandwidth: float,
    n_samples: int,
    kernel: str,
) -> torch.Tensor:
    """Evaluate a kernel on an edge list and assemble its dense matrix.

    Parameters
    ----------
    edges : torch.Tensor (M, 2)
        Integer source and destination indices for the directed edges.
    distances : torch.Tensor (M,)
        Nonnegative directed distances associated with ``edges``.
    bandwidth : float
        Strictly positive kernel bandwidth.
    n_samples : int
        Number of vertices in the resulting dense matrix.
    kernel : str
        Kernel family, either ``"gaussian"`` or ``"exponential"``.

    Returns
    -------
    torch.Tensor (N, N)
        Dense directed kernel matrix with zero diagonal and zero values for
        absent edges.
    """
    validate_bandwidth(bandwidth)
    validate_kernel(kernel)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (M, 2)")
    if distances.ndim != 1 or distances.shape[0] != edges.shape[0]:
        raise ValueError("distances must have shape (M,)")
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distances must be finite and nonnegative")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if edges.numel() and ((edges < 0).any() or (edges >= n_samples).any()):
        raise ValueError("edges contain indices outside [0, n_samples)")

    distance_matrix = torch.zeros(
        (n_samples, n_samples), device=distances.device, dtype=distances.dtype
    )
    distance_matrix[edges[:, 0], edges[:, 1]] = distances
    kernel_distances = distance_matrix[edges[:, 0], edges[:, 1]]
    if kernel == "gaussian":
        values = torch.exp(-(kernel_distances / bandwidth).square())
    else:
        values = torch.exp(-kernel_distances / bandwidth)
    kernel_matrix = torch.zeros_like(distance_matrix)
    kernel_matrix[edges[:, 0], edges[:, 1]] = values
    kernel_matrix.fill_diagonal_(0)
    return kernel_matrix


def gaussian_kernel(
    edges: torch.Tensor,
    distances: torch.Tensor,
    bandwidth: float,
    n_samples: int,
    dimension: int = 2,
) -> tuple[torch.Tensor, float, float]:
    """Construct a dense directed Gaussian kernel and its moments.

    This function is the graph-edge counterpart of :func:`evaluate_kernel`.
    It returns the first two moments as well because the experiment code
    historically consumed this three-value return signature.

    Parameters
    ----------
    edges : torch.Tensor (M, 2)
        Integer source and destination indices for the directed edges.
    distances : torch.Tensor (M,)
        Nonnegative directed distances associated with ``edges``.
    bandwidth : float
        Strictly positive Gaussian bandwidth.
    n_samples : int
        Number of vertices in the dense kernel matrix.
    dimension : int, default=2
        Dimension used to compute the radial kernel moments.

    Returns
    -------
    kernel : torch.Tensor (N, N)
        Dense directed Gaussian kernel with zero diagonal.
    mu_0 : float
        Zeroth radial moment of the Gaussian kernel.
    mu_1 : float
        First radial moment of the Gaussian kernel.
    """
    kernel = _edge_kernel(edges, distances, bandwidth, n_samples, "gaussian")
    mu_0, mu_1, _ = kernel_moments(dimension)
    return kernel, mu_0, mu_1


def laplacian_kernel(
    edges: torch.Tensor,
    distances: torch.Tensor,
    bandwidth: float,
    n_samples: int,
    dimension: int = 2,
) -> tuple[torch.Tensor, int, int]:
    """Construct a dense directed exponential kernel and its moments.

    Parameters
    ----------
    edges : torch.Tensor (M, 2)
        Integer source and destination indices for the directed edges.
    distances : torch.Tensor (M,)
        Nonnegative directed distances associated with ``edges``.
    bandwidth : float
        Strictly positive exponential-kernel bandwidth.
    n_samples : int
        Number of vertices in the dense kernel matrix.
    dimension : int, default=2
        Dimension used to compute the radial kernel moments.

    Returns
    -------
    kernel : torch.Tensor (N, N)
        Dense directed exponential kernel with zero diagonal.
    mu_0 : float
        Zeroth radial moment of the exponential kernel.
    mu_1 : float
        First radial moment of the exponential kernel.
    """
    kernel = _edge_kernel(edges, distances, bandwidth, n_samples, "exponential")
    return kernel, gamma(dimension), gamma(dimension + 1)


def validate_kernel(kernel: str) -> str:
    """Validate a supported kernel name.

    Parameters
    ----------
    kernel : str
        Kernel name to validate.

    Returns
    -------
    str
        The validated kernel name.
    """
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
    return kernel


def validate_bandwidth(bandwidth: float) -> float:
    """Validate and normalize a kernel bandwidth.

    Parameters
    ----------
    bandwidth : float
        Candidate positive, finite bandwidth.

    Returns
    -------
    float
        The bandwidth converted to a Python float.
    """
    if not torch.isfinite(torch.as_tensor(bandwidth)) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and strictly positive")
    return float(bandwidth)


def kernel_moments(m: int, kernel: str = "gaussian") -> tuple[float, float, float]:
    """Compute the first three radial moments of a kernel.

    Parameters
    ----------
    m : int
        Ambient or intrinsic dimension used in the radial integral.
    kernel : str, default="gaussian"
        Kernel name, either ``"gaussian"`` or ``"exponential"``.

    Returns
    -------
    tuple of float
        Zeroth, first, and second radial moments ``(mu_0, mu_1, mu_2)``.
    """
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    validate_kernel(kernel)
    if kernel == "gaussian":
        return tuple(0.5 * gamma((m + n) / 2) for n in (0, 1, 2))
    return tuple(gamma(m + n) for n in (0, 1, 2))


def evaluate_kernel(distances: torch.Tensor, bandwidth: float, kernel: str = "gaussian") -> torch.Tensor:
    """Evaluate a nonnegative radial kernel on directed distances.

    Parameters
    ----------
    distances : torch.Tensor (...,)
        Finite, nonnegative directed distances or edge lengths.
    bandwidth : float
        Positive kernel bandwidth.
    kernel : str, default="gaussian"
        Kernel name, either ``"gaussian"`` or ``"exponential"``.

    Returns
    -------
    torch.Tensor (...,)
        Kernel values with the same shape, dtype, and device as ``distances``.
    """
    validate_bandwidth(bandwidth)
    validate_kernel(kernel)
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distances must be finite and nonnegative")
    scaled = distances / bandwidth
    return torch.exp(-scaled.square() if kernel == "gaussian" else -scaled)


def randers_constants(m: int, kernel: str = "gaussian") -> tuple[float, float, float]:
    """Compute constants used by moment-Randers reconstruction.

    Parameters
    ----------
    m : int
        Intrinsic dimension of the embedding.
    kernel : str, default="gaussian"
        Kernel name, either ``"gaussian"`` or ``"exponential"``.

    Returns
    -------
    tuple of float
        Constants ``(c_2, C, kappa_K)`` from the kernel moments.
    """
    mu0, mu1, mu2 = kernel_moments(m, kernel)
    c2 = mu2 / (2.0 * m * mu0)
    C = (m + 1) * mu1 / (m * mu0)
    return c2, C, c2 / C**2
