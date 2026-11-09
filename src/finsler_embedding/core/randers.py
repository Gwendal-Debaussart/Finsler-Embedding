"""Moment-Randers reconstruction from directed operators."""

import torch

from .kernels import randers_constants
from .operators import carre_du_champ, construct_operators, truncated_pinv
from .results import RandersResult


def reconstruct_randers(
    kernel: torch.Tensor,
    bandwidth: float,
    coordinates: torch.Tensor,
    intrinsic_dim: int,
    kernel_name: str = "gaussian",
    theta: float = 1.0,
    jacobian: torch.Tensor | None = None,
) -> RandersResult:
    """Reconstruct moment-Randers quantities from a directed kernel.

    Parameters
    ----------
    kernel : torch.Tensor (N, N)
        Directed kernel matrix.
    bandwidth : float
        Positive bandwidth used to scale the symmetric and antisymmetric
        operators.
    coordinates : torch.Tensor (N, L)
        Embedding coordinates used to estimate moments.
    intrinsic_dim : int
        Intrinsic rank used for truncated pseudoinverses.
    kernel_name : str, default="gaussian"
        Kernel family used to compute asymptotic constants.
    theta : float, default=1.0
        Degree-normalization exponent.
    jacobian : torch.Tensor or None (N, L, intrinsic_dim), optional
        Optional embedding Jacobians associated with the coordinates.

    Returns
    -------
    RandersResult
        Typed symmetric and antisymmetric operators, moment estimates,
        admissibility quantities, and reconstructed Randers tensors.
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if coordinates.ndim != 2 or coordinates.shape[0] != kernel.shape[0]:
        raise ValueError("coordinates must have shape (N, n_components)")
    if not 1 <= intrinsic_dim <= coordinates.shape[1]:
        raise ValueError("intrinsic_dim must not exceed the coordinate dimension")
    symmetric_operator, antisymmetric_operator = construct_operators(
        kernel, bandwidth, theta=theta
    )
    c2, C, kappa = randers_constants(intrinsic_dim, kernel_name)
    gamma = carre_du_champ(symmetric_operator, coordinates)
    drift = antisymmetric_operator @ coordinates
    gamma_pseudoinverse = truncated_pinv(gamma, rank=intrinsic_dim)
    drift_norm_squared = kappa * torch.einsum(
        "nij,ni,nj->n", gamma_pseudoinverse, drift, drift
    )
    admissible = drift_norm_squared < 1.0 / (intrinsic_dim + 3)
    centroid = kappa**0.5 * drift
    cometric_inverse = gamma - (intrinsic_dim + 2) * kappa * torch.einsum(
        "ni,nj->nij", drift, drift
    )
    cometric = truncated_pinv(cometric_inverse, rank=intrinsic_dim)
    lambda_ = 1.0 - torch.einsum("nij,ni,nj->n", cometric, centroid, centroid)
    return RandersResult(
        kernel=kernel,
        symmetric_operator=symmetric_operator,
        antisymmetric_operator=antisymmetric_operator,
        coordinates=coordinates,
        carré_du_champ=gamma,
        drift=drift,
        gamma_pseudoinverse=gamma_pseudoinverse,
        drift_norm_squared=drift_norm_squared,
        admissible=admissible,
        centroid=centroid,
        cometric=cometric,
        lambda_=lambda_,
        c2=c2,
        C=C,
        kappa=kappa,
        jacobian=jacobian,
    )
