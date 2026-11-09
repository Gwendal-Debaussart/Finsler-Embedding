"""Typed results returned by the public construction pipeline."""

from dataclasses import dataclass

import torch


@dataclass
class OperatorResult:
    kernel: torch.Tensor
    symmetric_operator: torch.Tensor
    antisymmetric_operator: torch.Tensor
    carré_du_champ: torch.Tensor
    drift: torch.Tensor
    gamma_pseudoinverse: torch.Tensor


@dataclass
class EmbeddingResult:
    coordinates: torch.Tensor
    jacobian: torch.Tensor | None
    method: str


@dataclass
class RandersResult:
    kernel: torch.Tensor
    symmetric_operator: torch.Tensor
    antisymmetric_operator: torch.Tensor
    coordinates: torch.Tensor
    carré_du_champ: torch.Tensor
    drift: torch.Tensor
    gamma_pseudoinverse: torch.Tensor
    drift_norm_squared: torch.Tensor
    admissible: torch.Tensor
    centroid: torch.Tensor
    cometric: torch.Tensor
    lambda_: torch.Tensor
    c2: float
    C: float
    kappa: float
    jacobian: torch.Tensor | None = None
