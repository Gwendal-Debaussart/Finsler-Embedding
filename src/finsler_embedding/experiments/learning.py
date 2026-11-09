"""Reusable vector-field learning utilities for experiments."""

from collections.abc import Sequence

import torch
from einops import rearrange
from tqdm import tqdm

from geodesic_toolbox import CoMetric


def interpolate_V(
    X_low: torch.Tensor,
    V: torch.Tensor,
    dim: int,
    n_epochs: int = 500,
) -> tuple[torch.nn.Module, list[float]]:
    """Fit a neural network interpolant to a vector field.

    Parameters
    ----------
    X_low : torch.Tensor (N, D)
        Coordinates at which the vector field is observed.
    V : torch.Tensor (N, D)
        Vector-field values at ``X_low``.
    dim : int
        Coordinate and vector-field dimension.
    n_epochs : int, default=500
        Number of optimization steps.

    Returns
    -------
    model : torch.nn.Module
        Trained vector-field interpolant.
    list of float
        Training loss at every optimization step.
    """
    model = VEstimator(hidden_dims=[64, 64], dim=dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses: list[float] = []
    iterator = tqdm(range(n_epochs), desc="Interpolating V")
    for _ in iterator:
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X_low), V)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        iterator.set_postfix(loss=losses[-1])
    return model, losses


def solve_lstsq(
    f_grad_values: torch.Tensor, Lf_values: torch.Tensor, c_km: float
) -> torch.Tensor:
    """Recover a vector field from test-function gradients by least squares.

    Parameters
    ----------
    f_grad_values : torch.Tensor (K, N, D)
        Gradients of ``K`` test functions at ``N`` samples.
    Lf_values : torch.Tensor (K, N)
        Operator values for the test functions.
    c_km : float
        Scaling constant in the least-squares equations.

    Returns
    -------
    torch.Tensor (N, D)
        Estimated vector field at each sample.
    """
    K, N, D = f_grad_values.shape
    if Lf_values.shape != (K, N):
        raise ValueError(f"expected Lf_values shape {(K, N)}, got {Lf_values.shape}")
    design = c_km * rearrange(f_grad_values, "k n d -> n k d")
    target = rearrange(Lf_values, "k n -> n k 1")
    return rearrange(torch.linalg.lstsq(design, target).solution, "n d 1 -> n d")


class VEstimator(torch.nn.Module):
    """Small multilayer perceptron used to estimate a vector field."""

    def __init__(self, hidden_dims: Sequence[int], dim: int = 2):
        super().__init__()
        layers: list[torch.nn.Module] = []
        input_dim = dim
        for hidden_dim in hidden_dims:
            layers.extend([torch.nn.Linear(input_dim, hidden_dim), torch.nn.SiLU()])
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector-field model.

        Parameters
        ----------
        x : torch.Tensor (N, D)
            Input coordinates.

        Returns
        -------
        torch.Tensor (N, D)
            Estimated vector field.
        """
        return self.model(x)


class OmegaFromV(torch.nn.Module):
    """Convert a learned vector field into a one-form."""

    def __init__(self, v_estimator: VEstimator, cometric: CoMetric):
        super().__init__()
        self.v_estimator = v_estimator
        self.cometric = cometric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the learned one-form.

        Parameters
        ----------
        x : torch.Tensor (N, D)
            Input coordinates.

        Returns
        -------
        torch.Tensor (N, D)
            One-form values corresponding to the learned vector field.
        """
        from finsler_embedding.utils import v_to_b

        return v_to_b(self.v_estimator(x), self.cometric.metric_tensor(x))


def learn_v(
    X: torch.Tensor,
    f_values: torch.Tensor,
    Lf_values: torch.Tensor,
    f_grad_values: torch.Tensor,
    c_km: float,
    v_model: VEstimator,
    device: torch.device = torch.device("cpu"),
    b_size: int = 128,
    n_epochs: int = 1000,
    pbar: bool = True,
) -> list[float]:
    """Fit a vector-field model to operator and gradient observations.

    Parameters
    ----------
    X : torch.Tensor (N, D)
        Training sample coordinates.
    f_values : torch.Tensor (K, N)
        Test-function values.
    Lf_values : torch.Tensor (K, N)
        Operator values for the test functions.
    f_grad_values : torch.Tensor (K, N, D)
        Test-function gradients.
    c_km : float
        Scaling constant in the training equations.
    v_model : VEstimator
        Model optimized in place.
    device : torch.device, default=torch.device("cpu")
        Device used for sampled batch indices.
    b_size : int, default=128
        Number of spatial samples per optimization step.
    n_epochs : int, default=1000
        Number of optimization steps.
    pbar : bool, default=True
        Whether to display a progress bar.

    Returns
    -------
    list of float
        Training loss at every optimization step.
    """
    if b_size <= 0 or n_epochs < 0:
        raise ValueError("b_size must be positive and n_epochs nonnegative")
    optimizer = torch.optim.Adam(v_model.parameters(), lr=1e-3)
    losses: list[float] = []
    iterator = tqdm(range(n_epochs), desc="Learning v", disable=not pbar)
    for epoch in iterator:
        idx_i = torch.randint(0, X.shape[0], (b_size,), device=device)
        idx_k = (
            torch.arange(0, f_values.shape[0] - 1, device=device)
            if epoch == 0
            else torch.randint(0, f_values.shape[0], (b_size,), device=device)
        )
        X_i = X[idx_i].requires_grad_(True)
        lf_i = Lf_values[idx_k][:, idx_i].detach()
        grad_f_i = f_grad_values[idx_k][:, idx_i, :].detach()
        vector_field = v_model(X_i)
        predicted = c_km * torch.einsum("kbd,bd->kb", grad_f_i, vector_field)
        loss = torch.nn.functional.mse_loss(predicted, lf_i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        iterator.set_description(f"Loss: {loss.item():.4e}")
    return losses
