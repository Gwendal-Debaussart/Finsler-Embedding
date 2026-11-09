import torch

from finsler_embedding import from_distances, from_points
from finsler_embedding.core.kernels import evaluate_kernel, kernel_moments
from finsler_embedding.core.operators import (
    carre_du_champ,
    get_Q_inv_theta,
    get_W_thetas,
    leading_eigenvectors,
    truncated_pinv,
)


def test_kernel_values_and_moments():
    distances = torch.tensor([0.0, 1.0])
    assert torch.allclose(evaluate_kernel(distances, 2.0), torch.exp(-distances.square() / 4))
    assert len(kernel_moments(2, "exponential")) == 3


def test_operator_helpers():
    matrix = torch.diag(torch.tensor([2.0, 3.0]))
    inverse = truncated_pinv(matrix, rank=2)
    assert torch.allclose(matrix @ inverse @ matrix, matrix)

    operator = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
    coordinates = torch.tensor([[0.0], [1.0]])
    gamma = carre_du_champ(operator, coordinates)
    assert gamma.shape == (2, 1, 1)
    assert torch.allclose(gamma, gamma.transpose(1, 2))


def test_from_distances_and_optional_jacobian():
    points = torch.linspace(0, 1, 10).reshape(-1, 1)
    distances = torch.cdist(points, points)
    result = from_distances(distances, bandwidth=0.25, n_components=1)
    assert result.coordinates.shape == (10, 1)
    assert result.jacobian is None

    result = from_points(
        points,
        distances=distances,
        bandwidth=0.25,
        n_components=1,
        jacobian="local_least_squares",
        jacobian_params={"k": 3},
    )
    assert result.jacobian.shape == (10, 1, 1)


def test_spectral_embedding_uses_normalized_symmetric_operator():
    points = torch.linspace(0, 1, 10).reshape(-1, 1)
    distances = torch.cdist(points, points)
    result = from_distances(distances, bandwidth=0.25, n_components=1, theta=1.0)

    kernel = evaluate_kernel(distances, 0.25)
    kernel.fill_diagonal_(0)
    normalized_symmetric, _ = get_W_thetas(
        kernel, get_Q_inv_theta(kernel, theta=1.0)
    )
    expected = leading_eigenvectors(normalized_symmetric, 1)
    assert torch.allclose(result.coordinates.abs(), expected.abs())
