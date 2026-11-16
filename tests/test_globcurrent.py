import numpy as np
import torch

from finsler_embedding.experiments.globcurrent.curl import edge_asymmetries, potential_fit
from finsler_embedding.experiments.globcurrent.data import ZermeloMetric, tangent_frame, to_sphere


def uniform_metric(east: float, north: float = 0.0, ship_speed: float = 1.0) -> ZermeloMetric:
    lon, lat = np.linspace(10.0, 20.0, 6), np.linspace(-40.0, -30.0, 5)
    u, v = np.full((5, 6), east), np.full((5, 6), north)
    return ZermeloMetric(lon, lat, u, v, ship_speed)


def test_zermelo_metric_is_cheaper_with_the_current():
    metric = uniform_metric(east=0.5)
    lon, lat = torch.tensor([np.deg2rad(15.0)], dtype=torch.float32), torch.tensor([np.deg2rad(-35.0)], dtype=torch.float32)
    x = to_sphere(lon, lat)
    east = tangent_frame(lon, lat)[:, :, 0]  # unit vector towards the east, (1, 3)
    length = 10.0
    assert torch.allclose(metric(x, length * east), torch.tensor([length / 1.5]), rtol=1e-4)
    assert torch.allclose(metric(x, -length * east), torch.tensor([length / 0.5]), rtol=1e-4)
    # Across the current: sqrt(lambda) |v| / lambda with lambda = 1 - |w|^2
    north = tangent_frame(lon, lat)[:, :, 1]
    assert torch.allclose(metric(x, length * north), torch.tensor([length / (0.75**0.5)]), rtol=1e-4)


def test_zermelo_metric_without_current_is_the_euclidean_length():
    metric = uniform_metric(east=0.0)
    lon, lat = torch.tensor([np.deg2rad(12.0)], dtype=torch.float32), torch.tensor([np.deg2rad(-33.0)], dtype=torch.float32)
    v = torch.tensor([[1.0, 2.0, 2.0]])
    assert torch.allclose(metric(to_sphere(lon, lat), v), torch.tensor([3.0]), rtol=1e-5)


def test_potential_fit_separates_gradients_from_circulation():
    # Gradient of a potential: the edge asymmetries are differences, nothing is left over
    potential = np.array([0.0, 1.0, 3.0, 2.0])
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]])
    gradient = potential[pairs[:, 1]] - potential[pairs[:, 0]]
    _, curl_fraction = potential_fit(pairs, gradient, 4)
    assert curl_fraction < 1e-8
    # A circulation around a loop is orthogonal to every potential
    cycle = np.array([[0, 1], [1, 2], [0, 2]])
    _, curl_fraction = potential_fit(cycle, np.array([1.0, 1.0, -1.0]), 3)
    assert abs(curl_fraction - 1.0) < 1e-8


def test_edge_asymmetries_keeps_the_pairs_with_both_directions():
    edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1], [0, 2]])  # 0 -> 2 has no reverse edge
    lengths = torch.tensor([1.0, 3.0, 2.0, 2.0, 5.0])
    pairs, asymmetry = edge_asymmetries(edges, lengths, 3)
    assert pairs.tolist() == [[0, 1], [1, 2]]
    assert np.allclose(asymmetry, [-1.0, 0.0])  # (d_01 - d_10) / 2, (d_12 - d_21) / 2
