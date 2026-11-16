"""Single-cell velocity example: the pieces that need no data file."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from finsler_embedding.experiments.globcurrent.curl import potential_fit
from finsler_embedding.experiments.lineage_cells.velocity import (
    knn_average,
    push_forward,
    tangential_cosine,
    transition_matrix,
    wind,
    zermelo_distances,
)


def test_zermelo_distances_match_the_closed_form():
    w = np.array([[0.5, 0.0], [0.5, 0.0]])
    along = zermelo_distances(np.array([[0.0, 0.0], [1.0, 0.0]]), w)
    assert along[0, 1].item() == pytest.approx(1 / 1.5, rel=1e-5)  # with the wind
    assert along[1, 0].item() == pytest.approx(1 / 0.5, rel=1e-5)  # against it
    across = zermelo_distances(np.array([[0.0, 0.0], [0.0, 1.0]]), w)
    assert across[0, 1].item() == pytest.approx(1 / np.sqrt(0.75), rel=1e-5)
    assert along.diagonal().abs().max().item() == 0


def test_zermelo_distances_without_wind_are_euclidean():
    x = np.random.default_rng(0).normal(size=(6, 4))
    distances = zermelo_distances(x, np.zeros_like(x))
    assert torch.allclose(distances, torch.cdist(torch.as_tensor(x), torch.as_tensor(x)).float(), atol=1e-5)


def test_wind_is_admissible():
    displacement = np.random.default_rng(0).normal(size=(200, 5)) * np.linspace(0.1, 10, 200)[:, None]
    w = wind(displacement, beta=0.9)
    assert np.linalg.norm(w, axis=1).max() <= 0.9 + 1e-12
    assert np.quantile(np.linalg.norm(displacement, axis=1), 0.5) > 0  # sanity: the field is not degenerate


def test_transition_matrix_is_stochastic_and_push_forward_is_linear():
    cosines = csr_matrix(np.array([[0.0, 0.5, -0.2], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    t = transition_matrix(cosines)
    assert np.allclose(np.asarray(t.sum(axis=1)).ravel(), [1.0, 1.0, 0.0])  # a cell without neighbours has no row
    z = np.random.default_rng(0).normal(size=(3, 2))
    a = np.array([[2.0, 1.0], [0.0, -1.0]])
    assert np.allclose(push_forward(t, z @ a.T), push_forward(t, z) @ a.T)


def test_knn_average_rows_sum_to_one():
    x = np.random.default_rng(0).normal(size=(30, 3))
    for k in (1, 5):
        assert np.allclose(np.asarray(knn_average(x, k).sum(axis=1)).ravel(), 1.0)


def test_tangential_cosine_projects_on_the_local_plane():
    rng = np.random.default_rng(0)
    points = torch.as_tensor(np.c_[rng.normal(size=(100, 2)), 0.001 * rng.normal(size=100)], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0, 1.0])  # tangential part along the first axis, whatever the height
    cosine = tangential_cosine(points, direction, np.tile([1.0, 0.0, 0.0], (100, 1)))
    assert cosine.abs().min() > 0.99


def test_weighted_potential_fit():
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    gradient = np.array([1.0, 2.0, 3.0])  # phi = (0, 1, 3)
    assert potential_fit(pairs, gradient, 3)[1] == pytest.approx(0, abs=1e-12)
    cycle = np.array([1.0, 1.0, -1.0])  # circulation around the triangle
    weights = np.array([1.0, 2.0, 3.0])
    unweighted, weighted = potential_fit(pairs, cycle, 3)[1], potential_fit(pairs, cycle, 3, weights)[1]
    assert 0 < weighted < 1 and weighted != pytest.approx(unweighted, abs=1e-3)  # the weights matter
    assert potential_fit(pairs, cycle, 3, 7 * weights)[1] == pytest.approx(weighted, rel=1e-6)  # but not their scale
