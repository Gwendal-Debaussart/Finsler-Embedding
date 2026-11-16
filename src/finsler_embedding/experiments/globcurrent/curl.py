"""How much of the asymmetry of a directed graph is the gradient of a potential.

Finsler MDS (Dages et al., 2025) fits the canonical Randers metric ``|v| + alpha v_last``: the asymmetry of its
distances is ``d(x, y) - d(y, x) = 2 alpha (y_last - x_last)``, a difference of potentials, so it sums to zero
around every closed loop. Its drift, seen in the tangent plane of the embedded data, is the gradient of the
height of the surface. A current with vorticity has a nonzero circulation around a loop: whatever the
optimiser, no embedding of this family can reproduce it.

Two numbers measure this on the data, without any embedding:

* ``curl_fraction``: the share of the energy of the edge asymmetries ``a_ij = (d_ij - d_ji) / 2`` that no
  potential explains, ``min_phi sum (a_ij - (phi_j - phi_i))^2 / sum a_ij^2``;
* ``gradient_cosine``: the cosine between the true drift and the drift of the best potential, ``-grad phi``,
  the best direction a drift that is the gradient of a height can have (a ceiling for Finsler MDS).
"""

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors


def edge_asymmetries(edges: torch.Tensor, lengths: torch.Tensor, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs ``(i, j)``, ``i < j``, whose two directions are edges (P, 2), and their asymmetry ``(d_ij - d_ji) / 2`` (P,)."""
    i, j = edges[:, 0].numpy().astype(np.int64), edges[:, 1].numpy().astype(np.int64)
    d = lengths.numpy().astype(np.float64)
    key = i * n + j
    order = np.argsort(key)
    key, d = key[order], d[order]
    forward = i < j
    pairs = np.stack([i[forward], j[forward]], axis=1)
    back = pairs[:, 1] * n + pairs[:, 0]
    position = np.minimum(np.searchsorted(key, back), len(key) - 1)
    has_back = key[position] == back
    pairs = pairs[has_back]
    forward_length = d[np.searchsorted(key, pairs[:, 0] * n + pairs[:, 1])]
    return pairs, 0.5 * (forward_length - d[position[has_back]])


def potential_fit(pairs: np.ndarray, asymmetry: np.ndarray, n: int,
                  weights: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Potential ``phi`` (N,) with ``phi_j - phi_i ~ a_ij`` in least squares, and the fraction of ``sum a^2`` left over.

    ``weights`` (P,) weigh the squared errors (``w_ij^2 (a_ij - (phi_j - phi_i))^2``) and the energy of the fraction.
    With the inverse of the symmetric length of the edges the fraction does not depend on the sampling density.
    """
    weights = np.ones(len(pairs)) if weights is None else weights
    rows = np.repeat(np.arange(len(pairs)), 2)
    cols = pairs.ravel()
    values = np.tile([-1.0, 1.0], len(pairs)) * np.repeat(weights, 2)
    incidence = coo_matrix((values, (rows, cols)), shape=(len(pairs), n)).tocsr()
    target = asymmetry * weights
    phi = lsqr(incidence, target, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    residual = target - incidence @ phi
    return phi, float((residual**2).sum() / (target**2).sum())


def potential_drift(points: torch.Tensor, frame: torch.Tensor, phi: np.ndarray, k: int = 15,
                    neighbours: torch.Tensor | None = None) -> torch.Tensor:
    """Drift ``-grad phi`` (N, 2), in the tangent coordinates of ``frame`` (N, D, 2), by local least squares on the neighbours.

    The neighbours (N, k) are the ``k`` nearest ones in ``points`` unless given, for instance the neighbours in the graph
    of the data when ``points`` is a low-dimensional embedding whose own neighbourhoods are folded.
    """
    if neighbours is None:
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(points.numpy()).kneighbors(return_distance=False)[:, 1:]
    idx = torch.as_tensor(neighbours)
    dx = torch.einsum("nkd,ndm->nkm", points[idx] - points[:, None, :], frame)
    phi = torch.as_tensor(phi, dtype=points.dtype)
    dphi = (phi[idx] - phi[:, None])[..., None]
    return -torch.linalg.lstsq(dx, dphi).solution[..., 0]


def curl_diagnostics(points: torch.Tensor, frame: torch.Tensor, edges: torch.Tensor, lengths: torch.Tensor,
                     true_drift: torch.Tensor) -> dict[str, float]:
    """``curl_fraction`` of the edge asymmetries, and the cosine of the best potential drift with ``true_drift`` (N, 2)."""
    n = points.shape[0]
    pairs, asymmetry = edge_asymmetries(edges, lengths, n)
    phi, curl_fraction = potential_fit(pairs, asymmetry, n)
    drift = potential_drift(points, frame, phi)
    cosine = torch.nn.functional.cosine_similarity(drift, true_drift, dim=1)
    return {
        "curl_fraction": curl_fraction,
        "gradient_cosine": float(cosine.median()),
        "gradient_cosine_q10": float(cosine.quantile(0.1)),
    }
