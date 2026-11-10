"""
Vector field estimator of Perrault-Joncas and Meila (NIPS 2011).
"""

import numpy as np


def symmetric_operator(kernel: np.ndarray) -> np.ndarray:
    """``H_ss^(1)``: the density-normalized (alpha = 1) random walk of the symmetrized kernel (steps 1 to 5)."""
    S = 0.5 * (kernel + kernel.T)
    q = S.sum(axis=1)
    V = S / np.outer(q, q)
    return V / V.sum(axis=1, keepdims=True)


def asymmetric_operator(kernel: np.ndarray) -> np.ndarray:
    """``H_aa^(1)``: the same normalization of the directed kernel, with the out-degrees.

    The normalization divides by the out-degrees, so it is undefined at a node without out-edge, and at the nodes that point to it. The corresponding rows of the operator are NaN.
    """
    p = kernel.sum(axis=1)
    sink = p == 0
    undefined = sink | (kernel[:, sink] > 0).any(axis=1)
    p = np.where(sink, 1.0, p)
    T = kernel / np.outer(p, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = T / T.sum(axis=1, keepdims=True)
    H[undefined] = np.nan
    return H


def spectral_coordinates(kernel: np.ndarray, n_components: int) -> np.ndarray:
    """Coordinates of the embedding of the paper: eigenvectors 2 to ``n_components + 1`` of ``H_ss^(1)`` (step 6)."""
    S = 0.5 * (kernel + kernel.T)
    q = S.sum(axis=1)
    V = S / np.outer(q, q)
    d = V.sum(axis=1)
    # H_ss = D^{-1} V is similar to the symmetric matrix D^{-1/2} V D^{-1/2}
    values, vectors = np.linalg.eigh(V / np.sqrt(np.outer(d, d)))
    order = np.argsort(-values)[1 : n_components + 1]
    return vectors[:, order] / np.sqrt(d)[:, None]


def vector_field(kernel: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Vector field ``R`` (N, l) in the coordinates of an embedding, ``R = (H_aa - H_ss) Y / 2`` (step 13).

    Parameters
    ----------
    kernel : (N, N) directed affinity matrix ``W``. The rows of the result are NaN where the field is undefined
        (see :func:`asymmetric_operator`).
    coordinates : (N, l) coordinates ``Y`` of the nodes. The paper uses the eigenvectors of ``H_ss`` (see
        :func:`spectral_coordinates`), but the formula holds for any embedding of the manifold.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    difference = asymmetric_operator(kernel) - symmetric_operator(kernel)
    return 0.5 * difference @ np.asarray(coordinates, dtype=np.float64)
