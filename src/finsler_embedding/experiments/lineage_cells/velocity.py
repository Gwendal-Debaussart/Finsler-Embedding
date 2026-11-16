"""RNA velocity as a wind on the cells: directed distances for a single-cell dataset.

A single-cell dataset with RNA velocity (scVelo) gives, for every cell, the direction in which its expression
is changing, but no directed distance between cells. As for the ocean currents of ``globcurrent``, the
distances are *modelled* from the real vector field, by Zermelo navigation: a cell travels at unit speed in the
PCA space against a wind ``w`` proportional to its velocity, so that going along the velocity is cheaper than
going against it. The travel time of a segment is a Randers length (closed form in :func:`zermelo_distances`),
whose drift is ``w``.

What is real is the velocity field, with its structure (branching, cell cycle). What is modelled is that the
distances are the travel times in it. There is no ground truth beyond the velocity itself, so the methods are
compared on how well their drift follows it (:func:`push_forward`), and on how much of the asymmetry a potential
can explain (``finsler_embedding.experiments.globcurrent.curl``).

Any AnnData with ``obsm["X_pca"]``, ``uns["velocity_graph"]`` (and ``velocity_graph_neg``) and a cluster column
works: :func:`preprocess_with_scvelo` builds one from any dataset of ``scvelo.datasets``.
"""

from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

DEFAULT_DIR = Path(__file__).resolve().parents[4] / "data" / "cells"  # <repository>/data/cells, git-ignored


def preprocess_with_scvelo(dataset: str, out_path: str | Path, n_pcs: int = 30, n_neighbors: int = 30) -> None:
    """Runs the scVelo pipeline once on ``scvelo.datasets.<dataset>()`` and writes the AnnData (needs scvelo, scanpy).

    Same pipeline as the scVelo tutorial: normalisation, PCA, moments, deterministic velocity, velocity graph.
    """
    import warnings

    import scanpy as sc
    import scvelo as scv

    warnings.filterwarnings("ignore")
    adata = getattr(scv.datasets, dataset)()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    scv.tl.velocity(adata, mode="deterministic")
    scv.tl.velocity_graph(adata)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)


def largest_component(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of the cells in the largest connected component of the symmetrized k-NN graph of ``x`` (N, D)."""
    graph = kneighbors_graph(x, n_neighbors=k, mode="connectivity")
    _, labels = connected_components(graph, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def load_velocity_data(path: str | Path, cluster_key: str = "clusters", keep_clusters: list[str] | None = None,
                       n: int | None = None, seed: int = 0, k: int = 10, obs_columns: tuple[str, ...] = ()) -> dict:
    """Reads the PCA coordinates, the velocity cosines and the cell types of a preprocessed AnnData.

    Only the cells of ``keep_clusters`` are kept if given, then ``n`` of them are drawn at random if given. The cells
    outside the largest connected component of the k-NN graph are dropped (the directed shortest paths of Finsler
    MDS need a connected graph).

    Returns
    -------
    dict with ``x`` (N, D) PCA coordinates, ``umap`` (N, 2) or None, ``cosines`` (N, N) sparse matrix of the cosines
    between the velocity of a cell and its displacement to another, ``labels`` (N,) cell types, ``index`` (N,)
    positions of the cells in the file, and ``obs`` the requested columns (N,) of the cell annotations.
    """
    import anndata

    adata = anndata.read_h5ad(path, backed="r")
    try:
        x = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        umap = np.asarray(adata.obsm["X_umap"], dtype=np.float64) if "X_umap" in adata.obsm else None
        cosines = adata.uns["velocity_graph"].tocsr()
        if "velocity_graph_neg" in adata.uns:
            cosines = cosines + adata.uns["velocity_graph_neg"].tocsr()
        labels = adata.obs[cluster_key].astype(str).to_numpy()
        obs = {name: np.asarray(adata.obs[name]) for name in obs_columns}
    finally:
        adata.file.close()
    index = np.arange(x.shape[0])
    if keep_clusters is not None:
        index = index[np.isin(labels, keep_clusters)]
    if n is not None and n < len(index):
        index = np.sort(np.random.default_rng(seed).choice(index, n, replace=False))
    index = index[largest_component(x[index], k)]
    return {
        "x": x[index], "umap": None if umap is None else umap[index], "labels": labels[index],
        "cosines": cosines[index][:, index].tocsr(), "index": index, "obs": {k_: v[index] for k_, v in obs.items()},
    }


def transition_matrix(cosines: csr_matrix, scale: float = 10.0) -> csr_matrix:
    """Row-stochastic matrix ``T_ij`` proportional to ``exp(scale * cos_ij)`` on the stored entries (N, N).

    The softmax of the cosines of scVelo's velocity graph, in the spirit of its transition matrix.
    """
    t = cosines.tocsr().copy()
    t.data = np.exp(scale * t.data)
    total = np.asarray(t.sum(axis=1)).ravel()
    total[total == 0] = 1.0
    return (diags(1.0 / total) @ t).tocsr()


def push_forward(t: csr_matrix, z: np.ndarray, smoother: csr_matrix | None = None) -> np.ndarray:
    """Expected displacement ``sum_j T_ij (z_j - z_i)`` of every cell in any coordinates ``z`` (N, d), then smoothed.

    This is how scVelo draws the velocity in an embedding. It is linear in ``z``, so the velocity pushed into
    the coordinates of an embedding is the image of the velocity in the PCA space.
    """
    displacement = t @ z - z * np.asarray(t.sum(axis=1))
    return displacement if smoother is None else smoother @ displacement


def knn_average(x: np.ndarray, k: int) -> csr_matrix:
    """Averaging operator (N, N) over a cell and its ``k - 1`` nearest neighbours; the identity if ``k <= 1``."""
    n = x.shape[0]
    if k <= 1:
        return diags(np.ones(n)).tocsr()
    return (kneighbors_graph(x, n_neighbors=k, mode="connectivity", include_self=True) / k).tocsr()


def wind(displacement: np.ndarray, beta: float, quantile: float = 0.99) -> np.ndarray:
    """Wind (N, d) of norm at most ``beta < 1``: the displacements, scaled so that the ``quantile`` of their norms is ``beta``."""
    norms = np.linalg.norm(displacement, axis=1)
    w = beta * displacement / np.quantile(norms, quantile)
    w_norm = np.linalg.norm(w, axis=1)
    over = w_norm > beta
    w[over] *= (beta / w_norm[over])[:, None]
    return w


def zermelo_distances(x: np.ndarray, w: np.ndarray) -> torch.Tensor:
    """Directed distances (N, N), ``D[i, j]`` from cell ``i`` to cell ``j``, in the wind ``w`` (N, d) at unit ship speed.

    The length of the straight segment is the Randers norm of ``x_j - x_i`` in the wind averaged on the segment,
    ``w_ij = (w_i + w_j) / 2``: ``F(v) = (sqrt(lam |v|^2 + <w_ij, v>^2) - <w_ij, v>) / lam``, ``lam = 1 - |w_ij|^2``.
    Along the wind it is ``|v| / (1 + |w|)``, against it ``|v| / (1 - |w|)``.
    """
    x, w = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(w, dtype=torch.float64)
    cross = w @ x.T  # <w_i, x_j>
    own = (w * x).sum(1)  # <w_i, x_i>
    along = 0.5 * (cross - own[:, None] + own[None, :] - cross.T)  # <w_ij, x_j - x_i>
    w2 = 0.25 * (torch.einsum("id,id->i", w, w)[:, None] + torch.einsum("jd,jd->j", w, w)[None, :] + 2 * (w @ w.T))
    lam = (1.0 - w2).clamp_min(1e-12)
    d2 = torch.cdist(x, x).square()
    lengths = ((lam * d2 + along.square()).sqrt() - along) / lam
    lengths.fill_diagonal_(0)
    return lengths.to(torch.float32)


def tangential_cosine(points: torch.Tensor, direction: torch.Tensor, vectors: np.ndarray | torch.Tensor,
                      k: int = 15, dim: int = 2) -> torch.Tensor:
    """Cosine (N,) between ``vectors`` (N, d) and a fixed ``direction`` (d,) projected on the tangent plane of the cloud.

    The tangent plane at a point is spanned by the ``dim`` leading principal directions of its ``k`` neighbours.
    Finsler MDS has a constant drift along the last axis of its embedding: its drift on the embedded data is the
    tangential part of that axis.
    """
    idx = torch.from_numpy(NearestNeighbors(n_neighbors=k + 1).fit(points.numpy()).kneighbors(return_distance=False)[:, 1:])
    neighbours = points[idx]
    basis = torch.linalg.svd(neighbours - neighbours.mean(1, keepdim=True), full_matrices=False)[2][:, :dim]  # (N, dim, d)
    tangential = torch.einsum("nmd,nm->nd", basis, torch.einsum("nmd,d->nm", basis, direction.to(points.dtype)))
    return torch.nn.functional.cosine_similarity(tangential, torch.as_tensor(vectors, dtype=points.dtype), dim=1)
