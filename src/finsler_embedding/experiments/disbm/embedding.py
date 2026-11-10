"""Embedding and drift-recovery experiments on sampled DiSBM graphs."""

from dataclasses import dataclass

import numpy as np
import torch

from finsler_embedding.core.inputs import from_graph
from finsler_embedding.experiments.reproducibility import seed_everything

from .disbm import DiSBMConfig, graph_info
from .finsler_mds import canonical_c_norm, finsler_mds_embedding


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for DiSBM spectral embedding and drift recovery.

    Parameters
    ----------
    embedding_dim : int, default=2
        Number of spectral embedding coordinates.
    bandwidth : float, default=1.0
        Gaussian kernel bandwidth.
    method : {"spectral", "finsler_mds"}, default="spectral"
        ``"spectral"`` is the moment-Randers embedding. ``"finsler_mds"`` is the
        Finsler MDS baseline: the directed hop distances of the graph embedded in
        a canonical Randers space of dimension ``embedding_dim + 1``, whose last
        axis carries the constant drift.
    mds_alpha : float, default=0.5
        Finsler MDS only: strength of the canonical drift, in ``[0, 1)``.
    """

    embedding_dim: int = 2
    bandwidth: float = 1.0
    method: str = "spectral"
    mds_alpha: float = 0.5


@dataclass
class DirectionResult:
    """Directional DiSBM metrics for one sampled graph."""

    q: float
    cos_community: torch.Tensor
    cos_node: torch.Tensor
    c_norm: torch.Tensor
    admissible: torch.Tensor


def run_direction(
    p: float,
    model_config: DiSBMConfig,
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
    *,
    seed: int | None = None,
) -> DirectionResult:
    """Recover the drift direction from one sampled DiSBM graph.

    Parameters
    ----------
    p : float
        Forward inter-community edge probability.
    model_config : DiSBMConfig
        Graph generation configuration.
    embedding_config : EmbeddingConfig, default=EmbeddingConfig()
        Method (moment-Randers spectral embedding or Finsler MDS baseline) and
        its configuration.
    seed : int or None, optional
        Seed applied before graph sampling, and used by Finsler MDS for its
        initial configuration.

    Returns
    -------
    DirectionResult
        Community and node directional cosines, drift strength, and
        admissibility mask.
    """
    if seed is not None:
        seed_everything(seed)
    q = 1 - p - model_config.r
    _, _, labels, adjacency = graph_info(
        p=p,
        q=q,
        r=model_config.r,
        K=model_config.communities,
        N=model_config.n,
    )
    if embedding_config.method == "spectral":
        result = from_graph(
            adjacency,
            embedding_config.bandwidth,
            theta=1.0,
            n_components=embedding_config.embedding_dim,
        )
        coordinates, drift = result.coordinates, result.drift
        c_norm, admissible = result.drift_norm_squared.sqrt(), result.admissible
    elif embedding_config.method == "finsler_mds":
        fit = finsler_mds_embedding(
            adjacency, embedding_config.embedding_dim, embedding_config.mds_alpha, seed=seed
        )
        coordinates = fit.coordinates
        dim = coordinates.shape[1]

        drift = torch.zeros_like(coordinates)
        drift[:, -1] = -1.0
        c_norm = torch.full((coordinates.shape[0],), canonical_c_norm(fit.alpha, dim))
        admissible = c_norm**2 < 1.0 / (dim + 3)
    else:
        raise ValueError(
            f"method must be 'spectral' or 'finsler_mds', got {embedding_config.method!r}"
        )
    means = torch.stack(
        [coordinates[labels == k].mean(0) for k in range(model_config.communities)]
    )
    forward = means[(torch.arange(model_config.communities) + 1) % model_config.communities] - means[
        (torch.arange(model_config.communities) - 1) % model_config.communities
    ]
    community_drift = torch.stack(
        [drift[labels == k].mean(0) for k in range(model_config.communities)]
    )
    cosine = torch.nn.functional.cosine_similarity
    return DirectionResult(
        q=q,
        cos_community=cosine(community_drift, forward, dim=1),
        cos_node=cosine(drift, forward[labels], dim=1),
        c_norm=c_norm,
        admissible=admissible,
    )


def run_sweep(
    p_minus_q: np.ndarray,
    model_config: DiSBMConfig,
    embedding_config: EmbeddingConfig = EmbeddingConfig(),
    seeds: int = 5,
) -> dict[str, np.ndarray]:
    """Sweep flow asymmetry and aggregate node-level quantiles.

    Parameters
    ----------
    p_minus_q : numpy.ndarray (P,)
        Forward-minus-backward probability values.
    model_config : DiSBMConfig
        Graph generation configuration.
    embedding_config : EmbeddingConfig, default=EmbeddingConfig()
        Spectral embedding and kernel configuration.
    seeds : int, default=5
        Number of independent graph samples per asymmetry value.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``cos`` and ``c_norm`` arrays with shape ``(P, 3)`` containing
        10th-percentile, median, and 90th-percentile values.
    """
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    stats = {"cos": [], "c_norm": []}
    for delta in p_minus_q:
        p = (1 - model_config.r + float(delta)) / 2
        node_cos, node_norm = [], []
        for seed in range(seeds):
            result = run_direction(p, model_config, embedding_config, seed=seed)
            node_cos.append(result.cos_node.numpy())
            node_norm.append(result.c_norm.numpy())
        stats["cos"].append(np.quantile(np.concatenate(node_cos), [0.1, 0.5, 0.9]))
        stats["c_norm"].append(np.quantile(np.concatenate(node_norm), [0.1, 0.5, 0.9]))
    return {key: np.asarray(value) for key, value in stats.items()}
