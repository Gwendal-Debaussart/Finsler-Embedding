"""Command-line runner for reproducible DiSBM sweeps."""

import argparse
import itertools
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .disbm import DiSBMConfig
from .embedding import EmbeddingConfig, run_direction
from finsler_embedding.experiments.reproducibility import create_run_directory


def run_experiment(
    model_config: DiSBMConfig,
    embedding_config: EmbeddingConfig,
    p_values: list[float],
    seeds: int,
    output_dir: str | Path,
    methods: Sequence[str] | None = None,
) -> Path:
    """Run a DiSBM sweep and save configuration and metrics.

    Parameters
    ----------
    model_config : DiSBMConfig
        Graph generation configuration.
    embedding_config : EmbeddingConfig
        Embedding and kernel configuration.
    p_values : list of float
        Forward edge probabilities to evaluate.
    seeds : int
        Number of independent graph samples per probability.
    output_dir : str or pathlib.Path
        Parent directory for the timestamped run directory.
    methods : sequence of str or None
        Methods run on the same sampled graphs: ``"spectral"`` (moment-Randers)
        and/or ``"finsler_mds"`` (baseline). Defaults to
        ``[embedding_config.method]``.

    Returns
    -------
    pathlib.Path
        Path to the saved ``results.csv`` file.
    """
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    methods = list(methods) if methods else [embedding_config.method]
    run_dir = create_run_directory(
        output_dir,
        "disbm",
        {
            "model": model_config.__dict__,
            "embedding": embedding_config.__dict__,
            "methods": methods,
        },
    )
    rows: list[dict[str, float | int | str]] = []
    for p, seed, method in itertools.product(p_values, range(seeds), methods):
        # The graph only depends on (p, seed): every method sees the same samples.
        result = run_direction(
            p, model_config, replace(embedding_config, method=method), seed=seed
        )
        row = {
            "method": method,
            "p": p,
            "q": result.q,
            "p_minus_q": p - result.q,
            "seed": seed,
            "n": model_config.n,
            "communities": model_config.communities,
            "r": model_config.r,
            "embedding_dim": embedding_config.embedding_dim,
            "admissible": float(result.admissible.float().mean()),
            "cos_forward_community": float(result.cos_community.mean()),
            "cos_forward_node": float(result.cos_node.median()),
            "c_norm": float(result.c_norm.median()),
        }
        if method == "finsler_mds":
            row["mds_alpha"] = embedding_config.mds_alpha
        if abs(row["p_minus_q"]) > 1e-9:
            sign = 1.0 if row["p_minus_q"] > 0 else -1.0
            row.update(
                flow_cos_community=float((sign * result.cos_community).mean()),
                flow_accuracy_community=float(
                    (sign * result.cos_community > 0).float().mean()
                ),
                flow_cos_node=float((sign * result.cos_node).median()),
                flow_accuracy_node=float(
                    (sign * result.cos_node > 0).float().mean()
                ),
            )
        rows.append(row)
    path = run_dir / "results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    """Parse command-line options and run a DiSBM experiment sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--communities", type=int, default=15)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--method",
        nargs="+",
        choices=["spectral", "finsler_mds"],
        default=["spectral"],
        help="Methods run on the same sampled graphs: the moment-Randers spectral "
        "embedding and/or the Finsler MDS baseline.",
    )
    parser.add_argument(
        "--mds-alpha",
        type=float,
        default=EmbeddingConfig.mds_alpha,
        help="finsler_mds: strength of the canonical drift, in [0, 1).",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[(1 - 0.1 + delta) / 2 for delta in np.arange(-0.8, 0.81, 0.1)],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/disbm"))
    args = parser.parse_args()
    path = run_experiment(
        DiSBMConfig(
            n=args.n,
            communities=args.communities,
            r=args.r,
        ),
        EmbeddingConfig(
            embedding_dim=args.embedding_dim,
            bandwidth=args.bandwidth,
            method=args.method[0],
            mds_alpha=args.mds_alpha,
        ),
        args.p_values,
        args.seeds,
        args.output_dir,
        methods=args.method,
    )
    print(f"Saved DiSBM results to {path}")


if __name__ == "__main__":
    main()
