"""Run reproducible quantitative Swiss-roll evaluations."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finsler_embedding.eval_operators import EvalOperators, main as main_eval
from finsler_embedding.experiments.swiss_roll.table import generate_table

PAPER_BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]
PAPER_SAMPLE_SIZES = [1000, 2000, 4000]
PAPER_SEEDS = [0, 1, 2]
PAPER_RADIUS = 5.0
PAPER_EMBEDDING = "isomap"
PAPER_OUTPUT = Path("results/evaluation")


def parse_args() -> argparse.Namespace:
    """Parse Swiss-roll sweep configuration from the command line.

    Returns
    -------
    argparse.Namespace
        Requested beta values, sample sizes, seeds, graph parameters, and
        output path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=PAPER_BETAS,
        help="List of beta values to evaluate.",
    )
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=PAPER_SAMPLE_SIZES,
        help="List of sample sizes to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=PAPER_SEEDS,
        help="List of random seeds to use for evaluation.",
    )
    parser.add_argument("--radius", type=float, default=PAPER_RADIUS)
    parser.add_argument("--embedding", choices=["chart", "isomap"], default=PAPER_EMBEDDING)
    parser.add_argument("--output", type=Path, default=PAPER_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run the requested Swiss-roll evaluation sweep."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df_list = []
    for beta, n_samples, seed in itertools.product(
        args.betas, args.n_list, args.seeds
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        config: EvalOperators = EvalOperators(
            dataset="swiss_roll",
            n=n_samples,
            m=2,
            beta=beta,
            graph_type="radius",
            embedding_type=args.embedding,
            k=15,
            eps_scale=1.0,
            eps0=1.0,
            cut=0.0,
            radius=args.radius,
        )
        df = main_eval(config)
        df["seed"] = seed
        df_list.append(df)
        pd.concat(df_list, ignore_index=True).to_csv(
            args.output / "eval_operators_swiss_roll.csv", index=False
        )

    df_tot = pd.concat(df_list, ignore_index=True)
    df_tot.to_csv(args.output / "eval_operators_swiss_roll.csv", index=False)
    table_path = generate_table(
        df_tot,
        args.betas,
        args.n_list,
        args.output / "swiss_roll_table.tex",
        args.embedding,
    )
    print(f"Saved Swiss-roll table to {table_path}")


if __name__ == "__main__":
    main()
