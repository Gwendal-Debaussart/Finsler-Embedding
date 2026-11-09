"""
Table of the quantitative evaluation on the Swiss roll (Table 1 of the paper), as a function of the
strength ||b||_{A^-1} of the Randers drift (rows) and of the number of samples N (columns): median
over seeds of the per-run values computed by evaluate.py, on a radius graph that does not truncate
the kernel,
    - relative error on ||hat c||^2_{hat g_BL} (median over all the samples),
    - cosine between V_i and its limit c_1 J_i c(X_i) (median over all the samples),
    - fraction of samples declared admissible, ||hat c||^2_{hat g_BL} < 1 / (m + 3).

Missing runs are computed with evaluate.py, existing ones are read from <out>.

    python -m finsler_embedding.experiments.swiss_roll.table [--betas 0.1 0.3 0.5 0.7 0.9] [--n-list 1000 2000 4000]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

TABLES = {  # table -> [(csv column, decimals)]
    "main": [("relerr_c_norm_all", 2), ("cos_V_all", 2), ("admissible", 2)],
}


def generate_table(
    df: pd.DataFrame,
    betas: list[float],
    sample_sizes: list[int],
    output: Path,
    embedding: str = "isomap",
) -> Path:
    """Aggregate evaluation results and save the paper table as LaTeX.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-run evaluation results.
    betas : list of float
        Drift strengths represented by the table rows.
    sample_sizes : list of int
        Sample sizes represented by the table columns.
    output : pathlib.Path
        Destination for the generated LaTeX table.
    embedding : str, default="isomap"
        Embedding results to include.

    Returns
    -------
    pathlib.Path
        Path to the generated LaTeX table.
    """
    required = {"embedding_type", "n", "beta"}
    missing = required - set(df.columns)
    missing.update(column for column, _ in TABLES["main"] if column not in df.columns)
    if missing:
        raise ValueError(f"evaluation results are missing columns: {sorted(missing)}")
    filtered = df[
        (df["embedding_type"] == embedding)
        & df["n"].isin(sample_sizes)
    ].rename(columns={"n": "N"})
    lines: list[str] = []
    for name, columns in TABLES.items():
        grouped = filtered.groupby(["beta", "N"])[[column for column, _ in columns]]
        median = grouped.median()
        lines.append(f"% {name} table")
        for beta in betas:
            cells = [f"${beta:g}$"]
            for column, decimals in columns:
                cells.extend(
                    f"${median.loc[(beta, n), column]:.{decimals}f}$"
                    for n in sample_sizes
                )
            lines.append("    " + " & ".join(cells) + r" \\")
    output.write_text("\n".join(lines) + "\n")
    return output


def load(beta: float, args) -> pd.DataFrame:
    path = args.out / f"swiss_roll_radius{args.radius}_beta{beta}.csv"
    if not path.exists():
        subprocess.run(
            [sys.executable, "-m", "finsler_embedding.evaluate", "swiss_roll", "--beta", str(beta),
             "--graph", "radius", "--radius", str(args.radius), "--seeds", str(args.seeds),
             "--n-list", *map(str, args.n_list), "--embedding", args.embedding, "--device", args.device,
             "--out", str(args.out)],
            check=True,
        )
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--n-list", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--radius", type=float, default=5.0, help="Euclidean radius of the graph.")
    parser.add_argument("--embedding", default="isomap", choices=["chart", "isomap"])
    parser.add_argument("--device", default="cpu", help="Device for the Finsler distances of missing runs.")
    parser.add_argument("--out", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()

    df = pd.concat([load(beta, args) for beta in args.betas])
    output = args.out / "swiss_roll_table.tex"
    generate_table(df, args.betas, args.n_list, output, args.embedding)
    print(output.read_text())


if __name__ == "__main__":
    main()
