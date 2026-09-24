"""
Table of the quantitative evaluation on the Swiss roll (Table 1 of the paper), as a function of the
strength ||b||_{A^-1} of the Randers drift (rows) and of the number of samples N (columns): median
over seeds of the per-run values computed by evaluate.py, on a radius graph that does not truncate
the kernel,
    - relative error on ||hat c||^2_{hat g_BL} (median over all the samples),
    - cosine between V_i and its limit c_1 J_i c(X_i) (median over all the samples),
    - fraction of samples declared admissible, ||hat c||^2_{hat g_BL} < 1 / (m + 3).

Missing runs are computed with evaluate.py, existing ones are read from <out>.

    python -m finsler_embedding.swiss_roll_table [--betas 0.1 0.3 0.5 0.7 0.9] [--n-list 1000 2000 4000]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

TABLES = {  # table -> [(csv column, decimals)]
    "main": [("relerr_c_norm_all", 2), ("cos_V_all", 2), ("admissible", 2)],
}


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
    df = df[(df.embedding == args.embedding) & df.N.isin(args.n_list)]
    for name, columns in TABLES.items():
        g = df.groupby(["beta", "N"])[[c for c, _ in columns]]
        median, iqr = g.median(), g.quantile(0.75) - g.quantile(0.25)
        print(f"\n% {name} table, max interquartile range over seeds: {iqr.max().round(3).to_dict()}")
        for beta in args.betas:
            cells = [f"${beta:g}$"]
            for c, d in columns:
                cells += [f"${median.loc[(beta, N), c]:.{d}f}$" for N in args.n_list]
            print("    " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
