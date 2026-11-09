# Swiss-roll experiment

This directory contains the Swiss-roll dataset notebook and the quantitative
operator evaluation used for the paper results.

- [`runner.py`](./runner.py) runs the seeded evaluation sweep.
- [`table.py`](./table.py) generates the paper-style summary table from saved
  evaluation results.
- [`swissroll_dataset.ipynb`](./swissroll_dataset.ipynb) contains the
  interactive dataset and vector-field visualizations.

Run the quantitative sweep from the repository root with:

```bash
uv run python -m finsler_embedding.experiments.swiss_roll.runner
```

With no arguments, this directly reproduces the paper sweep:

- beta values: `0.1 0.3 0.5 0.7 0.9`
- sample sizes: `1000 2000 4000`
- seeds: `0 1 2`
- radius graph with radius `5.0`
- Isomap embedding

Use `--help` to override these values. For a small smoke test, run:

```bash
uv run python -m finsler_embedding.experiments.swiss_roll.runner \
  --betas 0.9 --n-list 100 --seeds 0 \
  --output results/swiss_roll-smoke
```

The runner saves both artifacts in the output directory:

- `eval_operators_swiss_roll.csv`: per-run metrics
- `swiss_roll_table.tex`: paper-style median summary table

The table can also be regenerated from saved results with:

```bash
uv run python -m finsler_embedding.experiments.swiss_roll.table
```
