# DiSBM experiment

This directory contains the reproducible directed stochastic block model
experiment used to measure recovery of the direction of the asymmetric flow.

The implementation has two independent layers:

- `disbm.py` contains only the cyclic DiSBM model configuration and graph
  generation functions.
- `embedding.py` contains spectral embedding, Randers recovery, and sweep
  analysis on a sampled graph.

This separation makes it possible to generate and inspect model instances
without running an embedding experiment, and to reuse the embedding analysis
with independently generated DiSBM graphs.

Communities form a cycle:

- `p`: edge probability from community `k` to `k + 1`
- `q`: edge probability from community `k` to `k - 1`
- `r`: within-community edge probability
- `p + q + r = 1`

The experiment samples a graph, constructs the directed Gaussian kernel,
computes the leading spectral embedding of the normalized symmetric operator,
and reconstructs the antisymmetric drift. Results report signed cosine with
the forward cycle direction and the estimated drift norm.

## Reproduce a sweep

From the repository root:

```bash
uv run python -m finsler_embedding.experiments.disbm.runner \
  --n 1000 \
  --communities 15 \
  --r 0.1 \
  --seeds 5 \
  --output-dir results/disbm
```

Each run creates a timestamped directory containing:

- `config.json`
- `metadata.json`
- `results.csv`

For the paper-style directional plots:

```bash
uv run python -m finsler_embedding.experiments.disbm.plot_direction \
  --N 1000 --K 15 --r 0.1 --seeds 5 \
  --out results/disbm/direction
```

The migrated notebook is
[`disbm_dataset.ipynb`](./disbm_dataset.ipynb).
