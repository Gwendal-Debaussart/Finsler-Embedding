# Finsler Embedding

This repository contains the code accompanying the paper on learning
low-dimensional Finsler/Randers structure from directed geometric data. The
method turns directed pairwise distances or directed graph edge lengths into
a kernel operator, extracts an embedding from its symmetric part, and
recovers the directional drift from its antisymmetric part.

The implementation is organized as a reusable Python API and as reproducible
experiment runners. The main pipeline is:

1. evaluate a directed Gaussian or exponential kernel;
2. normalize the kernel and split it into symmetric and antisymmetric
   operators;
3. compute a spectral or Isomap embedding;
4. estimate the carré du champ and directed drift; and
5. reconstruct moment-Randers quantities and admissibility diagnostics.

## Installation

The project requires Python 3.13 or newer and uses
[uv](https://docs.astral.sh/uv/) for environment and dependency management.
The `geodesic-toolbox` dependency is installed from a pinned commit of its
GitHub repository. Install the project from the repository root:

```bash
cd Finsler-Embedding
uv sync
```

This creates or updates the project environment in `.venv`. Run commands
inside that environment with `uv run`.

## Reproducing the paper experiments

All experiment commands are run from the repository root. The default
arguments of the runners are the paper configurations; use `--help` to
inspect or override them.

### Swiss roll

The default sweep evaluates the five values
`beta = 0.1, 0.3, 0.5, 0.7, 0.9`, sample sizes `1000, 2000, 4000`, and
seeds `0, 1, 2` using a radius graph with radius `5.0` and Isomap:

```bash
uv run python -m finsler_embedding.experiments.swiss_roll.runner
```

Results are written to `results/evaluation/`, including
`eval_operators_swiss_roll.csv` and the paper-style
`swiss_roll_table.tex`. A smaller smoke run is:

```bash
uv run python -m finsler_embedding.experiments.swiss_roll.runner \
  --betas 0.9 --n-list 100 --seeds 0 \
  --output results/swiss_roll-smoke
```

### Matsumoto surface

Run the Matsumoto surface evaluation with:

```bash
uv run python -m finsler_embedding.experiments.matsumoto.runner
```

The default output is `results/evaluation/matsumoto.csv`. A small run using
the chart embedding is:

```bash
uv run python -m finsler_embedding.experiments.matsumoto.runner \
  --n 100 --seeds 1 --embedding chart
```

Use `--out` to choose a different output directory.

### Directed stochastic block model (DiSBM)

The DiSBM sweep measures recovery of the direction of the asymmetric flow:

```bash
uv run python -m finsler_embedding.experiments.disbm.runner \
  --n 1000 \
  --communities 15 \
  --r 0.1 \
  --seeds 5 \
  --output-dir results/disbm
```

Each run creates a timestamped directory containing `config.json`,
`metadata.json`, and `results.csv`. To generate the directional plots:

```bash
uv run python -m finsler_embedding.experiments.disbm.plot_direction \
  --N 1000 --K 15 --r 0.1 --seeds 5 \
  --out results/disbm/direction
```

The notebooks in the dataset directories provide interactive visualizations:

- [`swissroll_dataset.ipynb`](src/finsler_embedding/experiments/swiss_roll/swissroll_dataset.ipynb)
- [`matsumoto_dataset.ipynb`](src/finsler_embedding/experiments/matsumoto/matsumoto_dataset.ipynb)
- [`disbm_dataset.ipynb`](src/finsler_embedding/experiments/disbm/disbm_dataset.ipynb)

## Using the method

The public entry points are exported from `finsler_embedding`. Inputs must be
finite, nonnegative `torch.Tensor` objects. For a directed distance matrix,
`distances[i, j]` is the distance from sample `i` to sample `j`:

```python
import torch

from finsler_embedding import from_distances

result = from_distances(
    directed_distances, # (N, N) Tensor
    bandwidth=0.25,
    kernel="gaussian",
    theta=1.0,
    n_components=2,
    embedding="spectral",
)

coordinates = result.coordinates
drift = result.drift
admissible = result.admissible
```

For point-cloud coordinates, pass the coordinates separately when using
Isomap or when estimating a Jacobian. A directed distance matrix is still
required; symmetric distances computed from the point cloud do not contain
directional information:

```python
from finsler_embedding import from_points

result = from_points(
    points, # (N, d)
    distances=directed_distances, # (N, N)
    bandwidth=0.25,
    embedding="isomap",
    embedding_params={"n_neighbors": 10},
    jacobian="local_least_squares",
    jacobian_params={"k": 15},
)

jacobian = result.jacobian
```

For a custom embedding callable, `embedding_input` selects whether the callable
receives the normalized symmetric kernel or the point-cloud coordinates. For example, to run UMAP directly on the input points:

```python
import torch
from umap import UMAP
from finsler_embedding import from_points


def umap_on_points(points, *, n_components, **params):
    coordinates = UMAP(n_components=n_components, **params).fit_transform(
        points.detach().cpu().numpy()
    )
    return torch.as_tensor(
        coordinates, dtype=points.dtype, device=points.device
    )

result = from_points(
    points,
    distances=directed_distances,
    bandwidth=0.25,
    embedding=umap_on_points,
    embedding_input="points",
    embedding_params={"n_neighbors": 10},
)
```

`embedding_input="points"` requires `from_points`; `from_distances` and
`from_graph` do not have point-cloud coordinates. The callable must return a
`torch.Tensor` with shape `(N, n_components)`. The built-in `"spectral"` and
`"isomap"` embeddings retain their existing input behavior.

For a directed graph, pass a matrix of edge lengths. A zero entry denotes a
missing edge:

```python
from finsler_embedding import from_graph

result = from_graph(
    adjacency=edge_lengths, # (N, N)
    bandwidth=1.0,
    kernel="exponential",
    n_components=2,
)
```

Supported kernels are `"gaussian"` and `"exponential"`. Supported built-in
embeddings are `"spectral"` and `"isomap"`; a callable embedding can also be
provided. Jacobian estimation is optional and supports
`"local_least_squares"`, `"polynomial"`, or a custom callable.

Each call returns a `RandersResult` dataclass. Its most commonly used fields
are:

| Field                      | Shape                                            | Meaning                       |
| -------------------------- | ------------------------------------------------ | ----------------------------- |
| `coordinates`            | `(N, n_components)`                            | learned embedding             |
| `kernel`                 | `(N, N)`                                       | directed kernel               |
| `symmetric_operator`     | `(N, N)`                                       | symmetric normalized operator |
| `antisymmetric_operator` | `(N, N)`                                       | directional operator          |
| `carré_du_champ`        | `(N, n_components, n_components)`              | local cometric estimate       |
| `drift`                  | `(N, n_components)`                            | estimated directed drift      |
| `drift_norm_squared`     | `(N,)`                                         | squared drift norm            |
| `admissible`             | `(N,)`                                         | Randers admissibility mask    |
| `jacobian`               | `(N, n_components, intrinsic_dim)` or `None` | optional Jacobian             |

Kernel utilities are also available directly:

```python
from finsler_embedding import evaluate_kernel, kernel_moments, randers_constants
```

## Repository layout

- `src/finsler_embedding/core/`: validated inputs, kernels, operators,
  embeddings, Jacobians, and Randers reconstruction;
- `src/finsler_embedding/experiments/`: reproducible experiment
  configurations and runners;
- `tests/`: public API and operator tests;
- `results/`: generated evaluation output (not required for installation).

## Development checks

```bash
uv lock --check
uv build
uv run pytest
uv run python -m compileall -q src/finsler_embedding
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
