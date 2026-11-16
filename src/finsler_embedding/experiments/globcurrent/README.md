# GlobCurrent experiment

This directory contains the reproducible ocean-current experiment used to compare the moment-Randers estimator with
Finsler MDS and Perrault-Joncas & Meila on a real velocity field with curl.

Finsler MDS (Dagès et al., 2025) fits the canonical Randers metric `|v| + alpha v_last`, whose asymmetry is a
difference of potentials and sums to zero around every closed loop. A current with eddies has a nonzero circulation,
so no embedding of that family reproduces it. The moment-Randers estimator has no such constraint.

The implementation has three independent layers:

- `data.py` downloads one day of currents and builds the directed distances and the ground truth (the dataset is
  registered as `globcurrent` in `experiments/legacy/evaluation.py`).
- `curl.py` contains the curl diagnostics, which do not use any embedding.
- `plot.py` draws the figures.

The currents are real, the distances are modelled. The data are the daily surface currents of the Agulhas region
(GlobCurrent, 0.25 degree grid, 2552 ocean nodes), from the
[Parcels example data](https://github.com/Parcels-code/parcels-data/tree/main/data/GlobCurrent_example_data). The
file of a day (about 55 kB) is downloaded on first use into `data/globcurrent/`, which is git-ignored. The default day
is 2002-01-01.

The file has no directed distances, so a vessel is assumed to sail at constant speed `s` relative to the water in the
current `W(x)`, with `|W| < s`. The travel time is a Randers metric, which gives the ground truth in closed form:

- `c = w = W / s`
- `g_BL^{-1} = I + 4 w w^T`
- `beta`: the strongest current of the day over the ship speed, the role of `||b||` in the synthetic metrics

The experiment samples ocean nodes, builds the same radius graph for ours and Perrault-Joncas & Meila (directed kNN
graph for Finsler MDS), and compares the drift of each method with the true drift mapped by the Jacobian of the
embedding. Results report the cosine with the true drift, the relative error on `||c||^2_{g_BL}`, and two diagnostics
that do not use any embedding:

- `curl_fraction`: the share of the energy of the edge asymmetries `(d_ij - d_ji) / 2` that no potential explains
- `gradient_ceiling`: the cosine of the true drift with the best drift that is the gradient of a potential, which is
  the form of the drift of Finsler MDS

## Reproduce the comparison

From the repository root:

```bash
uv run python -m finsler_embedding.experiments.finsler_mds_manifolds \
  --datasets globcurrent --betas 0.5 0.9 --seeds 3 --n 2000 \
  --out results/globcurrent/alpha0.5
```

Pass `--alpha 0.9` for the other value of the Finsler MDS parameter and `--thetas 1 0.5 0` for the three values of the
normalization exponent of our method (rows `ours`, `ours_theta0.5` and `ours_theta0`). Finsler MDS runs 2000 SMACOF
iterations from a random start, without early stop (`max_iter` of `finsler_mds_points`).

The run saves `manifolds.csv` in the output directory, with the median over seeds printed at the end. The two values
of `beta` use the same field and the same graph geometry, so they are not independent experiments.

For the figures (the currents, our embedding with the estimated drift, and Finsler MDS):

```bash
uv run python -m finsler_embedding.experiments.globcurrent.plot \
  --out results/globcurrent/figure/globcurrent
```

This writes `globcurrent_currents.pdf`, `globcurrent_proposed.pdf` and `globcurrent_finsler_mds.pdf`.

The notebook is [`globcurrent.ipynb`](./globcurrent.ipynb). It runs the same comparison step by step: one sample with
the figure, then the three seeds.
