# Sphere experiment

The sphere experiment evaluates the moment-Randers operators against the
analytic sphere ground truth.

Run a small evaluation with:

```bash
uv run python -m finsler_embedding.experiments.sphere.runner \
  --n 100 --seeds 1 --embedding chart
```

Results are written to `results/evaluation/sphere.csv` by default. Use
`--help` for graph, embedding, device, and output configuration.
