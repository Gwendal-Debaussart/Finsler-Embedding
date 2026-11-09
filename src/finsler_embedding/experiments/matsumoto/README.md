# Matsumoto surface experiment

This directory contains the interactive Matsumoto surface notebook:
[`matsumoto_dataset.ipynb`](./matsumoto_dataset.ipynb).

The notebook constructs directed Randers distances on a graph, uses the
Matsumoto surface coordinates as the point cloud for Isomap, and runs the
shared `core.inputs.from_points` pipeline.

Run a small quantitative evaluation with:

```bash
uv run python -m finsler_embedding.experiments.matsumoto.runner \
  --n 100 --seeds 1 --embedding chart
```

Use `--help` to change graph, embedding, device, or output settings.
