"""Compatibility entry point for quantitative experiment evaluation.

New experiment code belongs in the dataset packages under
``finsler_embedding.experiments``.  This module remains intentionally thin so
existing ``finsler-embedding <dataset>`` commands keep working.
"""

from .experiments.legacy.evaluation import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
