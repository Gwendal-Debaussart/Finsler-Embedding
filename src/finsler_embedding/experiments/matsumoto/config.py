"""Configuration for the Matsumoto quantitative experiment."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatsumotoConfig:
    """Parameters for a seeded Matsumoto evaluation sweep."""

    n: int = 2000
    seeds: int = 5
    graph: str = "knn"
    k: int = 10
    eps_scale: float = 1.0
    eps0: float = 0.5
    cut: float = 3.0
    radius: float | None = None
    device: str = "cpu"
    embeddings: tuple[str, ...] = ("chart", "isomap")
    output: Path = Path("results/evaluation")
