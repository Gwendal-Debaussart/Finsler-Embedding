"""Typed configuration objects shared by reproducible experiments."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for directed graph construction.

    Parameters
    ----------
    graph_type : str
        Graph strategy, either ``"knn"`` or ``"radius"``.
    k : int
        Number of neighbors for a k-nearest-neighbor graph.
    eps_scale : float
        Scale applied to the empirical k-neighbor bandwidth.
    eps0 : float
        Base scale for the asymptotic radius bandwidth.
    cut : float
        Radius multiplier when constructing a dynamic radius graph.
    radius : float or None
        Fixed Euclidean radius, or ``None`` for a dynamic radius.
    """

    graph_type: str = "knn"
    k: int = 10
    eps_scale: float = 1.0
    eps0: float = 0.5
    cut: float = 3.0
    radius: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Base configuration shared by seeded experiment runs.

    Parameters
    ----------
    seed : int
        Random seed used for Python, NumPy, and PyTorch.
    output_dir : pathlib.Path
        Directory in which run artifacts are stored.
    device : str
        Torch device used by the experiment.
    """

    seed: int = 0
    output_dir: Path = Path("results")
    device: str = "cpu"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the configuration.

        Returns
        -------
        dict
            Configuration fields with paths converted to strings.
        """
        values = asdict(self)
        values["output_dir"] = str(values["output_dir"])
        return values
