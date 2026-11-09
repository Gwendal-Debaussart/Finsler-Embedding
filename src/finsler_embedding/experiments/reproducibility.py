"""Reproducibility and run-artifact helpers for experiments."""

import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all supported random-number generators.

    Parameters
    ----------
    seed : int
        Nonnegative seed shared by Python, NumPy, and PyTorch.
    deterministic : bool, default=True
        Whether to request deterministic PyTorch algorithms where supported.

    Returns
    -------
    None
        Random-number generators are seeded in place.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def create_run_directory(
    output_dir: str | Path, experiment: str, config: Any
) -> Path:
    """Create a timestamped run directory and persist its configuration.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Parent directory for experiment artifacts.
    experiment : str
        Short experiment name used in the run-directory name.
    config : object
        Dataclass or mapping containing the run configuration.

    Returns
    -------
    pathlib.Path
        Newly created run directory containing ``config.json`` and
        ``metadata.json``.
    """
    if not experiment or "/" in experiment or "\\" in experiment:
        raise ValueError("experiment must be a nonempty simple name")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{experiment}-{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{experiment}-{timestamp}-{suffix}"
        suffix += 1
    run_dir.mkdir()
    values = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
    values = {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
    (run_dir / "config.json").write_text(json.dumps(values, indent=2, sort_keys=True))
    (run_dir / "metadata.json").write_text(
        json.dumps({"experiment": experiment, "created_at": timestamp}, indent=2)
    )
    return run_dir
