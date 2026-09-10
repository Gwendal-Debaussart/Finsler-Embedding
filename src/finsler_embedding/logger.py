import logging
import sys
from pathlib import Path


def setup_logger(name: str | None, log_file: Path = None, level=logging.INFO):
    """
    Set up a logger with both file and console handlers.

    Parameters:
    -----------
    name : str
        Name of the logger
    log_file : Path, optional
        Path to the log file. If None, only console logging is enabled.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def update_log_file(LOGGER: logging.Logger, new_log_file: Path):
    """
    Update the log file of an existing logger.

    Parameters:
    -----------
    LOGGER : logging.Logger
        The logger instance to update
    new_log_file : Path
        Path to the new log file
    """
    # Remove existing file handlers
    for handler in LOGGER.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            LOGGER.removeHandler(handler)

    # Add new file handler
    new_log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(new_log_file, mode="a")
    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
