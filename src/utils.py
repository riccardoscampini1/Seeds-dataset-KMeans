"""
Utility module for ML Project.

Contains logging configuration and custom exception classes.
"""

import logging
import sys
import os
from pathlib import Path


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "ml_project.log",
) -> logging.Logger:
    """Configure and return a named logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name (str): Logger name (typically __name__).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Configure logger if not already configured
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def ensure_directory_exists(directory: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory.
        
    Returns:
        str: Absolute path to the directory.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())


# Custom Exception Classes

class MLProjectError(Exception):
    """Base exception for ML Project."""
    pass


class DataLoadingError(MLProjectError):
    """Exception raised for errors during data loading."""
    pass


class PreprocessingError(MLProjectError):
    """Exception raised for errors during data preprocessing."""
    pass


class ModelError(MLProjectError):
    """Exception raised for errors during model training or prediction."""
    pass


class ModelTrainingError(ModelError):
    """Exception raised for errors during model training."""
    pass


class PersistenceError(MLProjectError):
    """Exception raised for errors during model persistence (save/load)."""
    pass


class ValidationError(MLProjectError):
    """Exception raised for data validation errors."""
    pass


class DataValidationError(MLProjectError):
    """Exception raised for data validation errors."""
    pass


__all__ = [
    'setup_logger',
    'get_logger',
    'ensure_directory_exists',
    'MLProjectError',
    'DataLoadingError',
    'DataValidationError',
    'PreprocessingError',
    'ModelError',
    'ModelTrainingError',
    'PersistenceError',
    'ValidationError',
]
