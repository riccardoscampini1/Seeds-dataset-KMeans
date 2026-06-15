"""
Metrics package - Evaluation metrics for ML models.

Contains utilities for evaluating supervised and unsupervised models.
"""

try:
    from .unsupervised_metrics import UnsupervisedMetrics
except ImportError:
    pass

__all__ = [
    'UnsupervisedMetrics',
]
