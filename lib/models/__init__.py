"""
Models package - Core model implementations.

Contains base class and specific model implementations.
"""

from .base import AbstractModel
from .k_means import GenericKMeans

try:
    from .decision_tree import DecisionTreeModel
except ImportError:
    pass

try:
    from .ensemble import EnsembleModel
except ImportError:
    pass

try:
    from .logreg import LogisticRegressionModel
except ImportError:
    pass

try:
    from .random_forest import RandomForestModel
except ImportError:
    pass

try:
    from .svm import SVMModel
except ImportError:
    pass

try:
    from .xgboost import XGBoostModel
except ImportError:
    pass

__all__ = [
    'AbstractModel',
    'GenericKMeans',
    'DecisionTreeModel',
    'EnsembleModel',
    'LogisticRegressionModel',
    'RandomForestModel',
    'SVMModel',
    'XGBoostModel',
]
