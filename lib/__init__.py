"""
Library package - Core ML utilities and classes.

This package contains all core machine learning classes and utilities.
"""

# Data loading
try:
    from .loader import GenericLoader
except ImportError:
    pass

# EDA
try:
    from .eda import GenericEda
except ImportError:
    pass

# Preprocessing
try:
    from .encoder import GenericEncoder
except ImportError:
    pass

try:
    from .outliers import GenericOutlierHandler
except ImportError:
    pass

try:
    from .multicollinearity import MulticollinearityHandler
except ImportError:
    pass

try:
    from .pca_reducer import PCAReducer
except ImportError:
    pass

# Models
try:
    from .models.k_means import GenericKMeans
except ImportError:
    pass

try:
    from .models.base import AbstractModel
except ImportError:
    pass

__all__ = [
    'GenericLoader',
    'GenericEda',
    'GenericEncoder',
    'GenericOutlierHandler',
    'MulticollinearityHandler',
    'PCAReducer',
    'GenericKMeans',
    'AbstractModel',
]
