"""
Dimensionality reduction module using PCA.

This module provides utilities for Principal Component Analysis (PCA)
using the PCAReducer class from lib.pca_reducer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from lib.pca_reducer import PCAReducer
from src.utils import get_logger, PreprocessingError

logger = get_logger(__name__)


class PCAProcessor:
    """
    Handles Principal Component Analysis (PCA) for dimensionality reduction.
    
    Uses lib.pca_reducer.PCAReducer for transformations.
    
    Supports:
    - Fixed number of components (int)
    - Variance retention (float 0-1)
    - Fit/Transform pattern with state preservation
    """
    
    def __init__(self):
        """Initialize the PCA Processor."""
        self.pca_reducer = None
        self.n_components = None
        self.variance_explained = None
        self.n_features_in = None
    
    def configure(
        self,
        n_components: Union[int, float] = 0.95,
        random_state: int = 42
    ) -> "PCAProcessor":
        """
        Configure PCA reducer.
        
        Args:
            n_components (int or float):
                - int: exact number of components to keep
                - float (0-1): fraction of variance to retain (e.g., 0.95 for 95%)
            random_state (int): Random seed for reproducibility.
            
        Returns:
            PCAProcessor: Self for method chaining.
        """
        self.pca_reducer = PCAReducer()
        self.pca_reducer.configure(n_components=n_components, random_state=random_state)
        logger.info(f"PCA configured with n_components={n_components}")
        
        return self
    
    def fit(self, X: pd.DataFrame) -> "PCAProcessor":
        """
        Fit PCA on the training data.
        
        Args:
            X (pd.DataFrame): Training dataframe (numeric features).
            
        Returns:
            PCAProcessor: Self for method chaining.
            
        Raises:
            PreprocessingError: If PCA not configured.
        """
        if self.pca_reducer is None:
            raise PreprocessingError("PCA not configured. Call configure() first.")
        
        logger.info(f"Fitting PCA on data with shape {X.shape}")
        self.pca_reducer.fit(X)
        
        report = self.pca_reducer.report()
        self.n_features_in = report['n_components_in']
        self.n_components = report['n_components_out']
        self.variance_explained = report['variance_explained']
        
        logger.info(
            f"PCA fitted: {self.n_features_in} features -> {self.n_components} components "
            f"(variance explained: {self.variance_explained*100:.2f}%)"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted PCA.
        
        Args:
            X (pd.DataFrame): Data to transform (numeric features).
            
        Returns:
            pd.DataFrame: Transformed data with columns PC1, PC2, etc.
            
        Raises:
            PreprocessingError: If PCA not fitted.
        """
        if self.pca_reducer is None or self.pca_reducer._pca is None:
            raise PreprocessingError("PCA not fitted. Call fit() first.")
        
        logger.info(f"Transforming data with shape {X.shape}")
        transformed = self.pca_reducer.transform(X)
        
        return transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X (pd.DataFrame): Data to fit and transform.
            
        Returns:
            pd.DataFrame: Transformed data.
        """
        return self.fit(X).transform(X)
    
    def get_report(self) -> dict:
        """
        Get detailed report about the fitted PCA.
        
        Returns:
            dict: Report with:
                - n_components_in: number of input features
                - n_components_out: number of components
                - variance_explained: total variance explained (0-1)
        """
        if self.pca_reducer is None:
            return {}
        
        return self.pca_reducer.report()
    
    def plot_explained_variance(
        self,
        figsize: Tuple[int, int] = (12, 5),
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot explained variance ratio for each component.
        
        Args:
            figsize (tuple): Figure size.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
            
        Raises:
            PreprocessingError: If PCA not fitted.
        """
        if self.pca_reducer is None or self.pca_reducer._pca is None:
            raise PreprocessingError("PCA not fitted. Call fit() first.")
        
        explained_var = self.pca_reducer._pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        axes[0].bar(
            range(1, len(explained_var) + 1),
            explained_var,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        axes[0].set_xlabel('Principal Component', fontweight='bold')
        axes[0].set_ylabel('Explained Variance Ratio', fontweight='bold')
        axes[0].set_title('Explained Variance by Component', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Cumulative variance
        axes[1].plot(
            range(1, len(cumsum_var) + 1),
            cumsum_var,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=6,
            color='darkgreen'
        )
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        axes[1].set_xlabel('Number of Components', fontweight='bold')
        axes[1].set_ylabel('Cumulative Explained Variance', fontweight='bold')
        axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_pca_2d(
        self,
        X_pca: pd.DataFrame,
        y: Optional[pd.Series] = None,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True
    ) -> Optional[plt.Figure]:
        """
        Plot 2D scatter plot of the first two principal components.
        
        Args:
            X_pca (pd.DataFrame): Transformed data (output of transform()).
            y (pd.Series, optional): Target variable for coloring points.
            figsize (tuple): Figure size.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        if X_pca.shape[1] < 2:
            raise PreprocessingError("Need at least 2 principal components for 2D plot.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pc1 = X_pca.iloc[:, 0]
        pc2 = X_pca.iloc[:, 1]
        
        if y is not None:
            scatter = ax.scatter(pc1, pc2, c=y, cmap='viridis', alpha=0.6, edgecolors='k', s=50)
            plt.colorbar(scatter, ax=ax, label='Target')
        else:
            ax.scatter(pc1, pc2, alpha=0.6, edgecolors='k', s=50, color='steelblue')
        
        ax.set_xlabel(f'PC1 ({self.pca_reducer._pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.pca_reducer._pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        ax.set_title('2D PCA Visualization', fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def print_report(self) -> None:
        """Print detailed PCA report."""
        report = self.get_report()
        
        if not report:
            print("No PCA report available. Fit PCA first.")
            return
        
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS REPORT")
        print("="*60)
        print(f"Input features: {report['n_components_in']}")
        print(f"Output components: {report['n_components_out']}")
        print(f"Variance explained: {report['variance_explained']*100:.2f}%")
        print("="*60 + "\n")
