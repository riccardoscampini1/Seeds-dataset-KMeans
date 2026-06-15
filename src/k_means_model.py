"""
K-Means clustering model wrapper.

This module provides a high-level interface for K-Means clustering
using the GenericKMeans class from lib.models.k_means.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from lib.models.k_means import GenericKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.utils import get_logger, ModelError

logger = get_logger(__name__)


class KMeansModel:
    """
    High-level K-Means clustering model.
    
    Wraps GenericKMeans from lib.models.k_means with additional utilities:
    - Model training with parameter search
    - Clustering and prediction
    - Evaluation metrics (Silhouette, Davies-Bouldin, etc.)
    - Visualization (elbow plot, cluster visualization)
    - Model persistence (save/load)
    """
    
    def __init__(self):
        """Initialize K-Means Model."""
        self.model = None
        self.X_train = None
        self.predictions = None
        self.best_params = None
        self.best_score = None
        self.inertias = {}
        self.silhouette_scores = {}
    
    def train(
        self,
        X_train: pd.DataFrame,
        param_grid: Optional[Dict] = None,
        random_state: int = 42,
        verbose: bool = True,
    ) -> "KMeansModel":
        """
        Train K-Means model with hyperparameter search.
        
        Searches through parameter grid and selects the model with
        the highest Silhouette Score.
        
        Args:
            X_train (pd.DataFrame): Training features (must be numeric).
            param_grid (dict, optional): Custom parameter grid. Uses default if None.
                Default: n_clusters=[2-8], init=['k-means++'], n_init=[10], max_iter=[300]
            random_state (int): Random seed for reproducibility.
            verbose (bool): Whether to print training progress.
            
        Returns:
            KMeansModel: Self for method chaining.
            
        Raises:
            ModelError: If training fails.
        """
        try:
            self.X_train = X_train.copy()
            
            if verbose:
                logger.info(f"Starting K-Means training on data with shape {X_train.shape}")
            
            # Initialize and train GenericKMeans
            self.model = GenericKMeans()
            self.model.train(
                X_train=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
                random_state=random_state,
                param_grid=param_grid
            )
            
            self.best_params = self.model.best_params
            self.best_score = self.model.best_score
            
            if verbose:
                logger.info(
                    f"K-Means training completed. "
                    f"Best params: {self.best_params}, "
                    f"Silhouette Score: {self.best_score:.4f}"
                )
            
            return self
            
        except Exception as e:
            error_msg = f"Failed to train K-Means model: {e}"
            logger.error(error_msg)
            raise ModelError(error_msg)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for data.
        
        Args:
            X (pd.DataFrame): Data to cluster.
            
        Returns:
            np.ndarray: Cluster labels.
            
        Raises:
            ModelError: If model not trained.
        """
        if self.model is None or self.model.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        logger.info(f"Making predictions on data with shape {X.shape}")
        return self.model.predict(X)
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and predict on same data in one step.
        
        Args:
            X (pd.DataFrame): Data to fit and predict.
            
        Returns:
            np.ndarray: Cluster labels.
        """
        self.train(X, verbose=False)
        return self.predict(X)
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centers from trained model.
        
        Returns:
            np.ndarray: Cluster centers with shape (n_clusters, n_features).
            
        Raises:
            ModelError: If model not trained.
        """
        if self.model is None or self.model.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        return self.model.model.cluster_centers_
    
    def get_inertia(self) -> float:
        """
        Get within-cluster sum of squares (inertia).
        
        Returns:
            float: Inertia value.
            
        Raises:
            ModelError: If model not trained.
        """
        if self.model is None or self.model.model is None:
            raise ModelError("Model not trained. Call train() first.")
        
        return self.model.model.inertia_
    
    def evaluate(self, X: pd.DataFrame, labels: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate clustering quality using multiple metrics.
        
        Args:
            X (pd.DataFrame): Data to evaluate.
            labels (np.ndarray, optional): Cluster labels. If None, uses predictions.
            
        Returns:
            Dict: Evaluation metrics including:
                - silhouette_score: Silhouette coefficient (-1 to 1, higher is better)
                - davies_bouldin_index: Davies-Bouldin Index (lower is better)
                - inertia: Within-cluster sum of squares (lower is better)
                - n_clusters: Number of clusters
        """
        if labels is None:
            labels = self.predict(X)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        metrics = {
            'silhouette_score': silhouette_score(X_array, labels),
            'davies_bouldin_index': davies_bouldin_score(X_array, labels),
            'inertia': self.get_inertia(),
            'n_clusters': len(np.unique(labels)),
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def plot_elbow(
        self,
        X: pd.DataFrame,
        n_clusters_range: Tuple[int, int] = (2, 10),
        figsize: Tuple[int, int] = (12, 4),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot elbow curve to determine optimal number of clusters.
        
        Args:
            X (pd.DataFrame): Training data.
            n_clusters_range (tuple): Range of clusters to test (min, max).
            figsize (tuple): Figure size.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        logger.info(f"Computing elbow curve for n_clusters {n_clusters_range[0]}-{n_clusters_range[1]}")
        
        inertias = []
        silhouette_scores = []
        K_range = range(n_clusters_range[0], n_clusters_range[1] + 1)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        for k in K_range:
            kmeans = GenericKMeans()
            param_grid = {
                'n_clusters': [k],
                'init': ['k-means++'],
                'n_init': [10],
                'max_iter': [300]
            }
            kmeans.train(X_array, param_grid=param_grid, random_state=42)
            
            inertias.append(kmeans.model.inertia_)
            
            labels = kmeans.model.predict(X_array)
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X_array, labels))
            else:
                silhouette_scores.append(-1)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Inertia plot
        axes[0].plot(K_range, inertias, marker='o', linestyle='-', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontweight='bold')
        axes[0].set_title('Elbow Curve - Inertia', fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Silhouette score plot
        axes[1].plot(K_range, silhouette_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='darkgreen')
        axes[1].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[1].set_ylabel('Silhouette Score', fontweight='bold')
        axes[1].set_title('Silhouette Score by Number of Clusters', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def plot_clusters_2d(
        self,
        X: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        pca_components: int = 2,
        figsize: Tuple[int, int] = (10, 8),
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot clusters in 2D using PCA or first two features.
        
        Args:
            X (pd.DataFrame): Data to plot.
            labels (np.ndarray, optional): Cluster labels. If None, uses predictions.
            pca_components (int): If > 0, use PCA for visualization.
            figsize (tuple): Figure size.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        if labels is None:
            labels = self.predict(X)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Reduce dimensions if necessary
        if X_array.shape[1] > 2 and pca_components > 0:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_components)
            X_plot = pca.fit_transform(X_array)
        else:
            X_plot = X_array[:, :2] if X_array.shape[1] >= 2 else X_array
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points colored by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_plot[mask, 0],
                X_plot[mask, 1],
                c=[colors[idx]],
                label=f'Cluster {label}',
                alpha=0.6,
                s=50,
                edgecolors='k'
            )
        
        # Plot cluster centers if available
        if X_array.shape[1] > 2 and pca_components > 0:
            centers_pca = pca.transform(self.get_cluster_centers())
            ax.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                c='red',
                marker='X',
                s=300,
                edgecolors='black',
                linewidths=2,
                label='Centroids'
            )
        else:
            centers = self.get_cluster_centers()
            if centers.shape[1] >= 2:
                ax.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    c='red',
                    marker='X',
                    s=300,
                    edgecolors='black',
                    linewidths=2,
                    label='Centroids'
                )
        
        ax.set_xlabel('Dimension 1', fontweight='bold')
        ax.set_ylabel('Dimension 2', fontweight='bold')
        ax.set_title('K-Means Clustering Visualization', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    def print_summary(self) -> None:
        """Print summary of trained model."""
        if self.model is None or self.model.model is None:
            print("No trained model. Call train() first.")
            return
        
        print("\n" + "="*60)
        print("K-MEANS MODEL SUMMARY")
        print("="*60)
        print(f"Best Parameters: {self.best_params}")
        print(f"Silhouette Score: {self.best_score:.4f}")
        print(f"Inertia: {self.get_inertia():.4f}")
        print(f"Number of Iterations: {self.model.model.n_iter_}")
        print("="*60 + "\n")
