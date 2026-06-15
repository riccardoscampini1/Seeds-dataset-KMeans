"""
Exploratory Data Analysis (EDA) module.

Provides utilities for outlier removal and correlation analysis
using classes from lib.outliers and lib.multicollinearity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

from lib.outliers import GenericOutlierHandler
from lib.multicollinearity import MulticollinearityHandler
from lib.eda import GenericEda


class EDAAnalyzer:
    """
    Exploratory Data Analysis tool for outlier detection and correlation analysis.
    
    Combines GenericOutlierHandler (from lib.outliers) and MulticollinearityHandler
    (from lib.multicollinearity) to provide comprehensive EDA capabilities.
    """

    def __init__(self):
        """Initialize the EDA Analyzer."""
        self.outlier_handler = None
        self.multicollinearity_handler = None
        self.outlier_report = None

    def remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        iqr_threshold: float = 1.5,
        cols: Optional[list] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove outliers using IQR method and drop rows containing them.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            iqr_threshold (float): IQR multiplier for bounds (default 1.5).
            cols (list, optional): Columns to check for outliers. If None, uses all numeric.
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 
                - Dataframe with outlier rows removed
                - Report dict with outlier statistics
        """
        self.outlier_handler = GenericOutlierHandler()
        self.outlier_handler.configure(
            detection='iqr',
            treatment='drop',
            iqr_threshold=iqr_threshold,
            cols=cols
        )

        # Generate report before dropping
        self.outlier_handler.fit(df)
        self.outlier_report = self.outlier_handler.report(df)

        # Apply transformation (drop rows with outliers)
        df_clean = self.outlier_handler.fit_transform(df)

        return df_clean, self.outlier_report

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 8),
        annot: bool = True,
        cmap: str = 'coolwarm',
        fmt: str = '.2f',
        title: str = 'Correlation Matrix',
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot the correlation matrix heatmap for numeric features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            figsize (tuple): Figure size (width, height).
            annot (bool): Whether to annotate cells with values.
            cmap (str): Colormap name.
            fmt (str): Format string for annotations.
            title (str): Plot title.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("Warning: No numeric columns found in dataframe.")
            return None

        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=annot,
            cmap=cmap,
            fmt=fmt,
            ax=ax,
            cbar_kws={'label': 'Correlation'},
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def detect_multicollinearity(
        self,
        df: pd.DataFrame,
        threshold: float = 0.85,
        drop: bool = False,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and optionally remove highly correlated features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            threshold (float): Correlation threshold (default 0.85).
            drop (bool): Whether to drop correlated columns (default False).
            
        Returns:
            Tuple[pd.DataFrame, Dict]:
                - Dataframe (with cols dropped if drop=True)
                - Report with correlated pairs and columns to drop
        """
        self.multicollinearity_handler = MulticollinearityHandler()
        self.multicollinearity_handler.configure(threshold=threshold, drop=drop)
        
        self.multicollinearity_handler.fit(df)
        report = self.multicollinearity_handler.report()
        
        df_result = self.multicollinearity_handler.transform(df)

        return df_result, report

    def print_outlier_report(self) -> None:
        """Print the outlier detection report."""
        if self.outlier_report:
            print("\n" + "="*60)
            print("OUTLIER DETECTION REPORT (IQR Method)")
            print("="*60)
            if not self.outlier_report:
                print("No outliers detected.")
            else:
                for col, stats in self.outlier_report.items():
                    print(f"\nColumn: {col}")
                    print(f"  Outliers found: {stats['outliers']}")
                    print(f"  Percentage: {stats['pct']}%")
                    print(f"  Lower bound: {stats['lower_bound']}")
                    print(f"  Upper bound: {stats['upper_bound']}")
            print("="*60 + "\n")
        else:
            print("No outlier report available. Run remove_outliers_iqr() first.")

    def print_multicollinearity_report(self) -> None:
        """Print the multicollinearity detection report."""
        if self.multicollinearity_handler:
            report = self.multicollinearity_handler.report()
            print("\n" + "="*60)
            print("MULTICOLLINEARITY DETECTION REPORT")
            print("="*60)
            
            if report['pairs']:
                print(f"\nHighly correlated pairs (|r| > threshold):")
                for pair in report['pairs']:
                    print(f"  {pair['col_a']:20s} <-> {pair['col_b']:20s} : r = {pair['r']:.4f}")
            else:
                print("\nNo highly correlated pairs detected.")
            
            if report['cols_to_drop']:
                print(f"\nColumns recommended for removal: {report['cols_to_drop']}")
            else:
                print("\nNo columns recommended for removal.")
            
            print("="*60 + "\n")
        else:
            print("No multicollinearity report available. Run detect_multicollinearity() first.")

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        bins: int = 30,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot histograms for all numeric features.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            bins (int): Number of bins for histograms.
            figsize (tuple, optional): Figure size. Auto-calculated if None.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("Warning: No numeric columns found in dataframe.")
            return None

        n_cols = numeric_df.shape[1]
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        if figsize is None:
            figsize = (15, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_df.columns):
            axes[idx].hist(numeric_df[col], bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
            axes[idx].set_title(f'Distribution: {col}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_boxplots_outliers(
        self,
        df: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Plot boxplots for all numeric features to visualize outliers.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            figsize (tuple, optional): Figure size. Auto-calculated if None.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("Warning: No numeric columns found in dataframe.")
            return None

        n_cols = numeric_df.shape[1]
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        if figsize is None:
            figsize = (15, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_df.columns):
            axes[idx].boxplot(numeric_df[col], vert=True)
            axes[idx].set_title(f'Boxplot: {col}', fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_comparative_distributions(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        title_suffix: str = "(Before vs After)",
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Compare distributions of a feature before and after preprocessing (e.g., outlier removal).
        
        Args:
            df_before (pd.DataFrame): Dataframe before preprocessing.
            df_after (pd.DataFrame): Dataframe after preprocessing.
            title_suffix (str): Suffix for plot title.
            figsize (tuple, optional): Figure size.
            show (bool): Whether to display the plot.
            
        Returns:
            plt.Figure: Matplotlib figure object.
        """
        numeric_cols = df_before.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col in df_after.columns]
        
        if len(numeric_cols) == 0:
            print("Warning: No numeric columns found in dataframe.")
            return None

        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        if figsize is None:
            figsize = (15, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(df_before[col], bins=25, alpha=0.6, label='Before', color='red', edgecolor='black')
            axes[idx].hist(df_after[col], bins=25, alpha=0.6, label='After', color='green', edgecolor='black')
            axes[idx].set_title(f'{col} {title_suffix}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)

        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def get_eda_summary(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict:
        """
        Get a comprehensive EDA summary using GenericEda class.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            target (pd.Series, optional): Target variable for class balance analysis.
            
        Returns:
            Dict: Comprehensive EDA summary.
        """
        eda = GenericEda(df)
        
        summary = {
            'summary_statistics': eda.summary(),
            'missing_values': eda.missing_report(),
            'correlation': eda.correlation(),
        }
        
        if target is not None:
            summary['class_balance'] = eda.class_balance(target)
        
        return summary
