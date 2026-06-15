"""
Feature scaling module for ML Project.

This module provides utilities for scaling and normalizing features
using the GenericEncoder class from lib.encoder.
Supports StandardScaler, MinMaxScaler, RobustScaler, and other scaling methods.
"""

import pandas as pd
from typing import List, Optional
from lib.encoder import GenericEncoder
from src.utils import get_logger, PreprocessingError

logger = get_logger(__name__)


class FeatureScaler:
    """
    Handles feature scaling and normalization using GenericEncoder.
    
    Scalers:
    - Standard Scaling (z-score normalization)
    - Min-Max Scaling (normalization)
    - Robust Scaling (resistant to outliers) - **PRIMARY METHOD**
    
    Uses lib.encoder.GenericEncoder for all scaling operations.
    """
    
    def __init__(self):
        """Initialize the FeatureScaler."""
        self.encoder = None
        self.scaled_columns = []
        self.scaler_type = None
    
    def scale_standard(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply standard scaling (z-score normalization).
        
        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): Columns to scale. If None, scale all numeric columns.
            
        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Applying standard scaling to {len(columns)} columns")
        
        df_subset = df[columns].copy()
        
        self.encoder = GenericEncoder()
        self.encoder.configure(num_strategy='standard', cat_strategy='ohe')
        scaled_data = self.encoder.fit_transform_features(df_subset)
        
        self.scaled_columns = columns
        self.scaler_type = 'standard'
        
        # Merge scaled columns with non-numeric columns from original df
        result = df.copy()
        result[columns] = scaled_data
        
        return result
    
    def scale_minmax(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        feature_range: tuple = (0, 1)
    ) -> pd.DataFrame:
        """
        Apply min-max scaling (normalization).
        
        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): Columns to scale.
            feature_range (tuple): Range for scaled features (note: GenericEncoder applies [0,1] by default).
            
        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Applying min-max scaling to {len(columns)} columns")
        
        df_subset = df[columns].copy()
        
        self.encoder = GenericEncoder()
        self.encoder.configure(num_strategy='minmax', cat_strategy='ohe')
        scaled_data = self.encoder.fit_transform_features(df_subset)
        
        self.scaled_columns = columns
        self.scaler_type = 'minmax'
        
        result = df.copy()
        result[columns] = scaled_data
        
        return result
    
    def scale_robust(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply robust scaling (resistant to outliers) - PRIMARY METHOD.
        
        Uses RobustScaler via GenericEncoder, which scales features using
        statistics that are robust to outliers (median and interquartile range).
        
        Args:
            df (pd.DataFrame): Input dataframe.
            columns (List[str], optional): Columns to scale. If None, scale all numeric columns.
            
        Returns:
            pd.DataFrame: Scaled dataframe.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Applying robust scaling to {len(columns)} columns")
        
        df_subset = df[columns].copy()
        
        self.encoder = GenericEncoder()
        self.encoder.configure(num_strategy='robust', cat_strategy='ohe')
        scaled_data = self.encoder.fit_transform_features(df_subset)
        
        self.scaled_columns = columns
        self.scaler_type = 'robust'
        
        result = df.copy()
        result[columns] = scaled_data
        
        return result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned scaling to new data.
        
        Args:
            df (pd.DataFrame): New dataframe to scale.
            
        Returns:
            pd.DataFrame: Scaled dataframe.
            
        Raises:
            PreprocessingError: If scaler not fitted yet.
        """
        if self.encoder is None:
            raise PreprocessingError("Scaler not fitted. Call fit method first.")
        
        df_subset = df[self.scaled_columns].copy()
        scaled_data = self.encoder.transform_features(df_subset)
        
        result = df.copy()
        result[self.scaled_columns] = scaled_data
        
        return result
    
    def get_scaler_info(self) -> dict:
        """
        Get information about the fitted scaler.
        
        Returns:
            dict: Information about scaler type and scaled columns.
        """
        return {
            'scaler_type': self.scaler_type,
            'scaled_columns': self.scaled_columns,
            'n_columns_scaled': len(self.scaled_columns),
        }
