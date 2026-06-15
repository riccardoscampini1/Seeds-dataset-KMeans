"""
Data loading module for ML Project.

This module provides utilities for loading data from various sources (CSV, Excel, databases, etc.)
and converting them into pandas DataFrames for further processing.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from src.utils import get_logger, DataLoadingError
from lib.loader import GenericLoader

logger = get_logger(__name__)


class DataLoader:
    """
    Handles loading data from various sources.
    
    Supports CSV, Excel, JSON, Parquet, and other formats.
    Implements error handling and logging for data loading operations.
    
    TODO: Add support for database connections (SQL)
    TODO: Add support for cloud storage (S3, GCS, Azure Blob)
    TODO: Implement data caching mechanism
    TODO: Implement incremental loading for large datasets
    TODO: Add support for data compression formats
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.df = None
        self.filepath = None
        self.metadata = {}
    
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            **kwargs: Additional arguments to pass to pd.read_csv().
            
        Returns:
            pd.DataFrame: Loaded dataframe.
            
        Raises:
            DataLoadingError: If loading fails.
            
        TODO: Implement automatic data type inference
        TODO: Implement handling of large files (chunking)
        TODO: Add encoding detection
        """
        try:
            logger.info(f"Loading CSV from: {filepath}")
            self.df = pd.read_csv(filepath, **kwargs)
            self.filepath = filepath
            self._update_metadata()
            logger.info(f"Successfully loaded {len(self.df)} rows from {filepath}")
            return self.df
        except FileNotFoundError:
            error_msg = f"CSV file not found: {filepath}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file: {e}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error while loading CSV: {e}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
    
    def load_excel(self, filepath: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.
        
        Args:
            filepath (str): Path to the Excel file.
            sheet_name (str): Name or index of the sheet to load.
            **kwargs: Additional arguments to pass to pd.read_excel().
            
        Returns:
            pd.DataFrame: Loaded dataframe.
            
        Raises:
            DataLoadingError: If loading fails.
            
        TODO: Implement multi-sheet loading
        TODO: Add support for different Excel formats
        """
        try:
            logger.info(f"Loading Excel from: {filepath}")
            self.df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            self.filepath = filepath
            self._update_metadata()
            logger.info(f"Successfully loaded {len(self.df)} rows from {filepath}")
            return self.df
        except Exception as e:
            error_msg = f"Failed to load Excel file: {e}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
    
    def load_json(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
            **kwargs: Additional arguments to pass to pd.read_json().
            
        Returns:
            pd.DataFrame: Loaded dataframe.
            
        Raises:
            DataLoadingError: If loading fails.
        """
        try:
            logger.info(f"Loading JSON from: {filepath}")
            self.df = pd.read_json(filepath, **kwargs)
            self.filepath = filepath
            self._update_metadata()
            logger.info(f"Successfully loaded {len(self.df)} rows from {filepath}")
            return self.df
        except Exception as e:
            error_msg = f"Failed to load JSON file: {e}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
    
    def load_parquet(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a Parquet file.
        
        Args:
            filepath (str): Path to the Parquet file.
            **kwargs: Additional arguments to pass to pd.read_parquet().
            
        Returns:
            pd.DataFrame: Loaded dataframe.
            
        Raises:
            DataLoadingError: If loading fails.
        """
        try:
            logger.info(f"Loading Parquet from: {filepath}")
            self.df = pd.read_parquet(filepath, **kwargs)
            self.filepath = filepath
            self._update_metadata()
            logger.info(f"Successfully loaded {len(self.df)} rows from {filepath}")
            return self.df
        except Exception as e:
            error_msg = f"Failed to load Parquet file: {e}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
    
    def _update_metadata(self) -> None:
        """
        Update metadata about the loaded data.
        
        TODO: Implement comprehensive metadata tracking
        TODO: Store data schema information
        TODO: Track data quality metrics
        """
        if self.df is not None:
            self.metadata = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
                'column_names': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict(),
            }
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded dataframe.
        
        Returns:
            pd.DataFrame: Loaded dataframe or None if nothing loaded.
        """
        return self.df
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded data.
        
        Returns:
            Dict: Metadata dictionary.
        """
        return self.metadata.copy()
    
    def load_seeds_dataset(self, filepath: str = 'data/seeds_dataset.txt') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the seeds dataset from a TXT file and split into features (X) and target (y).
        
        Uses GenericLoader with whitespace separator.
        
        Args:
            filepath (str): Path to the seeds_dataset.txt file.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (X, y) where:
                - X contains all feature columns
                - y contains only the 'class' column
                Both maintain the index.
                
        Raises:
            DataLoadingError: If loading fails.
        """
        try:
            columns = [
                'area', 'perimeter', 'compactness', 'length_kernel',
                'width_kernel', 'asymmetry_coefficient', 'length_kernel_groove', 'class'
            ]
            
            logger.info(f"Loading seeds dataset from: {filepath}")
            
            loader = GenericLoader(
                target_col='class',
                csv_path=filepath,
                sep=r'\s+',
                engine='python',
                header=None,
                names=columns
            )
            
            self.df = loader.load()
            self.filepath = filepath
            self._update_metadata()
            
            # Split into X (features) and y (target)
            X = self.df.drop(columns=['class'])
            y = self.df[['class']]
            
            logger.info(f"Successfully loaded {len(self.df)} rows")
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            return X, y
            
        except FileNotFoundError:
            error_msg = f"Seeds dataset file not found: {filepath}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load seeds dataset: {e}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg)
