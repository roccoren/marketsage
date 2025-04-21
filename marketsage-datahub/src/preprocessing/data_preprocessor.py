#!/usr/bin/env python3
"""
Data preprocessing module for the MarketSage DataHub component.
This module provides functionality for cleaning, normalizing, and transforming stock data.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats

# Setup logging
logger = logging.getLogger(__name__)


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data and return the processed data."""
        pass


class RemoveOutliers(PreprocessingStep):
    """Remove outliers from the data."""
    
    def __init__(self, method: str = "z_score", threshold: float = 3.0, columns: Optional[List[str]] = None):
        """
        Initialize outlier removal.
        
        Args:
            method (str): Method to use for outlier detection
            threshold (float): Threshold for outlier detection
            columns (List[str], optional): Columns to process
        """
        self.method = method
        self.threshold = threshold
        self.columns = columns
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Determine columns to process
        columns = self.columns or ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = [col for col in columns if col in result.columns]
        
        if not columns:
            logger.warning("No valid columns for outlier removal")
            return result
        
        # Process each column
        for column in columns:
            if self.method == "z_score":
                # Z-score method
                z_scores = stats.zscore(result[column], nan_policy='omit')
                mask = np.abs(z_scores) < self.threshold
                result.loc[~mask, column] = np.nan
            
            elif self.method == "iqr":
                # IQR method
                q1 = result[column].quantile(0.25)
                q3 = result[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - self.threshold * iqr
                upper_bound = q3 + self.threshold * iqr
                result.loc[(result[column] < lower_bound) | (result[column] > upper_bound), column] = np.nan
        
        logger.info(f"Removed outliers using {self.method} method with threshold {self.threshold}")
        return result


class FillMissingValues(PreprocessingStep):
    """Fill missing values in the data."""
    
    def __init__(self, method: str = "interpolate", limit: int = 5, columns: Optional[List[str]] = None):
        """
        Initialize missing value filling.
        
        Args:
            method (str): Method to use for filling missing values
            limit (int): Maximum number of consecutive NaN values to fill
            columns (List[str], optional): Columns to process
        """
        self.method = method
        self.limit = limit
        self.columns = columns
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with missing values filled
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Determine columns to process
        columns = self.columns or ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = [col for col in columns if col in result.columns]
        
        if not columns:
            logger.warning("No valid columns for filling missing values")
            return result
        
        # Process each column
        for column in columns:
            if self.method == "interpolate":
                # Linear interpolation
                result[column] = result[column].interpolate(method='linear', limit=self.limit)
            
            elif self.method == "ffill":
                # Forward fill
                result[column] = result[column].fillna(method='ffill', limit=self.limit)
            
            elif self.method == "bfill":
                # Backward fill
                result[column] = result[column].fillna(method='bfill', limit=self.limit)
            
            elif self.method == "mean":
                # Fill with mean
                mean_value = result[column].mean()
                result[column] = result[column].fillna(mean_value)
        
        logger.info(f"Filled missing values using {self.method} method with limit {self.limit}")
        return result


class Normalize(PreprocessingStep):
    """Normalize the data."""
    
    def __init__(self, method: str = "minmax", columns: Optional[List[str]] = None):
        """
        Initialize normalization.
        
        Args:
            method (str): Method to use for normalization
            columns (List[str], optional): Columns to process
        """
        self.method = method
        self.columns = columns
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Normalized data
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Determine columns to process
        columns = self.columns or ['Open', 'High', 'Low', 'Close', 'Volume']
        columns = [col for col in columns if col in result.columns]
        
        if not columns:
            logger.warning("No valid columns for normalization")
            return result
        
        # Process each column
        for column in columns:
            if self.method == "minmax":
                # Min-max scaling
                min_val = result[column].min()
                max_val = result[column].max()
                if max_val > min_val:
                    result[column] = (result[column] - min_val) / (max_val - min_val)
            
            elif self.method == "zscore":
                # Z-score normalization
                mean = result[column].mean()
                std = result[column].std()
                if std > 0:
                    result[column] = (result[column] - mean) / std
        
        logger.info(f"Normalized data using {self.method} method")
        return result


class CalculateTechnicalIndicators(PreprocessingStep):
    """Calculate technical indicators."""
    
    def __init__(self, indicators: List[str] = None):
        """
        Initialize technical indicator calculation.
        
        Args:
            indicators (List[str], optional): Indicators to calculate
        """
        self.indicators = indicators or ['sma', 'ema', 'rsi', 'macd']
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Check if we have the necessary columns
        if 'Close' not in result.columns:
            logger.warning("Close column not found, cannot calculate technical indicators")
            return result
        
        # Calculate indicators
        for indicator in self.indicators:
            if indicator == 'sma':
                # Simple Moving Average
                result['SMA_20'] = result['Close'].rolling(window=20).mean()
                result['SMA_50'] = result['Close'].rolling(window=50).mean()
            
            elif indicator == 'ema':
                # Exponential Moving Average
                result['EMA_20'] = result['Close'].ewm(span=20, adjust=False).mean()
                result['EMA_50'] = result['Close'].ewm(span=50, adjust=False).mean()
            
            elif indicator == 'rsi':
                # Relative Strength Index
                delta = result['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                result['RSI_14'] = 100 - (100 / (1 + rs))
            
            elif indicator == 'macd':
                # MACD
                ema_12 = result['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = result['Close'].ewm(span=26, adjust=False).mean()
                result['MACD'] = ema_12 - ema_26
                result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
                result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
        
        logger.info(f"Calculated technical indicators: {', '.join(self.indicators)}")
        return result


class DataPreprocessor:
    """
    Manager for data preprocessing operations.
    Handles cleaning, normalizing, and transforming stock data.
    """
    
    def __init__(self, storage_manager=None, config: Dict[str, Any] = None):
        """
        Initialize data preprocessor.
        
        Args:
            storage_manager: Storage manager for data persistence
            config (Dict[str, Any], optional): Configuration dictionary
        """
        self.config = config or {}
        self.storage_manager = storage_manager
        self.preprocessing_config = self.config.get("preprocessing", {})
        
        # Initialize preprocessing steps
        self.steps = self._init_preprocessing_steps()
    
    def _init_preprocessing_steps(self) -> Dict[str, PreprocessingStep]:
        """
        Initialize preprocessing steps based on configuration.
        
        Returns:
            Dict[str, PreprocessingStep]: Dictionary of preprocessing steps
        """
        steps = {}
        
        # Outlier detection
        outlier_config = self.preprocessing_config.get("outlier_detection", {})
        steps["remove_outliers"] = RemoveOutliers(
            method=outlier_config.get("method", "z_score"),
            threshold=outlier_config.get("threshold", 3.0)
        )
        
        # Missing value handling
        missing_config = self.preprocessing_config.get("missing_values", {})
        steps["fill_missing_values"] = FillMissingValues(
            method=missing_config.get("method", "interpolate"),
            limit=missing_config.get("limit", 5)
        )
        
        # Normalization
        norm_config = self.preprocessing_config.get("normalization", {})
        steps["normalize"] = Normalize(
            method=norm_config.get("method", "minmax")
        )
        
        # Technical indicators
        steps["calculate_indicators"] = CalculateTechnicalIndicators()
        
        return steps
    
    def preprocess_data(self, symbols: List[str], steps: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess data for multiple symbols.
        
        Args:
            symbols (List[str]): List of stock symbols
            steps (List[str], optional): List of preprocessing steps to apply
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of preprocessing results by symbol
        """
        results = {}
        
        # Determine steps to apply
        if steps is None:
            # Use default steps from config
            steps = self.preprocessing_config.get("default_steps", ["remove_outliers", "fill_missing_values"])
        
        for symbol in symbols:
            try:
                # Load data
                if not self.storage_manager:
                    logger.error(f"No storage manager available for {symbol}")
                    results[symbol] = {"success": False, "error": "No storage manager available"}
                    continue
                
                data = self.storage_manager.load_data(symbol)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    results[symbol] = {"success": False, "error": "No data found"}
                    continue
                
                # Apply preprocessing steps
                processed_data = data.copy()
                applied_steps = []
                
                for step_name in steps:
                    if step_name in self.steps:
                        step = self.steps[step_name]
                        processed_data = step.process(processed_data)
                        applied_steps.append(step_name)
                    else:
                        logger.warning(f"Unknown preprocessing step: {step_name}")
                
                # Store processed data
                if not processed_data.empty:
                    self.storage_manager.save_data(f"{symbol}_processed", processed_data)
                    logger.info(f"Stored processed data for {symbol}")
                
                results[symbol] = {
                    "success": True,
                    "rows": len(processed_data),
                    "applied_steps": applied_steps
                }
                
            except Exception as e:
                logger.error(f"Error preprocessing data for {symbol}: {e}")
                results[symbol] = {"success": False, "error": str(e)}
        
        return results
    
    def get_available_steps(self) -> List[str]:
        """
        Get list of available preprocessing steps.
        
        Returns:
            List[str]: List of available preprocessing step names
        """
        return list(self.steps.keys())