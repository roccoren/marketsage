#!/usr/bin/env python3
"""
Dataset module for stock trading prediction.

This module provides functionality to create and manage datasets for training
and evaluating deep learning models for stock trading prediction.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# Configure logger
logger = logging.getLogger(__name__)


class TimeSeriesSplit:
    """Time series cross-validation split for financial data."""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, gap: int = 0):
        """
        Initialize TimeSeriesSplit.
        
        Args:
            n_splits: Number of splits
            test_size: Proportion of data to use for testing in each split
            gap: Number of samples to exclude between training and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features DataFrame
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = int(n_samples * self.test_size)
        
        # Minimum training size is half of the test size
        min_train_size = max(2, test_size // 2)
        
        # Step size between splits
        step = (n_samples - test_size) // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            # Ensure we don't exceed the available data
            if i * step + min_train_size >= n_samples - test_size:
                break
            
            # Calculate the end index for training data
            train_end = i * step + min_train_size
            
            # Apply gap between train and test
            test_start = train_end + self.gap
            
            # Ensure we have enough data for testing
            if test_start + test_size > n_samples:
                break
            
            test_end = test_start + test_size
            
            # Get the indices for this split
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        return splits


class PurgedTimeSeriesSplit(TimeSeriesSplit):
    """Purged time series cross-validation to prevent data leakage."""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2, gap: int = 0, purge_size: int = 0):
        """
        Initialize PurgedTimeSeriesSplit.
        
        Args:
            n_splits: Number of splits
            test_size: Proportion of data to use for testing in each split
            gap: Number of samples to exclude between training and test sets
            purge_size: Number of samples to purge from training set before each test sample
        """
        super().__init__(n_splits, test_size, gap)
        self.purge_size = purge_size
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features DataFrame
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = int(n_samples * self.test_size)
        
        # Minimum training size is half of the test size
        min_train_size = max(2, test_size // 2)
        
        # Step size between splits
        step = (n_samples - test_size) // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            # Ensure we don't exceed the available data
            if i * step + min_train_size >= n_samples - test_size:
                break
            
            # Calculate the end index for training data
            train_end = i * step + min_train_size
            
            # Apply gap between train and test
            test_start = train_end + self.gap
            
            # Ensure we have enough data for testing
            if test_start + test_size > n_samples:
                break
            
            test_end = test_start + test_size
            
            # Get the indices for this split
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            # Purge training data that could cause leakage
            if self.purge_size > 0:
                # Remove samples from training that are too close to test samples
                purge_indices = []
                for test_idx in test_indices:
                    purge_start = max(0, test_idx - self.purge_size)
                    purge_end = test_idx
                    purge_indices.extend(range(purge_start, purge_end))
                
                # Remove purged indices from training
                train_indices = np.array([idx for idx in train_indices if idx not in purge_indices])
            
            splits.append((train_indices, test_indices))
        
        return splits


class WalkForwardSplit:
    """Walk-forward validation split for financial data."""
    
    def __init__(self, train_size: int, test_size: int, step: int = 1):
        """
        Initialize WalkForwardSplit.
        
        Args:
            train_size: Number of samples in the training window
            test_size: Number of samples in the test window
            step: Number of samples to step forward for each split
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features DataFrame
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        splits = []
        for i in range(0, n_samples - self.train_size - self.test_size + 1, self.step):
            train_start = i
            train_end = i + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            # Get the indices for this split
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        return splits


class TimeSeriesDataset:
    """Dataset for time series prediction."""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5, 
                 step: int = 1, target_type: str = 'classification'):
        """
        Initialize TimeSeriesDataset.
        
        Args:
            sequence_length: Number of time steps in each sequence
            prediction_horizon: Number of time steps ahead to predict
            step: Step size between sequences
            target_type: Type of target ('classification', 'regression', 'both')
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.step = step
        self.target_type = target_type
    
    def create_sequences(self, features: pd.DataFrame, prices: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: DataFrame with features
            prices: DataFrame with price data (required for target calculation)
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        if prices is None:
            # If no price data is provided, assume features includes 'Close' column
            if 'Close' in features.columns:
                prices = features[['Close']]
            else:
                raise ValueError("Price data is required for target calculation")
        
        # Ensure features and prices have the same index
        features = features.loc[prices.index.intersection(features.index)]
        prices = prices.loc[features.index]
        
        # Create sequences
        X = []
        y = []
        
        for i in range(0, len(features) - self.sequence_length - self.prediction_horizon + 1, self.step):
            # Extract sequence
            seq = features.iloc[i:i+self.sequence_length].values
            
            # Extract target based on prediction horizon
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            
            if target_idx < len(prices):
                # Calculate target based on type
                if self.target_type == 'classification':
                    # Calculate price change
                    current_price = prices.iloc[i + self.sequence_length - 1]['Close']
                    future_price = prices.iloc[target_idx]['Close']
                    price_change = (future_price - current_price) / current_price
                    
                    # Convert to class: 0 = sell, 1 = hold, 2 = buy
                    if price_change < -0.01:  # -1% threshold for sell
                        target = 0
                    elif price_change > 0.01:  # +1% threshold for buy
                        target = 2
                    else:
                        target = 1
                
                elif self.target_type == 'regression':
                    # Target is future price change percentage
                    current_price = prices.iloc[i + self.sequence_length - 1]['Close']
                    future_price = prices.iloc[target_idx]['Close']
                    target = (future_price - current_price) / current_price
                
                else:  # 'both'
                    # Calculate both classification and regression targets
                    current_price = prices.iloc[i + self.sequence_length - 1]['Close']
                    future_price = prices.iloc[target_idx]['Close']
                    price_change = (future_price - current_price) / current_price
                    
                    # Classification target
                    if price_change < -0.01:
                        class_target = 0
                    elif price_change > 0.01:
                        class_target = 2
                    else:
                        class_target = 1
                    
                    # Regression target
                    reg_target = price_change
                    
                    target = (class_target, reg_target)
                
                X.append(seq)
                y.append(target)
        
        return np.array(X), np.array(y)
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                         shuffle: bool = False, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            X: Sequence data
            y: Target data
            test_size: Proportion of data to use for testing
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for shuffling
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if shuffle:
            logger.warning("Shuffling time series data may lead to data leakage")
        
        return train_test_split(X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)
    
    def time_series_split(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, 
                         test_size: float = 0.2, gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data using time series cross-validation.
        
        Args:
            X: Sequence data
            y: Target data
            n_splits: Number of splits
            test_size: Proportion of data to use for testing in each split
            gap: Number of samples to exclude between training and test sets
            
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        splits = tscv.split(np.arange(len(X)))
        
        result = []
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            result.append((X_train, X_test, y_train, y_test))
        
        return result
    
    def purged_time_series_split(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, 
                               test_size: float = 0.2, gap: int = 0, purge_size: int = 0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data using purged time series cross-validation.
        
        Args:
            X: Sequence data
            y: Target data
            n_splits: Number of splits
            test_size: Proportion of data to use for testing in each split
            gap: Number of samples to exclude between training and test sets
            purge_size: Number of samples to purge from training set before each test sample
            
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        ptscv = PurgedTimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap, purge_size=purge_size)
        splits = ptscv.split(np.arange(len(X)))
        
        result = []
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            result.append((X_train, X_test, y_train, y_test))
        
        return result
    
    def walk_forward_split(self, X: np.ndarray, y: np.ndarray, train_size: int, 
                         test_size: int, step: int = 1) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data using walk-forward validation.
        
        Args:
            X: Sequence data
            y: Target data
            train_size: Number of samples in the training window
            test_size: Number of samples in the test window
            step: Number of samples to step forward for each split
            
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        wfs = WalkForwardSplit(train_size=train_size, test_size=test_size, step=step)
        splits = wfs.split(np.arange(len(X)))
        
        result = []
        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            result.append((X_train, X_test, y_train, y_test))
        
        return result


def prepare_dataset(features: pd.DataFrame, prices: pd.DataFrame = None, 
                   sequence_length: int = 60, prediction_horizon: int = 5, 
                   step: int = 1, target_type: str = 'classification', 
                   test_size: float = 0.2, validation_size: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Prepare dataset for training, validation, and testing.
    
    Args:
        features: DataFrame with features
        prices: DataFrame with price data (required for target calculation)
        sequence_length: Number of time steps in each sequence
        prediction_horizon: Number of time steps ahead to predict
        step: Step size between sequences
        target_type: Type of target ('classification', 'regression', 'both')
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        
    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Create dataset
    dataset = TimeSeriesDataset(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        step=step,
        target_type=target_type
    )
    
    # Create sequences
    X, y = dataset.create_sequences(features, prices)
    
    # Split into train and test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Split train into train and validation
    if validation_size > 0:
        val_idx = int(len(X_train) * (1 - validation_size))
        X_val = X_train[val_idx:]
        y_val = y_train[val_idx:]
        X_train = X_train[:val_idx]
        y_train = y_train[:val_idx]
    else:
        X_val = np.array([])
        y_val = np.array([])
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }