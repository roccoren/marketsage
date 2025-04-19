#!/usr/bin/env python3
"""
Base classes and interfaces for the trading prediction module.

This module defines abstract base classes and interfaces that serve as the foundation
for the trading prediction system, ensuring consistent APIs across different implementations.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical market data for a symbol.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
            
        Returns:
            DataFrame with historical data (datetime index, OHLCV columns)
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from this data source.
        
        Returns:
            List of available ticker symbols
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this data source.
        
        Returns:
            Data source name
        """
        pass


class MarketAdapter(ABC):
    """Abstract base class for market-specific adapters."""
    
    @abstractmethod
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of trading days for this market.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of trading days in YYYY-MM-DD format
        """
        pass
    
    @abstractmethod
    def is_trading_day(self, date: str) -> bool:
        """
        Check if a given date is a trading day in this market.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            True if trading day, False otherwise
        """
        pass
    
    @abstractmethod
    def get_market_hours(self) -> Dict[str, Any]:
        """
        Get market hours information.
        
        Returns:
            Dictionary with market hours information
        """
        pass
    
    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for this market.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Normalized symbol
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this market.
        
        Returns:
            Market name
        """
        pass


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering."""
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw market data.
        
        Args:
            data: DataFrame with raw market data
            
        Returns:
            DataFrame with engineered features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        pass


class ModelBase(ABC):
    """Abstract base class for prediction models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Train the model on historical data.
        
        Args:
            X: Features DataFrame
            y: Target DataFrame (can contain multiple columns for multi-output)
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Features DataFrame
            
        Returns:
            numpy.ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict with model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters.
        
        Args:
            params: Dictionary with parameters
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """
        Check if the model is trained.
        
        Returns:
            True if trained, False otherwise
        """
        pass


class SignalGenerator(ABC):
    """Abstract base class for trading signal generators."""
    
    @abstractmethod
    def generate_signals(self, predictions: np.ndarray, confidence: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate trading signals from model predictions.
        
        Args:
            predictions: Model predictions
            confidence: Prediction confidence (optional)
            
        Returns:
            DataFrame with trading signals
        """
        pass
    
    @abstractmethod
    def set_thresholds(self, buy_threshold: float, sell_threshold: float) -> None:
        """
        Set thresholds for signal generation.
        
        Args:
            buy_threshold: Threshold for buy signals
            sell_threshold: Threshold for sell signals
        """
        pass


class BacktestEngine(ABC):
    """Abstract base class for backtesting engines."""
    
    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest with trading signals.
        
        Args:
            data: Historical market data
            signals: Trading signals
            
        Returns:
            Dictionary with backtest results
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set backtest parameters.
        
        Args:
            params: Dictionary with parameters
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics from backtest.
        
        Returns:
            Dictionary with performance metrics
        """
        pass


class ModelEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    @abstractmethod
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def evaluate_trading(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate trading performance.
        
        Args:
            backtest_results: Results from backtest
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass


class ModelInterpreter(ABC):
    """Abstract base class for model interpreters."""
    
    @abstractmethod
    def get_feature_importance(self, model: ModelBase, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance from model.
        
        Args:
            model: Trained model
            X: Features DataFrame
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def explain_prediction(self, model: ModelBase, X: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Explain a specific prediction.
        
        Args:
            model: Trained model
            X: Features DataFrame
            index: Index of the prediction to explain
            
        Returns:
            Dictionary with explanation
        """
        pass
    
    @abstractmethod
    def visualize_interpretation(self, explanation: Dict[str, Any], save_path: Optional[str] = None) -> Any:
        """
        Visualize model interpretation.
        
        Args:
            explanation: Explanation from explain_prediction
            save_path: Path to save visualization (optional)
            
        Returns:
            Visualization object
        """
        pass


class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers."""
    
    @abstractmethod
    def optimize(self, model_class: type, param_space: Dict[str, Any], 
                X: pd.DataFrame, y: pd.DataFrame, 
                n_trials: int = 100, timeout: Optional[int] = None) -> Tuple[Dict[str, Any], Any]:
        """
        Optimize hyperparameters.
        
        Args:
            model_class: Model class to optimize
            param_space: Hyperparameter search space
            X: Features DataFrame
            y: Target DataFrame
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            
        Returns:
            Tuple of (best parameters, best model)
        """
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history.
        
        Returns:
            DataFrame with optimization history
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best parameters from optimization.
        
        Returns:
            Dictionary with best parameters
        """
        pass