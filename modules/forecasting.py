#!/usr/bin/env python3
"""
Forecasting module for the Qlib-based quantitative analysis project.
This module provides classes and functions for forecasting stock prices
using Qlib's model framework and ensemble techniques.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
# Import standard ML libraries instead of qlib
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
# LSTM model import temporarily commented out
# from qlib.contrib.model.pytorch.lstm import LSTMModel

from modules.storage import StorageManager

# Set global random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ForecastModel(ABC):
    """Abstract base class for forecasting models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on historical data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the trained model."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        pass


class LGBModel(ForecastModel):
    """LightGBM model using scikit-learn API."""
    
    def __init__(self, **kwargs):
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'n_estimators': 100,
            'random_state': 42  # Fixed random_state for reproducibility
        }
        self.params.update(kwargs)
        self.model = lgb.LGBMRegressor(**self.params)
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
    
    def set_params(self, params: Dict[str, Any]) -> None:
        self.model.set_params(**params)


# LSTM model class temporarily commented out
# class QlibLSTMModel(ForecastModel):
#     """LSTM model using Qlib's implementation."""
#
#     def __init__(self, **kwargs):
#         self.model = LSTMModel(**kwargs)
#         self.is_trained = False
#
#     def train(self, X: pd.DataFrame, y: pd.Series) -> None:
#         # Start recording with Qlib's workflow
#         with R.start(experiment_name="train_lstm"):
#             self.model.fit(X, y)
#             self.is_trained = True
#
#             # Log prediction scores
#             pred = self.model.predict(X)
#             SignalRecord(model=self.model).log_pred(pred)
#
#     def predict(self, X: pd.DataFrame) -> np.ndarray:
#         if not self.is_trained:
#             raise ValueError("Model must be trained before making predictions")
#         return self.model.predict(X)
#
#     def get_params(self) -> Dict[str, Any]:
#         return self.model.get_params()
#
#     def set_params(self, params: Dict[str, Any]) -> None:
#         self.model.set_params(**params)


class EnsembleModel(ForecastModel):
    """Ensemble forecasting model that combines multiple models."""
    
    def __init__(self, models: List[ForecastModel], weights: Optional[List[float]] = None,
                 dynamic_weighting: bool = True, error_metric: str = 'rmse'):
        """
        Initialize EnsembleModel.
        
        Args:
            models (List[ForecastModel]): List of models to ensemble
            weights (Optional[List[float]]): Initial weights for models
            dynamic_weighting (bool): Whether to use dynamic weighting based on performance
            error_metric (str): Metric for evaluating model performance ('rmse', 'mae', 'mape')
        """
        self.models = models
        self.dynamic_weighting = dynamic_weighting
        self.error_metric = error_metric
        
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.is_trained = False
        self.model_errors = None  # Store historical prediction errors
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on historical data using cross-validation."""
        # Reset random seeds before training
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        # Use TimeSeriesSplit for time series cross-validation
        # TimeSeriesSplit is deterministic as it doesn't shuffle data
        tscv = TimeSeriesSplit(n_splits=5)
        model_errors = [[] for _ in range(len(self.models))]
        
        # Train and evaluate each model using cross-validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train each model and calculate validation error
            for i, model in enumerate(self.models):
                model.train(X_train, y_train)
                val_pred = model.predict(X_val)
                error = self._calculate_error(y_val, val_pred)
                model_errors[i].append(error)
        
        # Calculate average error for each model
        self.model_errors = [np.mean(errors) for errors in model_errors]
        
        # Update weights based on model performance
        if self.dynamic_weighting:
            self._update_weights()
        
        # Finally, train all models on the full dataset
        for model in self.models:
            model.train(X, y)
            
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate weighted predictions from all models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = np.zeros((len(self.models), len(X)))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X) * self.weights[i]
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred
    
    def _calculate_error(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate error metric for model evaluation."""
        if self.error_metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.error_metric == 'mae':
            return np.abs(y_true - y_pred).mean()
        elif self.error_metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _update_weights(self) -> None:
        """Update model weights based on their performance."""
        if self.model_errors is None or len(self.model_errors) == 0:
            return
        
        # Convert errors to weights (lower error = higher weight)
        max_error = max(self.model_errors)
        weights = [max_error - error for error in self.model_errors]
        
        # Add small constant to avoid zero weights
        weights = [w + 1e-6 for w in weights]
        
        # Normalize weights to sum to 1
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def get_params(self) -> Dict[str, Any]:
        model_params = []
        for i, model in enumerate(self.models):
            model_params.append({
                'type': model.__class__.__name__,
                'weight': self.weights[i],
                'params': model.get_params()
            })
        return {'models': model_params}
    
    def set_params(self, params: Dict[str, Any]) -> None:
        if 'models' in params:
            model_params = params['models']
            if len(model_params) != len(self.models):
                raise ValueError("Number of model parameters must match number of models")
            for i, model_param in enumerate(model_params):
                if 'weight' in model_param:
                    self.weights[i] = model_param['weight']
                if 'params' in model_param:
                    self.models[i].set_params(model_param['params'])


class FeatureEngineering:
    """Feature engineering for time series forecasting."""
    
    def create_features(self, data: pd.DataFrame, window_sizes: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
        """Create technical features for time series forecasting."""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling statistics for multiple windows
        for window in window_sizes:
            # Price statistics
            df[f'close_ma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'close_min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['Close'].rolling(window=window).max()
            
            # Return statistics
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'returns_skew_{window}'] = df['returns'].rolling(window=window).skew()
            
            # Volume statistics
            df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'volume_std_{window}'] = df['Volume'].rolling(window=window).std()
            
            # Price momentum and volatility
            df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
            df[f'volatility_{window}d'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Technical indicators
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands - Simplified calculation to avoid DataFrame issues
        # Make sure all values are Series, not DataFrame
        close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        mid = close_series.rolling(window=20).mean()
        std = close_series.rolling(window=20).std()
        
        # Store as Series
        df['bb_upper'] = mid + 2 * std
        df['bb_lower'] = mid - 2 * std
        # Calculate width directly using Series
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / mid
        
        # Volume-price relationship
        df['volume_price_corr'] = df['Close'].rolling(window=20).corr(df['Volume'])
        df['volume_returns_corr'] = df['returns'].rolling(window=20).corr(df['Volume'])
        
        # Trend strength
        df['adx'] = FeatureEngineering._calculate_adx(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        df = data.copy()
        
        # Calculate True Range (TR)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Calculate Directional Movement
        df['DMplus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        df['DMminus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        # Calculate smoothed averages
        df['TR_avg'] = df['TR'].rolling(window=period).mean()
        df['DMplus_avg'] = df['DMplus'].rolling(window=period).mean()
        df['DMminus_avg'] = df['DMminus'].rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        df['DIplus'] = 100 * df['DMplus_avg'] / df['TR_avg']
        df['DIminus'] = 100 * df['DMminus_avg'] / df['TR_avg']
        
        # Calculate ADX
        df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
        adx = df['DX'].rolling(window=period).mean()
        
        return adx


class ForecastingManager:
    """
    Manager for forecasting operations.
    Handles model training, prediction, and optimization.
    """
    
    def __init__(self, storage_manager: Optional[StorageManager] = None, forecast_horizon: int = 10):
        self.storage_manager = storage_manager if storage_manager else StorageManager()
        self.forecast_horizon = forecast_horizon
        self.feature_engineering = FeatureEngineering()
        
        # Available models
        self.model_factories = {
            'lgb': lambda **kwargs: LGBModel(**kwargs),
            'deep': lambda **kwargs: self._create_deep_model(**kwargs),
            # LSTM model temporarily disabled
            # 'lstm': lambda **kwargs: QlibLSTMModel(**kwargs)
        }
    
    def _create_deep_model(self, **kwargs):
        """
        Create a deep learning model instance.
        This method is used to avoid circular imports.
        
        Args:
            **kwargs: Parameters to pass to the DeepLearningModel constructor
            
        Returns:
            DeepLearningModel instance
        """
        # Import here to avoid circular imports
        from modules.deep_learning_model import DeepLearningModel
        return DeepLearningModel(**kwargs)
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Create features
        features_df = self.feature_engineering.create_features(data)
        
        # Use next day's close price as the target
        features_df['target'] = features_df['Close'].shift(-1)
        
        # Drop rows with NaN target
        features_df = features_df.dropna()
        
        # Separate features and target
        X = features_df.drop(['target', 'Open', 'High', 'Low', 'Close', 'Volume', 'returns'], axis=1)
        y = features_df['target']
        # Clean feature names for LightGBM - replace special characters
        X = X.copy()
        clean_columns = []
        for col in X.columns:
            if isinstance(col, str):
                clean_col = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_') \
                              .replace('(', '_').replace(')', '_').replace(':', '_').replace(' ', '_') \
                              .replace(',', '_').replace('/', '_div_').replace('\\', '_')
            else:
                # If column is not a string (e.g., tuple), convert to string
                clean_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_') \
                              .replace('(', '_').replace(')', '_').replace(':', '_').replace(' ', '_') \
                              .replace(',', '_').replace('/', '_div_').replace('\\', '_')
            clean_columns.append(clean_col)
        
        X.columns = clean_columns
        
        return X, y
    
    def train_model(self, symbol: str, data: pd.DataFrame, model_type: str = 'ensemble', 
                    save: bool = True) -> ForecastModel:
        X, y = self.prepare_data(data)
        
        if model_type == 'ensemble':
            # Create an ensemble of all available models
            models = []
            for model_name, model_factory in self.model_factories.items():
                model = model_factory()
                models.append(model)
            
            model = EnsembleModel(models)
        else:
            if model_type not in self.model_factories:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model_factory = self.model_factories[model_type]
            model = model_factory()
        
        # Train the model
        model.train(X, y)
        
        # Save model parameters if requested
        if save and self.storage_manager:
            self.storage_manager.save_model_params(symbol, model_type, model.get_params())
        
        return model
    
    def generate_forecast(self, symbol: str, data: pd.DataFrame, model_type: str = 'ensemble',
                          use_stored_model: bool = True) -> Dict[str, Any]:
        # Try to load model parameters if requested
        model = None
        if use_stored_model and self.storage_manager:
            params = self.storage_manager.load_model_params(symbol, model_type)
            
            if params:
                # Create model with stored parameters
                if model_type == 'ensemble':
                    # Create ensemble model from stored parameters
                    models = []
                    weights = []
                    
                    for model_info in params.get('models', []):
                        model_type_name = model_info.get('type')
                        model_params = model_info.get('params', {})
                        weight = model_info.get('weight', 1.0)
                        
                        for factory_name, factory in self.model_factories.items():
                            if factory_name in model_type_name.lower():
                                model = factory()
                                model.set_params(model_params)
                                models.append(model)
                                weights.append(weight)
                                break
                    
                    if models:
                        model = EnsembleModel(models, weights)
                else:
                    # Create single model from stored parameters
                    if model_type in self.model_factories:
                        model = self.model_factories[model_type]()
                        model.set_params(params)
        
        # Train a new model if no stored model was found or loaded
        if model is None:
            model = self.train_model(symbol, data, model_type, save=True)
        # Ensure the model is trained even if loaded from stored parameters
        if not model.is_trained:
            X, y = self.prepare_data(data)
            model.train(X, y)
        
        # Prepare data for forecasting
        X, _ = self.prepare_data(data)
        
        # Generate forecast for the next periods (2 weeks = 10 trading days)
        last_features = X.iloc[-1:].copy()
        forecasts = []
        
        # Reset random seeds before generating forecasts for reproducibility
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
        for _ in range(self.forecast_horizon):
            try:
                # Predict next value
                predictions = model.predict(last_features)
                # Handle empty or scalar predictions
                if isinstance(predictions, np.ndarray) and len(predictions) > 0:
                    next_value = predictions[0]
                else:
                    next_value = predictions  # Handle scalar value
                forecasts.append(next_value)
            except Exception as e:
                import logging
                logging.error(f"Error in prediction: {e}")
                # Use last prediction or last price if no predictions yet
                if forecasts:
                    forecasts.append(forecasts[-1])
                else:
                    forecasts.append(last_price)
            
            # This is a simplified approach; in a real implementation,
            # we would need to update all features based on the predicted value
        
        # Calculate forecast metrics
        # Ensure last_price is a scalar value
        last_price = data['Close'].iloc[-1].item() if isinstance(data['Close'].iloc[-1], pd.Series) else data['Close'].iloc[-1]
        forecast_change = (forecasts[-1] / last_price - 1) * 100
        
        # Calculate daily forecast changes
        daily_changes = []
        prev_price = last_price
        for price in forecasts:
            # Ensure price is a scalar value
            if isinstance(price, (pd.Series, np.ndarray)):
                price = price.item() if hasattr(price, 'item') else price[0]
            daily_change = (price / prev_price - 1) * 100
            daily_changes.append(float(daily_change))  # Convert to float to ensure scalar value
            prev_price = price
        
        # Determine recommended actions based on forecast
        # For simplicity: >1% daily gain = Strong Buy, >0.5% = Buy, <-1% = Strong Sell, <-0.5% = Sell, else = Hold
        actions = []
        for change in daily_changes:
            # Ensure change is a scalar value, not a Series
            if isinstance(change, pd.Series):
                change = change.iloc[0]  # Get the first value if it's a Series
                
            if change > 1.0:
                actions.append("Strong Buy")
            elif change > 0.5:
                actions.append("Buy")
            elif change < -1.0:
                actions.append("Strong Sell")
            elif change < -0.5:
                actions.append("Sell")
            else:
                actions.append("Hold")
        
        # Calculate volatility and trend indicators
        volatility = np.std(daily_changes)
        trend = np.mean(daily_changes)
        
        # Determine overall recommendation
        if trend > 0.5 and volatility < 1.0:
            overall_action = "Buy - Positive trend with low volatility"
        elif trend > 0.5 and volatility >= 1.0:
            overall_action = "Cautious Buy - Positive trend but high volatility"
        elif trend < -0.5 and volatility < 1.0:
            overall_action = "Sell - Negative trend with low volatility"
        elif trend < -0.5 and volatility >= 1.0:
            overall_action = "Cautious Sell - Negative trend but high volatility"
        else:
            overall_action = "Hold - No strong trend detected"
        
        return {
            'symbol': symbol,
            'last_price': last_price,
            'forecast_values': forecasts,
            'forecast_horizon': self.forecast_horizon,
            'forecast_change_percent': forecast_change,
            'daily_changes': daily_changes,
            'daily_actions': actions,
            'volatility': volatility,
            'trend': trend,
            'overall_action': overall_action,
            'model_type': model_type
        }


# For backward compatibility with existing code
def forecast_stock(data):
    """
    Generate a two-week forecast for the provided stock data.
    This function is maintained for backward compatibility.
    """
    try:
        if data.empty:
            return None
        
        # Use the new forecasting manager
        manager = ForecastingManager(forecast_horizon=10)
        result = manager.generate_forecast("unknown", data, use_stored_model=False)
        
        return result['forecast_values']
    except Exception as e:
        print(f"Error forecasting stock data: {e}")
        return None