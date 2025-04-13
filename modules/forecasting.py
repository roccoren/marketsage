#!/usr/bin/env python3
"""
Forecasting module for the Qlib-based quantitative analysis project.
This module provides classes and functions for forecasting stock prices
using various models and ensemble techniques.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from modules.storage import StorageManager


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


class LinearRegressionModel(ForecastModel):
    """Linear regression forecasting model."""
    
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'coef': self.model.coef_.tolist() if self.is_trained else None,
            'intercept': float(self.model.intercept_) if self.is_trained else None
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        if 'coef' in params and 'intercept' in params:
            self.model.coef_ = np.array(params['coef'])
            self.model.intercept_ = params['intercept']
            self.is_trained = True


class RandomForestModel(ForecastModel):
    """Random forest forecasting model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **kwargs):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'feature_importances': self.model.feature_importances_.tolist() if self.is_trained else None
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        if 'n_estimators' in params:
            self.model.n_estimators = params['n_estimators']
        if 'max_depth' in params:
            self.model.max_depth = params['max_depth']


class EnsembleModel(ForecastModel):
    """Ensemble forecasting model that combines multiple models."""
    
    def __init__(self, models: List[ForecastModel], weights: Optional[List[float]] = None):
        self.models = models
        
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
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        for model in self.models:
            model.train(X, y)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions.append(pred * self.weights[i])
        
        return np.sum(predictions, axis=0)
    
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
    
    @staticmethod
    def create_features(data: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        
        # Rolling statistics
        for window in window_sizes:
            # Price rolling statistics
            df[f'close_ma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window=window).std()
            
            # Volume rolling statistics
            df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Price momentum
        df['momentum_1d'] = df['Close'] / df['Close'].shift(1) - 1
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df


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
            'linear': lambda **kwargs: LinearRegressionModel(**kwargs),
            'random_forest': lambda **kwargs: RandomForestModel(**kwargs)
        }
    
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
            X_train, y_train = self.prepare_data(data)
            model.train(X_train, y_train)
        
        # Prepare data for forecasting
        X, _ = self.prepare_data(data)
        
        # Generate forecast for the next periods (2 weeks = 10 trading days)
        last_features = X.iloc[-1:].copy()
        forecasts = []
        
        for _ in range(self.forecast_horizon):
            # Predict next value
            next_value = model.predict(last_features)[0]
            forecasts.append(next_value)
            
            # This is a simplified approach; in a real implementation,
            # we would need to update all features based on the predicted value
        
        # Calculate forecast metrics
        last_price = data['Close'].iloc[-1]
        forecast_change = (forecasts[-1] / last_price - 1) * 100
        
        # Calculate daily forecast changes
        daily_changes = []
        prev_price = last_price
        for price in forecasts:
            daily_change = (price / prev_price - 1) * 100
            daily_changes.append(daily_change)
            prev_price = price
        
        # Determine recommended actions based on forecast
        # For simplicity: >1% daily gain = Strong Buy, >0.5% = Buy, <-1% = Strong Sell, <-0.5% = Sell, else = Hold
        actions = []
        for change in daily_changes:
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