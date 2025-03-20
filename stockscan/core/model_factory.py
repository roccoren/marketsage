"""
Model factory for creating and managing prediction models
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple

logger = logging.getLogger(__name__)

def create_model(model_type: str, **kwargs) -> Optional[Any]:
    """
    Create a model for stock prediction
    
    Parameters
    ----------
    model_type : str
        Type of model to create. Options: 'LightGBM', 'XGBoost', 'CatBoost', 'LSTM'
    **kwargs : dict
        Additional arguments for model creation
        
    Returns
    -------
    model : object
        Model object or None if model creation failed
    """
    model_type = model_type.lower()
    
    try:
        if model_type == 'lightgbm':
            return create_lightgbm_model(**kwargs)
        elif model_type == 'xgboost':
            return create_xgboost_model(**kwargs)
        elif model_type == 'catboost':
            return create_catboost_model(**kwargs)
        elif model_type == 'lstm':
            return create_lstm_model(**kwargs)
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return None
    except Exception as e:
        logger.warning(f"Error creating {model_type} model: {e}")
        return None

def create_lightgbm_model(**kwargs) -> Any:
    """
    Create a LightGBM model
    
    Parameters
    ----------
    **kwargs : dict
        Additional arguments for model creation
        
    Returns
    -------
    model : object
        LightGBM model
    """
    try:
        import lightgbm as lgb
        
        # Default parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'max_depth': 6
        }
        
        # Update with user-provided parameters
        params.update(kwargs)
        
        return lgb.LGBMRegressor(**params)
    except ImportError:
        logger.warning("LightGBM not installed. Install with: pip install lightgbm")
        return None

def create_xgboost_model(**kwargs) -> Any:
    """
    Create an XGBoost model
    
    Parameters
    ----------
    **kwargs : dict
        Additional arguments for model creation
        
    Returns
    -------
    model : object
        XGBoost model
    """
    try:
        import xgboost as xgb
        
        # Default parameters
        params = {
            'objective': 'reg:squarederror',
            'verbosity': 0,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Update with user-provided parameters
        params.update(kwargs)
        
        return xgb.XGBRegressor(**params)
    except ImportError:
        logger.warning("XGBoost not installed. Install with: pip install xgboost")
        return None

def create_catboost_model(**kwargs) -> Any:
    """
    Create a CatBoost model
    
    Parameters
    ----------
    **kwargs : dict
        Additional arguments for model creation
        
    Returns
    -------
    model : object
        CatBoost model
    """
    try:
        from catboost import CatBoostRegressor
        
        # Default parameters
        params = {
            'loss_function': 'RMSE',
            'verbose': False,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': 42
        }
        
        # Update with user-provided parameters
        params.update(kwargs)
        
        return CatBoostRegressor(**params)
    except ImportError:
        logger.warning("CatBoost not installed. Skip training with CatBoost.")
        print("ModuleNotFoundError. CatBoostModel are skipped. (optional: maybe installing CatBoostModel can fix it.)")
        return None

def create_lstm_model(**kwargs) -> Any:
    """
    Create an LSTM model
    
    Parameters
    ----------
    **kwargs : dict
        Additional arguments for model creation
        
    Returns
    -------
    model : object
        LSTM model
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        # Default parameters
        params = {
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mse'
        }
        
        # Update with user-provided parameters
        params.update(kwargs)
        
        model = Sequential()
        model.add(LSTM(
            units=params['units'],
            return_sequences=True,
            input_shape=(None, 1),
            dropout=params['dropout'],
            recurrent_dropout=params['recurrent_dropout'],
            activation=params['activation']
        ))
        model.add(LSTM(units=params['units'] // 2))
        model.add(Dense(1))
        
        model.compile(
            optimizer=params['optimizer'],
            loss=params['loss']
        )
        
        return model
    except ImportError:
        logger.warning("TensorFlow not installed. Skip training with LSTM.")
        print("ModuleNotFoundError.  PyTorch models are skipped (optional: maybe installing pytorch can fix it).")
        return None 