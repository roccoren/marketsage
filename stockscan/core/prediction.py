"""
Prediction module for making stock predictions
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)

def predict_stock_movement(model: Any, data: pd.DataFrame, days_ahead: int = 1, window: int = 30) -> Optional[float]:
    """
    Predict stock movement using a trained model
    
    Parameters
    ----------
    model : object
        Trained prediction model
    data : pd.DataFrame
        Historical stock data
    days_ahead : int
        Number of days ahead to predict
    window : int
        Number of days to use for feature creation
        
    Returns
    -------
    prediction : float
        Predicted stock movement (positive = up, negative = down)
    """
    if data.empty:
        logger.warning("Empty data provided for prediction")
        return None
    
    try:
        # Prepare features for prediction
        X, y = prepare_features(data, days_ahead, window)
        
        if X.empty:
            logger.warning("No features could be created for prediction")
            return None
        
        # Check if model was trained with a subset of these features
        # This is a common issue when predicting with a model trained on a different feature set
        if hasattr(model, 'feature_name_') and model.feature_name_ is not None:
            # LightGBM style feature names
            required_features = model.feature_name_
            logger.debug(f"Model was trained with features: {required_features}")
            
            # Check if we have all required features
            missing_features = [f for f in required_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features required by the model: {missing_features}")
                return None
            
            # Use only the features the model was trained with
            X = X[required_features]
        elif hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            # Scikit-learn style feature names
            required_features = model.feature_names_in_
            logger.debug(f"Model was trained with features: {required_features}")
            
            # Check if we have all required features
            missing_features = [f for f in required_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features required by the model: {missing_features}")
                return None
            
            # Use only the features the model was trained with
            X = X[required_features]
        
        # Check model type and make prediction accordingly
        if hasattr(model, 'predict'):
            # For sklearn-like models
            pred = model.predict(X.iloc[-1:])
            return float(pred[0])
        elif str(type(model)).find('keras') != -1:
            # For Keras models (LSTM)
            # Reshape data for LSTM [samples, timesteps, features]
            X_lstm = X.iloc[-window:].values.reshape(1, window, X.shape[1])
            pred = model.predict(X_lstm)
            return float(pred[0][0])
        else:
            logger.warning(f"Unsupported model type: {type(model)}")
            return None
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None

def prepare_features(data: pd.DataFrame, days_ahead: int = 1, window: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features for prediction
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical stock data
    days_ahead : int
        Number of days ahead to predict
    window : int
        Number of days to use for feature creation
        
    Returns
    -------
    X : pd.DataFrame
        Feature data
    y : pd.DataFrame
        Target data
    """
    try:
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Log the dataframe structure for debugging
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        
        # Reset index if it's a DatetimeIndex to have it as a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Ensure there's enough data
        if len(df) < window + days_ahead:
            logger.warning(f"Not enough data for feature creation: {len(df)} rows available, {window + days_ahead} required")
            return pd.DataFrame(), pd.DataFrame()
        
        # Check if we have the necessary technical indicators
        required_indicators = ['ma5', 'ma10', 'ma20', 'rsi', 'macd']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
        # If indicators are missing, calculate them
        if missing_indicators:
            logger.info(f"Calculating missing indicators: {missing_indicators}")
            
            # Ensure we have required price data
            if 'close' not in df.columns:
                # Look for column names that might contain 'close'
                close_cols = [col for col in df.columns if 'close' in str(col).lower()]
                if close_cols:
                    df['close'] = df[close_cols[0]]
                else:
                    logger.error("No close price column found, cannot calculate indicators")
                    return pd.DataFrame(), pd.DataFrame()
            
            # Calculate missing indicators
            if 'ma5' in missing_indicators:
                df['ma5'] = df['close'].rolling(window=5).mean()
            if 'ma10' in missing_indicators:
                df['ma10'] = df['close'].rolling(window=10).mean()
            if 'ma20' in missing_indicators:
                df['ma20'] = df['close'].rolling(window=20).mean()
            
            if 'rsi' in missing_indicators:
                # Calculate RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss.replace(0, 1e-9)  # Avoid division by zero
                df['rsi'] = 100 - (100 / (1 + rs))
            
            if 'macd' in missing_indicators:
                # Calculate MACD
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
        
        # Create target variable: price change X days ahead
        if 'close' in df.columns:
            df['target'] = df['close'].pct_change(days_ahead).shift(-days_ahead)
        
        # Create additional derived features
        feature_columns = []
        
        # Add the base technical indicators
        for col in required_indicators:
            if col in df.columns:
                feature_columns.append(col)
        
        # Add derived features based on what we have
        if all(col in df.columns for col in ['ma5', 'ma10']):
            df['ma5_10_cross'] = df['ma5'] - df['ma10']
            feature_columns.append('ma5_10_cross')
            
        if all(col in df.columns for col in ['ma10', 'ma20']):
            df['ma10_20_cross'] = df['ma10'] - df['ma20']
            feature_columns.append('ma10_20_cross')
            
        if 'close' in df.columns and 'ma5' in df.columns:
            df['ma5_diff'] = (df['close'] - df['ma5']) / df['ma5'].replace(0, 1e-9)
            feature_columns.append('ma5_diff')
            
        if 'close' in df.columns and 'ma10' in df.columns:
            df['ma10_diff'] = (df['close'] - df['ma10']) / df['ma10'].replace(0, 1e-9)
            feature_columns.append('ma10_diff')
            
        if 'close' in df.columns and 'ma20' in df.columns:
            df['ma20_diff'] = (df['close'] - df['ma20']) / df['ma20'].replace(0, 1e-9)
            feature_columns.append('ma20_diff')
            
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            feature_columns.append('volume_change')
        
        # Fill missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        
        # Ensure all feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Some features are missing: {missing_features}")
            for col in missing_features:
                feature_columns.remove(col)
        
        if not feature_columns:
            logger.error("No valid features available for prediction")
            return pd.DataFrame(), pd.DataFrame()
        
        # Extract X and y
        X = df[feature_columns].copy()
        
        if 'target' in df.columns:
            y = df['target'].copy()
        else:
            # If we couldn't create a target, use a dummy
            logger.warning("Could not create target variable, using dummy")
            y = pd.Series(0, index=df.index)
        
        # Check for NaN values
        if X.isna().any().any():
            logger.warning(f"Feature dataframe contains NaN values, filling with zeros")
            X = X.fillna(0)
            
        if y.isna().any():
            logger.warning(f"Target variable contains NaN values, filling with zeros")
            y = y.fillna(0)
        
        # Align X and y
        if days_ahead > 0 and len(y) >= days_ahead:
            # Remove the last days_ahead rows from X since we don't have their targets
            X = X.iloc[:-days_ahead].copy()
            y = y.iloc[:-days_ahead].copy()
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        return X, y
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        logger.error(traceback.format_exc())  # Print full traceback
        return pd.DataFrame(), pd.DataFrame()

def evaluate_prediction(prediction: float) -> Dict[str, Any]:
    """
    Evaluate a prediction and provide interpretations
    
    Parameters
    ----------
    prediction : float
        Predicted stock movement
        
    Returns
    -------
    evaluation : dict
        Dictionary with evaluation metrics
    """
    try:
        result = {
            'prediction': prediction,
            'direction': 'up' if prediction > 0 else 'down',
            'strength': abs(prediction),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Determine signal strength
        if abs(prediction) > 0.02:
            result['signal'] = 'STRONG ' + ('BUY' if prediction > 0 else 'SELL')
        elif abs(prediction) > 0.01:
            result['signal'] = 'BUY' if prediction > 0 else 'SELL'
        elif abs(prediction) > 0.005:
            result['signal'] = 'WEAK ' + ('BUY' if prediction > 0 else 'SELL')
        else:
            result['signal'] = 'NEUTRAL'
        
        # Confidence level
        if abs(prediction) > 0.03:
            result['confidence'] = 'Very High'
        elif abs(prediction) > 0.02:
            result['confidence'] = 'High'
        elif abs(prediction) > 0.01:
            result['confidence'] = 'Medium'
        else:
            result['confidence'] = 'Low'
        
        return result
    except Exception as e:
        logger.error(f"Error evaluating prediction: {e}")
        return {'prediction': prediction, 'error': str(e)} 