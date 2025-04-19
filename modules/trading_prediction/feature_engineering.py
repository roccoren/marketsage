#!/usr/bin/env python3
"""
Feature engineering module for stock trading prediction.

This module provides functionality to transform raw market data into features
suitable for deep learning models, including technical indicators, price-based features,
volume-based features, and more.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Try to import TA-Lib, but provide fallbacks if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Using pandas-ta fallbacks.")
    try:
        import pandas_ta as ta
    except ImportError:
        logging.warning("pandas-ta not available. Using custom implementations.")

from modules.trading_prediction.base import FeatureEngineer

# Configure logger
logger = logging.getLogger(__name__)


class BaseFeatureEngineer(FeatureEngineer):
    """Base implementation for feature engineering."""
    
    def __init__(self):
        """Initialize BaseFeatureEngineer."""
        self._feature_functions = {}
        self._feature_names = []
        
        # Register default feature functions
        self._register_default_features()
    
    def _register_default_features(self):
        """Register default feature functions."""
        # Price-based features
        self.register_feature("returns", self._calculate_returns)
        self.register_feature("log_returns", self._calculate_log_returns)
        
        # Moving averages
        self.register_feature("sma_5", lambda df: self._calculate_sma(df, 5))
        self.register_feature("sma_20", lambda df: self._calculate_sma(df, 20))
        self.register_feature("sma_50", lambda df: self._calculate_sma(df, 50))
        
        self.register_feature("ema_5", lambda df: self._calculate_ema(df, 5))
        self.register_feature("ema_20", lambda df: self._calculate_ema(df, 20))
        
        # Technical indicators
        self.register_feature("rsi_14", lambda df: self._calculate_rsi(df, 14))
        self.register_feature("macd", self._calculate_macd)
        self.register_feature("bollinger_bands", self._calculate_bollinger_bands)
    
    def register_feature(self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """
        Register a feature function.
        
        Args:
            name: Feature name
            func: Function that calculates the feature
        """
        self._feature_functions[name] = func
        if name not in self._feature_names:
            self._feature_names.append(name)
    
    def unregister_feature(self, name: str) -> None:
        """
        Unregister a feature function.
        
        Args:
            name: Feature name
        """
        if name in self._feature_functions:
            del self._feature_functions[name]
        
        if name in self._feature_names:
            self._feature_names.remove(name)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.
        
        Returns:
            List of feature names
        """
        return self._feature_names.copy()
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw market data.
        
        Args:
            data: DataFrame with raw market data (OHLCV)
            
        Returns:
            DataFrame with engineered features
        """
        if data is None or data.empty:
            logger.warning("Empty data provided to feature engineering")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} missing from data")
                return pd.DataFrame()
        
        # Calculate returns first as many features depend on it
        df['returns'] = df['Close'].pct_change()
        
        # Apply each feature function
        features = pd.DataFrame(index=df.index)
        
        for name, func in self._feature_functions.items():
            try:
                feature_df = func(df)
                
                # If the function returns a DataFrame, merge it
                if isinstance(feature_df, pd.DataFrame):
                    # Avoid duplicate columns
                    for col in feature_df.columns:
                        if col not in features.columns:
                            features[col] = feature_df[col]
                # If it returns a Series, add it as a column
                elif isinstance(feature_df, pd.Series):
                    features[name] = feature_df
                # Otherwise, skip it
                else:
                    logger.warning(f"Feature function {name} returned an invalid type: {type(feature_df)}")
            except Exception as e:
                logger.error(f"Error calculating feature {name}: {e}")
        
        # Drop rows with NaN values (typically at the beginning due to lookback periods)
        features = features.dropna()
        
        return features
    
    # Price-based feature functions
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns."""
        returns = pd.DataFrame(index=df.index)
        returns['returns'] = df['Close'].pct_change()
        return returns
    
    def _calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate logarithmic returns."""
        log_returns = pd.DataFrame(index=df.index)
        log_returns['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        return log_returns
    
    def _calculate_sma(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Simple Moving Average."""
        sma = pd.DataFrame(index=df.index)
        sma[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        sma[f'sma_{window}_ratio'] = df['Close'] / sma[f'sma_{window}']
        return sma
    
    def _calculate_ema(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Exponential Moving Average."""
        ema = pd.DataFrame(index=df.index)
        ema[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        ema[f'ema_{window}_ratio'] = df['Close'] / ema[f'ema_{window}']
        return ema
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        if TALIB_AVAILABLE:
            try:
                rsi = pd.DataFrame(index=df.index)
                rsi[f'rsi_{window}'] = talib.RSI(df['Close'].values, timeperiod=window)
                return rsi
            except Exception as e:
                logger.warning(f"Error using TA-Lib for RSI: {e}. Using fallback.")
        
        try:
            # Try pandas-ta if available
            import pandas_ta as ta
            rsi = pd.DataFrame(index=df.index)
            rsi[f'rsi_{window}'] = ta.rsi(df['Close'], length=window)
            return rsi
        except ImportError:
            # Custom implementation
            rsi = pd.DataFrame(index=df.index)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
            
            # Calculate RS
            rs = gain / loss
            
            # Calculate RSI
            rsi[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            return rsi
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD."""
        if TALIB_AVAILABLE:
            try:
                macd = pd.DataFrame(index=df.index)
                macd_line, signal_line, histogram = talib.MACD(
                    df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9
                )
                macd['macd'] = macd_line
                macd['macd_signal'] = signal_line
                macd['macd_hist'] = histogram
                return macd
            except Exception as e:
                logger.warning(f"Error using TA-Lib for MACD: {e}. Using fallback.")
        
        try:
            # Try pandas-ta if available
            import pandas_ta as ta
            macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            
            # pandas-ta returns a DataFrame with specific column names
            macd = pd.DataFrame(index=df.index)
            macd['macd'] = macd_result[f'MACD_12_26_9']
            macd['macd_signal'] = macd_result[f'MACDs_12_26_9']
            macd['macd_hist'] = macd_result[f'MACDh_12_26_9']
            return macd
        except ImportError:
            # Custom implementation
            macd = pd.DataFrame(index=df.index)
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd['macd'] = ema12 - ema26
            macd['macd_signal'] = macd['macd'].ewm(span=9, adjust=False).mean()
            macd['macd_hist'] = macd['macd'] - macd['macd_signal']
            return macd
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        if TALIB_AVAILABLE:
            try:
                bb = pd.DataFrame(index=df.index)
                upper, middle, lower = talib.BBANDS(
                    df['Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                bb['bb_upper'] = upper
                bb['bb_middle'] = middle
                bb['bb_lower'] = lower
                bb['bb_width'] = (upper - lower) / middle
                bb['bb_pct'] = (df['Close'] - lower) / (upper - lower)
                return bb
            except Exception as e:
                logger.warning(f"Error using TA-Lib for Bollinger Bands: {e}. Using fallback.")
        
        try:
            # Try pandas-ta if available
            import pandas_ta as ta
            bb_result = ta.bbands(df['Close'], length=20, std=2)
            
            # pandas-ta returns a DataFrame with specific column names
            bb = pd.DataFrame(index=df.index)
            bb['bb_upper'] = bb_result[f'BBU_20_2.0']
            bb['bb_middle'] = bb_result[f'BBM_20_2.0']
            bb['bb_lower'] = bb_result[f'BBL_20_2.0']
            bb['bb_width'] = (bb['bb_upper'] - bb['bb_lower']) / bb['bb_middle']
            bb['bb_pct'] = (df['Close'] - bb['bb_lower']) / (bb['bb_upper'] - bb['bb_lower'])
            return bb
        except ImportError:
            # Custom implementation
            bb = pd.DataFrame(index=df.index)
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            bb['bb_upper'] = sma20 + (std20 * 2)
            bb['bb_middle'] = sma20
            bb['bb_lower'] = sma20 - (std20 * 2)
            bb['bb_width'] = (bb['bb_upper'] - bb['bb_lower']) / bb['bb_middle']
            bb['bb_pct'] = (df['Close'] - bb['bb_lower']) / (bb['bb_upper'] - bb['bb_lower'])
            return bb


class EnhancedFeatureEngineer(BaseFeatureEngineer):
    """Enhanced feature engineering with additional features."""
    
    def __init__(self):
        """Initialize EnhancedFeatureEngineer."""
        super().__init__()
        
        # Register additional features
        self._register_enhanced_features()
    
    def _register_enhanced_features(self):
        """Register enhanced features."""
        # Additional technical indicators
        self.register_feature("adx_14", lambda df: self._calculate_adx(df, 14))
        self.register_feature("obv", self._calculate_obv)
        
        # Market regime features
        self.register_feature("market_phase", self._calculate_market_phase)
    
    def _calculate_adx(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        if TALIB_AVAILABLE:
            try:
                adx = pd.DataFrame(index=df.index)
                adx[f'adx_{window}'] = talib.ADX(
                    df['High'].values, df['Low'].values, df['Close'].values, timeperiod=window
                )
                return adx
            except Exception as e:
                logger.warning(f"Error using TA-Lib for ADX: {e}. Using fallback.")
        
        try:
            # Try pandas-ta if available
            import pandas_ta as ta
            adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=window)
            
            adx = pd.DataFrame(index=df.index)
            adx[f'adx_{window}'] = adx_result[f'ADX_{window}']
            return adx
        except ImportError:
            # ADX is complex, return empty DataFrame with warning
            logger.warning("ADX calculation requires TA-Lib or pandas-ta. Skipping.")
            return pd.DataFrame(index=df.index)
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        if TALIB_AVAILABLE:
            try:
                obv = pd.DataFrame(index=df.index)
                obv['obv'] = talib.OBV(df['Close'].values, df['Volume'].values)
                return obv
            except Exception as e:
                logger.warning(f"Error using TA-Lib for OBV: {e}. Using fallback.")
        
        try:
            # Try pandas-ta if available
            import pandas_ta as ta
            obv_result = ta.obv(df['Close'], df['Volume'])
            
            obv = pd.DataFrame(index=df.index)
            obv['obv'] = obv_result
            return obv
        except ImportError:
            # Custom implementation
            obv = pd.DataFrame(index=df.index)
            obv_values = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv_values.append(obv_values[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv_values.append(obv_values[-1] - df['Volume'].iloc[i])
                else:
                    obv_values.append(obv_values[-1])
            obv['obv'] = obv_values
            return obv
    
    def _calculate_market_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market phase indicators (bull, bear, sideways)."""
        phase = pd.DataFrame(index=df.index)
        
        # Use 200-day SMA and 50-day SMA for trend determination
        sma_200 = df['Close'].rolling(window=200).mean()
        sma_50 = df['Close'].rolling(window=50).mean()
        
        # Bull market: price > SMA200 and SMA50 > SMA200
        # Bear market: price < SMA200 and SMA50 < SMA200
        # Sideways: everything else
        
        phase['bull_market'] = ((df['Close'] > sma_200) & (sma_50 > sma_200)).astype(int)
        phase['bear_market'] = ((df['Close'] < sma_200) & (sma_50 < sma_200)).astype(int)
        phase['sideways_market'] = ((~phase['bull_market'].astype(bool)) & (~phase['bear_market'].astype(bool))).astype(int)
        
        return phase


class FeatureNormalizer:
    """Feature normalization for trading prediction."""
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize FeatureNormalizer.
        
        Args:
            method: Normalization method ('minmax', 'zscore')
        """
        self.method = method
        self.scalers = {}
    
    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit normalizers on training data.
        
        Args:
            features: DataFrame with features
        """
        for col in features.columns:
            if self.method == 'minmax':
                self.scalers[col] = MinMaxScaler()
            else:  # zscore
                self.scalers[col] = StandardScaler()
            
            # Reshape for sklearn's fit method
            self.scalers[col].fit(features[col].values.reshape(-1, 1))
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted normalizers.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        normalized = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            if col in self.scalers:
                normalized[col] = self.scalers[col].transform(
                    features[col].values.reshape(-1, 1)
                ).flatten()
            else:
                # If no scaler is found for this column, copy as is
                normalized[col] = features[col]
        
        return normalized
    
    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Fit normalizers and transform features.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        self.fit(features)
        return self.transform(features)