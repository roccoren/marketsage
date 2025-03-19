"""
Technical indicator utilities for stock analysis

This module provides a comprehensive set of technical indicators for stock analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Utility class for calculating technical indicators.
    
    All methods are static to allow for use without instantiation.
    """
    
    @staticmethod
    def get_all_indicators(data: pd.DataFrame, params: Optional[Dict[str, Any]] = None) -> Dict[str, pd.Series]:
        """
        Calculate all indicators based on provided parameters
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        params : Dict[str, Any], optional
            Parameters for indicators, by default None
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary containing all calculated indicators
        """
        if params is None:
            params = {}
        
        indicators = {}
        
        # Simple Moving Averages (SMAs)
        sma_short_period = params.get('sma_short_period', 20)
        sma_long_period = params.get('sma_long_period', 50)
        indicators['sma_short'] = TechnicalIndicators.sma(data, sma_short_period)
        indicators['sma_long'] = TechnicalIndicators.sma(data, sma_long_period)
        
        # Exponential Moving Averages (EMAs)
        ema_short_period = params.get('ema_short_period', 12)
        ema_long_period = params.get('ema_long_period', 26)
        indicators['ema_short'] = TechnicalIndicators.ema(data, ema_short_period)
        indicators['ema_long'] = TechnicalIndicators.ema(data, ema_long_period)
        
        # SMA and EMA Crossover signals
        indicators['sma_crossover'] = TechnicalIndicators.crossover_signal(
            indicators['sma_short'], indicators['sma_long']
        )
        indicators['ema_crossover'] = TechnicalIndicators.crossover_signal(
            indicators['ema_short'], indicators['ema_long']
        )
        
        # Relative Strength Index (RSI)
        rsi_period = params.get('rsi_period', 14)
        indicators['rsi'] = TechnicalIndicators.rsi(data, rsi_period)
        
        # Bollinger Bands
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            data, bb_period, bb_std
        )
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # MACD
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            data, macd_fast, macd_slow, macd_signal
        )
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Average Directional Index (ADX)
        adx_period = params.get('adx_period', 14)
        adx, plus_di, minus_di = TechnicalIndicators.adx(data, adx_period)
        indicators['adx'] = adx
        indicators['plus_di'] = plus_di
        indicators['minus_di'] = minus_di
        
        # Commodity Channel Index (CCI)
        cci_period = params.get('cci_period', 20)
        indicators['cci'] = TechnicalIndicators.cci(data, cci_period)
        
        # Stochastic Oscillator
        stoch_k_period = params.get('stoch_k_period', 14)
        stoch_d_period = params.get('stoch_d_period', 3)
        stoch_k, stoch_d = TechnicalIndicators.stochastic(
            data, stoch_k_period, stoch_d_period
        )
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        # On-Balance Volume (OBV)
        indicators['obv'] = TechnicalIndicators.obv(data)
        
        # Average True Range (ATR)
        atr_period = params.get('atr_period', 14)
        indicators['atr'] = TechnicalIndicators.atr(data, atr_period)
        
        # Additional indicators based on parameters
        if params.get('calculate_volume_indicators', False):
            # Volume indicators
            volume_ma_period = params.get('volume_ma_period', 20)
            indicators['volume_sma'] = TechnicalIndicators.sma(
                data['Volume'], volume_ma_period
            )
            indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
        
        if params.get('calculate_price_channels', False):
            # Price channels
            channel_period = params.get('channel_period', 20)
            indicators['upper_channel'] = data['High'].rolling(channel_period).max()
            indicators['lower_channel'] = data['Low'].rolling(channel_period).min()
        
        # Clean up NaN values at the beginning
        for name, indicator in indicators.items():
            if indicator is not None and isinstance(indicator, pd.Series):
                indicators[name] = indicator.fillna(method='bfill')
        
        return indicators
    
    @staticmethod
    def sma(data: Union[pd.DataFrame, pd.Series], period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, pd.Series]
            Price data, either a DataFrame (will use 'Close' or 'close') or a Series
        period : int
            Moving average period
            
        Returns:
        --------
        pd.Series
            Simple Moving Average series
        """
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                return data['close'].rolling(window=period).mean()
            elif 'Close' in data.columns:
                return data['Close'].rolling(window=period).mean()
            else:
                raise ValueError("DataFrame must contain 'close' or 'Close' column")
        else:
            return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: Union[pd.DataFrame, pd.Series], period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, pd.Series]
            Price data, either a DataFrame (will use 'Close' or 'close') or a Series
        period : int
            Moving average period
            
        Returns:
        --------
        pd.Series
            Exponential Moving Average series
        """
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                return data['close'].ewm(span=period, adjust=False).mean()
            elif 'Close' in data.columns:
                return data['Close'].ewm(span=period, adjust=False).mean()
            else:
                raise ValueError("DataFrame must contain 'close' or 'Close' column")
        else:
            return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: Union[pd.DataFrame, pd.Series], period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, pd.Series]
            OHLCV data with datetime index, or price series
        period : int, optional
            RSI period, by default 14
            
        Returns:
        --------
        pd.Series
            RSI series
        """
        # Extract price series if DataFrame is provided
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                price = data['close']
            elif 'Close' in data.columns:
                price = data['Close']
            else:
                raise ValueError("DataFrame must contain 'close' or 'Close' column")
        else:
            price = data
            
        # Calculate price changes
        delta = price.diff()
        
        # Make two series: one for gains, one for losses
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate the EWMA
        roll_up = up.ewm(com=period-1, adjust=False).mean()
        roll_down = down.ewm(com=period-1, adjust=False).mean()
        
        # Calculate the RSI based on EWMA
        rs = roll_up / roll_down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        period : int, optional
            Moving average period, by default 20
        std_dev : float, optional
            Number of standard deviations, by default 2.0
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            Upper band, middle band, lower band
        """
        middle_band = TechnicalIndicators.sma(data, period)
        std = data['Close'].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def macd(data: Union[pd.DataFrame, pd.Series], fast_period: int = 12, slow_period: int = 26, 
           signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, pd.Series]
            OHLCV data with datetime index, or price series
        fast_period : int, optional
            Fast EMA period, by default 12
        slow_period : int, optional
            Slow EMA period, by default 26
        signal_period : int, optional
            Signal line period, by default 9
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            MACD line, signal line, histogram
        """
        # Handle both DataFrame and Series inputs
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                price_series = data['close']
            elif 'Close' in data.columns:
                price_series = data['Close']
            else:
                raise ValueError("DataFrame must contain 'close' or 'Close' column")
        else:
            price_series = data
        
        # Calculate EMAs
        fast_ema = TechnicalIndicators.ema(price_series, fast_period)
        slow_ema = TechnicalIndicators.ema(price_series, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        period : int, optional
            ADX period, by default 14
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            ADX, +DI, -DI
        """
        # Calculate True Range
        tr1 = pd.Series(data['High'] - data['Low']).abs()
        tr2 = pd.Series(data['High'] - data['Close'].shift()).abs()
        tr3 = pd.Series(data['Low'] - data['Close'].shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DM and -DM
        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)
        
        for i in range(1, len(data)):
            high_diff = data['High'].iloc[i] - data['High'].iloc[i-1]
            low_diff = data['Low'].iloc[i-1] - data['Low'].iloc[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm.iloc[i] = high_diff
            elif low_diff > high_diff and low_diff > 0:
                minus_dm.iloc[i] = low_diff
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        
        # Calculate DX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
        
        # Calculate ADX
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def cci(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        period : int, optional
            CCI period, by default 20
            
        Returns:
        --------
        pd.Series
            CCI series
        """
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        moving_avg = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        cci = (typical_price - moving_avg) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        k_period : int, optional
            %K period, by default 14
        d_period : int, optional
            %D period, by default 3
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            %K, %D
        """
        # Calculate %K
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        
        k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
            
        Returns:
        --------
        pd.Series
            OBV series
        """
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def crossover_signal(fast_line: pd.Series, slow_line: pd.Series) -> pd.Series:
        """
        Generate crossover signals
        
        Parameters:
        -----------
        fast_line : pd.Series
            Faster moving series (e.g., shorter period MA)
        slow_line : pd.Series
            Slower moving series (e.g., longer period MA)
            
        Returns:
        --------
        pd.Series
            Series with values:
            1 for bullish crossover (fast crosses above slow)
            -1 for bearish crossover (fast crosses below slow)
            0 for no crossover
        """
        # Compute crossovers
        signals = pd.Series(0, index=fast_line.index)
        
        # Create boolean series for crossover conditions
        fast_above_slow = fast_line > slow_line
        
        # Detect changes in relative position
        crossover = fast_above_slow.astype(int).diff()
        
        # Set signal values
        signals[crossover == 1] = 1    # Bullish crossover
        signals[crossover == -1] = -1  # Bearish crossover
        
        return signals
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
        period : int, optional
            ATR period, by default 14
            
        Returns:
        --------
        pd.Series
            ATR series
        """
        # Calculate True Range
        tr1 = pd.Series(data['High'] - data['Low']).abs()
        tr2 = pd.Series(data['High'] - data['Close'].shift()).abs()
        tr3 = pd.Series(data['Low'] - data['Close'].shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using simple moving average
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, 
                         window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Detect regular and hidden divergences between price and indicator
        
        Parameters:
        -----------
        price : pd.Series
            Price series (usually Close)
        indicator : pd.Series
            Technical indicator series (RSI, MACD, etc.)
        window : int, optional
            Lookback window to find local extrema, by default 20
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            regular_divergence, hidden_divergence series with values:
            1 for bullish divergence
            -1 for bearish divergence
            0 for no divergence
        """
        # Initialize output series
        regular_divergence = pd.Series(0, index=price.index)
        hidden_divergence = pd.Series(0, index=price.index)
        
        # Function to find local extrema
        def find_extrema(series, window):
            # Find local maxima
            maxima = pd.Series(0, index=series.index)
            for i in range(window, len(series) - window):
                if series.iloc[i] == series.iloc[i-window:i+window+1].max():
                    maxima.iloc[i] = 1
            
            # Find local minima
            minima = pd.Series(0, index=series.index)
            for i in range(window, len(series) - window):
                if series.iloc[i] == series.iloc[i-window:i+window+1].min():
                    minima.iloc[i] = 1
            
            return maxima, minima
        
        # Find price extrema
        price_maxima, price_minima = find_extrema(price, window)
        
        # Find indicator extrema
        ind_maxima, ind_minima = find_extrema(indicator, window)
        
        # Detect regular bearish divergence: price makes higher high, indicator makes lower high
        for i in range(window*2, len(price)):
            if price_maxima.iloc[i] == 1:
                # Find previous price maximum within window
                prev_max_idx = price_maxima.iloc[i-window*2:i].idxmax()
                if price.iloc[i] > price.iloc[prev_max_idx] and ind_maxima.iloc[i] == 1:
                    prev_ind_max_idx = ind_maxima.iloc[i-window*2:i].idxmax()
                    if indicator.iloc[i] < indicator.iloc[prev_ind_max_idx]:
                        regular_divergence.iloc[i] = -1  # Bearish
        
        # Detect regular bullish divergence: price makes lower low, indicator makes higher low
        for i in range(window*2, len(price)):
            if price_minima.iloc[i] == 1:
                # Find previous price minimum within window
                prev_min_idx = price_minima.iloc[i-window*2:i].idxmax()
                if price.iloc[i] < price.iloc[prev_min_idx] and ind_minima.iloc[i] == 1:
                    prev_ind_min_idx = ind_minima.iloc[i-window*2:i].idxmax()
                    if indicator.iloc[i] > indicator.iloc[prev_ind_min_idx]:
                        regular_divergence.iloc[i] = 1  # Bullish
        
        # Hidden divergences (continuation patterns)
        # Hidden bullish: price makes higher low, indicator makes lower low
        for i in range(window*2, len(price)):
            if price_minima.iloc[i] == 1:
                prev_min_idx = price_minima.iloc[i-window*2:i].idxmax()
                if price.iloc[i] > price.iloc[prev_min_idx] and ind_minima.iloc[i] == 1:
                    prev_ind_min_idx = ind_minima.iloc[i-window*2:i].idxmax()
                    if indicator.iloc[i] < indicator.iloc[prev_ind_min_idx]:
                        hidden_divergence.iloc[i] = 1  # Bullish
        
        # Hidden bearish: price makes lower high, indicator makes higher high
        for i in range(window*2, len(price)):
            if price_maxima.iloc[i] == 1:
                prev_max_idx = price_maxima.iloc[i-window*2:i].idxmax()
                if price.iloc[i] < price.iloc[prev_max_idx] and ind_maxima.iloc[i] == 1:
                    prev_ind_max_idx = ind_maxima.iloc[i-window*2:i].idxmax()
                    if indicator.iloc[i] > indicator.iloc[prev_ind_max_idx]:
                        hidden_divergence.iloc[i] = -1  # Bearish
        
        return regular_divergence, hidden_divergence