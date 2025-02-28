import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    @staticmethod
    def get_all_indicators(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Moving Averages
        indicators['sma_short'] = TechnicalIndicators.sma(data['Close'], params['sma_short_period'])
        indicators['sma_long'] = TechnicalIndicators.sma(data['Close'], params['sma_long_period'])
        indicators['ema_short'] = TechnicalIndicators.ema(data['Close'], params['ema_short_period'])
        indicators['ema_long'] = TechnicalIndicators.ema(data['Close'], params['ema_long_period'])
        
        # Crossover signals
        indicators['sma_crossover'] = TechnicalIndicators.crossover_signal(
            indicators['sma_short'],
            indicators['sma_long']
        )
        indicators['ema_crossover'] = TechnicalIndicators.crossover_signal(
            indicators['ema_short'],
            indicators['ema_long']
        )
        
        # RSI
        indicators['rsi'] = TechnicalIndicators.rsi(data['Close'], params['rsi_period'])
        
        # Bollinger Bands
        bb = TechnicalIndicators.bollinger_bands(
            data['Close'],
            params['bb_period'],
            params['bb_std']
        )
        indicators.update(bb)
        
        # MACD
        macd = TechnicalIndicators.macd(
            data['Close'],
            params['macd_fast'],
            params['macd_slow'],
            params['macd_signal']
        )
        indicators.update(macd)

        # ATR for trailing stops and position sizing
        indicators['atr'] = TechnicalIndicators.atr(
            data['High'],
            data['Low'],
            data['Close'],
            params['atr_period']
        )
        
        return indicators

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int,
        std_dev: float
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        bb_middle = series.rolling(window=period).mean()
        bb_std = series.rolling(window=period).std()
        
        bb_upper = bb_middle + (bb_std * std_dev)
        bb_lower = bb_middle - (bb_std * std_dev)
        
        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower
        }

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int,
        slow_period: int,
        signal_period: int
    ) -> Dict[str, pd.Series]:
        """Calculate MACD and Signal Line"""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate crossover signal
        macd_crossover = pd.Series(0, index=series.index)
        macd_crossover.loc[macd_line > signal_line] = 1
        macd_crossover.loc[macd_line < signal_line] = -1
        
        return {
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_crossover': macd_crossover
        }

    @staticmethod
    def crossover_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
        """Generate crossover signal (-1, 0, 1)"""
        signal = pd.Series(0, index=fast.index)
        signal.loc[fast > slow] = 1
        signal.loc[fast < slow] = -1
        return signal

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        # True Range calculation
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Calculate ATR using EMA of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()
        return atr