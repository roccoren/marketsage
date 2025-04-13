#!/usr/bin/env python3
"""
Indicators module for the Qlib-based quantitative analysis project.
This module provides custom technical indicators and risk management metrics
that can be used in forecasting and backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod

from modules.config_manager import get_config
from modules.logging_manager import get_logger


# Configure logger
logger = get_logger(__name__)


class Indicator(ABC):
    """Abstract base class for indicators."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the indicator value.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.Series: Indicator values
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        pass


class MovingAverage(Indicator):
    """Simple moving average indicator."""
    
    def __init__(self, window: int = 20, column: str = 'Close'):
        """
        Initialize moving average indicator.
        
        Args:
            window (int): Window size
            column (str): Column to use for calculation
        """
        self.window = window
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate moving average.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.Series: Moving average values
        """
        return data[self.column].rolling(window=self.window).mean()
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"MA_{self.window}"


class ExponentialMovingAverage(Indicator):
    """Exponential moving average indicator."""
    
    def __init__(self, span: int = 20, column: str = 'Close'):
        """
        Initialize exponential moving average indicator.
        
        Args:
            span (int): Span for EMA calculation
            column (str): Column to use for calculation
        """
        self.span = span
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate exponential moving average.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.Series: Exponential moving average values
        """
        return data[self.column].ewm(span=self.span, adjust=False).mean()
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"EMA_{self.span}"


class RelativeStrengthIndex(Indicator):
    """Relative Strength Index (RSI) indicator."""
    
    def __init__(self, window: int = 14, column: str = 'Close'):
        """
        Initialize RSI indicator.
        
        Args:
            window (int): Window size
            column (str): Column to use for calculation
        """
        self.window = window
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.Series: RSI values
        """
        delta = data[self.column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.window).mean()
        avg_loss = loss.rolling(window=self.window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"RSI_{self.window}"


class BollingerBands(Indicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, window: int = 20, num_std: float = 2.0, column: str = 'Close'):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            window (int): Window size
            num_std (float): Number of standard deviations
            column (str): Column to use for calculation
        """
        self.window = window
        self.num_std = num_std
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame with middle, upper, and lower bands
        """
        middle_band = data[self.column].rolling(window=self.window).mean()
        std_dev = data[self.column].rolling(window=self.window).std()
        
        upper_band = middle_band + (std_dev * self.num_std)
        lower_band = middle_band - (std_dev * self.num_std)
        
        return pd.DataFrame({
            f"BB_middle_{self.window}": middle_band,
            f"BB_upper_{self.window}": upper_band,
            f"BB_lower_{self.window}": lower_band
        })
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"BB_{self.window}"


class MACD(Indicator):
    """Moving Average Convergence Divergence (MACD) indicator."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close'):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal EMA period
            column (str): Column to use for calculation
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame with MACD line, signal line, and histogram
        """
        fast_ema = data[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.column].ewm(span=self.slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            "MACD_line": macd_line,
            "MACD_signal": signal_line,
            "MACD_histogram": histogram
        })
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}"


class StochasticOscillator(Indicator):
    """Stochastic Oscillator indicator."""
    
    def __init__(self, k_period: int = 14, d_period: int = 3):
        """
        Initialize Stochastic Oscillator indicator.
        
        Args:
            k_period (int): K period
            d_period (int): D period
        """
        self.k_period = k_period
        self.d_period = d_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame with %K and %D values
        """
        low_min = data['Low'].rolling(window=self.k_period).min()
        high_max = data['High'].rolling(window=self.k_period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=self.d_period).mean()
        
        return pd.DataFrame({
            "Stoch_%K": k_percent,
            "Stoch_%D": d_percent
        })
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"Stoch_{self.k_period}_{self.d_period}"


class AverageTrueRange(Indicator):
    """Average True Range (ATR) indicator."""
    
    def __init__(self, window: int = 14):
        """
        Initialize ATR indicator.
        
        Args:
            window (int): Window size
        """
        self.window = window
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.Series: ATR values
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.window).mean()
        
        return atr
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return f"ATR_{self.window}"


class OnBalanceVolume(Indicator):
    """On-Balance Volume (OBV) indicator."""
    
    def __init__(self):
        """Initialize OBV indicator."""
        pass
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate OBV.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.Series: OBV values
        """
        obv = pd.Series(index=data.index, dtype='float64')
        obv.iloc[0] = 0
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @property
    def name(self) -> str:
        """
        Get the indicator name.
        
        Returns:
            str: Indicator name
        """
        return "OBV"


class RiskMetrics:
    """Risk management metrics."""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns (pd.Series): Historical returns
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            float: Value at Risk
        """
        # Sort returns in ascending order
        sorted_returns = returns.sort_values()
        
        # Find the index at the specified confidence level
        index = int((1 - confidence_level) * len(sorted_returns))
        
        # Return the value at that index
        return abs(sorted_returns.iloc[index])
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns (pd.Series): Historical returns
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            float: Conditional Value at Risk
        """
        # Calculate VaR
        var = RiskMetrics.calculate_var(returns, confidence_level)
        
        # Calculate CVaR as the mean of returns beyond VaR
        cvar_returns = returns[returns <= -var]
        
        if len(cvar_returns) > 0:
            return abs(cvar_returns.mean())
        else:
            return var  # Fallback if no returns beyond VaR
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices (pd.Series): Historical prices
            
        Returns:
            float: Maximum drawdown as a percentage
        """
        # Calculate running maximum
        running_max = prices.cummax()
        
        # Calculate drawdown
        drawdown = (prices / running_max - 1) * 100
        
        # Return the maximum drawdown
        return abs(drawdown.min())
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pd.Series): Historical returns
            risk_free_rate (float): Risk-free rate
            annualization_factor (int): Annualization factor (252 for daily returns)
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        return excess_returns.mean() / excess_returns.std() * np.sqrt(annualization_factor)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns (pd.Series): Historical returns
            risk_free_rate (float): Risk-free rate
            annualization_factor (int): Annualization factor (252 for daily returns)
            
        Returns:
            float: Sortino ratio
        """
        excess_returns = returns - risk_free_rate / annualization_factor
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            return excess_returns.mean() / downside_returns.std() * np.sqrt(annualization_factor)
        else:
            return np.nan  # Cannot calculate if no downside returns or std is 0
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, prices: pd.Series, annualization_factor: int = 252) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns (pd.Series): Historical returns
            prices (pd.Series): Historical prices
            annualization_factor (int): Annualization factor (252 for daily returns)
            
        Returns:
            float: Calmar ratio
        """
        annual_return = returns.mean() * annualization_factor
        max_drawdown = RiskMetrics.calculate_max_drawdown(prices)
        
        if max_drawdown > 0:
            return annual_return / max_drawdown
        else:
            return np.nan  # Cannot calculate if max drawdown is 0


class IndicatorRegistry:
    """Registry for custom indicators."""
    
    def __init__(self):
        """Initialize indicator registry."""
        self.indicators: Dict[str, Indicator] = {}
        
        # Register built-in indicators
        self._register_builtin_indicators()
    
    def _register_builtin_indicators(self) -> None:
        """Register built-in indicators."""
        self.register_indicator("ma_20", MovingAverage(window=20))
        self.register_indicator("ma_50", MovingAverage(window=50))
        self.register_indicator("ma_200", MovingAverage(window=200))
        
        self.register_indicator("ema_12", ExponentialMovingAverage(span=12))
        self.register_indicator("ema_26", ExponentialMovingAverage(span=26))
        
        self.register_indicator("rsi_14", RelativeStrengthIndex(window=14))
        
        self.register_indicator("bb_20", BollingerBands(window=20))
        
        self.register_indicator("macd", MACD())
        
        self.register_indicator("stoch", StochasticOscillator())
        
        self.register_indicator("atr_14", AverageTrueRange(window=14))
        
        self.register_indicator("obv", OnBalanceVolume())
    
    def register_indicator(self, name: str, indicator: Indicator) -> None:
        """
        Register a custom indicator.
        
        Args:
            name (str): Indicator name
            indicator (Indicator): Indicator instance
        """
        self.indicators[name] = indicator
        logger.info(f"Registered indicator: {name}")
    
    def get_indicator(self, name: str) -> Optional[Indicator]:
        """
        Get an indicator by name.
        
        Args:
            name (str): Indicator name
            
        Returns:
            Optional[Indicator]: Indicator instance or None if not found
        """
        return self.indicators.get(name)
    
    def calculate_indicator(self, name: str, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate an indicator by name.
        
        Args:
            name (str): Indicator name
            data (pd.DataFrame): Historical price data
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Indicator values
        """
        indicator = self.get_indicator(name)
        
        if indicator is None:
            raise ValueError(f"Indicator not found: {name}")
        
        return indicator.calculate(data)
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all registered indicators.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame with all indicator values
        """
        result = data.copy()
        
        for name, indicator in self.indicators.items():
            try:
                values = indicator.calculate(data)
                
                if isinstance(values, pd.Series):
                    result[indicator.name] = values
                elif isinstance(values, pd.DataFrame):
                    for col in values.columns:
                        result[col] = values[col]
            except Exception as e:
                logger.error(f"Error calculating indicator {name}: {e}")
        
        return result
    
    def list_indicators(self) -> List[str]:
        """
        List all registered indicators.
        
        Returns:
            List[str]: List of indicator names
        """
        return list(self.indicators.keys())


# Default indicator registry
_default_indicator_registry = None


def get_indicator_registry() -> IndicatorRegistry:
    """
    Get the default indicator registry.
    
    Returns:
        IndicatorRegistry: Default indicator registry
    """
    global _default_indicator_registry
    
    if _default_indicator_registry is None:
        _default_indicator_registry = IndicatorRegistry()
    
    return _default_indicator_registry


def register_indicator(name: str, indicator: Indicator) -> None:
    """
    Register a custom indicator with the default registry.
    
    Args:
        name (str): Indicator name
        indicator (Indicator): Indicator instance
    """
    get_indicator_registry().register_indicator(name, indicator)


def get_indicator(name: str) -> Optional[Indicator]:
    """
    Get an indicator by name from the default registry.
    
    Args:
        name (str): Indicator name
        
    Returns:
        Optional[Indicator]: Indicator instance or None if not found
    """
    return get_indicator_registry().get_indicator(name)


def calculate_indicator(name: str, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate an indicator by name using the default registry.
    
    Args:
        name (str): Indicator name
        data (pd.DataFrame): Historical price data
        
    Returns:
        Union[pd.Series, pd.DataFrame]: Indicator values
    """
    return get_indicator_registry().calculate_indicator(name, data)


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all registered indicators using the default registry.
    
    Args:
        data (pd.DataFrame): Historical price data
        
    Returns:
        pd.DataFrame: DataFrame with all indicator values
    """
    return get_indicator_registry().calculate_all_indicators(data)


def list_indicators() -> List[str]:
    """
    List all registered indicators in the default registry.
    
    Returns:
        List[str]: List of indicator names
    """
    return get_indicator_registry().list_indicators()