"""
Price target prediction functionality

This module provides functions to predict buy and sell price targets
based on technical indicators and historical price patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_price_targets(df: pd.DataFrame, lookback_period: int = 30, 
                           support_resistance_periods: int = 20,
                           volatility_factor: float = 1.5) -> Dict[str, float]:
    """
    Calculate buy and sell price targets based on technical indicators

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with stock data including technical indicators
    lookback_period : int
        Period to look back for historical patterns
    support_resistance_periods : int
        Number of periods to use for support/resistance calculations
    volatility_factor : float
        Multiplier for volatility to determine price ranges

    Returns
    -------
    Dict[str, float]
        Dictionary with buy_price, sell_price, and stop_loss targets
    """
    if df.empty or len(df) < lookback_period:
        return {
            'buy_price': None,
            'sell_price': None,
            'stop_loss': None
        }
    
    # Get the latest available data
    latest_data = df.iloc[-1]
    latest_close = latest_data['Close']
    
    # 1. Support and resistance levels based on recent price history
    recent_df = df.iloc[-support_resistance_periods:]
    support_level = recent_df['Low'].min()
    resistance_level = recent_df['High'].max()
    
    # 2. Volatility measurement (ATR - Average True Range)
    # Calculate true range
    tr1 = recent_df['High'] - recent_df['Low']
    tr2 = abs(recent_df['High'] - recent_df['Close'].shift(1))
    tr3 = abs(recent_df['Low'] - recent_df['Close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.mean()
    
    # 3. Price targets based on technical analysis
    
    # Buy zone: Support zone or below current price if trending up
    rsi = latest_data.get('rsi', 50)
    macd_hist = latest_data.get('hist', 0)
    price_trend = latest_data['Close'] > latest_data.get('ma20', latest_data['Close'])
    
    # Buy logic
    if rsi < 40 or (price_trend and macd_hist > 0):
        # If oversold or uptrending with positive MACD
        buy_price = max(latest_close * 0.98, support_level)
    else:
        # Otherwise look for a deeper pullback
        buy_price = max(support_level, latest_close - atr)
    
    # Sell zone: Resistance zone or profit target
    if rsi > 70 or macd_hist < 0:
        # If overbought or diverging MACD
        sell_price = min(latest_close * 1.02, resistance_level)
    else:
        # Otherwise set a higher target
        sell_price = min(resistance_level, latest_close + (atr * 2))
    
    # Stop loss: Based on volatility and support levels
    stop_loss = max(support_level * 0.95, latest_close - (atr * volatility_factor))

    # 4. Bollinger Bands for price targets
    if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
        # Use Bollinger Bands to refine targets
        bb_lower = latest_data['bb_lower']
        bb_upper = latest_data['bb_upper']
        
        # If close to lower band, use it as buy zone
        if latest_close < latest_data['bb_middle']:
            buy_price = max(buy_price, bb_lower)
        
        # If close to upper band, use it as sell zone
        if latest_close > latest_data['bb_middle']:
            sell_price = min(sell_price, bb_upper)
    
    # Round to 2 decimal places
    return {
        'buy_price': round(buy_price, 2),
        'sell_price': round(sell_price, 2),
        'stop_loss': round(stop_loss, 2)
    }

def get_entry_exit_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals based on technical indicators
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with stock data including technical indicators
        
    Returns
    -------
    pd.DataFrame
        DataFrame with buy/sell signals
    """
    # Copy the dataframe to avoid modifying the original
    signals = df.copy()
    
    # Initialize signal columns
    signals['buy_signal'] = 0
    signals['sell_signal'] = 0
    
    # RSI oversold/overbought signals
    signals.loc[signals['rsi'] < 30, 'buy_signal'] += 1
    signals.loc[signals['rsi'] > 70, 'sell_signal'] += 1
    
    # MACD crossing signals
    signals['macd_cross'] = np.where(
        (signals['macd'] > signals['signal']) & 
        (signals['macd'].shift(1) <= signals['signal'].shift(1)),
        1, 0
    )
    signals['macd_cross_down'] = np.where(
        (signals['macd'] < signals['signal']) & 
        (signals['macd'].shift(1) >= signals['signal'].shift(1)),
        1, 0
    )
    signals.loc[signals['macd_cross'] == 1, 'buy_signal'] += 1
    signals.loc[signals['macd_cross_down'] == 1, 'sell_signal'] += 1
    
    # Bollinger Band signals
    if all(x in signals.columns for x in ['bb_lower', 'bb_middle', 'bb_upper']):
        # Price crossing below lower band
        signals.loc[signals['Close'] < signals['bb_lower'], 'buy_signal'] += 1
        # Price crossing above upper band
        signals.loc[signals['Close'] > signals['bb_upper'], 'sell_signal'] += 1
    
    # Moving average crossovers
    if all(x in signals.columns for x in ['ma10', 'ma20', 'ma50']):
        # Fast MA crossing above slow MA
        signals['ma_cross'] = np.where(
            (signals['ma10'] > signals['ma20']) & 
            (signals['ma10'].shift(1) <= signals['ma20'].shift(1)),
            1, 0
        )
        signals['ma_cross_down'] = np.where(
            (signals['ma10'] < signals['ma20']) & 
            (signals['ma10'].shift(1) >= signals['ma20'].shift(1)),
            1, 0
        )
        signals.loc[signals['ma_cross'] == 1, 'buy_signal'] += 1
        signals.loc[signals['ma_cross_down'] == 1, 'sell_signal'] += 1
    
    # Strong signals are those with multiple indicators confirming
    signals['strong_buy'] = signals['buy_signal'] >= 2
    signals['strong_sell'] = signals['sell_signal'] >= 2
    
    return signals

def predict_price_movement(df: pd.DataFrame, forecast_periods: int = 5) -> Dict[str, any]:
    """
    Predict future price movement direction and magnitude
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with stock data including technical indicators
    forecast_periods : int
        Number of periods ahead to forecast
        
    Returns
    -------
    Dict
        Dictionary with prediction details
    """
    if df.empty or len(df) < 20:  # Need sufficient history
        return {
            'direction': 'neutral',
            'magnitude': 0,
            'confidence': 0,
            'forecast_price': None
        }
    
    # Get the latest data
    latest = df.iloc[-1]
    
    # Current technical setup
    rsi = latest.get('rsi', 50)
    macd_hist = latest.get('hist', 0)
    close = latest['Close']
    
    # Determine trend direction from moving averages
    ma_trend = 0
    if 'ma10' in latest and 'ma20' in latest and 'ma50' in latest:
        if latest['ma10'] > latest['ma20'] > latest['ma50']:
            ma_trend = 2  # Strong uptrend
        elif latest['ma10'] > latest['ma20']:
            ma_trend = 1  # Moderate uptrend
        elif latest['ma10'] < latest['ma20'] < latest['ma50']:
            ma_trend = -2  # Strong downtrend
        elif latest['ma10'] < latest['ma20']:
            ma_trend = -1  # Moderate downtrend
    
    # Momentum from RSI
    rsi_signal = 0
    if rsi < 30:
        rsi_signal = 1  # Oversold, potential bounce
    elif rsi > 70:
        rsi_signal = -1  # Overbought, potential decline
    elif rsi > 50 and rsi < 70:
        rsi_signal = 0.5  # Bullish momentum
    elif rsi < 50 and rsi > 30:
        rsi_signal = -0.5  # Bearish momentum
    
    # MACD momentum
    macd_signal = np.sign(macd_hist) * min(abs(macd_hist) / 0.5, 1)
    
    # Combine signals with weightings
    trend_weight = 0.5
    rsi_weight = 0.3
    macd_weight = 0.2
    
    combined_signal = (
        (ma_trend * trend_weight) + 
        (rsi_signal * rsi_weight) + 
        (macd_signal * macd_weight)
    )
    
    # Determine direction
    if combined_signal > 0.5:
        direction = 'bullish'
    elif combined_signal < -0.5:
        direction = 'bearish'
    else:
        direction = 'neutral'
    
    # Estimate magnitude of movement
    # Use average true range (ATR) as volatility measure
    recent_df = df.iloc[-20:]
    tr1 = recent_df['High'] - recent_df['Low']
    tr2 = abs(recent_df['High'] - recent_df['Close'].shift(1))
    tr3 = abs(recent_df['Low'] - recent_df['Close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.mean()
    
    # Scale magnitude by signal strength and forecast period
    magnitude = abs(combined_signal) * atr * np.sqrt(forecast_periods)
    
    # Confidence based on signal consistency
    confidence = min(abs(combined_signal) * 100, 100)
    
    # Forecast price
    if direction == 'bullish':
        forecast_price = close + magnitude
    elif direction == 'bearish':
        forecast_price = close - magnitude
    else:
        forecast_price = close
    
    return {
        'direction': direction,
        'magnitude': round(magnitude, 2),
        'confidence': round(confidence, 1),
        'forecast_price': round(forecast_price, 2)
    }

def analyze_stock_for_trading(df: pd.DataFrame) -> Dict[str, any]:
    """
    Complete analysis of a stock for trading decisions
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with stock data including technical indicators
        
    Returns
    -------
    Dict
        Dictionary with trading analysis
    """
    if df.empty:
        return {
            'status': 'insufficient_data',
            'message': 'Not enough data for analysis'
        }
    
    # Get the latest day's data
    current_price = df['Close'].iloc[-1]
    
    # Get price targets
    price_targets = calculate_price_targets(df)
    
    # Get entry/exit signals
    signals_df = get_entry_exit_signals(df)
    latest_signals = signals_df.iloc[-1]
    
    # Get price movement prediction
    price_prediction = predict_price_movement(df)
    
    # Determine current trading action
    action = 'hold'  # Default action
    
    if latest_signals['strong_buy']:
        action = 'buy'
    elif latest_signals['strong_sell']:
        action = 'sell'
    
    # Calculate risk/reward ratio
    if price_targets['buy_price'] and price_targets['sell_price'] and price_targets['stop_loss']:
        potential_gain = price_targets['sell_price'] - current_price
        potential_loss = current_price - price_targets['stop_loss']
        risk_reward = round(potential_gain / potential_loss, 2) if potential_loss > 0 else None
    else:
        risk_reward = None
    
    # Compile results
    return {
        'status': 'success',
        'current_price': round(current_price, 2),
        'action': action,
        'price_targets': price_targets,
        'prediction': price_prediction,
        'risk_reward_ratio': risk_reward,
        'buy_signal_strength': int(latest_signals['buy_signal']),
        'sell_signal_strength': int(latest_signals['sell_signal']),
        'timestamp': df.index[-1].strftime('%Y-%m-%d')
    } 