import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from .base import Strategy
from utils.logging_config import configure_logging, load_debug_config

# Load debug configuration
debug_config = load_debug_config()
debug_enabled = debug_config.get('macd', False)

# Configure module logger
logger = configure_logging(
    __name__,
    log_level=debug_config.get('logLevel', 'INFO'),
    log_file=debug_config.get('logFiles', {}).get('macd'),
    debug_mode=debug_enabled
)

class SignalType:
    BUY = "BUY"
    SELL = "SELL"

class MACDStrategy(Strategy):
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name="MACD")
        self.exchanges = ['NYSE', 'NASDAQ', 'HKEX', 'ASX']
        self.base_dir = "data"
        """
        Initialize MACD strategy with configurable parameters.
        
        Args:
            params: Dictionary of strategy parameters
                - fast_period (int): Fast EMA period (default: 12)
                - slow_period (int): Slow EMA period (default: 26)
                - signal_period (int): Signal line EMA period (default: 9)
                - rsi_period (int): RSI calculation period (default: 14)
                - rsi_overbought (float): RSI overbought threshold (default: 70)
                - rsi_oversold (float): RSI oversold threshold (default: 30)
                - volume_ma_period (int): Volume moving average period (default: 5)
                - volume_threshold (float): Volume increase threshold (default: 1.01)
        """
        self.params = params or {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'volume_ma_period': 5,
            'volume_threshold': 1.01
        }

    def calculate_signal_strength(self, macd_hist: float, volume_ratio: float, price_change: float) -> float:
        """Calculate signal strength based on MACD histogram, volume, and price change."""
        try:
            # Normalize components
            hist_score = min(abs(macd_hist) / 2.0, 1.0)
            vol_score = min((volume_ratio - 1.0) / 2.0, 1.0)
            price_score = min(abs(price_change) * 10, 1.0)
            
            # Calculate weighted average
            strength = (0.5 * hist_score) + (0.3 * vol_score) + (0.2 * price_score)
            
            if debug_enabled:
                logger.debug(f"Signal strength calculation:")
                logger.debug(f"  Histogram: {macd_hist:.4f} -> {hist_score:.4f}")
                logger.debug(f"  Volume ratio: {volume_ratio:.4f} -> {vol_score:.4f}")
                logger.debug(f"  Price change: {price_change:.4f} -> {price_score:.4f}")
                logger.debug(f"  Final strength: {strength:.4f}")
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0.0

    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD technical indicator components."""
        if debug_enabled:
            logger.debug("Calculate MACD called")
            logger.debug(f"Input DataFrame shape: {data.shape}")
        
        try:
            if debug_enabled:
                logger.debug(f"DataFrame columns: {data.columns.tolist()}")
                logger.debug(f"Column types: {data.dtypes}")
            
            if 'Close' in data.columns:
                close = pd.to_numeric(data.loc[:, 'Close'], errors='coerce')
            else:
                close = data.iloc[:, 0]
                if debug_enabled:
                    logger.debug("Using first column as Close price data")
                
        except Exception as e:
            logger.error(f"Error accessing Close price data: {type(e)}: {str(e)}")
            raise
        
        if close.isna().any():
            nan_count = close.isna().sum()
            logger.warning(f"Found {nan_count} NaN values in Close prices")
            close = close.fillna(method='ffill').fillna(method='bfill')
            
        try:
            exp1 = close.ewm(span=self.params['fast_period'], adjust=False).mean()
            exp2 = close.ewm(span=self.params['slow_period'], adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.params['signal_period'], adjust=False).mean()
            
            if debug_enabled:
                logger.debug(f"MACD calculation completed - Range: {macd.min():.4f} to {macd.max():.4f}")
            
            hist = 2 * (macd - signal)
            
            return pd.DataFrame({
                'MACD': macd,
                'Signal': signal,
                'Histogram': hist
            })
            
        except Exception as e:
            logger.exception("Error calculating MACD components")
            raise

    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI for market condition filtering."""
        if 'Close' not in data.columns:
            logger.error("Required column 'Close' not found for RSI calculation")
            raise KeyError("Required column 'Close' not found")

        close = pd.to_numeric(data['Close'], errors='coerce')
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.ewm(com=self.params['rsi_period']-1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.params['rsi_period']-1, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def detect_recent_golden_cross(self, data: pd.DataFrame, lookback_days: int = 5) -> bool:
        """Detect if a MACD golden cross occurred within the lookback period."""
        min_required = max(self.params['slow_period'] + self.params['signal_period'], lookback_days)
        if len(data) < min_required:
            logger.warning("Insufficient data for golden cross detection")
            return False
            
        macd_data = self.calculate_macd(data)
        lookback_size = lookback_days + 1
        recent_macd = macd_data['MACD'].tail(lookback_size)
        recent_signal = macd_data['Signal'].tail(lookback_size)
        
        for i in range(1, len(recent_macd)):
            prev_macd = recent_macd.iloc[i-1]
            prev_signal = recent_signal.iloc[i-1]
            curr_macd = recent_macd.iloc[i]
            curr_signal = recent_signal.iloc[i]
            
            epsilon = 1e-10
            if (prev_macd <= prev_signal + epsilon) and (curr_macd > curr_signal + epsilon):
                if debug_enabled:
                    logger.debug(f"Golden cross detected: {curr_macd:.4f} crossed above {curr_signal:.4f}")
                else:
                    logger.info("Golden cross detected")
                return True
        
        if debug_enabled:
            logger.debug("No golden cross detected in lookback period")
        return False

    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """Analyze price data for MACD signals."""
        if len(data) < self.params['slow_period'] + self.params['signal_period']:
            logger.warning("Insufficient data for MACD calculation")
            return []

        if debug_enabled:
            logger.debug(f"Analyzing data from {data.index[0]} to {data.index[-1]}")

        signals = []
        macd_data = self.calculate_macd(data)
        
        # Calculate technical indicators
        volume = data['Volume']
        vol_ma = pd.to_numeric(volume, errors='coerce').rolling(window=self.params['volume_ma_period']).mean()
        
        # Calculate volume ratios
        volume_ratios = pd.Series(index=volume.index)
        for date in volume.index[5:]:
            recent_vol = volume.loc[date]
            past_avg = volume.loc[date-pd.Timedelta(days=3):date-pd.Timedelta(days=1)].mean()
            volume_ratios.loc[date] = recent_vol / past_avg if past_avg > 0 else 0
                
        price_change_1d = data['Close'].pct_change()
        price_change_5d = data['Close'].pct_change(periods=5)
        rsi = self.calculate_rsi(data)

        # Get last 7 days of data
        end_date = data.index[-1]
        last_7_days = data.loc[end_date - pd.Timedelta(days=7):end_date]
        if len(last_7_days) < 2:
            logger.warning("Insufficient data for 7-day analysis")
            return []
            
        historical_values = {
            'dates': last_7_days.index.strftime('%Y-%m-%d').tolist(),
            'macd_values': macd_data['MACD'].loc[last_7_days.index].tolist(),
            'signal_values': macd_data['Signal'].loc[last_7_days.index].tolist(),
            'histogram_values': macd_data['Histogram'].loc[last_7_days.index].tolist()
        }

        # Detect crossovers
        for i in range(1, len(last_7_days)):
            current_date = last_7_days.index[i]
            previous_date = last_7_days.index[i-1]
            
            current_dif = macd_data['MACD'].loc[current_date]
            current_dea = macd_data['Signal'].loc[current_date]
            previous_dif = macd_data['MACD'].loc[previous_date]
            previous_dea = macd_data['Signal'].loc[previous_date]
            current_hist = macd_data['Histogram'].loc[current_date]
            
            current_rsi = rsi.loc[current_date]
            current_volume = volume.loc[current_date]
            current_volume_ma = vol_ma.loc[current_date]
            current_volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 0
            
            if debug_enabled:
                logger.debug(f"Checking {current_date.strftime('%Y-%m-%d')} - RSI: {current_rsi:.2f}, Volume ratio: {current_volume_ratio:.2f}")
            
            epsilon = 1e-4
            trend_threshold = 0.05
            trend_strength = abs(current_dif - current_dea)
            prev_trend_strength = abs(previous_dif - previous_dea)
            
            # Buy signal conditions
            if (previous_dif <= previous_dea and
                current_dif > current_dea and
                trend_strength > trend_threshold and
                trend_strength > prev_trend_strength and
                current_volume_ratio > self.params['volume_threshold'] and
                current_rsi < self.params['rsi_overbought']):
                
                if debug_enabled:
                    logger.debug(f"BUY signal found - Trend strength: {trend_strength:.4f}")
                else:
                    logger.info(f"BUY signal detected at {current_date.strftime('%Y-%m-%d')}")
                
                strength = self.calculate_signal_strength(
                    macd_hist=current_hist,
                    volume_ratio=current_volume_ratio,
                    price_change=price_change_1d.loc[current_date]
                )
                
                signal = self._create_signal_dict(
                    data=last_7_days.loc[[current_date]],
                    macd_data=macd_data.loc[[current_date]],
                    volume=volume.loc[[current_date]],
                    vol_ma=vol_ma.loc[[current_date]],
                    volume_ratio=pd.Series([current_volume_ratio], index=[current_date]),
                    price_changes=(
                        price_change_1d.loc[[current_date]],
                        price_change_5d.loc[[current_date]]
                    ),
                    rsi=current_rsi,
                    strength=strength,
                    signal_type=SignalType.BUY,
                    historical_values=historical_values,
                    signal_date=current_date
                )
                signals.append(signal)
            
            # Sell signal conditions
            elif (previous_dif >= previous_dea and
                  current_dif < current_dea and
                  trend_strength > trend_threshold and
                  trend_strength > prev_trend_strength and
                  current_volume_ratio > self.params['volume_threshold'] and
                  current_rsi > self.params['rsi_oversold']):
                
                if debug_enabled:
                    logger.debug(f"SELL signal found - Trend strength: {trend_strength:.4f}")
                else:
                    logger.info(f"SELL signal detected at {current_date.strftime('%Y-%m-%d')}")
                
                strength = self.calculate_signal_strength(
                    macd_hist=-current_hist,
                    volume_ratio=current_volume_ratio,
                    price_change=-price_change_1d.loc[current_date]
                )
                
                signal = self._create_signal_dict(
                    data=last_7_days.loc[[current_date]],
                    macd_data=macd_data.loc[[current_date]],
                    volume=volume.loc[[current_date]],
                    vol_ma=vol_ma.loc[[current_date]],
                    volume_ratio=pd.Series([current_volume_ratio], index=[current_date]),
                    price_changes=(
                        price_change_1d.loc[[current_date]],
                        price_change_5d.loc[[current_date]]
                    ),
                    rsi=current_rsi,
                    strength=strength,
                    signal_type=SignalType.SELL,
                    historical_values=historical_values,
                    signal_date=current_date
                )
                signals.append(signal)
        
        return signals

    def _create_signal_dict(self, data, macd_data, volume, vol_ma, volume_ratio, 
                         price_changes, rsi, strength, signal_type,
                         historical_values, signal_date) -> Dict:
        """Helper method to create signal dictionary."""
        price_change_1d, price_change_5d = price_changes
        return {
            'Price': float(data['Close'].iloc[-1]),
            'PriceChange1d': float(price_change_1d.iloc[-1]),
            'PriceChange5d': float(price_change_5d.iloc[-1]),
            'Volume': float(volume.iloc[-1]),
            'AverageVolume': float(vol_ma.iloc[-1]),
            'VolumeRatio': float(volume_ratio.iloc[-1]),
            'MACD': float(macd_data['MACD'].iloc[-1]),
            'Signal': float(macd_data['Signal'].iloc[-1]),
            'Histogram': float(macd_data['Histogram'].iloc[-1]),
            'History': {
                'dates': historical_values['dates'],
                'macd_values': historical_values['macd_values'],
                'signal_values': historical_values['signal_values'],
                'histogram_values': historical_values['histogram_values']
            },
            'RSI': rsi,
            'SignalStrength': float(strength),
            'Type': signal_type,
            'SignalDate': signal_date.strftime('%Y-%m-%d')
        }

    def filter_stocks(self) -> List[Dict]:
        """Process all stocks across exchanges and filter based on MACD signals."""
        filtered_stocks = []
        
        for exchange in self.exchanges:
            if debug_enabled:
                logger.debug(f"Processing {exchange} stocks")
            
            exchange_dir = os.path.join(self.base_dir, exchange)
            if not os.path.exists(exchange_dir):
                logger.warning(f"Exchange directory not found: {exchange_dir}")
                continue
                
            stock_files = [f for f in os.listdir(exchange_dir) if f.endswith('.json')]
            
            if debug_enabled:
                logger.debug(f"Found {len(stock_files)} stock files in {exchange}")
            
            for stock_file in stock_files:
                try:
                    symbol = stock_file.replace('.json', '')
                    file_path = os.path.join(exchange_dir, stock_file)
                    
                    with open(file_path, 'r') as f:
                        stock_data = json.load(f)
                    
                    if not stock_data.get('history'):
                        logger.warning(f"Invalid/missing data for {exchange}:{symbol}")
                        continue
                    
                    df = pd.DataFrame(stock_data['history'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.columns = df.columns.str.capitalize()
                    
                    signals = self.analyze(df)
                    
                    for signal in signals:
                        filtered_signal = {
                            'Exchange': exchange,
                            'Symbol': symbol,
                            'Price': signal['Price'],
                            'Volume': signal['Volume'],
                            'MACD': signal['MACD'],
                            'Signal': signal['Signal'],
                            'Histogram': signal['Histogram'],
                            'Type': signal['Type'],
                            'SignalDate': signal['SignalDate'],
                            'RSI': signal['RSI'],
                            'SignalStrength': signal['SignalStrength']
                        }
                        filtered_stocks.append(filtered_signal)
                        
                except Exception as e:
                    logger.error(f"Error processing {exchange}:{symbol}: {str(e)}")
                    continue
        
        return filtered_stocks