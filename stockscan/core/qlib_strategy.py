import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union, Any

# Import Qlib modules
import qlib
from qlib.data import D

logger = logging.getLogger(__name__)

class QlibStrategyRunner:
    """
    Simplified Qlib-based strategy runner
    """
    def __init__(self, region: str = "us", config_file: Optional[str] = None):
        """
        Initialize the Qlib Strategy Runner.
        
        Args:
            region: Region code ('us', 'cn', etc.)
            config_file: Path to the Qlib configuration file
        """
        self.region = region
        self.config_file = config_file
        self.initialized = False
        
    def init_qlib(self, data_path: Optional[str] = None):
        """
        Initialize Qlib with the specified configuration.
        
        Args:
            data_path: Path to the data directory
        """
        if self.initialized:
            return
            
        try:
            # Use config file if provided
            if self.config_file and os.path.exists(self.config_file):
                import yaml
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                qlib.init(**config)
                logger.info(f"Initialized Qlib using config file: {self.config_file}")
                self.initialized = True
                return
                
            # Set up default data path if not provided
            if data_path is None:
                data_path = os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "us_data")
                os.makedirs(data_path, exist_ok=True)
            
            # Initialize Qlib
            if self.region.lower() == "cn":
                # For Chinese market data
                from qlib.config import REG_CN
                qlib.init(provider_uri=data_path, region=REG_CN)
            else:
                # For US market data
                qlib.init(provider_uri=data_path)
                
            self.initialized = True
            logger.info(f"Qlib initialized with data path: {data_path}")
            
        except Exception as e:
            logger.error(f"Error initializing Qlib: {str(e)}")
            raise
            
    def load_data(self, symbol: str, data_dir: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load stock data from JSON file.
        
        Args:
            symbol: Stock symbol
            data_dir: Data directory
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Set default end date if not provided
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            # Construct file path
            file_path = os.path.join(data_dir, self.region.upper(), f"{symbol}.json")
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                return pd.DataFrame()
                
            # Load JSON data
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Convert to DataFrame
            if 'history' in data and isinstance(data['history'], list):
                df = pd.DataFrame(data['history'])
            else:
                logger.error(f"Invalid data format in {file_path}")
                return pd.DataFrame()
                
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            # Sort by date
            df = df.sort_values('date')
            
            # Set date as index
            df = df.set_index('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def run_macd_strategy(self, symbol: str, data_dir: str, 
                         start_date: str, end_date: Optional[str] = None,
                         fast_period: int = 12, slow_period: int = 26, 
                         signal_period: int = 9) -> Dict[str, Any]:
        """
        Run MACD strategy on stock data.
        
        Args:
            symbol: Stock symbol
            data_dir: Data directory
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal period
            
        Returns:
            Dictionary with strategy results
        """
        try:
            # Load data
            df = self.load_data(symbol, data_dir, start_date, end_date)
            
            if df.empty:
                logger.error(f"No data available for {symbol}")
                return {
                    'success': False,
                    'message': f"No data available for {symbol}"
                }
                
            # Calculate MACD
            # Fast EMA
            df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
            # Slow EMA
            df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
            # MACD Line
            df['macd'] = df['ema_fast'] - df['ema_slow']
            # Signal Line
            df['signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            # Histogram
            df['histogram'] = df['macd'] - df['signal']
            
            # Generate trading signals
            # Buy when MACD crosses above Signal
            df['signal_value'] = 0
            df.loc[df['macd'] > df['signal'], 'signal_value'] = 1  # Buy signal
            df.loc[df['macd'] < df['signal'], 'signal_value'] = -1  # Sell signal
            
            # Detect signal changes (crossovers)
            df['signal_change'] = df['signal_value'].diff()
            
            # Find buy and sell signals
            buy_signals = df[df['signal_change'] == 2].copy()  # Changed from -1 to 1
            sell_signals = df[df['signal_change'] == -2].copy()  # Changed from 1 to -1
            
            # Prepare results
            signals = []
            
            # Add buy signals
            for idx, row in buy_signals.iterrows():
                signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'buy',
                    'price': row['close'],
                    'macd': row['macd'],
                    'signal': row['signal'],
                    'histogram': row['histogram']
                })
                
            # Add sell signals
            for idx, row in sell_signals.iterrows():
                signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'sell',
                    'price': row['close'],
                    'macd': row['macd'],
                    'signal': row['signal'],
                    'histogram': row['histogram']
                })
                
            # Sort signals by date
            signals = sorted(signals, key=lambda x: x['date'])
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(df, signals)
            
            # Calculate current signal
            current_signal = 'neutral'
            if not df.empty:
                last_row = df.iloc[-1]
                if last_row['macd'] > last_row['signal']:
                    current_signal = 'buy'
                elif last_row['macd'] < last_row['signal']:
                    current_signal = 'sell'
                    
            return {
                'success': True,
                'symbol': symbol,
                'period': f"{start_date} to {end_date}",
                'signals': signals,
                'metrics': metrics,
                'current_signal': current_signal,
                'parameters': {
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'signal_period': signal_period
                }
            }
            
        except Exception as e:
            logger.error(f"Error running MACD strategy for {symbol}: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
            
    def run_rsi_strategy(self, symbol: str, data_dir: str,
                       start_date: str, end_date: Optional[str] = None,
                       rsi_period: int = 14, overbought: int = 70,
                       oversold: int = 30) -> Dict[str, Any]:
        """
        Run RSI strategy on stock data.
        
        Args:
            symbol: Stock symbol
            data_dir: Data directory
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rsi_period: RSI period
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            Dictionary with strategy results
        """
        try:
            # Load data
            df = self.load_data(symbol, data_dir, start_date, end_date)
            
            if df.empty:
                logger.error(f"No data available for {symbol}")
                return {
                    'success': False,
                    'message': f"No data available for {symbol}"
                }
                
            # Calculate RSI
            # Calculate price changes
            df['price_change'] = df['close'].diff()
            
            # Calculate gains and losses
            df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
            df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
            
            # Calculate average gain and loss
            df['avg_gain'] = df['gain'].rolling(window=rsi_period).mean()
            df['avg_loss'] = df['loss'].rolling(window=rsi_period).mean()
            
            # Calculate RS and RSI
            df['rs'] = df['avg_gain'] / df['avg_loss']
            df['rsi'] = 100 - (100 / (1 + df['rs']))
            
            # Generate trading signals
            df['signal_value'] = 0
            df.loc[df['rsi'] < oversold, 'signal_value'] = 1  # Buy signal when RSI below oversold
            df.loc[df['rsi'] > overbought, 'signal_value'] = -1  # Sell signal when RSI above overbought
            
            # Detect signal changes
            df['prev_signal'] = df['signal_value'].shift(1)
            df['signal_change'] = df['signal_value'] - df['prev_signal']
            
            # Find buy and sell signals
            buy_signals = df[(df['signal_change'] != 0) & (df['signal_value'] == 1)].copy()
            sell_signals = df[(df['signal_change'] != 0) & (df['signal_value'] == -1)].copy()
            
            # Prepare results
            signals = []
            
            # Add buy signals
            for idx, row in buy_signals.iterrows():
                signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'buy',
                    'price': row['close'],
                    'rsi': row['rsi'],
                })
                
            # Add sell signals
            for idx, row in sell_signals.iterrows():
                signals.append({
                    'date': idx.strftime('%Y-%m-%d'),
                    'type': 'sell',
                    'price': row['close'],
                    'rsi': row['rsi'],
                })
                
            # Sort signals by date
            signals = sorted(signals, key=lambda x: x['date'])
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(df, signals)
            
            # Calculate current signal
            current_signal = 'neutral'
            if not df.empty:
                last_row = df.iloc[-1]
                if last_row['rsi'] < oversold:
                    current_signal = 'buy'
                elif last_row['rsi'] > overbought:
                    current_signal = 'sell'
                    
            return {
                'success': True,
                'symbol': symbol,
                'period': f"{start_date} to {end_date}",
                'signals': signals,
                'metrics': metrics,
                'current_signal': current_signal,
                'parameters': {
                    'rsi_period': rsi_period,
                    'overbought': overbought,
                    'oversold': oversold
                }
            }
            
        except Exception as e:
            logger.error(f"Error running RSI strategy for {symbol}: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
            
    def _calculate_performance_metrics(self, df: pd.DataFrame, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics from signals.
        
        Args:
            df: DataFrame with stock data
            signals: List of signals
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Skip if no signals
            if not signals:
                return {
                    'total_signals': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                }
                
            # Count buy and sell signals
            buy_signals = [signal for signal in signals if signal['type'] == 'buy']
            sell_signals = [signal for signal in signals if signal['type'] == 'sell']
            
            metrics = {
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
            }
            
            # Calculate strategy performance (simplified)
            # Assume we trade with fixed position size
            if len(buy_signals) > 0 and len(sell_signals) > 0:
                # Create a list of all transactions
                trades = []
                
                # Simulate trading
                position = 0
                entry_price = 0
                entry_date = None
                
                for signal in signals:
                    if signal['type'] == 'buy' and position == 0:
                        # Enter long position
                        position = 1
                        entry_price = signal['price']
                        entry_date = signal['date']
                    elif signal['type'] == 'sell' and position == 1:
                        # Close long position
                        position = 0
                        exit_price = signal['price']
                        exit_date = signal['date']
                        
                        # Calculate profit
                        profit_pct = (exit_price - entry_price) / entry_price
                        
                        # Add trade
                        trades.append({
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': exit_date,
                            'exit_price': exit_price,
                            'profit_pct': profit_pct
                        })
                        
                # Calculate trade metrics
                if trades:
                    wins = [trade for trade in trades if trade['profit_pct'] > 0]
                    losses = [trade for trade in trades if trade['profit_pct'] <= 0]
                    
                    # Calculate win rate
                    metrics['total_trades'] = len(trades)
                    metrics['winning_trades'] = len(wins)
                    metrics['losing_trades'] = len(losses)
                    metrics['win_rate'] = len(wins) / len(trades) if len(trades) > 0 else 0
                    
                    # Calculate average profit
                    if wins:
                        metrics['avg_win_pct'] = sum(trade['profit_pct'] for trade in wins) / len(wins)
                    if losses:
                        metrics['avg_loss_pct'] = sum(trade['profit_pct'] for trade in losses) / len(losses)
                        
                    # Calculate cumulative return
                    metrics['cumulative_return_pct'] = (1 + sum(trade['profit_pct'] for trade in trades)) - 1
                        
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'total_signals': len(signals) if signals else 0
            } 