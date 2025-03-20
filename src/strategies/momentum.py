"""
Momentum trading strategy implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class MomentumStrategy:
    def __init__(self, symbols: List[str], start_date: datetime, end_date: datetime):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        
        # Strategy parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.ma_short = 10
        self.ma_medium = 20
        self.ma_long = 50
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate trading signals based on technical indicators
        """
        signals = {}
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
                
            df = data[symbol].copy()
            
            # Calculate technical indicators
            self._calculate_indicators(df)
            
            # Generate signals based on multiple indicators
            signal = pd.Series(0, index=df.index)
            
            # RSI signals
            rsi_signal = pd.Series(0, index=df.index)
            rsi_signal[df['rsi'] < self.rsi_oversold] = 1  # Buy signal
            rsi_signal[df['rsi'] > self.rsi_overbought] = -1  # Sell signal
            
            # MACD signals
            macd_signal = pd.Series(0, index=df.index)
            macd_signal[df['macd'] > df['signal']] = 1  # Buy signal
            macd_signal[df['macd'] < df['signal']] = -1  # Sell signal
            
            # Bollinger Bands signals
            bb_signal = pd.Series(0, index=df.index)
            bb_signal[df['Close'] < df['bb_lower']] = 1  # Buy signal
            bb_signal[df['Close'] > df['bb_upper']] = -1  # Sell signal
            
            # Moving Average signals
            ma_signal = pd.Series(0, index=df.index)
            ma_signal[(df['ma10'] > df['ma20']) & (df['ma20'] > df['ma50'])] = 1  # Buy signal
            ma_signal[(df['ma10'] < df['ma20']) & (df['ma20'] < df['ma50'])] = -1  # Sell signal
            
            # Combine signals with weights
            signal = (
                rsi_signal * 0.3 +
                macd_signal * 0.3 +
                bb_signal * 0.2 +
                ma_signal * 0.2
            )
            
            # Normalize signals to [-1, 1] range
            signal = signal.clip(-1, 1)
            
            # Add trend strength
            trend_strength = abs(df['ma10'] / df['ma20'] - 1)
            signal = signal * (1 + trend_strength)
            
            signals[symbol] = signal
            
        return signals
    
    def predict_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        Generate predictions for future price movements
        """
        predictions = {}
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
                
            df = data[symbol].copy()
            
            # Calculate technical indicators
            self._calculate_indicators(df)
            
            # Generate base signals
            signals = self.generate_signals({symbol: df})
            signal = signals[symbol]
            
            # Calculate prediction confidence
            confidence = pd.Series(0.0, index=df.index)
            
            # RSI confidence
            rsi_confidence = 1 - abs(df['rsi'] - 50) / 50  # Higher confidence when RSI is closer to 50
            
            # MACD confidence
            macd_confidence = abs(df['macd'] - df['signal']) / df['Close']  # Normalized by price
            
            # Trend confidence
            trend_confidence = abs(df['ma10'] / df['ma20'] - 1)  # Stronger trend = higher confidence
            
            # Combine confidence metrics
            confidence = (
                rsi_confidence * 0.3 +
                macd_confidence * 0.3 +
                trend_confidence * 0.4
            )
            
            # Scale confidence to [0, 1] range
            confidence = confidence.clip(0, 1)
            
            # Generate predictions with confidence
            prediction = signal * confidence
            
            # Add momentum factor
            returns = df['Close'].pct_change()
            momentum = returns.rolling(window=20).mean()
            prediction = prediction * (1 + momentum)
            
            predictions[symbol] = prediction
            
        return predictions
    
    def _calculate_indicators(self, df: pd.DataFrame):
        """
        Calculate technical indicators for the given dataframe
        """
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=self.bb_period).mean()
        bb_std = df['Close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)
        
        # Calculate Moving Averages
        df['ma10'] = df['Close'].rolling(window=self.ma_short).mean()
        df['ma20'] = df['Close'].rolling(window=self.ma_medium).mean()
        df['ma50'] = df['Close'].rolling(window=self.ma_long).mean() 