"""
TrendFollowingAlgorithm.py - Trend Following Strategy

This module implements a trend following strategy based on moving average crossovers
using the LEAN framework. It goes long when the fast MA crosses above the slow MA,
and short (or exits) when the fast MA crosses below the slow MA.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import LEAN components
try:
    import clr
    clr.AddReference("QuantConnect.Common")
    clr.AddReference("QuantConnect.Algorithm")
    clr.AddReference("QuantConnect.Indicators")
    
    from QuantConnect import *
    from QuantConnect.Algorithm import *
    from QuantConnect.Data import *
    from QuantConnect.Data.Market import *
    from QuantConnect.Indicators import *
except ImportError:
    logging.warning("LEAN framework import failed")

from src.StockScan.Algorithm.BaseAlgorithm import BaseAlgorithm

# Set up logger
logger = logging.getLogger(__name__)


class TrendFollowingAlgorithm(BaseAlgorithm):
    """
    Trend following algorithm using moving average crossovers.
    
    This algorithm implements a trend following strategy based on moving average crossovers.
    It goes long when the fast moving average crosses above the slow moving average,
    and exits (or goes short if enabled) when the fast moving average crosses below
    the slow moving average.
    
    Additional features include:
    - Volume confirmation for trade entries
    - ATR-based stop loss and take profit
    - Risk management controls
    """
    
    def Initialize(self) -> None:
        """Initialize the algorithm settings and state."""
        # Call base class Initialize
        super().Initialize()
        
        # Algorithm parameters
        self.FastPeriod = 10  # Fast moving average period
        self.SlowPeriod = 30  # Slow moving average period
        self.VolumeThreshold = 1.5  # Volume increase threshold for confirmation
        self.EnableShort = False  # Whether to allow short positions
        self.UseATRForStops = True  # Whether to use ATR for stop losses
        self.ATRPeriod = 14  # ATR period for volatility calculation
        self.ATRMultiplier = 2.0  # Multiplier for ATR-based stops
        
        # Set warm-up period to ensure indicators are ready
        self.SetWarmUp(self.SlowPeriod)
        
        # Initialize indicators dict
        self.indicators = {}
    
    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        """
        Event handler for security changes.
        This is called when securities are added or removed from the algorithm.
        
        Args:
            changes: The security changes
        """
        # Initialize indicators for added securities
        for security in changes.AddedSecurities:
            self.initialize_indicators(security.Symbol)
    
    def initialize_indicators(self, symbol: Symbol) -> None:
        """
        Initialize indicators for a specific symbol.
        
        Args:
            symbol: Symbol to initialize indicators for
        """
        # Create indicators
        fast_ma = ExponentialMovingAverage(symbol, self.FastPeriod)
        slow_ma = ExponentialMovingAverage(symbol, self.SlowPeriod)
        
        # Create helper indicators
        atr = AverageTrueRange(symbol, self.ATRPeriod) 
        volume_ma = SimpleMovingAverage(symbol, 20)  # 20-day volume average
        rsi = RelativeStrengthIndex(symbol, 14)  # RSI for additional confirmation
        
        # Store indicators in dictionary
        self.indicators[symbol] = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'atr': atr,
            'volume_ma': volume_ma,
            'rsi': rsi,
            'last_crossover': 0,  # 1 for bullish, -1 for bearish, 0 for neutral
            'position': 0  # 1 for long, -1 for short, 0 for no position
        }
        
        # Register indicators with consolidator
        self.RegisterIndicator(symbol, fast_ma, Resolution.Daily)
        self.RegisterIndicator(symbol, slow_ma, Resolution.Daily)
        self.RegisterIndicator(symbol, atr, Resolution.Daily)
        self.RegisterIndicator(symbol, volume_ma, Resolution.Daily)
        self.RegisterIndicator(symbol, rsi, Resolution.Daily)
        
        # Plot indicators
        self.PlotIndicator(f"{symbol}", "Fast MA", fast_ma)
        self.PlotIndicator(f"{symbol}", "Slow MA", slow_ma)
        self.PlotIndicator(f"{symbol}2", "ATR", atr)
        self.PlotIndicator(f"{symbol}2", "RSI", rsi)
    
    def OnData(self, data: Slice) -> None:
        """
        Event handler for market data updates.
        This is the main entry point for trading logic.
        
        Args:
            data: The new data slice
        """
        # Skip during warmup
        if self.IsWarmingUp:
            self.Debug("Algorithm is warming up...")
            return
        
        # Process each symbol
        for symbol in self.symbols:
            # Skip if we don't have data for this symbol or indicators are not ready
            if not self.indicators.get(symbol) or not data.ContainsKey(symbol):
                continue
            
            # Get indicators for this symbol
            indicators = self.indicators[symbol]
            
            # Skip if indicators aren't ready
            if not indicators['fast_ma'].IsReady or not indicators['slow_ma'].IsReady:
                continue
            
            # Get indicator values
            fast_value = indicators['fast_ma'].Current.Value
            slow_value = indicators['slow_ma'].Current.Value
            
            # Calculate crossover - positive means fast is above slow (bullish)
            current_crossover = 1 if fast_value > slow_value else -1
            previous_crossover = indicators['last_crossover']
            
            # Update last crossover value for next time
            indicators['last_crossover'] = current_crossover
            
            # Get current position
            current_position = self.Portfolio[symbol].Quantity
            position_sign = 1 if current_position > 0 else (-1 if current_position < 0 else 0)
            indicators['position'] = position_sign
            
            # Check for crossover signal
            crossover_signal = 0
            
            # Bullish crossover (fast crosses above slow)
            if current_crossover == 1 and previous_crossover == -1:
                crossover_signal = 1
            
            # Bearish crossover (fast crosses below slow)
            elif current_crossover == -1 and previous_crossover == 1:
                crossover_signal = -1
                
            # Log current status
            self.Debug(f"{symbol}: Fast MA = {fast_value:.2f}, Slow MA = {slow_value:.2f}, Signal = {crossover_signal}")
            
            # Get volume confirmation
            current_volume = data[symbol].Volume
            avg_volume = indicators['volume_ma'].Current.Value
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Get ATR for volatility-based stops
            current_atr = indicators['atr'].Current.Value
            current_price = data[symbol].Price
            
            # Check for trade signals
            if crossover_signal != 0:
                # Volume confirmation
                volume_confirmed = volume_ratio >= self.VolumeThreshold
                
                # Entry logic
                if crossover_signal == 1 and position_sign <= 0:  # Bullish signal
                    # Check if we should enter long
                    if self.check_risk_limits() and volume_confirmed:
                        self.Debug(f"BULLISH SIGNAL for {symbol} - Entering long position")
                        
                        # Calculate position size
                        quantity = self.calculate_position_size(symbol)
                        
                        # Set stop loss and take profit
                        if self.UseATRForStops:
                            self.stop_loss_pct = (current_atr * self.ATRMultiplier) / current_price
                            self.take_profit_pct = self.stop_loss_pct * 2  # 2:1 reward/risk ratio
                        
                        # Close any existing short position first
                        if position_sign < 0:
                            self.Liquidate(symbol)
                        
                        # Place long trade
                        self.place_trade(symbol, 'long', quantity)
                        
                        # Log trade details
                        self.Debug(f"Entered long position in {symbol}: {quantity} shares at ${current_price:.2f}")
                        self.Debug(f"Stop loss: ${current_price * (1 - self.stop_loss_pct):.2f}, Take profit: ${current_price * (1 + self.take_profit_pct):.2f}")
                
                elif crossover_signal == -1 and position_sign >= 0:  # Bearish signal
                    # If we have a long position, exit it
                    if position_sign > 0:
                        self.Debug(f"BEARISH SIGNAL for {symbol} - Exiting long position")
                        self.Liquidate(symbol)
                    
                    # If we allow short positions and volume confirms, enter short
                    if self.EnableShort and self.check_risk_limits() and volume_confirmed:
                        self.Debug(f"BEARISH SIGNAL for {symbol} - Entering short position")
                        
                        # Calculate position size
                        quantity = self.calculate_position_size(symbol)
                        
                        # Set stop loss and take profit
                        if self.UseATRForStops:
                            self.stop_loss_pct = (current_atr * self.ATRMultiplier) / current_price
                            self.take_profit_pct = self.stop_loss_pct * 2  # 2:1 reward/risk ratio
                        
                        # Place short trade
                        self.place_trade(symbol, 'short', quantity)
                        
                        # Log trade details
                        self.Debug(f"Entered short position in {symbol}: {quantity} shares at ${current_price:.2f}")
                        self.Debug(f"Stop loss: ${current_price * (1 + self.stop_loss_pct):.2f}, Take profit: ${current_price * (1 - self.take_profit_pct):.2f}")
    
    def OnEndOfDay(self, symbol: Symbol = None) -> None:
        """
        Event handler called at the end of each trading day.
        
        Args:
            symbol: The symbol for which the day has ended
        """
        # Call base class implementation
        super().OnEndOfDay(symbol)
        
        # Log positions at end of day
        self.Debug("----- End of Day Positions -----")
        for s in self.symbols:
            if self.Portfolio[s].Invested:
                position = self.Portfolio[s]
                self.Debug(f"{s}: {position.Quantity} shares at ${position.AveragePrice:.2f}, P&L: ${position.UnrealizedProfit:.2f}")
    
    def OnEndOfAlgorithm(self) -> None:
        """Event handler called at the end of the algorithm execution."""
        # Call base class implementation
        super().OnEndOfAlgorithm()
        
        # Calculate and log additional metrics
        win_rate = self.metrics['winning_trades'] / max(1, self.metrics['total_trades'])
        profit_factor = abs(self.metrics['avg_win'] * self.metrics['winning_trades']) / abs(self.metrics['avg_loss'] * max(1, self.metrics['losing_trades'])) if self.metrics['avg_loss'] != 0 else float('inf')
        
        # Log additional performance metrics
        self.Log(f"Strategy: Trend Following (Fast Period: {self.FastPeriod}, Slow Period: {self.SlowPeriod})")
        self.Log(f"Win Rate: {win_rate:.2%}")
        self.Log(f"Profit Factor: {profit_factor:.2f}")
        self.Log(f"Average Win: ${self.metrics['avg_win']:.2f}")
        self.Log(f"Average Loss: ${self.metrics['avg_loss']:.2f}") 