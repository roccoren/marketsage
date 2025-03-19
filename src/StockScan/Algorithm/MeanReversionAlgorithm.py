"""
MeanReversionAlgorithm.py - Mean Reversion Strategy

This module implements a mean reversion strategy based on RSI (Relative Strength Index)
using the LEAN framework. It goes long when the RSI is oversold and exits (or shorts)
when the RSI is overbought.
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


class MeanReversionAlgorithm(BaseAlgorithm):
    """
    Mean reversion algorithm using RSI (Relative Strength Index).
    
    This algorithm implements a mean reversion strategy based on RSI.
    It goes long when the RSI falls below the oversold threshold,
    and exits (or goes short if enabled) when the RSI rises above
    the overbought threshold.
    
    Additional features include:
    - Bollinger Bands for additional confirmation
    - ATR-based stop loss and take profit
    - Risk management controls
    """
    
    def Initialize(self) -> None:
        """Initialize the algorithm settings and state."""
        # Call base class Initialize
        super().Initialize()
        
        # Algorithm parameters
        self.RsiPeriod = 14  # RSI period
        self.OversoldThreshold = 30  # RSI oversold threshold
        self.OverboughtThreshold = 70  # RSI overbought threshold
        self.BollingerPeriod = 20  # Bollinger Bands period
        self.BollingerStdDev = 2.0  # Bollinger Bands standard deviation
        self.UseBollingerConfirmation = True  # Whether to use Bollinger Bands for confirmation
        self.EnableShort = False  # Whether to allow short positions
        self.UseATRForStops = True  # Whether to use ATR for stop losses
        self.ATRPeriod = 14  # ATR period for volatility calculation
        self.ATRMultiplier = 2.0  # Multiplier for ATR-based stops
        
        # Set warm-up period to ensure indicators are ready
        self.SetWarmUp(max(self.RsiPeriod, self.BollingerPeriod) * 2)
        
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
        # Create RSI indicator
        rsi = RelativeStrengthIndex(symbol, self.RsiPeriod)
        
        # Create Bollinger Bands indicator
        bb = BollingerBands(symbol, self.BollingerPeriod, self.BollingerStdDev)
        
        # Create helper indicators
        atr = AverageTrueRange(symbol, self.ATRPeriod)
        ema50 = ExponentialMovingAverage(symbol, 50)  # Medium-term trend
        volume_ma = SimpleMovingAverage(symbol, 20)  # 20-day volume average
        
        # Store indicators in dictionary
        self.indicators[symbol] = {
            'rsi': rsi,
            'bb': bb,
            'atr': atr,
            'ema50': ema50,
            'volume_ma': volume_ma,
            'previous_rsi': 0,  # Previous RSI value for crossover detection
            'position': 0  # 1 for long, -1 for short, 0 for no position
        }
        
        # Register indicators with consolidator
        self.RegisterIndicator(symbol, rsi, Resolution.Daily)
        self.RegisterIndicator(symbol, bb, Resolution.Daily)
        self.RegisterIndicator(symbol, atr, Resolution.Daily)
        self.RegisterIndicator(symbol, ema50, Resolution.Daily)
        self.RegisterIndicator(symbol, volume_ma, Resolution.Daily)
        
        # Plot indicators
        self.PlotIndicator(f"{symbol}", "RSI", rsi)
        self.PlotIndicator(f"{symbol}", "Bollinger Middle", bb.MiddleBand)
        self.PlotIndicator(f"{symbol}", "Bollinger Upper", bb.UpperBand)
        self.PlotIndicator(f"{symbol}", "Bollinger Lower", bb.LowerBand)
        self.PlotIndicator(f"{symbol}2", "EMA50", ema50)
        self.PlotIndicator(f"{symbol}2", "ATR", atr)
    
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
            if not indicators['rsi'].IsReady or not indicators['bb'].IsReady:
                continue
            
            # Get indicator values
            current_rsi = indicators['rsi'].Current.Value
            previous_rsi = indicators['previous_rsi']
            
            # Update previous RSI value for next time
            indicators['previous_rsi'] = current_rsi
            
            # Get Bollinger Bands values
            bb_middle = indicators['bb'].MiddleBand.Current.Value
            bb_upper = indicators['bb'].UpperBand.Current.Value
            bb_lower = indicators['bb'].LowerBand.Current.Value
            
            # Get current position
            current_position = self.Portfolio[symbol].Quantity
            position_sign = 1 if current_position > 0 else (-1 if current_position < 0 else 0)
            indicators['position'] = position_sign
            
            # Get current price and other data
            current_price = data[symbol].Price
            current_atr = indicators['atr'].Current.Value
            current_ema50 = indicators['ema50'].Current.Value
            
            # Get volume confirmation
            current_volume = data[symbol].Volume
            avg_volume = indicators['volume_ma'].Current.Value
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Log current status
            self.Debug(f"{symbol}: RSI = {current_rsi:.2f}, Price = ${current_price:.2f}, BB Mid = ${bb_middle:.2f}")
            
            # Check for oversold condition (buy signal)
            if current_rsi < self.OversoldThreshold and previous_rsi >= self.OversoldThreshold:
                # Additional confirmations
                price_below_lower_band = current_price <= bb_lower  # Price below lower Bollinger Band
                price_above_ema50 = current_price > current_ema50  # Medium-term uptrend
                volume_confirmed = volume_ratio >= 1.2  # Above average volume
                
                # Check if we should enter long
                if position_sign <= 0 and self.check_risk_limits():
                    # Check if we need Bollinger Band confirmation
                    bollinger_confirmed = True
                    if self.UseBollingerConfirmation:
                        bollinger_confirmed = price_below_lower_band
                    
                    # Final confirmation check
                    if bollinger_confirmed and volume_confirmed:
                        self.Debug(f"OVERSOLD SIGNAL for {symbol} - Entering long position")
                        
                        # Calculate position size
                        quantity = self.calculate_position_size(symbol)
                        
                        # Set stop loss and take profit
                        if self.UseATRForStops:
                            self.stop_loss_pct = (current_atr * self.ATRMultiplier) / current_price
                            self.take_profit_pct = self.stop_loss_pct * 1.5  # 1.5:1 reward/risk ratio
                        
                        # Close any existing short position first
                        if position_sign < 0:
                            self.Liquidate(symbol)
                        
                        # Place long trade
                        self.place_trade(symbol, 'long', quantity)
                        
                        # Log trade details
                        self.Debug(f"Entered long position in {symbol}: {quantity} shares at ${current_price:.2f}")
                        self.Debug(f"Stop loss: ${current_price * (1 - self.stop_loss_pct):.2f}, Take profit: ${current_price * (1 + self.take_profit_pct):.2f}")
            
            # Check for overbought condition (sell/short signal)
            elif current_rsi > self.OverboughtThreshold and previous_rsi <= self.OverboughtThreshold:
                # Additional confirmations
                price_above_upper_band = current_price >= bb_upper  # Price above upper Bollinger Band
                price_below_ema50 = current_price < current_ema50  # Medium-term downtrend
                volume_confirmed = volume_ratio >= 1.2  # Above average volume
                
                # If we have a long position, exit it
                if position_sign > 0:
                    self.Debug(f"OVERBOUGHT SIGNAL for {symbol} - Exiting long position")
                    self.Liquidate(symbol)
                
                # If we allow short positions and all conditions align, enter short
                if self.EnableShort and position_sign >= 0 and self.check_risk_limits():
                    # Check if we need Bollinger Band confirmation
                    bollinger_confirmed = True
                    if self.UseBollingerConfirmation:
                        bollinger_confirmed = price_above_upper_band
                    
                    # Final confirmation check
                    if bollinger_confirmed and volume_confirmed and price_below_ema50:
                        self.Debug(f"OVERBOUGHT SIGNAL for {symbol} - Entering short position")
                        
                        # Calculate position size
                        quantity = self.calculate_position_size(symbol)
                        
                        # Set stop loss and take profit
                        if self.UseATRForStops:
                            self.stop_loss_pct = (current_atr * self.ATRMultiplier) / current_price
                            self.take_profit_pct = self.stop_loss_pct * 1.5  # 1.5:1 reward/risk ratio
                        
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
        self.Debug("----- End of Day Positions and Indicators -----")
        for s in self.symbols:
            if self.indicators.get(s) and self.indicators[s]['rsi'].IsReady:
                rsi_value = self.indicators[s]['rsi'].Current.Value
                
                if self.Portfolio[s].Invested:
                    position = self.Portfolio[s]
                    self.Debug(f"{s}: {position.Quantity} shares at ${position.AveragePrice:.2f}, P&L: ${position.UnrealizedProfit:.2f}, RSI: {rsi_value:.2f}")
                else:
                    self.Debug(f"{s}: No position, RSI: {rsi_value:.2f}")
    
    def OnEndOfAlgorithm(self) -> None:
        """Event handler called at the end of the algorithm execution."""
        # Call base class implementation
        super().OnEndOfAlgorithm()
        
        # Calculate and log additional metrics
        win_rate = self.metrics['winning_trades'] / max(1, self.metrics['total_trades'])
        profit_factor = abs(self.metrics['avg_win'] * self.metrics['winning_trades']) / abs(self.metrics['avg_loss'] * max(1, self.metrics['losing_trades'])) if self.metrics['avg_loss'] != 0 else float('inf')
        
        # Log additional performance metrics
        self.Log(f"Strategy: Mean Reversion (RSI Period: {self.RsiPeriod}, Oversold: {self.OversoldThreshold}, Overbought: {self.OverboughtThreshold})")
        self.Log(f"Win Rate: {win_rate:.2%}")
        self.Log(f"Profit Factor: {profit_factor:.2f}")
        self.Log(f"Average Win: ${self.metrics['avg_win']:.2f}")
        self.Log(f"Average Loss: ${self.metrics['avg_loss']:.2f}") 