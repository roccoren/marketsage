"""
Mean Reversion Algorithm using LEAN

This module implements a mean reversion strategy based on RSI (Relative Strength Index)
using the QuantConnect LEAN algorithmic trading framework.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from clr import AddReference

# Import LEAN namespaces
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Indicators")

from QuantConnect import *
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Data.Market import TradeBar
from QuantConnect.Parameters import *
from QuantConnect.Indicators import *

from src.market_system.lean.algorithms.base_algorithm import BaseLeanAlgorithm


class MeanReversionAlgorithm(BaseLeanAlgorithm):
    """
    A mean reversion algorithm based on RSI (Relative Strength Index) using LEAN.
    
    This algorithm generates buy signals when the RSI falls below the oversold threshold,
    and sell signals when the RSI rises above the overbought threshold.
    """
    
    def Initialize(self) -> None:
        """Initialize the algorithm and strategy parameters."""
        # Call parent initialize method
        super().Initialize()
        
        # Strategy-specific parameters
        self.rsi_period = 14
        self.oversold_threshold = 30
        self.overbought_threshold = 70
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.10
        
        # Set up indicators dictionary
        self.indicators = {}
        
        # Set up a default symbol
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbols.append(self.symbol)
        
        # Initialize indicators for default symbol
        self.initialize_indicators(self.symbol)
        
        # Set warmup period to ensure indicators are ready
        self.SetWarmUp(self.rsi_period * 2)  # 2x RSI period for adequate warmup
    
    def initialize_indicators(self, symbol: Symbol) -> None:
        """
        Initialize indicators for a specific symbol.
        
        Args:
            symbol: Symbol to initialize indicators for
        """
        # Create RSI indicator
        rsi = RelativeStrengthIndex(self.rsi_period)
        
        # Store indicators in dictionary
        self.indicators[symbol] = {
            'rsi': rsi,
            'previous_rsi': 0,
            'atr': AverageTrueRange(14),  # For volatility-based stops
            'volume_ma': SimpleMovingAverage(20)  # For volume confirmation
        }
        
        # Register indicators with consolidator
        self.RegisterIndicator(symbol, rsi, Resolution.Daily)
        self.RegisterIndicator(symbol, self.indicators[symbol]['atr'], Resolution.Daily)
        self.RegisterIndicator(symbol, self.indicators[symbol]['volume_ma'], Resolution.Daily)
    
    def OnData(self, data: TradeBar) -> None:
        """
        Process market data and execute trading logic.
        
        Args:
            data: Market data received
        """
        # Skip during warmup
        if self.IsWarmingUp:
            return
        
        # Process each symbol
        for symbol in self.symbols:
            # Skip if we don't have data for this symbol
            if not data.ContainsKey(symbol):
                continue
            
            # Get the indicators for this symbol
            symbol_indicators = self.indicators[symbol]
            
            # Get indicator values
            current_rsi = symbol_indicators['rsi'].Current.Value
            previous_rsi = symbol_indicators['previous_rsi']
            current_atr = symbol_indicators['atr'].Current.Value
            
            # Get volume indicators
            current_volume = data[symbol].Volume
            avg_volume = symbol_indicators['volume_ma'].Current.Value
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Update previous RSI value for next iteration
            symbol_indicators['previous_rsi'] = current_rsi
            
            # Log current indicator values
            self.Debug(f"{symbol}: RSI = {current_rsi:.2f}, ATR = {current_atr:.2f}")
            self.Debug(f"{symbol}: Volume Ratio = {volume_ratio:.2f}")
            
            # Check if we should enter a position
            if not self.Portfolio[symbol].Invested:
                # Check for oversold condition (buy signal)
                if current_rsi < self.oversold_threshold and previous_rsi >= self.oversold_threshold:
                    # Additional volume confirmation
                    if volume_ratio >= 1.2:  # Require 20% more volume than average
                        self.Debug(f"BUY SIGNAL for {symbol} - RSI Oversold ({current_rsi:.2f})")
                        
                        # Calculate position size
                        quantity = self.calculate_position_size(symbol)
                        
                        # Check risk limits
                        if self.check_risk_limits():
                            # Place long trade with adjusted stop loss and take profit
                            current_price = self.Securities[symbol].Price
                            self.stop_loss = current_price * (1 - self.stop_loss_pct)
                            self.take_profit = current_price * (1 + self.take_profit_pct)
                            
                            # Place trade
                            self.place_trade(symbol, 'long', quantity)
                            
                            # Log trade
                            self.Debug(f"LONG {quantity} shares of {symbol} at {current_price}")
                
                # Check for overbought condition (short signal if short selling is allowed)
                elif current_rsi > self.overbought_threshold and previous_rsi <= self.overbought_threshold:
                    # Only take short positions if explicitly enabled
                    if self.Portfolio.MarginRemaining > 0 and self.Settings.FreePortfolioValue > 0:
                        # Additional volume confirmation
                        if volume_ratio >= 1.2:  # Require 20% more volume than average
                            self.Debug(f"SHORT SIGNAL for {symbol} - RSI Overbought ({current_rsi:.2f})")
                            
                            # Calculate position size
                            quantity = self.calculate_position_size(symbol)
                            
                            # Check risk limits
                            if self.check_risk_limits():
                                # Place short trade with adjusted stop loss and take profit
                                current_price = self.Securities[symbol].Price
                                self.stop_loss = current_price * (1 + self.stop_loss_pct)
                                self.take_profit = current_price * (1 - self.take_profit_pct)
                                
                                # Place trade
                                self.place_trade(symbol, 'short', quantity)
                                
                                # Log trade
                                self.Debug(f"SHORT {quantity} shares of {symbol} at {current_price}")
            
            # Check if we should exit a position
            else:
                # Check if we are long
                if self.Portfolio[symbol].IsLong:
                    # Exit on overbought condition
                    if current_rsi > self.overbought_threshold:
                        self.Debug(f"SELL SIGNAL for {symbol} - RSI Overbought (exit long) ({current_rsi:.2f})")
                        
                        # Close position
                        self.Liquidate(symbol)
                        
                        # Log trade
                        self.Debug(f"CLOSED LONG position of {symbol} at {self.Securities[symbol].Price}")
                
                # Check if we are short
                elif self.Portfolio[symbol].IsShort:
                    # Exit on oversold condition
                    if current_rsi < self.oversold_threshold:
                        self.Debug(f"BUY SIGNAL for {symbol} - RSI Oversold (exit short) ({current_rsi:.2f})")
                        
                        # Close position
                        self.Liquidate(symbol)
                        
                        # Log trade
                        self.Debug(f"CLOSED SHORT position of {symbol} at {self.Securities[symbol].Price}")
    
    def OnOrderEvent(self, orderEvent):
        """
        Handle order execution events.
        
        Args:
            orderEvent: Order event information
        """
        self.Debug(f"Order {orderEvent.OrderId} {orderEvent.Status}: {orderEvent.FillQuantity} shares of {orderEvent.Symbol} at {orderEvent.FillPrice}")
        
        # Record trade for metrics if order is filled
        if orderEvent.Status == OrderStatus.Filled:
            # Calculate profit if this is a closing order
            symbol = orderEvent.Symbol
            if not self.Portfolio[symbol].Invested:
                trade_profit = self.Portfolio.TotalUnrealizedProfit
                
                # Update metrics
                self.update_metrics({
                    'symbol': symbol,
                    'profit': trade_profit,
                    'percent_return': trade_profit / self.Portfolio.TotalPortfolioValue
                })
    
    def OnEndOfAlgorithm(self):
        """
        Actions to perform at the end of the algorithm run.
        """
        self.Log("Algorithm Complete")
        self.Log(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue}")
        self.Log(f"Total Trades: {self.metrics['total_trades']}")
        
        if self.metrics['total_trades'] > 0:
            win_rate = self.metrics['winning_trades'] / self.metrics['total_trades']
            self.Log(f"Win Rate: {win_rate:.2%}")
            self.Log(f"Average Win: {self.metrics['avg_win']:.2%}")
            self.Log(f"Average Loss: {self.metrics['avg_loss']:.2%}")
            self.Log(f"Maximum Drawdown: {self.metrics['max_drawdown']:.2%}") 