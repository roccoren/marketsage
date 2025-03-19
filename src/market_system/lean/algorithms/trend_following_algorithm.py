"""
Trend Following Algorithm

This algorithm implements a trend following strategy using moving averages.
It goes long when the fast moving average crosses above the slow moving average,
and short when the fast moving average crosses below the slow moving average.
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


class TrendFollowingAlgorithm(BaseLeanAlgorithm):
    """
    Trend following algorithm using moving average crossovers.
    """
    
    def Initialize(self) -> None:
        """Initialize the algorithm with default settings."""
        # Call parent initialize method
        super().Initialize()
        
        # Set default parameters (can be overridden)
        self.FastPeriod = 10
        self.SlowPeriod = 30
        self.StopLossPercentage = 0.05
        self.TakeProfitPercentage = 0.1
        self.PositionSize = 1.0  # Full position size by default
        self.symbols = []  # Will be set by caller
        
        # Set default timeframe
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime.now()
        self.SetStartDate(self.start_date.year, self.start_date.month, self.start_date.day)
        self.SetEndDate(self.end_date.year, self.end_date.month, self.end_date.day)
        
        # Set cash
        self.SetCash(100000)
        
        # Initialize indicators dict
        self.fast_ma = {}
        self.slow_ma = {}
        self.consolidators = {}
        self.previous_state = {}
        
        # Set up a default symbol
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbols.append(self.symbol)
        
        # Initialize indicators for default symbol
        self.initialize_indicators(self.symbol)
        
        # Set warmup period to ensure indicators are ready
        self.SetWarmUp(self.slow_period)
    
    def initialize_indicators(self, symbol: Symbol) -> None:
        """
        Initialize indicators for a specific symbol.
        
        Args:
            symbol: Symbol to initialize indicators for
        """
        # Create moving averages
        fast_ma = SimpleMovingAverage(self.fast_period)
        slow_ma = SimpleMovingAverage(self.slow_period)
        
        # Create moving average crossover
        mac = MovingAverageCrossOver(fast_ma, slow_ma)
        
        # Store indicators in dictionary
        self.indicators[symbol] = {
            'fast_ma': fast_ma, 
            'slow_ma': slow_ma,
            'crossover': mac,
            'previous_crossover_value': 0,
            'volume_ma': SimpleMovingAverage(20)  # For volume comparison
        }
        
        # Register indicators with consolidator
        self.RegisterIndicator(symbol, fast_ma, Resolution.Daily)
        self.RegisterIndicator(symbol, slow_ma, Resolution.Daily)
        self.RegisterIndicator(symbol, mac, Resolution.Daily)
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
            fast_ma_value = symbol_indicators['fast_ma'].Current.Value
            slow_ma_value = symbol_indicators['slow_ma'].Current.Value
            crossover_value = symbol_indicators['crossover'].Current.Value
            previous_crossover_value = symbol_indicators['previous_crossover_value']
            
            # Get volume indicators
            current_volume = data[symbol].Volume
            avg_volume = symbol_indicators['volume_ma'].Current.Value
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Update previous crossover value for next iteration
            symbol_indicators['previous_crossover_value'] = crossover_value
            
            # Log current indicator values
            self.Debug(f"{symbol}: Fast MA = {fast_ma_value:.2f}, Slow MA = {slow_ma_value:.2f}, Crossover = {crossover_value:.2f}")
            self.Debug(f"{symbol}: Volume Ratio = {volume_ratio:.2f}")
            
            # Check if we should enter a position
            if not self.Portfolio[symbol].Invested:
                # Check for buy signal - crossover went from negative to positive
                if crossover_value > 0 and previous_crossover_value <= 0:
                    # Additional volume confirmation
                    if volume_ratio >= self.volume_threshold:
                        self.Debug(f"BUY SIGNAL for {symbol}")
                        
                        # Calculate position size
                        quantity = self.calculate_position_size(symbol)
                        
                        # Check risk limits
                        if self.check_risk_limits():
                            # Place trade
                            self.place_trade(symbol, 'long', quantity)
                            
                            # Log trade
                            self.Debug(f"LONG {quantity} shares of {symbol} at {self.Securities[symbol].Price}")
            
            # Check if we should exit a position
            else:
                # Check if we are long
                if self.Portfolio[symbol].IsLong:
                    # Check for sell signal - crossover went from positive to negative
                    if crossover_value < 0 and previous_crossover_value >= 0:
                        self.Debug(f"SELL SIGNAL for {symbol} (exit long)")
                        
                        # Close position
                        self.Liquidate(symbol)
                        
                        # Log trade
                        self.Debug(f"CLOSED LONG position of {symbol} at {self.Securities[symbol].Price}")
                
                # Check if we are short
                elif self.Portfolio[symbol].IsShort:
                    # Check for buy signal - crossover went from negative to positive
                    if crossover_value > 0 and previous_crossover_value <= 0:
                        self.Debug(f"BUY SIGNAL for {symbol} (exit short)")
                        
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

    def add_symbol(self, symbol_str):
        """Add a symbol to the algorithm for trading."""
        symbol = self.AddEquity(symbol_str, Resolution.Daily).Symbol
        self.symbols.append(symbol)
        
        # Initialize indicators for this symbol
        self.fast_ma[symbol] = self.SMA(symbol, self.FastPeriod, Resolution.Daily)
        self.slow_ma[symbol] = self.SMA(symbol, self.SlowPeriod, Resolution.Daily)
        self.previous_state[symbol] = None
    
    def OnEndOfDay(self, symbol):
        """
        Event handler for end of day.
        
        Args:
            symbol: Symbol that finished the day
        """
        if symbol in self.symbols:
            # Log portfolio statistics
            self.Log(f"End of day for {symbol}")
            self.Log(f"Price: {self.Securities[symbol].Close}")
            
            if self.fast_ma[symbol].IsReady and self.slow_ma[symbol].IsReady:
                self.Log(f"Fast MA: {self.fast_ma[symbol].Current.Value}")
                self.Log(f"Slow MA: {self.slow_ma[symbol].Current.Value}")
            
            self.Log(f"Holdings: {self.Portfolio[symbol].Quantity}")
            self.Log(f"Unrealized P&L: {self.Portfolio[symbol].UnrealizedProfit}") 