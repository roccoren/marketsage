"""
LEAN Strategy Wizard

This module provides an interactive wizard for creating new LEAN trading strategies.
It generates strategy templates based on user-selected patterns and configurations.
"""

import os
import re
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Templates directory - this would contain template files in practice
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Ensure template directory exists
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Default template location if not found in the package
DEFAULT_TEMPLATE_FILE = os.path.join(TEMPLATE_DIR, "basic_strategy.py.template")

# Strategy patterns
STRATEGY_PATTERNS = {
    "moving_average_crossover": {
        "name": "Moving Average Crossover",
        "description": "Strategy that trades based on moving average crossovers",
        "parameters": [
            {"name": "fast_period", "type": "int", "default": 10, "description": "Fast moving average period"},
            {"name": "slow_period", "type": "int", "default": 30, "description": "Slow moving average period"},
            {"name": "signal_period", "type": "int", "default": 9, "description": "Signal period for MACD"}
        ],
        "indicators": ["Simple Moving Average", "Exponential Moving Average"],
        "template": "moving_average_crossover.py.template"
    },
    "mean_reversion": {
        "name": "Mean Reversion",
        "description": "Strategy that trades based on price reversion to the mean",
        "parameters": [
            {"name": "lookback_period", "type": "int", "default": 20, "description": "Lookback period for mean calculation"},
            {"name": "entry_z_score", "type": "float", "default": 2.0, "description": "Z-score for trade entry"},
            {"name": "exit_z_score", "type": "float", "default": 0.5, "description": "Z-score for trade exit"}
        ],
        "indicators": ["Bollinger Bands", "RSI"],
        "template": "mean_reversion.py.template"
    },
    "trend_following": {
        "name": "Trend Following",
        "description": "Strategy that follows market trends based on indicators",
        "parameters": [
            {"name": "atr_period", "type": "int", "default": 14, "description": "ATR calculation period"},
            {"name": "atr_multiplier", "type": "float", "default": 3.0, "description": "ATR multiplier for stop loss"},
            {"name": "trend_period", "type": "int", "default": 100, "description": "Period for trend detection"}
        ],
        "indicators": ["Average True Range", "Directional Movement Index"],
        "template": "trend_following.py.template"
    },
    "breakout": {
        "name": "Breakout",
        "description": "Strategy that trades breakouts from price ranges or patterns",
        "parameters": [
            {"name": "channel_period", "type": "int", "default": 20, "description": "Period for channel calculation"},
            {"name": "volatility_period", "type": "int", "default": 10, "description": "Period for volatility calculation"},
            {"name": "filter_period", "type": "int", "default": 200, "description": "Period for trend filter"}
        ],
        "indicators": ["Donchian Channels", "Volatility"],
        "template": "breakout.py.template"
    },
    "momentum": {
        "name": "Momentum",
        "description": "Strategy that trades based on price momentum",
        "parameters": [
            {"name": "momentum_period", "type": "int", "default": 10, "description": "Period for momentum calculation"},
            {"name": "threshold", "type": "float", "default": 0.01, "description": "Momentum threshold for entry"},
            {"name": "holding_period", "type": "int", "default": 5, "description": "Position holding period"}
        ],
        "indicators": ["Rate of Change", "RSI", "MACD"],
        "template": "momentum.py.template"
    },
    "machine_learning": {
        "name": "Machine Learning",
        "description": "Strategy that uses machine learning models for prediction",
        "parameters": [
            {"name": "lookback_window", "type": "int", "default": 10, "description": "Historical window size for features"},
            {"name": "training_window", "type": "int", "default": 252, "description": "Training window in days"},
            {"name": "confidence_threshold", "type": "float", "default": 0.65, "description": "Threshold for trade entry"}
        ],
        "indicators": ["Custom ML indicators"],
        "template": "machine_learning.py.template"
    },
    "custom": {
        "name": "Custom Strategy",
        "description": "Create a completely custom strategy from scratch",
        "parameters": [],
        "indicators": [],
        "template": "custom_strategy.py.template"
    }
}

# Basic strategy template (will be used if template file not found)
BASIC_STRATEGY_TEMPLATE = '''"""
{strategy_name}

{strategy_description}
"""

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *

class {class_name}(QCAlgorithm):
    """
    {strategy_name}
    
    {strategy_description}
    """
    
    def Initialize(self):
        """
        Initialize the strategy and set required parameters
        """
        # Set start and end dates
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetEndDate({end_year}, {end_month}, {end_day})  # Set End Date
        
        # Set starting cash
        self.SetCash({initial_capital})
        
        # Set strategy parameters
{parameters_init}
        
        # Add required data
{symbols_init}
        
        # Initialize indicators
{indicators_init}
        
        # Set benchmark
        self.SetBenchmark("{benchmark}")
        
        # Set execution model
        self.SetExecution(ImmediateExecutionModel())
        
        # Set portfolio construction model
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # Set risk management model
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
        
        # Schedule rebalancing
{rebalancing_schedule}
        
        # Initialize any other required variables
        self.previous_signals = {{}}
        
    def OnData(self, data):
        """
        Event handler for market data updates
        """
{on_data_implementation}
    
    def Rebalance(self):
        """
        Rebalance portfolio based on current signals
        """
{rebalance_implementation}
    
    def GetSignal(self, symbol):
        """
        Calculate trading signal for a symbol
        """
{get_signal_implementation}
    
    def OnOrderEvent(self, orderEvent):
        """
        Event handler for order status changes
        """
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {{orderEvent.Symbol}}, Quantity: {{orderEvent.FillQuantity}}, Price: {{orderEvent.FillPrice}}")
    
    def OnEndOfAlgorithm(self):
        """
        Final cleanup when algorithm ends
        """
        self.Log("Algorithm completed.")
'''

# Moving Average Crossover template
MOVING_AVERAGE_TEMPLATE = '''"""
{strategy_name}

{strategy_description}
"""

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

import numpy as np
from datetime import datetime, timedelta
from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *

class {class_name}(QCAlgorithm):
    """
    Moving Average Crossover Strategy
    
    A classic trend-following strategy that generates signals based on fast and slow moving average crossovers
    """
    
    def Initialize(self):
        """Initialize the algorithm"""
        # Set start and end dates
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetEndDate({end_year}, {end_month}, {end_day})  # Set End Date
        
        # Set starting cash
        self.SetCash({initial_capital})
        
        # Set strategy parameters
        self.fast_period = {fast_period}  # Fast moving average period
        self.slow_period = {slow_period}  # Slow moving average period
        
        # Add required data
{symbols_init}
        
        self.symbols = [s.Value for s in self.Securities.Keys]
        
        # Initialize indicators
        self.fast_ma = {{}}
        self.slow_ma = {{}}
        
        for symbol in self.symbols:
            self.fast_ma[symbol] = self.EMA(symbol, self.fast_period)
            self.slow_ma[symbol] = self.EMA(symbol, self.slow_period)
        
        # Set benchmark
        self.SetBenchmark("{benchmark}")
        
        # Set execution model
        self.SetExecution(ImmediateExecutionModel())
        
        # Set portfolio construction model
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        
        # Set risk management model
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.05))
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(9, 30), self.Rebalance)
        
        # Initialize any other required variables
        self.previous_positions = {{}}
        for symbol in self.symbols:
            self.previous_positions[symbol] = 0
    
    def OnData(self, data):
        """OnData event is the primary entry point for your algorithm."""
        # We'll handle trading in the Rebalance method
        pass
    
    def Rebalance(self):
        """Rebalance portfolio based on current signals"""
        for symbol in self.symbols:
            if not self.fast_ma[symbol].IsReady or not self.slow_ma[symbol].IsReady:
                continue
                
            fast_value = self.fast_ma[symbol].Current.Value
            slow_value = self.slow_ma[symbol].Current.Value
            
            # Calculate signal - simple crossover
            if fast_value > slow_value:
                self.SetHoldings(symbol, 1.0 / len(self.symbols))
                if self.previous_positions[symbol] <= 0:
                    self.Log(f"BUY signal for {{symbol}}: Fast MA ({{fast_value:.2f}}) crossed above Slow MA ({{slow_value:.2f}})")
                self.previous_positions[symbol] = 1
            elif fast_value < slow_value:
                self.SetHoldings(symbol, -1.0 / len(self.symbols))  # Short position
                if self.previous_positions[symbol] >= 0:
                    self.Log(f"SELL signal for {{symbol}}: Fast MA ({{fast_value:.2f}}) crossed below Slow MA ({{slow_value:.2f}})")
                self.previous_positions[symbol] = -1
    
    def OnOrderEvent(self, orderEvent):
        """Order event handler"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {{orderEvent.Symbol}}, Quantity: {{orderEvent.FillQuantity}}, Price: {{orderEvent.FillPrice}}")
'''

# Mean Reversion template
MEAN_REVERSION_TEMPLATE = '''"""
{strategy_name}

{strategy_description}
"""

from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")

import numpy as np
from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
from QuantConnect.Indicators import *

class {class_name}(QCAlgorithm):
    """
    Mean Reversion Strategy
    
    Trades based on price deviations from the mean, using z-scores to identify 
    overbought and oversold conditions
    """
    
    def Initialize(self):
        """Initialize the algorithm"""
        # Set start and end dates
        self.SetStartDate({start_year}, {start_month}, {start_day})  # Set Start Date
        self.SetEndDate({end_year}, {end_month}, {end_day})  # Set End Date
        
        # Set starting cash
        self.SetCash({initial_capital})
        
        # Set strategy parameters
        self.lookback_period = {lookback_period}  # Period for mean calculation
        self.entry_z_score = {entry_z_score}      # Entry threshold (standard deviations)
        self.exit_z_score = {exit_z_score}        # Exit threshold (standard deviations)
        
        # Add required data
{symbols_init}
        
        self.symbols = [s.Value for s in self.Securities.Keys]
        
        # Initialize indicators
        self.sma = {{}}
        self.std = {{}}
        self.price_history = {{}}
        
        for symbol in self.symbols:
            self.sma[symbol] = SimpleMovingAverage(self.lookback_period)
            self.std[symbol] = StandardDeviation(self.lookback_period)
            self.price_history[symbol] = RollingWindow[float](self.lookback_period)
            self.RegisterIndicator(symbol, self.sma[symbol], Resolution.Daily)
            self.RegisterIndicator(symbol, self.std[symbol], Resolution.Daily)
        
        # Set benchmark
        self.SetBenchmark("{benchmark}")
        
        # Schedule rebalancing
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(10, 0), self.Rebalance)
        
        # Initialize position tracking
        self.positions = {{}}
        for symbol in self.symbols:
            self.positions[symbol] = 0
    
    def OnData(self, data):
        """OnData event is the primary entry point for your algorithm."""
        for symbol in self.symbols:
            if symbol not in data or not data[symbol]:
                continue
                
            # Update price history
            price = data[symbol].Close
            self.price_history[symbol].Add(price)
    
    def Rebalance(self):
        """Rebalance portfolio based on mean reversion signals"""
        for symbol in self.symbols:
            if not self.sma[symbol].IsReady or not self.std[symbol].IsReady or not self.price_history[symbol].IsReady:
                continue
            
            # Calculate z-score
            current_price = self.Securities[symbol].Price
            mean_price = self.sma[symbol].Current.Value
            std_dev = self.std[symbol].Current.Value
            
            # Avoid division by zero
            if std_dev == 0:
                continue
                
            z_score = (current_price - mean_price) / std_dev
            
            # Trading logic
            if z_score > self.entry_z_score and self.positions[symbol] >= 0:
                # Price is too high - go short
                self.SetHoldings(symbol, -1.0 / len(self.symbols))
                self.positions[symbol] = -1
                self.Log(f"SHORT signal for {{symbol}}: Z-Score = {{z_score:.2f}}")
            
            elif z_score < -self.entry_z_score and self.positions[symbol] <= 0:
                # Price is too low - go long
                self.SetHoldings(symbol, 1.0 / len(self.symbols))
                self.positions[symbol] = 1
                self.Log(f"LONG signal for {{symbol}}: Z-Score = {{z_score:.2f}}")
            
            elif abs(z_score) < self.exit_z_score and self.positions[symbol] != 0:
                # Price is back to normal - exit position
                self.Liquidate(symbol)
                self.positions[symbol] = 0
                self.Log(f"EXIT signal for {{symbol}}: Z-Score = {{z_score:.2f}}")
    
    def OnOrderEvent(self, orderEvent):
        """Order event handler"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Order filled: {{orderEvent.Symbol}}, Quantity: {{orderEvent.FillQuantity}}, Price: {{orderEvent.FillPrice}}")
'''

class StrategyWizard:
    """
    Interactive wizard for creating new LEAN trading strategies.
    """
    
    def __init__(self, 
                 output_dir: str = None,
                 template_dir: str = None,
                 default_capital: float = 100000,
                 default_symbols: List[str] = None):
        """
        Initialize the strategy wizard.
        
        Args:
            output_dir: Directory where strategy files will be created
            template_dir: Directory containing strategy templates
            default_capital: Default initial capital for new strategies
            default_symbols: Default symbols to include in strategies
        """
        # Setup directories
        self.output_dir = output_dir or os.path.join(os.getcwd(), "strategies")
        self.template_dir = template_dir or TEMPLATE_DIR
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Default values
        self.default_capital = default_capital
        self.default_symbols = default_symbols or ["SPY", "AAPL", "MSFT", "AMZN", "GOOG"]
        
        # Add built-in templates to template directory
        self._ensure_default_templates()
    
    def _ensure_default_templates(self):
        """Make sure default templates are available."""
        # Create basic template
        basic_template_path = os.path.join(self.template_dir, "basic_strategy.py.template")
        if not os.path.exists(basic_template_path):
            os.makedirs(os.path.dirname(basic_template_path), exist_ok=True)
            with open(basic_template_path, "w") as f:
                f.write(BASIC_STRATEGY_TEMPLATE)
        
        # Create moving average template
        ma_template_path = os.path.join(self.template_dir, "moving_average_crossover.py.template")
        if not os.path.exists(ma_template_path):
            with open(ma_template_path, "w") as f:
                f.write(MOVING_AVERAGE_TEMPLATE)
        
        # Create mean reversion template
        mr_template_path = os.path.join(self.template_dir, "mean_reversion.py.template")
        if not os.path.exists(mr_template_path):
            with open(mr_template_path, "w") as f:
                f.write(MEAN_REVERSION_TEMPLATE)
    
    def list_strategy_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        List available strategy patterns.
        
        Returns:
            Dictionary of strategy patterns with details
        """
        return STRATEGY_PATTERNS
    
    def create_strategy(self,
                        strategy_name: str,
                        pattern_key: str,
                        description: str = None,
                        symbols: List[str] = None,
                        parameters: Dict[str, Any] = None,
                        start_date: str = None,
                        end_date: str = None,
                        initial_capital: float = None,
                        benchmark: str = "SPY") -> str:
        """
        Create a new strategy based on a template.
        
        Args:
            strategy_name: Name of the strategy (will be used as file name and class name)
            pattern_key: Key of the strategy pattern to use
            description: Description of the strategy
            symbols: List of symbols to include
            parameters: Dictionary of parameters to override defaults
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Initial capital for backtesting
            benchmark: Benchmark symbol
            
        Returns:
            Path to the created strategy file
        """
        # Validate inputs
        if pattern_key not in STRATEGY_PATTERNS:
            raise ValueError(f"Unknown strategy pattern: {pattern_key}. Available patterns: {', '.join(STRATEGY_PATTERNS.keys())}")
        
        # Set defaults if not provided
        now = datetime.now()
        start_date = start_date or (now - timedelta(days=365*3)).strftime("%Y-%m-%d")
        end_date = end_date or now.strftime("%Y-%m-%d")
        
        symbols = symbols or self.default_symbols
        initial_capital = initial_capital or self.default_capital
        description = description or STRATEGY_PATTERNS[pattern_key]["description"]
        
        # Parse dates
        start_year, start_month, start_day = map(int, start_date.split("-"))
        end_year, end_month, end_day = map(int, end_date.split("-"))
        
        # Clean strategy name for file name
        file_name = re.sub(r'[^a-zA-Z0-9_]', '_', strategy_name)
        file_name = re.sub(r'_+', '_', file_name).lower()
        
        # Clean strategy name for class name
        class_name = ''.join(word.capitalize() for word in re.sub(r'[^a-zA-Z0-9]', ' ', strategy_name).split())
        
        # Merge default parameters with provided parameters
        default_params = {}
        for param in STRATEGY_PATTERNS[pattern_key]["parameters"]:
            default_params[param["name"]] = param["default"]
        
        if parameters:
            default_params.update(parameters)
        
        # Generate parameter initialization code
        parameters_init = ""
        for param_name, param_value in default_params.items():
            if isinstance(param_value, str):
                param_value = f'"{param_value}"'
            parameters_init += f"        self.{param_name} = {param_value}\n"
        
        # Generate symbols initialization code
        symbols_init = ""
        for symbol in symbols:
            symbols_init += f'        self.AddEquity("{symbol}", Resolution.Daily)\n'
        
        # Generate indicators initialization code based on pattern
        indicators_init = "        # Add indicators here\n"
        if pattern_key == "moving_average_crossover":
            indicators_init = """        self.fast_ma = {}
        self.slow_ma = {}
        
        for symbol in self.Securities.Keys:
            symbolStr = str(symbol.Value)
            self.fast_ma[symbolStr] = self.EMA(symbolStr, self.fast_period)
            self.slow_ma[symbolStr] = self.SMA(symbolStr, self.slow_period)
"""
        elif pattern_key == "mean_reversion":
            indicators_init = """        self.sma = {}
        self.upper_band = {}
        self.lower_band = {}
        
        for symbol in self.Securities.Keys:
            symbolStr = str(symbol.Value)
            self.sma[symbolStr] = self.SMA(symbolStr, self.lookback_period)
            self.upper_band[symbolStr] = self.BB(symbolStr, self.lookback_period, self.entry_z_score).UpperBand
            self.lower_band[symbolStr] = self.BB(symbolStr, self.lookback_period, self.entry_z_score).LowerBand
"""
        
        # Generate rebalancing schedule
        rebalancing_schedule = '        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(9, 30), self.Rebalance)'
        
        # Generate OnData implementation
        on_data_implementation = "        # Basic signal processing can go here\n        pass"
        
        # Generate Rebalance implementation
        rebalance_implementation = """        # Loop through each symbol to generate signals
        for symbol in self.Securities.Keys:
            signal = self.GetSignal(symbol)
            if signal > 0:
                self.SetHoldings(symbol, 0.95 / len(self.Securities))
            elif signal < 0:
                self.Liquidate(symbol)
"""
        
        # Generate GetSignal implementation
        get_signal_implementation = """        # Replace with your signal logic
        return 0  # Neutral signal
"""
        
        # Try to load the template file
        template_filename = STRATEGY_PATTERNS[pattern_key]["template"]
        template_path = os.path.join(self.template_dir, template_filename)
        
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template = f.read()
        else:
            # Use basic template if specific one not found
            template = BASIC_STRATEGY_TEMPLATE
        
        # Format the template with our values
        strategy_code = template.format(
            strategy_name=strategy_name,
            strategy_description=description,
            class_name=class_name,
            start_year=start_year,
            start_month=start_month,
            start_day=start_day,
            end_year=end_year,
            end_month=end_month,
            end_day=end_day,
            initial_capital=initial_capital,
            parameters_init=parameters_init,
            symbols_init=symbols_init,
            indicators_init=indicators_init,
            benchmark=benchmark,
            rebalancing_schedule=rebalancing_schedule,
            on_data_implementation=on_data_implementation,
            rebalance_implementation=rebalance_implementation,
            get_signal_implementation=get_signal_implementation,
            # Add pattern-specific parameters
            **default_params
        )
        
        # Create the strategy file
        output_filename = f"{file_name}.py"
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, "w") as f:
            f.write(strategy_code)
        
        logger.info(f"Created strategy file: {output_path}")
        return output_path

def create_strategy(strategy_name: str,
                   pattern: str,
                   description: str = None,
                   symbols: List[str] = None,
                   parameters: Dict[str, Any] = None,
                   output_dir: str = None,
                   start_date: str = None,
                   end_date: str = None,
                   initial_capital: float = 100000,
                   benchmark: str = "SPY") -> str:
    """
    Convenience function to create a strategy without instantiating the wizard.
    
    Args:
        strategy_name: Name of the strategy
        pattern: Strategy pattern to use
        description: Description of the strategy
        symbols: List of symbols to include
        parameters: Dictionary of parameters to override defaults
        output_dir: Directory where strategy file will be created
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_capital: Initial capital for backtesting
        benchmark: Benchmark symbol
        
    Returns:
        Path to the created strategy file
    """
    wizard = StrategyWizard(output_dir=output_dir)
    return wizard.create_strategy(
        strategy_name=strategy_name,
        pattern_key=pattern,
        description=description,
        symbols=symbols,
        parameters=parameters,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        benchmark=benchmark
    )

def list_available_patterns() -> Dict[str, str]:
    """
    List available strategy patterns with descriptions.
    
    Returns:
        Dictionary mapping pattern keys to descriptions
    """
    return {k: v["description"] for k, v in STRATEGY_PATTERNS.items()} 