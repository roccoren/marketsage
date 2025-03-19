import pandas as pd
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
from strategies.base import Strategy
from utils.technical_indicators import TechnicalIndicators

class OptimizedMultiStrategy(Strategy):
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__()
        # Default parameters
        self.params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'sma_short_period': 10,
            'sma_long_period': 50,
            'ema_short_period': 10,
            'ema_long_period': 50,
            'atr_period': 14,
            'min_score': 2  # Minimum number of indicators confirming to generate signal
        }
        # Update with custom parameters if provided
        if params:
            self.params.update(params)
            
    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """
        Analyze price data and generate trading signals based on multiple indicators
        """
        if len(data) < 50:  # Minimum data points needed
            return []
            
        # Calculate all technical indicators
        indicators = TechnicalIndicators.get_all_indicators(data, self.params)
        
        # Generate signals based on multiple conditions
        signals = []
        current_bar = data.iloc[-1]
        
        # Calculate signal scores
        buy_score = 0
        sell_score = 0
        
        # 1. MACD Crossover
        if indicators['macd_crossover'].iloc[-1] == 1:
            buy_score += 1
        elif indicators['macd_crossover'].iloc[-1] == -1:
            sell_score += 1
            
        # 2. RSI Conditions
        rsi_current = indicators['rsi'].iloc[-1]
        if rsi_current < self.params['rsi_oversold']:
            buy_score += 1
        elif rsi_current > self.params['rsi_overbought']:
            sell_score += 1
            
        # 3. Bollinger Bands
        if current_bar['Close'] < indicators['bb_lower'].iloc[-1]:
            buy_score += 1
        elif current_bar['Close'] > indicators['bb_upper'].iloc[-1]:
            sell_score += 1
            
        # 4. Moving Average Crossovers
        if indicators['sma_crossover'].iloc[-1] == 1:
            buy_score += 1
        elif indicators['sma_crossover'].iloc[-1] == -1:
            sell_score += 1
            
        # Generate signals if minimum score threshold is met
        if buy_score >= self.params['min_score']:
            signals.append({
                'Type': 'BUY',
                'Price': current_bar['Close'],
                'Score': buy_score,
                'ATR': indicators['atr'].iloc[-1]
            })
        elif sell_score >= self.params['min_score']:
            signals.append({
                'Type': 'SELL',
                'Price': current_bar['Close'],
                'Score': sell_score,
                'ATR': indicators['atr'].iloc[-1]
            })
            
        return signals

    @staticmethod
    def _process_single_stock(args):
        """Helper method for parallel processing"""
        data, params = args
        strategy = OptimizedMultiStrategy(params)
        signals = strategy.analyze(data)
        return signals

    @classmethod
    def parallel_analyze(cls, stock_data_dict: Dict[str, pd.DataFrame], params: Dict[str, Any] = None) -> Dict[str, List]:
        """
        Analyze multiple stocks in parallel
        
        Args:
            stock_data_dict: Dictionary of stock symbols to DataFrames
            params: Strategy parameters
            
        Returns:
            Dictionary of stock symbols to signals
        """
        # Prepare arguments for parallel processing
        process_args = [(data, params) for data in stock_data_dict.values()]
        
        results = {}
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor() as executor:
            signals_list = executor.map(cls._process_single_stock, process_args)
            
            # Combine results
            for symbol, signals in zip(stock_data_dict.keys(), signals_list):
                if signals:  # Only include stocks with signals
                    results[symbol] = signals
                    
        return results

    def optimize_parameters(self, training_data: pd.DataFrame, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            training_data: Historical price data for optimization
            param_grid: Dictionary of parameter names to lists of values to try
            
        Returns:
            Dictionary of optimized parameters
        """
        best_params = self.params.copy()
        best_sharpe = -np.inf
        
        from itertools import product
        param_combinations = [dict(zip(param_grid.keys(), values)) 
                            for values in product(*param_grid.values())]
        
        for params in param_combinations:
            # Update strategy parameters
            test_params = self.params.copy()
            test_params.update(params)
            
            # Create test strategy instance
            test_strategy = OptimizedMultiStrategy(test_params)
            
            # Run backtest with current parameters
            from ..strategies.backtester import Backtester
            backtester = Backtester(test_strategy)
            result = backtester.run(training_data)
            
            # Update best parameters if Sharpe ratio improved
            if result.metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = result.metrics['sharpe_ratio']
                best_params.update(params)
                
        return best_params