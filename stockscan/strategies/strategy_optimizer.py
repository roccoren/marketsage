import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from itertools import product
from .optimizer import StrategyOptimizer
from strategies.multi_indicator_strategy import MultiIndicatorStrategy
from strategies.backtester import Backtester
import logging

logger = logging.getLogger(__name__)

class AdvancedStrategyOptimizer:
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],  # Dictionary of market data (symbol -> DataFrame)
        train_size: float = 0.8,
        window_size: int = 252,  # 1 year of trading days
        step_size: int = 63,     # ~3 months step
        min_window_size: int = 126  # ~6 months minimum
    ):
        self.data_dict = data_dict
        self.train_size = train_size
        self.window_size = window_size
        self.step_size = step_size
        self.min_window_size = min_window_size
        
        # Define parameter search space
        self.param_grid = {
            # Moving Averages
            'sma_short_period': [20, 50],
            'sma_long_period': [100, 200],
            'ema_short_period': [20, 50],
            'ema_long_period': [100, 200],
            
            # RSI
            'rsi_period': list(range(10, 15)),
            'rsi_oversold': list(range(25, 31)),
            'rsi_overbought': list(range(70, 76)),
            
            # MACD
            'macd_fast': list(range(8, 13)),
            'macd_slow': list(range(20, 27)),
            'macd_signal': list(range(7, 10)),
            
            # Bollinger Bands
            'bb_period': list(range(15, 26)),
            'bb_std': [round(x, 1) for x in np.arange(1.8, 2.3, 0.1)]
        }
        
        # Strategy weights for different market regimes
        self.regime_weights = {
            'TRENDING': {
                'trend': 0.4,
                'momentum': 0.3,
                'volatility': 0.1,
                'macd': 0.2
            },
            'VOLATILE': {
                'trend': 0.2,
                'momentum': 0.3,
                'volatility': 0.3,
                'macd': 0.2
            },
            'RANGING': {
                'trend': 0.2,
                'momentum': 0.4,
                'volatility': 0.2,
                'macd': 0.2
            }
        }

    def detect_market_regime(self, data: pd.DataFrame, lookback: int = 126) -> str:
        """Detect market regime based on price action characteristics"""
        returns = data['Close'].pct_change()
        std = returns.rolling(lookback).std()
        trend = data['Close'].rolling(lookback).mean()
        
        current_volatility = std.iloc[-1]
        avg_volatility = std.mean()
        
        # Check if price is trending
        is_trending = abs(trend.pct_change(lookback).iloc[-1]) > 0.1
        is_volatile = current_volatility > 1.5 * avg_volatility
        
        if is_volatile:
            return 'VOLATILE'
        elif is_trending:
            return 'TRENDING'
        else:
            return 'RANGING'

    def _generate_windows(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate rolling windows for backtesting"""
        windows = []
        for start_idx in range(0, len(data) - self.min_window_size, self.step_size):
            end_idx = min(start_idx + self.window_size, len(data))
            if end_idx - start_idx < self.min_window_size:
                break
                
            window_data = data.iloc[start_idx:end_idx]
            train_idx = int(len(window_data) * self.train_size)
            
            train_data = window_data.iloc[:train_idx]
            test_data = window_data.iloc[train_idx:]
            
            windows.append((train_data, test_data))
            
        return windows

    def optimize_strategy(self, initial_capital: float = 100000) -> Dict[str, Any]:
        """Run comprehensive strategy optimization"""
        logger.info("Starting comprehensive strategy optimization...")
        
        all_results = []
        market_performances = {}
        
        # Cross-market validation
        for symbol, data in self.data_dict.items():
            logger.info(f"\nOptimizing for {symbol}...")
            
            # Generate rolling windows
            windows = self._generate_windows(data)
            window_results = []
            
            for window_idx, (train_data, test_data) in enumerate(windows):
                logger.info(f"Processing window {window_idx + 1}/{len(windows)}")
                
                # Detect market regime for this window
                regime = self.detect_market_regime(train_data)
                logger.info(f"Detected market regime: {regime}")
                
                # Adjust weights based on regime
                weights = self.regime_weights[regime]
                
                # Create optimizer instance
                optimizer = StrategyOptimizer(
                    strategy_class=MultiIndicatorStrategy,
                    param_grid=self.param_grid,
                    data=train_data,
                    train_size=self.train_size
                )
                
                # Run optimization
                opt_results = optimizer.optimize(initial_capital=initial_capital)
                
                if opt_results['best_params'] is not None:
                    # Test optimized parameters
                    strategy = MultiIndicatorStrategy(
                        **opt_results['best_params'],
                        **weights
                    )
                    backtester = Backtester(strategy, initial_capital=initial_capital)
                    test_results = backtester.run(test_data)
                    
                    window_results.append({
                        'window_idx': window_idx,
                        'regime': regime,
                        'train_score': opt_results['best_train_score'],
                        'test_score': test_results.metrics['sharpe_ratio'],
                        'params': opt_results['best_params'],
                        'metrics': test_results.metrics
                    })
            
            # Aggregate results for this market
            if window_results:
                market_performances[symbol] = {
                    'avg_sharpe': np.mean([r['test_score'] for r in window_results]),
                    'avg_profit_factor': np.mean([r['metrics']['profit_factor'] for r in window_results]),
                    'avg_win_rate': np.mean([r['metrics']['win_rate'] for r in window_results]),
                    'best_regime_params': {
                        regime: next((r['params'] for r in window_results if r['regime'] == regime), None)
                        for regime in self.regime_weights.keys()
                    }
                }
                all_results.extend(window_results)
        
        # Find most stable parameters across markets
        param_stability = self._analyze_parameter_stability(all_results)
        
        return {
            'market_performances': market_performances,
            'param_stability': param_stability,
            'regime_specific_params': self._get_regime_specific_params(all_results)
        }

    def _analyze_parameter_stability(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze parameter stability across different markets and regimes"""
        param_values = {param: [] for param in self.param_grid.keys()}
        
        for result in results:
            for param, value in result['params'].items():
                param_values[param].append(value)
                
        stability_metrics = {}
        for param, values in param_values.items():
            values = np.array(values)
            stability_metrics[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),
                'range': np.ptp(values)
            }
            
        return stability_metrics

    def _get_regime_specific_params(self, results: List[Dict]) -> Dict[str, Dict]:
        """Get best parameters for each market regime"""
        regime_results = {regime: [] for regime in self.regime_weights.keys()}
        
        for result in results:
            regime_results[result['regime']].append({
                'params': result['params'],
                'score': result['test_score']
            })
            
        best_params = {}
        for regime, regime_data in regime_results.items():
            if regime_data:
                best_result = max(regime_data, key=lambda x: x['score'])
                best_params[regime] = best_result['params']
            
        return best_params

    def get_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed optimization report"""
        report = []
        report.append("=== Strategy Optimization Report ===\n")
        
        # Market Performance Summary
        report.append("Market Performance Summary:")
        for symbol, perf in results['market_performances'].items():
            report.append(f"\n{symbol}:")
            report.append(f"- Average Sharpe Ratio: {perf['avg_sharpe']:.2f}")
            report.append(f"- Average Profit Factor: {perf['avg_profit_factor']:.2f}")
            report.append(f"- Average Win Rate: {perf['avg_win_rate']:.2%}")
        
        # Parameter Stability Analysis
        report.append("\nParameter Stability Analysis:")
        stability = results['param_stability']
        for param, metrics in stability.items():
            report.append(f"\n{param}:")
            report.append(f"- Mean: {metrics['mean']:.2f}")
            report.append(f"- Std Dev: {metrics['std']:.2f}")
            report.append(f"- CV: {metrics['cv']:.2f}")
            
        # Regime-Specific Parameters
        report.append("\nRegime-Specific Optimal Parameters:")
        for regime, params in results['regime_specific_params'].items():
            report.append(f"\n{regime}:")
            for param, value in params.items():
                report.append(f"- {param}: {value}")
                
        return "\n".join(report)