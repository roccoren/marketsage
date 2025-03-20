"""
StrategyManager - Handles strategy implementation, signal generation, and portfolio construction
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from qlib.contrib.strategy import TopkDropoutStrategy, EnhancedIndexingStrategy
from qlib.backtest import backtest

def get_ts_baseline(benchmark: pd.Series) -> dict:
    """
    Calculate baseline performance metrics for a benchmark series
    
    Parameters
    ----------
    benchmark : pd.Series
        Benchmark series with datetime index and price values
        
    Returns
    -------
    dict
        Dictionary of benchmark performance metrics
    """
    # Calculate returns
    returns = benchmark.pct_change().dropna()
    
    # Calculate metrics
    annualized_return = np.mean(returns) * 252
    annualized_volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else float('nan')
    max_drawdown = (benchmark / benchmark.cummax() - 1).min()
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

class StrategyManager:
    """
    Manager class for implementing trading strategies in Qlib
    """
    
    def __init__(self):
        """
        Initialize the StrategyManager
        """
        self.strategies = {}
        self.strategy_configs = {}
        
    def create_topk_strategy(self, 
                           strategy_name: str, 
                           topk: int = 50, 
                           n_drop: int = 5, 
                           **kwargs) -> TopkDropoutStrategy:
        """
        Create a TopkDropoutStrategy
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy
        topk : int
            Number of stocks to buy
        n_drop : int
            Number of stocks to sell
        **kwargs : Dict[str, Any]
            Additional parameters for the strategy
            
        Returns
        -------
        TopkDropoutStrategy
            The created strategy
        """
        strategy = TopkDropoutStrategy(
            topk=topk,
            n_drop=n_drop,
            **kwargs
        )
        
        self.strategies[strategy_name] = strategy
        self.strategy_configs[strategy_name] = {
            'type': 'TopkDropoutStrategy',
            'params': {
                'topk': topk,
                'n_drop': n_drop,
                **kwargs
            }
        }
        
        return strategy
    
    def create_enhanced_indexing_strategy(self, 
                                       strategy_name: str, 
                                       market_view: pd.Series, 
                                       risk_degree: float = 0.95, 
                                       **kwargs) -> EnhancedIndexingStrategy:
        """
        Create an EnhancedIndexingStrategy
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy
        market_view : pd.Series
            Market view for the strategy
        risk_degree : float
            Risk degree for the strategy
        **kwargs : Dict[str, Any]
            Additional parameters for the strategy
            
        Returns
        -------
        EnhancedIndexingStrategy
            The created strategy
        """
        strategy = EnhancedIndexingStrategy(
            market_view=market_view,
            risk_degree=risk_degree,
            **kwargs
        )
        
        self.strategies[strategy_name] = strategy
        self.strategy_configs[strategy_name] = {
            'type': 'EnhancedIndexingStrategy',
            'params': {
                'risk_degree': risk_degree,
                **kwargs
            }
        }
        
        return strategy
    
    def generate_signals(self, 
                       strategy_name: str, 
                       pred_scores: pd.Series) -> pd.DataFrame:
        """
        Generate signals using a strategy
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy to use
        pred_scores : pd.Series
            Prediction scores to generate signals from
            
        Returns
        -------
        pd.DataFrame
            DataFrame of signals
        """
        if strategy_name not in self.strategies:
            raise KeyError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        return strategy.generate_trade_decision(score=pred_scores)
    
    def backtest_strategy(self, 
                       strategy_name: str, 
                       pred_scores: pd.Series, 
                       benchmark: Optional[pd.Series] = None, 
                       **kwargs) -> Dict[str, Any]:
        """
        Backtest a strategy
        
        Parameters
        ----------
        strategy_name : str
            Name of the strategy to backtest
        pred_scores : pd.Series
            Prediction scores to use for backtesting
        benchmark : Optional[pd.Series]
            Benchmark for comparison
        **kwargs : Dict[str, Any]
            Additional parameters for backtesting
            
        Returns
        -------
        Dict[str, Any]
            Backtest results
        """
        if strategy_name not in self.strategies:
            raise KeyError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        
        # Generate signals
        signals = self.generate_signals(strategy_name, pred_scores)
        
        # Run backtest
        backtest_result = backtest(
            pred=pred_scores,
            strategy=strategy,
            **kwargs
        )
        
        # Calculate benchmark performance if provided
        if benchmark is not None:
            benchmark_result = get_ts_baseline(benchmark)
            backtest_result['benchmark'] = benchmark_result
            
        return backtest_result
    
    def evaluate_strategy(self, 
                       backtest_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a strategy from backtest results
        
        Parameters
        ----------
        backtest_result : Dict[str, Any]
            Backtest results
            
        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        # Extract portfolio metrics
        portfolio = backtest_result.get('portfolio', None)
        
        if portfolio is None:
            return {}
            
        # Calculate common metrics
        returns = portfolio.loc[:, 'return'].values
        metrics = {
            'annualized_return': np.mean(returns) * 252,
            'annualized_volatility': np.std(returns) * np.sqrt(252),
            'max_drawdown': (portfolio['value'] / portfolio['value'].cummax() - 1).min(),
            'total_turnover': portfolio.loc[:, 'turnover'].sum(),
            'average_turnover': portfolio.loc[:, 'turnover'].mean(),
        }
        
        # Calculate Sharpe ratio
        if metrics['annualized_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
        else:
            metrics['sharpe_ratio'] = float('nan')
            
        return metrics 