"""
Qlib backtesting system for stock analysis
"""

import os
import pandas as pd
import numpy as np
import qlib
from qlib.backtest import backtest
from qlib.contrib.strategy import TopkDropoutStrategy, EnhancedIndexingStrategy
from qlib.contrib.evaluate import risk_analysis, backtest_daily
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)

class QlibBacktest:
    """Qlib backtesting system for stock analysis"""
    
    def __init__(
        self,
        provider_uri: str,
        region: str = 'us',
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        instruments: Optional[List[str]] = None,
        benchmark: Optional[str] = None,
        account: float = 100000000.0,
        strategy_type: str = 'topk',
        strategy_params: Optional[Dict] = None
    ):
        """
        Initialize the backtesting system
        
        Parameters
        ----------
        provider_uri : str
            Path to Qlib data
        region : str, optional
            Market region (default: 'us')
        start_time : str, optional
            Start time for backtesting (YYYY-MM-DD)
        end_time : str, optional
            End time for backtesting (YYYY-MM-DD)
        instruments : List[str], optional
            List of stock symbols to backtest
        benchmark : str, optional
            Benchmark symbol for comparison
        account : float, optional
            Initial account value (default: 100M)
        strategy_type : str, optional
            Type of strategy to use (default: 'topk')
        strategy_params : Dict, optional
            Additional parameters for the strategy
        """
        self.provider_uri = provider_uri
        self.region = region
        self.start_time = start_time
        self.end_time = end_time
        self.instruments = instruments
        self.benchmark = benchmark
        self.account = account
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params or {}
        
        # Initialize Qlib
        qlib.init(provider_uri=provider_uri, region=region)
        
        # Create strategy
        self.strategy = self._create_strategy()
        
    def _create_strategy(self) -> Union[TopkDropoutStrategy, EnhancedIndexingStrategy]:
        """Create the trading strategy"""
        if self.strategy_type == 'topk':
            return TopkDropoutStrategy(
                signal='<PRED>',
                topk=self.strategy_params.get('topk', 50),
                n_drop=self.strategy_params.get('n_drop', 5)
            )
        elif self.strategy_type == 'enhanced_indexing':
            return EnhancedIndexingStrategy(
                signal='<PRED>',
                topk=self.strategy_params.get('topk', 50),
                n_drop=self.strategy_params.get('n_drop', 5)
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def run_backtest(
        self,
        predictions: pd.Series,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        instruments: Optional[List[str]] = None,
        benchmark: Optional[str] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run backtest with the given predictions
        
        Parameters
        ----------
        predictions : pd.Series
            Prediction scores for each stock
        start_time : str, optional
            Override start time for backtesting
        end_time : str, optional
            Override end time for backtesting
        instruments : List[str], optional
            Override list of instruments
        benchmark : str, optional
            Override benchmark symbol
            
        Returns
        -------
        Tuple[Dict, pd.DataFrame]
            Dictionary of backtest metrics and DataFrame of portfolio positions
        """
        # Use provided parameters or defaults
        start_time = start_time or self.start_time
        end_time = end_time or self.end_time
        instruments = instruments or self.instruments
        benchmark = benchmark or self.benchmark
        
        if not all([start_time, end_time, instruments]):
            raise ValueError("Missing required parameters for backtesting")
        
        logger.info(f"Running backtest from {start_time} to {end_time}")
        logger.info(f"Using {len(instruments)} instruments")
        
        # Run backtest
        backtest_result = backtest(
            pred=predictions,
            strategy=self.strategy,
            start_time=start_time,
            end_time=end_time,
            account=self.account,
            benchmark=benchmark,
            exchange_kwargs={
                'limit_threshold': 0.095,
                'deal_price': 'close',
                'open_cost': 0.0005,
                'close_cost': 0.0015,
                'min_cost': 5
            }
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(backtest_result)
        
        return metrics, backtest_result
    
    def _calculate_metrics(self, backtest_result: Dict) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Parameters
        ----------
        backtest_result : Dict
            Results from Qlib backtest
            
        Returns
        -------
        Dict
            Dictionary of performance metrics
        """
        portfolio = backtest_result.get('portfolio', None)
        if portfolio is None:
            return {}
        
        # Calculate returns
        returns = portfolio.loc[:, 'return'].values
        
        # Calculate metrics
        metrics = {
            'annualized_return': np.mean(returns) * 252,
            'annualized_volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': (portfolio['value'] / portfolio['value'].cummax() - 1).min(),
            'total_turnover': portfolio.loc[:, 'turnover'].sum(),
            'average_turnover': portfolio.loc[:, 'turnover'].mean(),
            'win_rate': (returns > 0).mean(),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
        }
        
        # Calculate risk metrics
        risk_metrics = risk_analysis(returns)
        metrics.update(risk_metrics)
        
        return metrics
    
    def plot_results(self, backtest_result: Dict, save_path: Optional[str] = None):
        """
        Plot backtest results
        
        Parameters
        ----------
        backtest_result : Dict
            Results from Qlib backtest
        save_path : str, optional
            Path to save the plot
        """
        portfolio = backtest_result.get('portfolio', None)
        if portfolio is None:
            logger.warning("No portfolio data available for plotting")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot portfolio value
        ax1.plot(portfolio.index, portfolio['value'], label='Portfolio Value')
        if 'benchmark' in portfolio:
            ax1.plot(portfolio.index, portfolio['benchmark'], label='Benchmark', alpha=0.5)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot returns
        ax2.plot(portfolio.index, portfolio['return'], label='Daily Returns')
        ax2.set_title('Daily Returns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_results(self, backtest_result: Dict, metrics: Dict, output_dir: str):
        """
        Save backtest results and metrics
        
        Parameters
        ----------
        backtest_result : Dict
            Results from Qlib backtest
        metrics : Dict
            Performance metrics
        output_dir : str
            Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio data
        portfolio = backtest_result.get('portfolio', None)
        if portfolio is not None:
            portfolio.to_csv(os.path.join(output_dir, 'portfolio.csv'))
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot results
        self.plot_results(backtest_result, os.path.join(output_dir, 'backtest_results.png')) 