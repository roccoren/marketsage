"""
Qlib backtesting system for stock analysis
"""

import os
import pandas as pd
import numpy as np
import qlib
from qlib.backtest import backtest
from typing import Dict, List, Optional, Tuple, Union
import logging
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

from src.data.loader import DataLoader
from src.prediction.price_targets import (
    calculate_price_targets, 
    predict_price_movement,
    analyze_stock_for_trading
)

logger = logging.getLogger(__name__)

class QlibBacktest:
    """
    Qlib backtesting system for stock analysis
    
    This class uses the Qlib backtesting framework to run backtests on stock data.
    """
    
    def __init__(
        self,
        start_time: str,
        end_time: str,
        instruments: Union[List[str], str],
        benchmark: str = "SPY",
        account: float = 100000.0,
        provider_uri: str = "~/.qlib/qlib_data/cn_data",
        strategy_type: str = "topk",
        strategy_params: Optional[Dict] = None,
    ):
        """
        Initialize the backtesting system
        
        Parameters
        ----------
        start_time : str
            Start time for backtest
        end_time : str
            End time for backtest
        instruments : Union[List[str], str]
            Instruments to use in backtest
            If str, it will be treated as a file with instruments separated by newlines
        benchmark : str
            Benchmark to use
        account : float
            Initial account value
        provider_uri : str
            Path to Qlib data provider
        strategy_type : str
            Type of strategy to use (topk or enhanced_indexing)
        strategy_params : Optional[Dict]
            Parameters for the strategy
        """
        self.start_time = start_time
        self.end_time = end_time
        self.account = account
        self.provider_uri = provider_uri
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params if strategy_params else {}
        
        # Parse instruments
        self.instruments = self._parse_instruments(instruments)
        
        # Add benchmark to instruments if not already there
        if benchmark not in self.instruments:
            self.instruments.append(benchmark)
        
        self.benchmark = benchmark
        logger.info(f"Using benchmark: {self.benchmark}")
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join("outputs", "backtest")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = logger
        self.logger.info(f"Initialized QlibBacktest with {len(self.instruments)} instruments")
    
    def _parse_instruments(self, instruments: Union[List[str], str]) -> List[str]:
        """
        Parse instruments
        
        Parameters
        ----------
        instruments : Union[List[str], str]
            Instruments to use in backtest
            If str, it will be treated as a file with instruments separated by newlines
            
        Returns
        -------
        List[str]
            List of instruments
        """
        if isinstance(instruments, list):
            return instruments
        elif isinstance(instruments, str):
            if os.path.exists(instruments):
                with open(instruments, "r") as f:
                    return [l.strip() for l in f.readlines() if l.strip()]
            else:
                return [instruments]
        else:
            raise ValueError(f"Invalid instruments type: {type(instruments)}")

    def run_backtest(self, historical_start: Optional[str] = None, historical_end: Optional[str] = None) -> Dict:
        """
        Run backtest
        
        Parameters
        ----------
        historical_start : Optional[str]
            Start time for historical data loading (for training)
        historical_end : Optional[str]
            End time for historical data loading (for training)
            
        Returns
        -------
        Dict
            Backtest results
        """
        # Configure historical data period
        if historical_start is None:
            historical_start = self.start_time
            
        if historical_end is None:
            historical_end = self.end_time
        
        logger.info(f"Running backtest from {self.start_time} to {self.end_time}")
        logger.info(f"Using historical data from {historical_start} to {historical_end}")
        
        # Load data
        loader = DataLoader()
        data = loader.load_data(
            symbols=self.instruments,
            start_date=historical_start,
            end_date=self.end_time,  # Need to load data until the end of backtest
            include_technicals=True
        )
        
        # Create strategy
        if self.strategy_type == "topk":
            topk = self.strategy_params.get("topk", 5)
            n_drop = self.strategy_params.get("n_drop", 1)
            strategy = DummyStrategy(
                data, 
                topk=topk, 
                n_drop=n_drop, 
                start_time=self.start_time,
                end_time=self.end_time
            )
            logger.info(f"Using TopK strategy with topk={topk}, n_drop={n_drop}")
        else:
            raise ValueError(f"Invalid strategy type: {self.strategy_type}")
        
        # Run strategy to get positions
        positions = strategy.get_positions()
        
        # Calculate metrics
        metrics = self._calculate_metrics(data, positions)
        
        # Save results
        self._save_results(data, positions, metrics)
        
        # Plot portfolio
        self._plot_portfolio(data, positions)
        
        return metrics
    
    def _calculate_metrics(self, data: Dict[str, pd.DataFrame], positions: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Data dictionary with stock data
        positions : pd.DataFrame
            Positions dataframe with dates as index and stock tickers as columns
            
        Returns
        -------
        Dict
            Dictionary of performance metrics
        """
        # Filter data for backtest period
        benchmark_data = data[self.benchmark]
        benchmark_data = benchmark_data[
            (benchmark_data.index >= self.start_time) &
            (benchmark_data.index <= self.end_time)
        ]
        
        # Get portfolio value over time
        portfolio_value = self._calculate_portfolio_value(data, positions)
        
        # Calculate returns
        portfolio_returns = portfolio_value.pct_change().fillna(0)
        benchmark_returns = benchmark_data["Close"].pct_change().fillna(0)
        
        # Calculate metrics
        total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
        benchmark_total_return = benchmark_data["Close"].iloc[-1] / benchmark_data["Close"].iloc[0] - 1
        
        # Annualized return
        days = (pd.to_datetime(self.end_time) - pd.to_datetime(self.start_time)).days
        annual_factor = 365 / days
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # Volatility (annualized)
        volatility = portfolio_returns.std() * (252 ** 0.5)  # Assuming 252 trading days
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        max_drawdown = ((cum_returns.cummax() - cum_returns) / cum_returns.cummax()).max()
        
        # Turnover
        turnover = positions.diff().abs().sum().sum() / len(positions)
        
        # Calculate price targets and predictions for the last date
        price_predictions = {}
        for symbol in self.instruments:
            if symbol == self.benchmark:
                continue
                
            if symbol in data and not data[symbol].empty:
                stock_data = data[symbol]
                analysis = analyze_stock_for_trading(stock_data)
                price_predictions[symbol] = {
                    'action': analysis['action'],
                    'current_price': analysis['current_price'],
                    'buy_price': analysis['price_targets']['buy_price'],
                    'sell_price': analysis['price_targets']['sell_price'],
                    'stop_loss': analysis['price_targets']['stop_loss'],
                    'forecast': analysis['prediction']['forecast_price'],
                    'direction': analysis['prediction']['direction'],
                    'confidence': analysis['prediction']['confidence']
                }
        
        return {
            "total_return": float(total_return),
            "benchmark_return": float(benchmark_total_return), 
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "turnover": float(turnover),
            "price_predictions": price_predictions
        }
    
    def _calculate_portfolio_value(self, data: Dict[str, pd.DataFrame], positions: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio value over time
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Data dictionary with stock data
        positions : pd.DataFrame
            Positions dataframe with dates as index and stock tickers as columns
            
        Returns
        -------
        pd.Series
            Portfolio value over time
        """
        # Initialize portfolio value
        portfolio_value = pd.Series(index=positions.index, dtype=float)
        
        # Fill portfolio value
        for date in positions.index:
            date_positions = positions.loc[date]
            date_value = 0
            
            for stock, position in date_positions.items():
                if position > 0 and stock in data:
                    stock_data = data[stock]
                    if date in stock_data.index:
                        date_value += position * stock_data.loc[date, "Close"]
            
            portfolio_value[date] = date_value if date_value > 0 else self.account
        
        return portfolio_value
    
    def _save_results(self, data: Dict[str, pd.DataFrame], positions: pd.DataFrame, metrics: Dict) -> None:
        """
        Save backtest results to files
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Data dictionary with stock data
        positions : pd.DataFrame
            Positions dataframe with dates as index and stock tickers as columns
        metrics : Dict
            Dictionary of performance metrics
        """
        # Save metrics
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save positions
        positions.to_csv(os.path.join(self.output_dir, "positions.csv"))
        
        # Save portfolio value
        portfolio_value = self._calculate_portfolio_value(data, positions)
        portfolio_value.to_csv(os.path.join(self.output_dir, "portfolio_value.csv"))
        
        # Save price predictions
        if "price_predictions" in metrics:
            with open(os.path.join(self.output_dir, "price_targets.json"), "w") as f:
                json.dump(metrics["price_predictions"], f, indent=4)
        
        # Create a plot of portfolio value vs benchmark
        self._plot_portfolio(data, positions)
    
    def _plot_portfolio(self, data: Dict[str, pd.DataFrame], positions: pd.DataFrame) -> None:
        """
        Plot portfolio value vs benchmark
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Data dictionary with stock data
        positions : pd.DataFrame
            Positions dataframe with dates as index and stock tickers as columns
        """
        # Get portfolio value
        portfolio_value = self._calculate_portfolio_value(data, positions)
        
        # Get benchmark data
        benchmark_data = data[self.benchmark]
        benchmark_data = benchmark_data[
            (benchmark_data.index >= self.start_time) &
            (benchmark_data.index <= self.end_time)
        ]
        
        # Normalize values
        portfolio_value = portfolio_value / portfolio_value.iloc[0]
        benchmark_value = benchmark_data["Close"] / benchmark_data["Close"].iloc[0]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value.index, portfolio_value, label="Portfolio")
        plt.plot(benchmark_value.index, benchmark_value, label=f"Benchmark ({self.benchmark})")
        plt.xlabel("Date")
        plt.ylabel("Value (normalized)")
        plt.title("Portfolio Performance")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.output_dir, "performance.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved performance plot to {plot_path}")
        
        # Plot returns
        portfolio_returns = portfolio_value.pct_change().fillna(0)
        benchmark_returns = benchmark_value.pct_change().fillna(0)
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_returns.index, portfolio_returns.cumsum(), label="Portfolio")
        plt.plot(benchmark_returns.index, benchmark_returns.cumsum(), label=f"Benchmark ({self.benchmark})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.title("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        
        plot_returns_path = os.path.join(self.output_dir, "returns.png")
        plt.savefig(plot_returns_path)
        plt.close()
        self.logger.info(f"Saved returns plot to {plot_returns_path}")

class DummyStrategy:
    """
    Dummy strategy for backtesting
    """
    def __init__(
        self, 
        data: Dict[str, pd.DataFrame], 
        topk: int = 5, 
        n_drop: int = 1,
        start_time: str = None,
        end_time: str = None
    ):
        """
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Data dictionary with stock data
        topk : int
            Number of top stocks to select
        n_drop : int
            Number of stocks to drop in each rebalance
        start_time : str
            Start time for backtest
        end_time : str
            End time for backtest
        """
        self.data = data
        self.topk = topk
        self.n_drop = n_drop
        self.start_time = start_time
        self.end_time = end_time
        
        # Get common dates
        self.dates = self._get_common_dates()
        
        # Create trading dates (monthly rebalance)
        self.trading_dates = self._create_trading_dates()
        
    def _get_common_dates(self) -> pd.DatetimeIndex:
        """
        Get common dates across all stocks
        
        Returns
        -------
        pd.DatetimeIndex
            Common dates
        """
        # Get unique dates
        all_dates = set()
        for symbol, df in self.data.items():
            all_dates.update(df.index)
        
        # Convert to DatetimeIndex
        dates = pd.DatetimeIndex(sorted(all_dates))
        
        # Filter by start and end time
        if self.start_time:
            dates = dates[dates >= self.start_time]
        if self.end_time:
            dates = dates[dates <= self.end_time]
        
        return dates
    
    def _create_trading_dates(self) -> List[str]:
        """
        Create trading dates (monthly rebalance)
        
        Returns
        -------
        List[str]
            List of trading dates
        """
        # Extract months
        months = pd.Series(self.dates).dt.to_period("M").unique()
        
        # Get last trading day of each month
        trading_dates = []
        for month in months:
            month_end = month.asfreq("D", "end")
            
            # Find the closest trading day to month end
            month_dates = [d for d in self.dates if d.strftime("%Y-%m") == month.strftime("%Y-%m")]
            if month_dates:
                trading_dates.append(max(month_dates))
        
        return sorted(trading_dates)
    
    def get_positions(self) -> pd.DataFrame:
        """
        Get positions for backtest
        
        Returns
        -------
        pd.DataFrame
            Positions dataframe with dates as index and stock tickers as columns
        """
        # Initialize positions DataFrame
        positions = pd.DataFrame(index=self.dates, columns=list(self.data.keys()))
        positions = positions.fillna(0)
        
        # Initialize portfolio
        current_stocks = []
        
        # Loop through trading dates
        for t_idx, trade_date in enumerate(self.trading_dates):
            # Skip the last trading date (no need to rebalance)
            if t_idx == len(self.trading_dates) - 1:
                continue
            
            # Get prediction date (current date) and next trading date
            pred_date = trade_date
            next_trade_date = self.trading_dates[t_idx + 1]
            
            # Predict best stocks
            pred_stocks = self._predict_best_stocks(pred_date)
            
            # Get positions to sell
            sell_stocks = self._get_stocks_to_sell(current_stocks, pred_stocks)
            
            # Get positions to buy
            buy_stocks = self._get_stocks_to_buy(current_stocks, pred_stocks, sell_stocks)
            
            # Update current stocks
            current_stocks = list(set(current_stocks) - set(sell_stocks))
            current_stocks.extend(buy_stocks)
            
            # Update positions
            positions = self._update_positions(
                positions, 
                current_stocks, 
                trade_date, 
                next_trade_date
            )
        
        return positions
    
    def _predict_best_stocks(self, date: pd.Timestamp) -> List[str]:
        """
        Predict best stocks for given date based on momentum
        
        Parameters
        ----------
        date : pd.Timestamp
            Prediction date
            
        Returns
        -------
        List[str]
            List of best stocks
        """
        # Get returns for each stock
        stock_metrics = {}
        for symbol, df in self.data.items():
            # Skip if date not in dataframe
            if date not in df.index:
                continue
            
            # Skip tickers with _len_ in name (leveraged ETFs)
            if "_" in symbol:
                continue

            # Get date index
            date_idx = df.index.get_loc(date)
            
            # Skip if not enough history
            if date_idx < 20:
                continue
            
            # Get 1-month return
            month_ago_idx = max(0, date_idx - 20)
            month_return = df['Close'].iloc[date_idx] / df['Close'].iloc[month_ago_idx] - 1
            
            # Get price targets and prediction
            stock_df = df.iloc[:date_idx+1]
            analysis = analyze_stock_for_trading(stock_df)
            
            # Score based on momentum, price targets, and prediction
            score = 0
            
            # Momentum component
            score += month_return * 2
            
            # RSI component (favor RSI between 50-70)
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[date_idx]
                if 50 <= rsi <= 70:
                    score += 0.3
                elif rsi < 30:
                    score += 0.2  # Oversold, may bounce
            
            # MACD component
            if all(x in df.columns for x in ['macd', 'signal']):
                if df['macd'].iloc[date_idx] > df['signal'].iloc[date_idx]:
                    score += 0.3
            
            # Price prediction component
            if analysis['prediction']['direction'] == 'bullish':
                score += 0.4 * (analysis['prediction']['confidence'] / 100)
            
            # Risk/reward component
            if analysis['risk_reward_ratio'] and analysis['risk_reward_ratio'] > 1.5:
                score += 0.3
            
            stock_metrics[symbol] = score
        
        # Sort by score and get top symbols
        sorted_stocks = sorted(stock_metrics.items(), key=lambda x: x[1], reverse=True)
        top_stocks = [s[0] for s in sorted_stocks[:self.topk]]
        
        return top_stocks
    
    def _get_stocks_to_sell(self, current_stocks: List[str], pred_stocks: List[str]) -> List[str]:
        """
        Get stocks to sell
        
        Parameters
        ----------
        current_stocks : List[str]
            Current stocks in portfolio
        pred_stocks : List[str]
            Predicted best stocks
            
        Returns
        -------
        List[str]
            Stocks to sell
        """
        # Stocks not in predicted list
        outdated = [s for s in current_stocks if s not in pred_stocks]
        
        # Limit number of stocks to sell
        return outdated[:self.n_drop]
    
    def _get_stocks_to_buy(
        self, 
        current_stocks: List[str], 
        pred_stocks: List[str], 
        sell_stocks: List[str]
    ) -> List[str]:
        """
        Get stocks to buy
        
        Parameters
        ----------
        current_stocks : List[str]
            Current stocks in portfolio
        pred_stocks : List[str]
            Predicted best stocks
        sell_stocks : List[str]
            Stocks to sell
            
        Returns
        -------
        List[str]
            Stocks to buy
        """
        # Number of stocks to buy = number of stocks sold
        n_to_buy = len(sell_stocks)
        
        # Get new stocks from predicted list
        new_stocks = [s for s in pred_stocks if s not in current_stocks]
        
        return new_stocks[:n_to_buy]
    
    def _update_positions(
        self, 
        positions: pd.DataFrame, 
        current_stocks: List[str], 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Update positions
        
        Parameters
        ----------
        positions : pd.DataFrame
            Positions dataframe
        current_stocks : List[str]
            Current stocks in portfolio
        start_date : pd.Timestamp
            Start date for positions
        end_date : pd.Timestamp
            End date for positions
            
        Returns
        -------
        pd.DataFrame
            Updated positions dataframe
        """
        # Get dates between start and end
        dates = [d for d in self.dates if start_date <= d <= end_date]
        
        # Equal weight allocation
        weight = 1.0 / len(current_stocks) if current_stocks else 0
        
        # Update positions
        for date in dates:
            for stock in current_stocks:
                positions.loc[date, stock] = weight
        
        return positions 