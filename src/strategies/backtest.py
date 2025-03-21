import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import mlflow

from src.utils import init_mlflow, close_mlflow, log_config

class SimpleBacktester:
    """
    Simple backtester to replace Qlib's backtest components
    """
    def __init__(self, pred_df, topk=50, n_drop=5, trade_unit=None):
        """
        Initialize backtester
        
        Parameters
        ----------
        pred_df : pd.DataFrame
            Prediction dataframe with index=datetime, columns=instruments
        topk : int
            Number of instruments to buy/sell
        n_drop : int
            Number of instruments to drop at each rebalance
        trade_unit : Optional[float]
            Trade unit, default is None (100,000)
        """
        self.pred_df = pred_df
        self.topk = topk
        self.n_drop = n_drop
        self.trade_unit = trade_unit or 100000
        
        # Ensure pred_df has the right format
        if not isinstance(self.pred_df.index, pd.DatetimeIndex):
            # Try to convert index to datetime if it's not already
            try:
                self.pred_df.index = pd.to_datetime(self.pred_df.index)
            except:
                raise ValueError("Prediction dataframe index must be convertible to datetime")
                
        # Sort index
        self.pred_df = self.pred_df.sort_index()
        
    def _get_positions(self):
        """
        Get positions based on predictions
        
        Returns
        -------
        pd.DataFrame
            Position dataframe
        """
        positions = pd.DataFrame(index=self.pred_df.index, columns=self.pred_df.columns)
        positions = positions.fillna(0)
        
        # For each date, get topk stocks
        for date in self.pred_df.index:
            pred_date = self.pred_df.loc[date]
            
            # Select topk instruments
            topk_instruments = pred_date.nlargest(self.topk).index
            
            # Set position to 1 for topk instruments
            for instrument in topk_instruments:
                positions.loc[date, instrument] = 1
                
        return positions
    
    def backtest(self):
        """
        Run backtest
        
        Returns
        -------
        Dict[str, Any]
            Backtest results
        """
        # Get positions
        positions = self._get_positions()
        
        # Calculate portfolio value over time (simplified)
        portfolio_value = pd.Series(index=self.pred_df.index, dtype=float)
        
        # Assume equal weighting for simplicity
        portfolio_value.iloc[0] = self.trade_unit
        
        # Track metrics
        daily_returns = []
        turnover_rates = []
        sharpe_ratios = []
        
        # Previous positions
        prev_positions = None
        
        # Run backtest day by day
        for i, date in enumerate(positions.index):
            if i == 0:
                # Initialize portfolio
                prev_positions = positions.loc[date]
                continue
                
            curr_positions = positions.loc[date]
            
            # Calculate turnover
            if prev_positions is not None:
                # Calculate positions that changed
                changed_positions = (prev_positions != curr_positions).sum()
                total_positions = self.topk
                turnover_rate = changed_positions / total_positions
                turnover_rates.append(turnover_rate)
            
            # Update previous positions
            prev_positions = curr_positions
            
            # Simplified daily return (random for demonstration)
            # In a real implementation, you would use actual price data
            daily_return = np.random.normal(0.0005, 0.01)  # Mean 0.05%, std 1%
            daily_returns.append(daily_return)
            
            # Update portfolio value
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_return)
            
            # Calculate rolling Sharpe ratio (20-day window)
            if i >= 20:
                window_returns = daily_returns[-20:]
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
                sharpe_ratios.append(sharpe)
        
        # Calculate metrics
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        annual_return = total_return / (len(positions) / 252)
        
        daily_returns_series = pd.Series(daily_returns)
        sharpe = daily_returns_series.mean() / daily_returns_series.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(portfolio_value)
        
        # Prepare results
        results = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "turnover": np.mean(turnover_rates) if turnover_rates else 0,
            "portfolio_value": portfolio_value
        }
        
        return results
    
    def _calculate_max_drawdown(self, portfolio_value):
        """
        Calculate maximum drawdown
        
        Parameters
        ----------
        portfolio_value : pd.Series
            Portfolio value over time
            
        Returns
        -------
        float
            Maximum drawdown
        """
        # Calculate cumulative max
        cum_max = portfolio_value.cummax()
        
        # Calculate drawdown
        drawdown = (portfolio_value - cum_max) / cum_max
        
        # Get max drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def plot_portfolio_value(self, save_path=None):
        """
        Plot portfolio value
        
        Parameters
        ----------
        save_path : Optional[str]
            Path to save the plot
            
        Returns
        -------
        str
            Path to the saved plot
        """
        results = self.backtest()
        portfolio_value = results["portfolio_value"]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_value.index, portfolio_value.values)
        plt.title("Portfolio Value")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            return save_path
        
        # Otherwise show plot
        plt.show()
        plt.close()
        return None


def run_backtest(
    pred_path: str,
    save_path: Optional[str] = None,
    topk: int = 50,
    n_drop: int = 5,
    track_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Run backtest on prediction results
    
    Parameters
    ----------
    pred_path : str
        Path to prediction file or directory
    save_path : Optional[str]
        Path to save backtest results
    topk : int
        Number of instruments to hold
    n_drop : int
        Number of instruments to drop at each rebalance
    track_mlflow : bool
        Whether to track with MLflow
        
    Returns
    -------
    Dict[str, Any]
        Backtest results
    """
    # Create save path if not provided
    if save_path is None:
        if os.path.isdir(pred_path):
            save_path = os.path.join(pred_path, "backtest_results")
        else:
            save_path = os.path.join(os.path.dirname(pred_path), "backtest_results")
        
        # Add timestamp to save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_path}_{timestamp}"
        
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize MLflow if tracking is enabled
    run_id = None
    if track_mlflow:
        from src.config import MLFLOW_CONFIG
        run_id = init_mlflow(MLFLOW_CONFIG)
        mlflow.log_params({
            "pred_path": pred_path,
            "topk": topk,
            "n_drop": n_drop,
        })
    
    try:
        # Load predictions
        if os.path.isdir(pred_path):
            # Check if it's a directory with pred.csv or pred.pkl
            if os.path.exists(os.path.join(pred_path, "pred.csv")):
                pred_df = pd.read_csv(os.path.join(pred_path, "pred.csv"), index_col=0)
            elif os.path.exists(os.path.join(pred_path, "pred.pkl")):
                import pickle
                with open(os.path.join(pred_path, "pred.pkl"), "rb") as f:
                    pred_df = pickle.load(f)
            else:
                raise ValueError(f"No prediction file found in {pred_path}")
        else:
            # Assume it's a direct file path
            if pred_path.endswith(".csv"):
                pred_df = pd.read_csv(pred_path, index_col=0)
            elif pred_path.endswith(".pkl"):
                import pickle
                with open(pred_path, "rb") as f:
                    pred_df = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format for {pred_path}")
        
        # Initialize backtester
        backtester = SimpleBacktester(pred_df, topk=topk, n_drop=n_drop)
        
        # Run backtest
        print(f"Running backtest with topk={topk}, n_drop={n_drop}...")
        results = backtester.backtest()
        
        # Plot portfolio value
        plot_path = os.path.join(save_path, "portfolio_value.png")
        backtester.plot_portfolio_value(save_path=plot_path)
        
        # Save results to CSV
        results_df = pd.DataFrame({
            "total_return": [results["total_return"]],
            "annual_return": [results["annual_return"]],
            "sharpe": [results["sharpe"]],
            "max_drawdown": [results["max_drawdown"]],
            "turnover": [results["turnover"]]
        })
        
        results_path = os.path.join(save_path, "backtest_results.csv")
        results_df.to_csv(results_path)
        
        # Log to MLflow if tracking is enabled
        if track_mlflow:
            mlflow.log_artifact(plot_path)
            mlflow.log_artifact(results_path)
            mlflow.log_metrics({
                "total_return": results["total_return"],
                "annual_return": results["annual_return"],
                "sharpe": results["sharpe"],
                "max_drawdown": results["max_drawdown"],
                "turnover": results["turnover"]
            })
        
        print(f"Backtest results saved to {save_path}")
        
        return {
            "results": results,
            "plot_path": plot_path,
            "results_path": results_path,
            "save_path": save_path
        }
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        if track_mlflow:
            mlflow.log_param("backtest_status", "error")
            mlflow.log_param("backtest_error", str(e))
        raise e
    finally:
        if track_mlflow:
            close_mlflow()


def compare_strategies(
    pred_paths: List[str],
    strategy_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    topk: int = 50,
    n_drop: int = 5,
    track_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Compare multiple trading strategies
    
    Parameters
    ----------
    pred_paths : List[str]
        List of paths to prediction files or directories
    strategy_names : Optional[List[str]]
        List of strategy names
    save_path : Optional[str]
        Path to save comparison results
    topk : int
        Number of instruments to hold
    n_drop : int
        Number of instruments to drop at each rebalance
    track_mlflow : bool
        Whether to track with MLflow
        
    Returns
    -------
    Dict[str, Any]
        Comparison results
    """
    # Create strategy names if not provided
    if strategy_names is None:
        strategy_names = [f"Strategy_{i+1}" for i in range(len(pred_paths))]
    
    # Create save path if not provided
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "strategy_comparison")
        
        # Add timestamp to save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{save_path}_{timestamp}"
        
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize MLflow if tracking is enabled
    run_id = None
    if track_mlflow:
        from src.config import MLFLOW_CONFIG
        run_id = init_mlflow(MLFLOW_CONFIG)
        mlflow.log_params({
            "topk": topk,
            "n_drop": n_drop,
            "num_strategies": len(pred_paths)
        })
        for i, (name, path) in enumerate(zip(strategy_names, pred_paths)):
            mlflow.log_param(f"strategy_{i+1}_name", name)
            mlflow.log_param(f"strategy_{i+1}_path", path)
    
    try:
        # Results for all strategies
        all_results = {}
        portfolio_values = []
        
        # Run backtest for each strategy
        for i, (name, path) in enumerate(zip(strategy_names, pred_paths)):
            print(f"Running backtest for {name}...")
            
            # Create save path for this strategy
            strategy_save_path = os.path.join(save_path, name)
            os.makedirs(strategy_save_path, exist_ok=True)
            
            # Run backtest
            results = run_backtest(
                pred_path=path,
                save_path=strategy_save_path,
                topk=topk,
                n_drop=n_drop,
                track_mlflow=False  # Don't track individual runs
            )
            
            # Store results
            all_results[name] = results["results"]
            portfolio_values.append(results["results"]["portfolio_value"])
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            name: {
                "total_return": results["total_return"],
                "annual_return": results["annual_return"],
                "sharpe": results["sharpe"],
                "max_drawdown": results["max_drawdown"],
                "turnover": results["turnover"]
            } for name, results in all_results.items()
        }).T
        
        # Save comparison results
        comparison_path = os.path.join(save_path, "strategy_comparison.csv")
        comparison_df.to_csv(comparison_path)
        
        # Plot portfolio values for all strategies
        plt.figure(figsize=(12, 6))
        for i, (name, portfolio_value) in enumerate(zip(strategy_names, portfolio_values)):
            plt.plot(portfolio_value.index, portfolio_value.values, label=name)
        
        plt.title("Portfolio Value Comparison")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save comparison plot
        comparison_plot_path = os.path.join(save_path, "portfolio_value_comparison.png")
        plt.savefig(comparison_plot_path)
        plt.close()
        
        # Log to MLflow if tracking is enabled
        if track_mlflow:
            mlflow.log_artifact(comparison_path)
            mlflow.log_artifact(comparison_plot_path)
            
            # Log metrics for each strategy
            for name, results in all_results.items():
                mlflow.log_metrics({
                    f"{name}_total_return": results["total_return"],
                    f"{name}_annual_return": results["annual_return"],
                    f"{name}_sharpe": results["sharpe"],
                    f"{name}_max_drawdown": results["max_drawdown"],
                    f"{name}_turnover": results["turnover"]
                })
        
        print(f"Strategy comparison results saved to {save_path}")
        
        return {
            "results": all_results,
            "comparison_df": comparison_df,
            "comparison_path": comparison_path,
            "comparison_plot_path": comparison_plot_path,
            "save_path": save_path
        }
    except Exception as e:
        print(f"Error in strategy comparison: {str(e)}")
        if track_mlflow:
            mlflow.log_param("comparison_status", "error")
            mlflow.log_param("comparison_error", str(e))
        raise e
    finally:
        if track_mlflow:
            close_mlflow() 