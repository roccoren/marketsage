"""
LEAN Strategy Benchmarking

This module provides functionality for comparing multiple LEAN strategies against
each other and against market benchmarks. It facilitates running multiple backtests
with the same data and time periods, normalizing results, and generating comparative
performance metrics and visualizations.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import asyncio
import concurrent.futures

# Import LEAN components
from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode

logger = logging.getLogger(__name__)

# Standard performance metrics to compute
PERFORMANCE_METRICS = [
    'sharpe_ratio',
    'sortino_ratio',
    'drawdown',
    'max_drawdown',
    'annual_return',
    'annual_volatility',
    'calmar_ratio',
    'omega_ratio',
    'information_ratio',
    'alpha',
    'beta',
    'total_trades',
    'win_rate',
    'profit_loss_ratio',
    'total_return'
]

class BenchmarkResult:
    """
    Container for benchmark results, storing performance metrics for multiple strategies.
    
    Attributes:
        strategies: Dictionary mapping strategy names to their performance metrics
        benchmark_strategy: Name of the benchmark strategy
        benchmark_metrics: Performance metrics for the benchmark
        symbols: List of symbols used in the benchmark
        start_date: Start date of the backtest
        end_date: End date of the backtest
        created_at: Timestamp of when the benchmark was run
    """
    
    def __init__(self, 
                strategies: Dict[str, Dict[str, Any]], 
                symbols: List[str],
                start_date: str,
                end_date: str,
                benchmark_strategy: Optional[str] = None):
        """
        Initialize benchmark result.
        
        Args:
            strategies: Dictionary mapping strategy names to their performance metrics
            symbols: List of symbols used in the benchmark
            start_date: Start date of the backtest
            end_date: End date of the backtest
            benchmark_strategy: Name of the benchmark strategy
        """
        self.strategies = strategies
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_strategy = benchmark_strategy
        self.created_at = datetime.now()
        
        # Extract benchmark metrics if available
        self.benchmark_metrics = None
        if benchmark_strategy and benchmark_strategy in strategies:
            self.benchmark_metrics = strategies[benchmark_strategy]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary representation."""
        return {
            'strategies': self.strategies,
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'benchmark_strategy': self.benchmark_strategy,
            'created_at': self.created_at.isoformat()
        }
    
    def save(self, file_path: str):
        """
        Save benchmark results to a file.
        
        Args:
            file_path: Path to save the results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare data to save
        data = self.to_dict()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Benchmark results saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'BenchmarkResult':
        """
        Load benchmark results from a file.
        
        Args:
            file_path: Path to load the results from
            
        Returns:
            BenchmarkResult object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        result = cls(
            strategies=data['strategies'],
            symbols=data['symbols'],
            start_date=data['start_date'],
            end_date=data['end_date'],
            benchmark_strategy=data.get('benchmark_strategy')
        )
        
        # Parse created_at if available
        if 'created_at' in data:
            result.created_at = datetime.fromisoformat(data['created_at'])
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert benchmark results to a pandas DataFrame.
        
        Returns:
            DataFrame with metrics for all strategies
        """
        # Extract metrics for all strategies
        metrics_data = []
        
        for strategy_name, metrics in self.strategies.items():
            strategy_data = {'strategy': strategy_name}
            strategy_data.update(metrics)
            metrics_data.append(strategy_data)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Set strategy as index
        if 'strategy' in df.columns:
            df = df.set_index('strategy')
        
        return df
    
    def plot_returns(self, figsize: Tuple[int, int] = (12, 8), 
                    title: str = 'Cumulative Returns Comparison', 
                    save_path: Optional[str] = None):
        """
        Plot cumulative returns for all strategies.
        
        Args:
            figsize: Figure size (width, height)
            title: Plot title
            save_path: Path to save the plot (None to display only)
            
        Returns:
            Matplotlib figure
        """
        # Check if equity curves are available
        equity_curves = {}
        for strategy_name, metrics in self.strategies.items():
            if 'equity_curve' in metrics:
                equity_curves[strategy_name] = metrics['equity_curve']
        
        if not equity_curves:
            logger.warning("No equity curves available in benchmark results. Cannot plot returns.")
            return None
        
        # Create a figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each equity curve
        for strategy_name, curve in equity_curves.items():
            # Convert to DataFrame if it's a list or dict
            if isinstance(curve, list):
                # Assume it's a list of (date, value) tuples
                curve_df = pd.DataFrame(curve, columns=['date', 'value'])
                curve_df['date'] = pd.to_datetime(curve_df['date'])
                curve_df = curve_df.set_index('date')
            elif isinstance(curve, dict):
                # Assume it's a dict with dates as keys and values as values
                curve_df = pd.DataFrame(list(curve.items()), columns=['date', 'value'])
                curve_df['date'] = pd.to_datetime(curve_df['date'])
                curve_df = curve_df.set_index('date')
            else:
                # Assume it's already a properly formatted DataFrame
                curve_df = pd.DataFrame(curve)
            
            # Plot the curve
            is_benchmark = strategy_name == self.benchmark_strategy
            ax.plot(curve_df.index, curve_df['value'], 
                   label=strategy_name, 
                   linewidth=3 if is_benchmark else 2,
                   linestyle='--' if is_benchmark else '-')
        
        # Add labels and title
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Rotate date labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns plot saved to {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, metrics: List[str] = None, 
                              figsize: Tuple[int, int] = (14, 10),
                              title: str = 'Performance Metrics Comparison',
                              save_path: Optional[str] = None):
        """
        Plot a comparison of performance metrics across strategies.
        
        Args:
            metrics: List of metrics to compare (None for all available)
            figsize: Figure size (width, height)
            title: Plot title
            save_path: Path to save the plot (None to display only)
            
        Returns:
            Matplotlib figure
        """
        # Get data as DataFrame
        df = self.to_dataframe()
        
        # Select metrics
        if metrics is None:
            # Use common numerical metrics, excluding equity curves and other non-numeric data
            metrics = [col for col in df.columns if df[col].dtype in [np.float64, np.int64, float, int]]
        else:
            # Filter to available metrics
            metrics = [m for m in metrics if m in df.columns]
        
        if not metrics:
            logger.warning("No suitable metrics available for plotting.")
            return None
        
        # Create a figure
        fig, axes = plt.subplots(nrows=len(metrics), figsize=figsize, sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Sort by metric value
            sorted_df = df.sort_values(by=metric)
            
            # Create bar colors
            colors = ['skyblue'] * len(sorted_df)
            if self.benchmark_strategy in sorted_df.index:
                benchmark_idx = sorted_df.index.get_loc(self.benchmark_strategy)
                colors[benchmark_idx] = 'orange'
            
            # Plot
            sorted_df[metric].plot(kind='barh', ax=ax, color=colors)
            
            # Add metric name as y-label
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            
            # Add gridlines
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for j, v in enumerate(sorted_df[metric]):
                if pd.notnull(v):  # Skip NaN values
                    ax.text(max(v + 0.01, 0.01) if v >= 0 else v - 0.03, 
                           j, f'{v:.4f}', va='center', fontsize=9)
        
        # Add title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_dir: str, report_name: str = None):
        """
        Generate a comprehensive HTML report of benchmark results.
        
        Args:
            output_dir: Directory to save the report and its resources
            report_name: Name of the report (None for automatic naming)
            
        Returns:
            Path to the generated report
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a report name if not provided
        if not report_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_name = f"benchmark_report_{timestamp}"
        
        # Define paths
        report_path = os.path.join(output_dir, f"{report_name}.html")
        returns_plot_path = os.path.join(output_dir, f"{report_name}_returns.png")
        metrics_plot_path = os.path.join(output_dir, f"{report_name}_metrics.png")
        
        # Create plots and save them
        self.plot_returns(save_path=returns_plot_path)
        self.plot_metrics_comparison(save_path=metrics_plot_path)
        
        # Get metrics data
        metrics_df = self.to_dataframe()
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Strategy Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ margin-bottom: 30px; }}
                .section {{ margin-bottom: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 50px; font-size: 0.8em; color: #777; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Strategy Benchmark Report</h1>
                <p><strong>Date Range:</strong> {self.start_date} to {self.end_date}</p>
                <p><strong>Symbols:</strong> {', '.join(self.symbols)}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Total Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Annual Return</th>
                        <th>Win Rate</th>
                    </tr>
        """
        
        # Add rows for each strategy
        for strategy_name, metrics in self.strategies.items():
            html_content += f"""
                    <tr>
                        <td>{'🔹 ' if strategy_name == self.benchmark_strategy else ''}{strategy_name}</td>
                        <td>{metrics.get('total_return', 'N/A'):.4f}</td>
                        <td>{metrics.get('sharpe_ratio', 'N/A'):.4f}</td>
                        <td>{metrics.get('max_drawdown', 'N/A'):.4f}</td>
                        <td>{metrics.get('annual_return', 'N/A'):.4f}</td>
                        <td>{metrics.get('win_rate', 'N/A'):.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Cumulative Returns</h2>
                <div class="plot">
                    <img src="{}" alt="Cumulative Returns Plot">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics Comparison</h2>
                <div class="plot">
                    <img src="{}" alt="Metrics Comparison Plot">
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Metrics</h2>
                <table>
        """.format(os.path.basename(returns_plot_path), os.path.basename(metrics_plot_path))
        
        # Add header row with all metrics
        metrics_cols = [col for col in metrics_df.columns if col != 'equity_curve']
        html_content += "                    <tr>\n"
        html_content += "                        <th>Strategy</th>\n"
        for col in metrics_cols:
            html_content += f"                        <th>{col.replace('_', ' ').title()}</th>\n"
        html_content += "                    </tr>\n"
        
        # Add rows for each strategy
        for strategy_name in metrics_df.index:
            html_content += "                    <tr>\n"
            html_content += f"                        <td>{'🔹 ' if strategy_name == self.benchmark_strategy else ''}{strategy_name}</td>\n"
            for col in metrics_cols:
                value = metrics_df.loc[strategy_name, col]
                if isinstance(value, (int, float)):
                    html_content += f"                        <td>{value:.4f}</td>\n"
                else:
                    html_content += f"                        <td>{value}</td>\n"
            html_content += "                    </tr>\n"
        
        html_content += """
                </table>
            </div>
            
            <div class="footer">
                <p>Generated using LEAN Strategy Benchmarking Tool</p>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Benchmark report generated at {report_path}")
        return report_path

class StrategyBenchmark:
    """
    Benchmark multiple LEAN trading strategies against each other and against
    standard market benchmarks.
    """
    
    def __init__(self, 
                 data_folder: Optional[str] = None,
                 results_folder: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the strategy benchmark.
        
        Args:
            data_folder: Path to data folder
            results_folder: Path to results folder
            config_path: Path to configuration file
        """
        # Initialize LEAN adapter
        self.lean_adapter = LeanAdapter(
            config_path=config_path,
            mode=BacktestMode.LEAN,
            data_folder=data_folder,
            results_folder=results_folder
        )
        
        self.data_folder = data_folder
        self.results_folder = results_folder or 'results/benchmarks'
        
        # Ensure the results folder exists
        os.makedirs(self.results_folder, exist_ok=True)
    
    def run_benchmark(self, 
                     algorithm_names: List[str],
                     symbols: List[str],
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime],
                     benchmark_algorithm: Optional[str] = None,
                     parameters: Optional[Dict[str, Dict[str, Any]]] = None,
                     initial_capital: float = 100000,
                     resolution: str = 'Daily',
                     parallel: bool = True,
                     max_workers: int = None,
                     include_equity_curves: bool = True) -> BenchmarkResult:
        """
        Run backtests for multiple algorithms with the same parameters and compare results.
        
        Args:
            algorithm_names: List of algorithm names to benchmark
            symbols: List of symbols to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            benchmark_algorithm: Name of the algorithm to use as benchmark (None for no benchmark)
            parameters: Dictionary mapping algorithm names to their parameters
            initial_capital: Initial capital for backtesting
            resolution: Data resolution for backtesting
            parallel: Whether to run backtests in parallel
            max_workers: Maximum number of worker processes for parallel execution
            include_equity_curves: Whether to include equity curves in the results
            
        Returns:
            BenchmarkResult containing the results of all backtests
        """
        # Handle dates
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
            
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Verify algorithms exist
        available_algorithms = self.lean_adapter.get_available_algorithms()
        for algorithm in algorithm_names:
            if algorithm not in available_algorithms:
                raise ValueError(f"Algorithm '{algorithm}' not found. Available algorithms: {', '.join(available_algorithms)}")
        
        # Initialize default parameters if not provided
        if parameters is None:
            parameters = {}
        
        # Define a function to run a single backtest
        def run_backtest(algorithm_name):
            try:
                logger.info(f"Running backtest for {algorithm_name}")
                
                # Get parameters for this algorithm
                algo_params = parameters.get(algorithm_name, {})
                
                # Run backtest
                result = self.lean_adapter.run_backtest(
                    algorithm_name=algorithm_name,
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    resolution=resolution,
                    parameters=algo_params
                )
                
                # Extract metrics
                metrics = result.get('metrics', {})
                
                # Add equity curve if requested
                if include_equity_curves and 'equity_curve' in result:
                    metrics['equity_curve'] = result['equity_curve']
                
                return algorithm_name, metrics
                
            except Exception as e:
                logger.error(f"Error running backtest for {algorithm_name}: {e}")
                return algorithm_name, {'error': str(e)}
        
        # Run backtests
        start_time = time.time()
        
        if parallel:
            # Parallel execution
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_backtest, algo): algo for algo in algorithm_names}
                results = {}
                
                for future in concurrent.futures.as_completed(futures):
                    algo = futures[future]
                    try:
                        algo_name, metrics = future.result()
                        results[algo_name] = metrics
                    except Exception as e:
                        logger.error(f"Error processing results for {algo}: {e}")
                        results[algo] = {'error': str(e)}
        else:
            # Sequential execution
            results = {}
            for algo in algorithm_names:
                algo_name, metrics = run_backtest(algo)
                results[algo_name] = metrics
        
        run_time = time.time() - start_time
        logger.info(f"Benchmark completed in {run_time:.2f} seconds")
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            strategies=results,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            benchmark_strategy=benchmark_algorithm
        )
        
        return benchmark_result
    
    def compare_strategy_parameters(self,
                                   algorithm_name: str,
                                   parameter_sets: Dict[str, Dict[str, Any]],
                                   symbols: List[str],
                                   start_date: Union[str, datetime],
                                   end_date: Union[str, datetime],
                                   benchmark_algorithm: Optional[str] = None,
                                   benchmark_params: Optional[Dict[str, Any]] = None,
                                   initial_capital: float = 100000,
                                   resolution: str = 'Daily',
                                   parallel: bool = True,
                                   max_workers: int = None) -> BenchmarkResult:
        """
        Compare the performance of a single strategy with different parameter sets.
        
        Args:
            algorithm_name: Name of the algorithm to test
            parameter_sets: Dictionary mapping parameter set names to parameter dictionaries
            symbols: List of symbols to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            benchmark_algorithm: Name of the algorithm to use as benchmark
            benchmark_params: Parameters for the benchmark algorithm
            initial_capital: Initial capital for backtesting
            resolution: Data resolution for backtesting
            parallel: Whether to run backtests in parallel
            max_workers: Maximum number of worker processes for parallel execution
            
        Returns:
            BenchmarkResult containing the results of all parameter sets
        """
        # Create algorithm names for each parameter set
        algorithm_names = []
        parameters = {}
        
        for param_set_name, params in parameter_sets.items():
            full_name = f"{algorithm_name}_{param_set_name}"
            algorithm_names.append(full_name)
            parameters[full_name] = params
        
        # Add benchmark if provided
        if benchmark_algorithm:
            if benchmark_algorithm not in self.lean_adapter.get_available_algorithms():
                raise ValueError(f"Benchmark algorithm '{benchmark_algorithm}' not found.")
            
            algorithm_names.append(benchmark_algorithm)
            if benchmark_params:
                parameters[benchmark_algorithm] = benchmark_params
        
        # Run benchmark
        benchmark_result = self.run_benchmark(
            algorithm_names=algorithm_names,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            benchmark_algorithm=benchmark_algorithm,
            parameters=parameters,
            initial_capital=initial_capital,
            resolution=resolution,
            parallel=parallel,
            max_workers=max_workers
        )
        
        return benchmark_result
    
    def create_standard_benchmark(self, 
                                benchmark_type: str = 'spy',
                                symbols: List[str] = None,
                                start_date: Union[str, datetime] = None,
                                end_date: Union[str, datetime] = None) -> str:
        """
        Create a standard benchmark algorithm based on common investment strategies.
        
        Args:
            benchmark_type: Type of benchmark to create
                - 'spy': S&P 500 ETF
                - 'balanced': 60/40 stocks/bonds
                - 'equal_weight': Equal weight portfolio
                - 'buy_and_hold': Buy and hold strategy
            symbols: List of symbols for the benchmark (ignored for some benchmarks)
            start_date: Start date for backtesting (for template creation)
            end_date: End date for backtesting (for template creation)
            
        Returns:
            Name of the created benchmark algorithm
        """
        from src.market_system.lean.strategy_wizard import StrategyWizard
        
        # Set defaults
        if symbols is None:
            if benchmark_type == 'spy':
                symbols = ['SPY']
            elif benchmark_type == 'balanced':
                symbols = ['SPY', 'TLT']  # S&P 500 and Long-Term Treasury ETFs
            else:
                symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'AGG']  # Common ETF mix
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        elif isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Initialize strategy wizard
        wizard = StrategyWizard(output_dir=os.path.join(self.lean_adapter.algorithms_dir))
        
        # Create the benchmark strategy
        if benchmark_type == 'spy':
            # SPY benchmark
            return wizard.create_strategy(
                strategy_name="SPYBenchmark",
                pattern_key="custom",
                description="S&P 500 ETF benchmark",
                symbols=['SPY'],
                start_date=start_date,
                end_date=end_date
            )
        elif benchmark_type == 'balanced':
            # 60/40 portfolio
            return wizard.create_strategy(
                strategy_name="BalancedPortfolioBenchmark",
                pattern_key="custom",
                description="60/40 stocks/bonds balanced portfolio benchmark",
                symbols=['SPY', 'TLT'],
                parameters={'stock_weight': 0.6, 'bond_weight': 0.4},
                start_date=start_date,
                end_date=end_date
            )
        elif benchmark_type == 'equal_weight':
            # Equal weight portfolio
            return wizard.create_strategy(
                strategy_name="EqualWeightBenchmark",
                pattern_key="custom",
                description="Equal weight portfolio benchmark",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
        elif benchmark_type == 'buy_and_hold':
            # Buy and hold
            return wizard.create_strategy(
                strategy_name="BuyAndHoldBenchmark",
                pattern_key="custom",
                description="Buy and hold benchmark",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

def run_benchmark(algorithm_names: List[str],
                 symbols: List[str],
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 benchmark_algorithm: Optional[str] = None,
                 parameters: Optional[Dict[str, Dict[str, Any]]] = None,
                 data_folder: Optional[str] = None,
                 results_folder: Optional[str] = None,
                 config_path: Optional[str] = None,
                 save_results: bool = True,
                 save_path: Optional[str] = None,
                 generate_report: bool = True,
                 report_dir: Optional[str] = None) -> BenchmarkResult:
    """
    Convenience function to benchmark multiple strategies.
    
    Args:
        algorithm_names: List of algorithm names to benchmark
        symbols: List of symbols to backtest
        start_date: Start date for backtesting
        end_date: End date for backtesting
        benchmark_algorithm: Name of the algorithm to use as benchmark
        parameters: Dictionary mapping algorithm names to their parameters
        data_folder: Path to data folder
        results_folder: Path to results folder
        config_path: Path to configuration file
        save_results: Whether to save the benchmark results
        save_path: Path to save the benchmark results (if None, uses default path)
        generate_report: Whether to generate an HTML report
        report_dir: Directory to save the report (if None, uses results_folder)
        
    Returns:
        BenchmarkResult containing the results of all backtests
    """
    # Create benchmarker
    benchmarker = StrategyBenchmark(
        data_folder=data_folder,
        results_folder=results_folder,
        config_path=config_path
    )
    
    # Run benchmark
    result = benchmarker.run_benchmark(
        algorithm_names=algorithm_names,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        benchmark_algorithm=benchmark_algorithm,
        parameters=parameters
    )
    
    # Save results if requested
    if save_results:
        if save_path is None:
            # Default path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = results_folder or 'results/benchmarks'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/benchmark_{timestamp}.json"
        
        result.save(save_path)
    
    # Generate report if requested
    if generate_report:
        if report_dir is None:
            report_dir = results_folder or 'results/benchmarks/reports'
        
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result.generate_report(report_dir, f"benchmark_report_{timestamp}")
    
    return result

def compare_parameters(algorithm_name: str,
                      parameter_sets: Dict[str, Dict[str, Any]],
                      symbols: List[str],
                      start_date: Union[str, datetime],
                      end_date: Union[str, datetime],
                      benchmark_algorithm: Optional[str] = None,
                      data_folder: Optional[str] = None,
                      results_folder: Optional[str] = None,
                      config_path: Optional[str] = None,
                      save_results: bool = True,
                      generate_report: bool = True) -> BenchmarkResult:
    """
    Convenience function to compare strategy parameters.
    
    Args:
        algorithm_name: Name of the algorithm to test
        parameter_sets: Dictionary mapping parameter set names to parameter dictionaries
        symbols: List of symbols to backtest
        start_date: Start date for backtesting
        end_date: End date for backtesting
        benchmark_algorithm: Name of the algorithm to use as benchmark
        data_folder: Path to data folder
        results_folder: Path to results folder
        config_path: Path to configuration file
        save_results: Whether to save the benchmark results
        generate_report: Whether to generate an HTML report
        
    Returns:
        BenchmarkResult containing the results of all parameter sets
    """
    # Create benchmarker
    benchmarker = StrategyBenchmark(
        data_folder=data_folder,
        results_folder=results_folder,
        config_path=config_path
    )
    
    # Run parameter comparison
    result = benchmarker.compare_strategy_parameters(
        algorithm_name=algorithm_name,
        parameter_sets=parameter_sets,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        benchmark_algorithm=benchmark_algorithm
    )
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = results_folder or 'results/benchmarks'
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{algorithm_name}_params_comparison_{timestamp}.json"
        result.save(save_path)
    
    # Generate report if requested
    if generate_report:
        report_dir = results_folder or 'results/benchmarks/reports'
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result.generate_report(report_dir, f"{algorithm_name}_params_report_{timestamp}")
    
    return result 