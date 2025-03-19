"""
LeanEngine.py - LEAN Engine Implementation

This module provides an adapter for QuantConnect LEAN backtesting engine.
It implements the IEngine interface and provides methods to run backtests 
and analyze results.
"""

import os
import json
import logging
import tempfile
import shutil
import importlib
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Import LEAN components using pythonnet
try:
    import clr
    clr.AddReference("QuantConnect.Common")
    clr.AddReference("QuantConnect.Algorithm")
    clr.AddReference("QuantConnect.Lean.Engine")
    
    from QuantConnect import *
    from QuantConnect.Lean.Engine import Engine
    from QuantConnect.Lean.Engine.Results import BacktestingResultHandler
    from QuantConnect.Packets import BacktestSettings
    
    LEAN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"QuantConnect LEAN framework import failed: {e}")
    logging.warning("Some features may not be available")
    LEAN_AVAILABLE = False

from src.StockScan.Interfaces.IEngine import IEngine, IBacktester

# Set up logger
logger = logging.getLogger(__name__)


class LeanEngineAdapter(IBacktester):
    """
    LEAN engine adapter implementing the IBacktester interface.
    This provides a standardized way to run backtests using the LEAN engine.
    """
    
    def __init__(self, 
                data_folder: Optional[str] = None,
                results_folder: Optional[str] = None,
                config_path: Optional[str] = None):
        """
        Initialize the LEAN engine adapter.
        
        Args:
            data_folder: Path to the data folder
            results_folder: Path to the results folder
            config_path: Path to the configuration file
        """
        self.data_folder = data_folder or os.path.join(os.getcwd(), "data")
        self.results_folder = results_folder or os.path.join(os.getcwd(), "results")
        self.config_path = config_path
        
        # Ensure folders exist
        Path(self.data_folder).mkdir(parents=True, exist_ok=True)
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize algorithm registry
        self.algorithms = self._discover_algorithms()
        
        # Check if LEAN is available
        if not LEAN_AVAILABLE:
            logger.error("LEAN framework not available. Please install the required dependencies.")
            logger.error("pip install quantconnect-lean quantconnect-stubs pythonnet")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Dictionary containing configuration
        """
        default_config = {
            "default_cash": 100000,
            "default_market": "usa",
            "default_resolution": "daily",
            "timeframe": "daily",
            "cache_data": True,
            "use_benchmark": True,
            "benchmark_symbol": "SPY",
            "slippage_model": "percentage",
            "slippage_value": 0.001,
            "commission_model": "fixed",
            "commission_value": 1.0
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge user config with defaults
                    config = {**default_config, **user_config}
                    logger.info(f"Loaded configuration from {self.config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        logger.info("Using default configuration")
        return default_config
    
    def _discover_algorithms(self) -> Dict[str, Any]:
        """
        Discover available LEAN algorithms in the project.
        
        Returns:
            Dictionary mapping algorithm names to algorithm classes
        """
        algorithms = {}
        
        try:
            # Import algorithm modules from StockScan.Algorithm
            algorithms_module = importlib.import_module("src.StockScan.Algorithm")
            
            # Find algorithm files in the Algorithm directory
            algorithm_path = Path(os.path.dirname(algorithms_module.__file__))
            for py_file in algorithm_path.glob("*.py"):
                if py_file.stem in ['__init__', 'BaseAlgorithm']:
                    continue
                
                try:
                    # Import the module
                    module_name = f"src.StockScan.Algorithm.{py_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find all classes that inherit from BaseAlgorithm or QCAlgorithm
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and (
                            hasattr(obj, 'Initialize') and 
                            hasattr(obj, 'OnData') and 
                            hasattr(obj, 'SetStartDate')
                        ):
                            algorithms[name] = obj
                            logger.info(f"Discovered algorithm: {name}")
                except Exception as e:
                    logger.error(f"Error importing algorithm module {py_file.stem}: {e}")
        except Exception as e:
            logger.error(f"Error discovering algorithms: {e}")
        
        return algorithms
    
    def Initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the engine with the given configuration.
        
        Args:
            config: Engine configuration
        """
        # Update config with provided values
        self.config.update(config)
        
        # Initialize algorithm registry (in case new algorithms were added)
        self.algorithms = self._discover_algorithms()
        
        logger.info(f"Initialized LEAN engine with {len(self.algorithms)} algorithms")
    
    def Run(self, algorithm: Any, settings: Any) -> Any:
        """
        Run the engine with the given algorithm and settings.
        
        Args:
            algorithm: The algorithm to run
            settings: The settings for the run
            
        Returns:
            The results of the run
        """
        # Create engine
        engine = Engine()
        
        # Run backtest
        result = engine.Run(algorithm, settings)
        
        return result
    
    def StopEngine(self) -> None:
        """
        Stop the engine if it's running.
        """
        # In LEAN, we don't need to explicitly stop the engine for backtests
        pass
    
    def get_available_algorithms(self) -> List[str]:
        """
        Get a list of available algorithm names.
        
        Returns:
            List of algorithm names
        """
        return list(self.algorithms.keys())
    
    def RunBacktest(self, 
                  algorithm_name: str,
                  symbols: List[str],
                  start_date: Union[str, datetime],
                  end_date: Union[str, datetime],
                  initial_capital: float = 100000,
                  resolution: str = "Daily",
                  parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a backtest with the specified parameters.
        
        Args:
            algorithm_name: Name of the algorithm to use
            symbols: List of symbols to backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            resolution: Data resolution
            parameters: Additional parameters for the algorithm
            
        Returns:
            Dictionary containing backtest results
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Get algorithm class
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {', '.join(self.algorithms.keys())}")
        
        algorithm_class = self.algorithms[algorithm_name]
        
        # Run LEAN backtest
        return self._run_lean_backtest(algorithm_class, symbols, start_date, end_date, initial_capital, resolution, parameters)
    
    def _run_lean_backtest(self, 
                         algorithm_class,
                         symbols: List[str],
                         start_date: datetime,
                         end_date: datetime,
                         initial_capital: float,
                         resolution: str,
                         parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a backtest using LEAN's engine.
        
        Args:
            algorithm_class: LEAN algorithm class
            symbols: List of symbols to backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            resolution: Data resolution
            parameters: Additional parameters for the algorithm
            
        Returns:
            Dictionary containing backtest results
        """
        # Create temporary folder for backtest
        backtest_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir = os.path.join(tempfile.gettempdir(), backtest_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Create algorithm instance
            algorithm = algorithm_class()
            
            # Set parameters
            if parameters:
                for key, value in parameters.items():
                    if hasattr(algorithm, key):
                        setattr(algorithm, key, value)
            
            # Configure algorithm
            algorithm.SetStartDate(start_date.year, start_date.month, start_date.day)
            algorithm.SetEndDate(end_date.year, end_date.month, end_date.day)
            algorithm.SetCash(initial_capital)
            
            # Add specified symbols
            algorithm.symbols = []
            for symbol in symbols:
                algorithm.add_symbol(symbol)
            
            # Configure data folder if needed
            data_folder = self.data_folder
            results_folder = os.path.join(self.results_folder, backtest_id)
            os.makedirs(results_folder, exist_ok=True)
            
            # Create LEAN backtest settings
            settings = BacktestSettings()
            settings.Name = backtest_id
            settings.BacktestId = backtest_id
            
            # Run backtest
            lean_result = self.Run(algorithm, settings)
            
            # Process results
            results = self._process_lean_results(lean_result, results_folder)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running LEAN backtest: {e}")
            raise
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _process_lean_results(self, lean_result, results_folder: str) -> Dict[str, Any]:
        """
        Process LEAN backtest results into a standardized format.
        
        Args:
            lean_result: Result object from LEAN backtest
            results_folder: Folder containing result files
            
        Returns:
            Dictionary containing processed results
        """
        processed_results = {
            'backtest_id': lean_result.BacktestId,
            'algorithm_name': lean_result.AlgorithmName,
            'start_date': lean_result.PeriodStart,
            'end_date': lean_result.PeriodFinish,
            'initial_capital': lean_result.InitialCapital,
            'final_capital': lean_result.FinalCapital,
            'total_trades': lean_result.TotalOrders,
            'runtime': lean_result.Runtime,
        }
        
        # Process performance metrics
        metrics = {
            'total_return': (lean_result.FinalCapital / lean_result.InitialCapital) - 1,
            'annual_return': lean_result.AnnualReturn,
            'sharpe_ratio': lean_result.SharpeRatio,
            'max_drawdown': lean_result.Drawdown,
            'win_rate': lean_result.WinRate if hasattr(lean_result, 'WinRate') else None,
            'loss_rate': lean_result.LossRate if hasattr(lean_result, 'LossRate') else None,
            'profit_loss_ratio': lean_result.ProfitLossRatio if hasattr(lean_result, 'ProfitLossRatio') else None,
        }
        
        processed_results['metrics'] = metrics
        
        # Add any result files
        processed_results['result_files'] = {}
        if os.path.exists(results_folder):
            for file in os.listdir(results_folder):
                file_path = os.path.join(results_folder, file)
                if os.path.isfile(file_path):
                    processed_results['result_files'][file] = file_path
        
        return processed_results
    
    def OptimizeParameters(self, 
                         algorithm_name: str,
                         symbols: List[str],
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime],
                         parameter_space: Dict[str, List[Any]],
                         initial_capital: float = 100000,
                         resolution: str = "Daily",
                         optimization_metric: str = "SharpeRatio",
                         max_concurrent_backtests: int = 4) -> Dict[str, Any]:
        """
        Optimize algorithm parameters using grid search.
        
        Args:
            algorithm_name: Name of the algorithm to optimize
            symbols: List of symbols to backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            parameter_space: Dictionary mapping parameter names to lists of values to try
            initial_capital: Initial capital for the backtest
            resolution: Data resolution
            optimization_metric: Metric to optimize (e.g., SharpeRatio, TotalReturn)
            max_concurrent_backtests: Maximum number of concurrent backtests to run
            
        Returns:
            Dictionary containing optimization results
        """
        # Implementation of parameter optimization using grid search
        import itertools
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate parameter combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Store results
        optimization_results = []
        
        # Run backtests for each parameter combination
        for combo in param_combinations:
            # Create parameters dict
            params = {param_names[i]: combo[i] for i in range(len(param_names))}
            
            # Run backtest with these parameters
            result = self.RunBacktest(
                algorithm_name=algorithm_name,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                resolution=resolution,
                parameters=params
            )
            
            # Add parameters to result
            result['parameters'] = params
            
            # Add to results list
            optimization_results.append(result)
        
        # Find best result based on optimization metric
        if optimization_metric == "SharpeRatio":
            best_result = max(optimization_results, key=lambda x: x['metrics']['sharpe_ratio'])
        elif optimization_metric == "TotalReturn":
            best_result = max(optimization_results, key=lambda x: x['metrics']['total_return'])
        elif optimization_metric == "MaxDrawdown":
            best_result = min(optimization_results, key=lambda x: x['metrics']['max_drawdown'])
        else:
            # Default to Sharpe ratio
            best_result = max(optimization_results, key=lambda x: x['metrics']['sharpe_ratio'])
        
        # Return optimization summary
        return {
            'best_result': best_result,
            'all_results': optimization_results,
            'parameter_space': parameter_space,
            'optimization_metric': optimization_metric,
            'total_combinations': len(param_combinations)
        }
    
    def AnalyzeResults(self, backtest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the results of a backtest.
        
        Args:
            backtest_result: The result of a backtest
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract results for analysis
        metrics = backtest_result['metrics']
        
        # Calculate additional metrics
        analysis = {
            'summary': {
                'total_return': metrics['total_return'],
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'win_rate': metrics['win_rate'],
                'profit_loss_ratio': metrics['profit_loss_ratio']
            },
            'risk_metrics': {
                'volatility': metrics.get('volatility', None),
                'beta': metrics.get('beta', None),
                'alpha': metrics.get('alpha', None),
                'information_ratio': metrics.get('information_ratio', None),
                'sortino_ratio': metrics.get('sortino_ratio', None)
            },
            'trade_statistics': {
                'total_trades': backtest_result['total_trades'],
                'average_win': metrics.get('average_win', None),
                'average_loss': metrics.get('average_loss', None),
                'largest_win': metrics.get('largest_win', None),
                'largest_loss': metrics.get('largest_loss', None)
            }
        }
        
        return analysis 