"""
LEAN Adapter

This module provides an adapter for the QuantConnect LEAN framework,
allowing seamless integration of LEAN's capabilities into the workflow.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import tempfile
import shutil
import logging
import importlib
import inspect
from enum import Enum

# Import LEAN components using pythonnet if available
LEAN_AVAILABLE = False
try:
    import clr
    try:
        clr.AddReference("QuantConnect.Common")
        clr.AddReference("QuantConnect.Algorithm")
        clr.AddReference("QuantConnect.Indicators")
        clr.AddReference("QuantConnect.Algorithm.Framework")
        
        from QuantConnect import *
        from QuantConnect.Algorithm import *
        from QuantConnect.Data import *
        from QuantConnect.Data.Market import *
        from QuantConnect.Parameters import *
        
        LEAN_AVAILABLE = True
    except Exception as e:
        logging.warning(f"QuantConnect LEAN framework import failed: {e}")
        logging.warning("Some features may not be available")
except ImportError:
    logging.warning("QuantConnect LEAN framework import failed: No module named 'clr'")
    logging.warning("Some features may not be available")


logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Enum for backtest execution modes."""
    LEAN = "lean"


class LeanAdapter:
    """
    Adapter class for the QuantConnect LEAN framework.
    """
    
    def __init__(self, 
                config_path: Optional[str] = None,
                mode: BacktestMode = BacktestMode.LEAN,
                data_folder: Optional[str] = None,
                results_folder: Optional[str] = None):
        """
        Initialize the LEAN adapter.
        
        Args:
            config_path: Path to the configuration file
            mode: Backtest execution mode
            data_folder: Path to the data folder
            results_folder: Path to the results folder
        """
        self.config_path = config_path
        self.mode = mode
        self.data_folder = data_folder or os.path.join(os.getcwd(), "data")
        self.results_folder = results_folder or os.path.join(os.getcwd(), "results")
        
        # Ensure folders exist
        Path(self.data_folder).mkdir(parents=True, exist_ok=True)
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Check if LEAN is available
        if not LEAN_AVAILABLE:
            logger.error("LEAN framework not available. Please install the required dependencies.")
        
        # Initialize algorithm registry
        self.algorithms = self._discover_algorithms()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Dictionary containing configuration
        """
        default_config = {
            "api_key": "",
            "api_secret": "",
            "organization_id": "",
            "project_id": "",
            "default_cash": 100000,
            "default_market": "usa",
            "default_resolution": "daily",
            "default_algorithm": "TrendFollowingAlgorithm",
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
            # Import all algorithm modules from market_system.lean.algorithms
            algorithms_module = importlib.import_module("src.market_system.lean.algorithms")
            
            # Find all modules in the algorithms package
            for _, module_name, _ in inspect.iter_modules(algorithms_module.__path__):
                try:
                    # Import the module
                    module = importlib.import_module(f"src.market_system.lean.algorithms.{module_name}")
                    
                    # Find all classes that inherit from QCAlgorithm or BaseLeanAlgorithm
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and (
                            hasattr(obj, 'Initialize') and 
                            hasattr(obj, 'OnData') and 
                            hasattr(obj, 'SetStartDate')
                        ):
                            algorithms[name] = obj
                            logger.info(f"Discovered algorithm: {name}")
                except Exception as e:
                    logger.error(f"Error importing algorithm module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error discovering algorithms: {e}")
        
        return algorithms
    
    def get_available_algorithms(self) -> List[str]:
        """
        Get a list of available algorithm names.
        
        Returns:
            List of algorithm names
        """
        return list(self.algorithms.keys())
    
    def run_backtest(self, 
                   algorithm_name: str,
                   symbols: List[str],
                   start_date: Union[str, datetime],
                   end_date: Union[str, datetime],
                   initial_capital: float = 100000,
                   resolution: str = "Daily",
                   parameters: Optional[Dict] = None) -> Dict:
        """
        Run a backtest using the specified algorithm and parameters.
        
        Args:
            algorithm_name: Name of the algorithm to use
            symbols: List of symbols to backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            resolution: Data resolution (Daily, Hour, Minute)
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
                          parameters: Optional[Dict]) -> Dict:
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
            
            # Prepare symbols
            algorithm.symbols = []
            for symbol in symbols:
                algorithm.add_symbol(symbol)
            
            # Configure data folder if needed
            data_folder = self.data_folder
            results_folder = os.path.join(self.results_folder, backtest_id)
            os.makedirs(results_folder, exist_ok=True)
            
            # Set up configuration for engine
            config = {
                'algorithm-type-name': algorithm_class.__name__,
                'algorithm-language': 'Python',
                'algorithm-location': algorithm_class.__module__,
                'data-folder': data_folder,
                'results-destination-folder': results_folder,
                'cash-amount': initial_capital,
                'security-type': 'Equity',
                'market': 'usa',
                'resolution': resolution,
                'start-date': start_date.strftime('%Y-%m-%d'),
                'end-date': end_date.strftime('%Y-%m-%d'),
                'leaky-bucket-rate': 120,
                'leaky-bucket-capacity': 100,
                'live-mode': False
            }
            
            # Create LEAN backtest settings
            from QuantConnect.Interfaces import IBacktestSettings
            from QuantConnect.Packets import BacktestSettings
            
            settings = BacktestSettings()
            settings.Name = backtest_id
            settings.BacktestId = backtest_id
            
            # Create backtest engine
            from QuantConnect.Lean.Engine import Engine
            from QuantConnect.Lean.Engine.Results import BacktestingResultHandler
            
            engine = Engine()
            
            # Run backtest
            lean_result = engine.Run(algorithm, settings)
            
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
    
    def _process_lean_results(self, lean_result, results_folder: str) -> Dict:
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
    
    def convert_data_to_lean(self, data_path: str, symbol: str, output_path: Optional[str] = None) -> str:
        """
        Convert data from existing format to LEAN format.
        
        Args:
            data_path: Path to the data file
            symbol: Symbol name
            output_path: Optional output path
            
        Returns:
            Path to the converted data file
        """
        try:
            # Read data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path, parse_dates=['timestamp'])
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Create output path if not provided
            if output_path is None:
                output_dir = os.path.join(self.data_folder, 'lean', 'equity', 'usa', 'daily')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{symbol}.csv")
            
            # Convert to LEAN format
            lean_df = pd.DataFrame({
                'time': df['timestamp'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume']
            })
            
            # Write to file
            lean_df.to_csv(output_path, index=False)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting data to LEAN format: {e}")
            raise 