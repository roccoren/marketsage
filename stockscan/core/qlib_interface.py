import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union, Any

# Import Qlib modules
import qlib
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest, executor
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data import D

logger = logging.getLogger(__name__)

class QlibInterface:
    """
    Interface for using Qlib for data processing and strategy execution
    """
    def __init__(self, data_provider: str = "yahoo", region: str = "us"):
        """
        Initialize the Qlib interface.
        
        Args:
            data_provider: Data provider ('yahoo', 'iex', etc.)
            region: Region code ('us', 'cn', etc.)
        """
        self.provider = data_provider
        self.region = region
        self.initialized = False
        
    def init_qlib(self, data_path: Optional[str] = None):
        """
        Initialize Qlib with the specified configuration.
        
        Args:
            data_path: Path to the data directory
        """
        if self.initialized:
            return
            
        try:
            # Set up default data path if not provided
            if data_path is None:
                data_path = os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "cn_data")
                os.makedirs(data_path, exist_ok=True)
            
            # Initialize Qlib
            if self.region.lower() == "cn":
                # For Chinese market data
                qlib.init(provider_uri=data_path, region=REG_CN)
            else:
                # For US market data
                qlib.init(provider_uri=data_path)
                
            self.initialized = True
            logger.info(f"Qlib initialized with data path: {data_path}")
            
        except Exception as e:
            logger.error(f"Error initializing Qlib: {str(e)}")
            raise
            
    def download_data(self, symbols: List[str], start_date: str, end_date: Optional[str] = None,
                     target_dir: Optional[str] = None):
        """
        Download stock data for specified symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            target_dir: Target directory to save data
        """
        try:
            from qlib.data.data import Cal
            from qlib.utils import get_trading_date_by_shift
            
            # Ensure Qlib is initialized
            if not self.initialized:
                self.init_qlib(target_dir)
                
            # Set default end date if not provided
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Create universe file
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
                universe_path = os.path.join(target_dir, "universe.txt")
                with open(universe_path, "w") as f:
                    for symbol in symbols:
                        f.write(f"{symbol}\n")
                logger.info(f"Created universe file with {len(symbols)} symbols at {universe_path}")
            
            # Download data for each symbol
            results = {}
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol}")
                    # Use Qlib data API to get stock data
                    df = D.features([symbol], ['$close', '$open', '$high', '$low', '$volume'], 
                                  start_time=start_date, end_time=end_date)
                    
                    # Rename columns to standard format
                    df = df.reset_index()
                    df = df.rename(columns={
                        'datetime': 'date',
                        '$open': 'open',
                        '$close': 'close',
                        '$high': 'high',
                        '$low': 'low',
                        '$volume': 'volume'
                    })
                    
                    results[symbol] = df
                    logger.info(f"Downloaded {len(df)} records for {symbol}")
                    
                    # Save data if target directory is provided
                    if target_dir:
                        symbol_dir = os.path.join(target_dir, self.region.upper())
                        os.makedirs(symbol_dir, exist_ok=True)
                        df.to_json(os.path.join(symbol_dir, f"{symbol}.json"), orient="records")
                        
                except Exception as e:
                    logger.error(f"Error downloading data for {symbol}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in download_data: {str(e)}")
            raise
            
    def run_strategy(self, strategy_name: str, symbols: List[str], 
                   start_date: str, end_date: Optional[str] = None, 
                   strategy_params: Optional[Dict[str, Any]] = None):
        """
        Run a trading strategy on the specified symbols.
        
        Args:
            strategy_name: Name of the strategy
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy_params: Strategy parameters
            
        Returns:
            Strategy backtest results
        """
        try:
            # Ensure Qlib is initialized
            if not self.initialized:
                self.init_qlib()
                
            # Set default end date if not provided
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            # Set default strategy parameters
            if strategy_params is None:
                strategy_params = {}
                
            logger.info(f"Running {strategy_name} strategy on {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Create universe
            universe = symbols
            
            # Initialize model based on strategy
            if strategy_name.lower() == "gbdt":
                model = LGBModel(**strategy_params)
            else:
                # Default to LGBModel if strategy not recognized
                logger.warning(f"Strategy {strategy_name} not recognized, using default GBDT model")
                model = LGBModel()
                
            # Prepare dataset
            # In real implementation, you would create proper Qlib datasets
            # This is a simplified example
            
            # Run model
            model.fit(data_handler=D.features(
                instruments=universe,
                fields=['$close', '$open', '$high', '$low', '$volume', '$vwap', '$factor'],
                start_time=start_date,
                end_time=end_date,
            ))
            
            # Generate trading signals
            signals = model.predict()
            
            # Create strategy
            strategy = TopkDropoutStrategy(
                topk=strategy_params.get('topk', 5),
                n_drop=strategy_params.get('n_drop', 2),
            )
            
            # Run backtest
            backtest_config = {
                "start_time": start_date,
                "end_time": end_date,
                "account": strategy_params.get('initial_capital', 100000),
                "benchmark": strategy_params.get('benchmark', symbols[0] if symbols else "SPY"),
                "exchange_kwargs": {
                    "freq": strategy_params.get('freq', "day"),
                    "limit_threshold": strategy_params.get('limit_threshold', 0.095),
                    "deal_price": strategy_params.get('deal_price', "close"),
                    "open_cost": strategy_params.get('open_cost', 0.0005),
                    "close_cost": strategy_params.get('close_cost', 0.0015),
                    "min_cost": strategy_params.get('min_cost', 5),
                },
            }
            
            # Execute backtest
            executor_config = {
                "time_per_step": strategy_params.get('time_per_step', "day"),
                "generate_portfolio_metrics": True,
            }
            
            # Create executor
            _executor = executor.SimulatorExecutor(**executor_config)
            
            # Run backtest
            result = backtest(
                executor=_executor,
                strategy=strategy,
                **backtest_config
            )
            
            # Process and return results
            return {
                "strategy": strategy_name,
                "symbols": symbols,
                "period": f"{start_date} to {end_date}",
                "metrics": result.metrics,
                "positions": result.positions,
                "portfolio": result.portfolio,
            }
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            raise
            
    def scan_market(self, strategy_name: str, symbols: List[str], 
                  date: Optional[str] = None, top_k: int = 10,
                  strategy_params: Optional[Dict[str, Any]] = None):
        """
        Scan market for trading signals based on the specified strategy.
        
        Args:
            strategy_name: Name of the strategy
            symbols: List of stock symbols to scan
            date: Date to scan (YYYY-MM-DD)
            top_k: Number of top signals to return
            strategy_params: Strategy parameters
            
        Returns:
            Dictionary with buy/sell signals
        """
        try:
            # Ensure Qlib is initialized
            if not self.initialized:
                self.init_qlib()
                
            # Set default date if not provided
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
                
            # Set default strategy parameters
            if strategy_params is None:
                strategy_params = {}
                
            logger.info(f"Scanning {len(symbols)} symbols using {strategy_name} strategy for {date}")
            
            # Create dataset for scanning
            # In real implementation, you would create proper Qlib datasets
            
            # Initialize model based on strategy
            if strategy_name.lower() == "gbdt":
                model = LGBModel(**strategy_params)
            else:
                # Default to LGBModel if strategy not recognized
                logger.warning(f"Strategy {strategy_name} not recognized, using default GBDT model")
                model = LGBModel()
                
            # Use a lookback period for training
            from datetime import datetime, timedelta
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start_date = (date_obj - timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year lookback
            
            # Fit model
            model.fit(data_handler=D.features(
                instruments=symbols,
                fields=['$close', '$open', '$high', '$low', '$volume', '$vwap', '$factor'],
                start_time=start_date,
                end_time=date,
            ))
            
            # Generate trading signals
            signals = model.predict()
            
            # Process signals
            signal_df = pd.DataFrame({
                'symbol': signals.index,
                'score': signals.values
            })
            
            # Sort signals
            buy_signals = signal_df.sort_values('score', ascending=False).head(top_k)
            sell_signals = signal_df.sort_values('score', ascending=True).head(top_k)
            
            return {
                "strategy": strategy_name,
                "date": date,
                "buy_signals": buy_signals.to_dict('records'),
                "sell_signals": sell_signals.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error scanning market: {str(e)}")
            raise 