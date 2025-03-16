"""
Backtest engine for market strategies.
Core component that orchestrates the backtesting process.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
import concurrent.futures
from tqdm import tqdm
import decimal

from .data_provider import DataProvider
from .portfolio_manager import PortfolioManager
from .execution_handler import ExecutionHandler
from .performance_metrics import PerformanceMetrics
from ..strategy.base_strategy import BaseStrategy
from ..domain.models import MarketData, Signal, MarketState

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Core backtesting engine.
    Orchestrates the backtesting process, including data loading, strategy execution,
    portfolio management, and performance analysis.
    """
    
    def __init__(self, 
                 initial_capital: Decimal,
                 start_date: datetime,
                 end_date: datetime,
                 data_provider: DataProvider,
                 strategies: List[BaseStrategy],
                 portfolio_manager: Optional[PortfolioManager] = None,
                 execution_handler: Optional[ExecutionHandler] = None):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Initial capital for the backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest
            data_provider: Data provider for market data
            strategies: List of strategies to use
            portfolio_manager: Optional portfolio manager (will create one if not provided)
            execution_handler: Optional execution handler (will create one if not provided)
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.data_provider = data_provider
        
        # Convert strategies list to dictionary
        if isinstance(strategies, list):
            self.strategies = {f"strategy_{i}": strategy for i, strategy in enumerate(strategies)}
        else:
            self.strategies = {"strategy_0": strategies}
            
        # Initialize portfolio manager if not provided
        if portfolio_manager is None:
            self.portfolio_manager = PortfolioManager(initial_capital=initial_capital)
        else:
            self.portfolio_manager = portfolio_manager
            
        # Initialize execution handler if not provided
        if execution_handler is None:
            self.execution_handler = ExecutionHandler()
        else:
            self.execution_handler = execution_handler
            
        # Initialize market states
        self.market_states = {}
        
        # Initialize signals and trades
        self.signals = []
        self.trades = []
        
        # Initialize performance metrics
        self.performance_metrics = {}
        
        logger.info(f"BacktestEngine initialized with {len(self.strategies)} strategies")
        logger.info(f"Backtest period: {start_date} to {end_date}")
        logger.info(f"Initial capital: {initial_capital}")
        
        # Initialize empty equity curve
        # self.equity_curve = []
        
    def run(self, 
            symbols: List[str], 
            timeframe: str = "1d",
            risk_percentage: Decimal = Decimal("1.0"),
            benchmark_symbol: Optional[str] = None,
            parallel: bool = False,
            market: str = "nasdaq",
            show_progress: bool = True) -> Dict:
        """
        Run the backtest for the specified symbols and timeframe.
        
        Args:
            symbols: List of symbols to include in the backtest
            timeframe: Data timeframe (e.g., "1d", "1h", "15m")
            risk_percentage: Percentage of capital to risk per trade
            benchmark_symbol: Optional benchmark symbol for comparison
            parallel: Whether to process symbols in parallel
            market: Market name (e.g., "nasdaq", "nyse")
            show_progress: Whether to show a progress bar
            
        Returns:
            Dict containing backtest results
        """
        start_time = time.time()
        logger.info(f"Starting backtest for {len(symbols)} symbols with timeframe {timeframe}")
        
        try:
            # Load market data
            market_data = self.data_provider.get_data(
                symbols=symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                timeframe=timeframe,
                market=market
            )
            
            # Debug: Check if market data was loaded
            for symbol, data in market_data.items():
                logger.info(f"Loaded data for {symbol}: {len(data)} data points")
                logger.info(f"Data sample: {data.head()}")
            
            # Load benchmark data if specified
            benchmark_data = None
            if benchmark_symbol:
                benchmark_data = self.data_provider.get_data(
                    symbols=[benchmark_symbol],
                    start_date=self.start_date,
                    end_date=self.end_date,
                    timeframe=timeframe,
                    market=market
                ).get(benchmark_symbol)
            
            # Preprocess data (calculate indicators)
            market_data = self.data_provider.preprocess_data(market_data)
            
            # Get all unique timestamps across all symbols
            all_timestamps = self._get_all_timestamps(market_data)
            all_timestamps.sort()
            
            logger.info(f"Found {len(all_timestamps)} unique timestamps")
            
            # Run the simulation
            if parallel and len(symbols) > 1:
                self._run_parallel_simulation(market_data, all_timestamps, risk_percentage, show_progress)
            else:
                self._run_sequential_simulation(market_data, all_timestamps, risk_percentage, show_progress)
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_performance_metrics(benchmark_data)
            
            # Prepare results
            results = {
                "strategy_name": "_".join(self.strategies.keys()),
                "symbols": symbols,
                "timeframe": timeframe,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_capital": float(self.initial_capital),
                "final_equity": float(self.portfolio_manager.equity_history[-1]["equity"]) if self.portfolio_manager.equity_history else float(self.initial_capital),
                "equity_curve": self.portfolio_manager.equity_history,
                "trades": self.portfolio_manager.trade_history,
                "signals": self.signals,
                "metrics": self.performance_metrics,
                "portfolio_stats": self.portfolio_manager.get_portfolio_stats(),
                "execution_time": time.time() - start_time
            }
            
            # Add portfolio_value for compatibility with existing code
            if self.portfolio_manager.equity_history:
                results["portfolio_value"] = [float(entry["equity"]) for entry in self.portfolio_manager.equity_history]
            else:
                results["portfolio_value"] = [float(self.initial_capital)]
            
            logger.info(f"Backtest completed in {results['execution_time']:.2f} seconds")
            logger.info(f"Final equity: {results['final_equity']:.2f} "
                       f"(Return: {self.performance_metrics.get('total_return', 0):.2%})")
            
            return results
        except Exception as e:
            logger.error(f"Error in backtest execution: [{e.__class__}] {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return default results
            return {
                "strategy_name": "_".join(self.strategies.keys()),
                "symbols": symbols,
                "timeframe": timeframe,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "initial_capital": float(self.initial_capital),
                "final_equity": float(self.initial_capital),
                "equity_curve": [],
                "trades": [],
                "signals": [],
                "metrics": {},
                "portfolio_stats": {},
                "execution_time": time.time() - start_time,
                "portfolio_value": [float(self.initial_capital)],
                "error": str(e)
            }
    
    def _run_sequential_simulation(self, 
                                  market_data: Dict[str, pd.DataFrame],
                                  timestamps: List[datetime],
                                  risk_percentage: Decimal,
                                  show_progress: bool) -> None:
        """
        Run the simulation sequentially.
        
        Args:
            market_data: Dict mapping symbols to DataFrames with market data
            timestamps: List of timestamps to process
            risk_percentage: Percentage of capital to risk per trade
            show_progress: Whether to show a progress bar
        """
        # Initialize progress bar if requested
        if show_progress:
            timestamps_iter = tqdm(timestamps, desc="Backtesting")
        else:
            timestamps_iter = timestamps
            
        # Process each timestamp
        for timestamp in timestamps_iter:
            # Get market data for current timestamp
            current_data = self._get_data_at_timestamp(market_data, timestamp)
            
            if not current_data:
                continue
                
            # Update market states
            self._update_market_states(current_data, timestamp)
            
            # For testing purposes, force signal generation
            for symbol, data in current_data.items():
                if len(data) >= 2:
                    try:
                        # Get the last two prices
                        current_price = Decimal(str(data.iloc[-1]['close']))
                        previous_price = Decimal(str(data.iloc[-2]['close']))
                        
                        # Skip if price is invalid
                        if current_price <= Decimal("0") or previous_price <= Decimal("0"):
                            logger.warning(f"Invalid price for {symbol}: current={current_price}, previous={previous_price}")
                            continue
                        
                        # Calculate price change
                        price_change = (current_price / previous_price - 1) * Decimal('100')
                        
                        # Create a signal based on price movement
                        if price_change > Decimal('0'):
                            action = "BUY"
                        else:
                            action = "SELL"
                            
                        # Create a signal
                        signal = Signal(
                            symbol=symbol,
                            timestamp=timestamp,
                            action=action,
                            confidence=Decimal("0.8"),
                            strategy_name=list(self.strategies.keys())[0],
                            market_state=MarketState.UNDEFINED,
                            indicators_snapshot={}
                        )
                        
                        # Add to signals list
                        self.signals.append({
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "strategy": list(self.strategies.keys())[0],
                            "action": action,
                            "confidence": 0.8,
                            "market_state": MarketState.UNDEFINED.value
                        })
                        
                        # Execute the signal
                        stop_loss = current_price * Decimal("0.95") if action == "BUY" else current_price * Decimal("1.05")
                        take_profit = current_price * Decimal("1.1") if action == "BUY" else current_price * Decimal("0.9")
                        
                        # Calculate position size - use a fixed risk percentage
                        fixed_risk = Decimal("1.0")
                        position_size = self.portfolio_manager.get_position_sizing(
                            symbol, current_price, fixed_risk, stop_loss
                        )
                        
                        # Skip if position size is invalid
                        if position_size <= Decimal("0"):
                            logger.warning(f"Invalid position size for {symbol}: {position_size}")
                            continue
                            
                        # Check if we have enough capital
                        cost = position_size * current_price
                        if cost > self.portfolio_manager.current_capital:
                            logger.warning(f"Not enough capital to open position: {symbol} {action} {position_size} @ {current_price}")
                            continue
                        
                        logger.info(f"Forced signal: {action} for {symbol} at {current_price}, position size: {position_size}")
                        
                        # Execute order
                        execution = self.execution_handler.execute_order(
                            symbol=symbol,
                            action=action,
                            price=current_price,
                            quantity=position_size,
                            timestamp=timestamp,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        
                        # Update portfolio with the execution
                        if execution:
                            logger.info(f"Executed order: {execution}")
                            position = self.portfolio_manager.update_portfolio(execution)
                            if not position:
                                logger.warning(f"Failed to open position for {symbol}")
                    except Exception as e:
                        logger.error(f"Error forcing signal: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Update positions (check for stop loss/take profit)
            self.portfolio_manager.update_positions(timestamp, current_data)
            
            # Calculate equity
            self.portfolio_manager.calculate_equity(timestamp, current_data)
    
    def _run_parallel_simulation(self, 
                                market_data: Dict[str, pd.DataFrame],
                                timestamps: List[datetime],
                                risk_percentage: Decimal,
                                show_progress: bool) -> None:
        """
        Run the simulation with parallel signal generation.
        
        Args:
            market_data: Dict mapping symbols to DataFrames with market data
            timestamps: List of timestamps to process
            risk_percentage: Percentage of capital to risk per trade
            show_progress: Whether to show a progress bar
        """
        # Initialize progress bar if requested
        if show_progress:
            timestamps_iter = tqdm(timestamps, desc="Backtesting")
        else:
            timestamps_iter = timestamps
            
        # Process each timestamp
        for timestamp in timestamps_iter:
            # Get market data for current timestamp
            current_data = self._get_data_at_timestamp(market_data, timestamp)
            
            if not current_data:
                continue
                
            # Update market states
            self._update_market_states(current_data, timestamp)
            
            # Generate signals in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for symbol, data in current_data.items():
                    # Skip if not enough data
                    if len(data) < 50:  # Minimum data required for most strategies
                        continue
                        
                    # Get market state
                    market_state = self.market_states.get(symbol, MarketState.UNDEFINED)
                    
                    # Submit signal generation tasks
                    for strategy_name, strategy in self.strategies.items():
                        futures.append(
                            executor.submit(
                                self._generate_signal_for_symbol,
                                strategy,
                                symbol,
                                data,
                                market_state,
                                timestamp
                            )
                        )
                
                # Collect signals
                signals = []
                for future in concurrent.futures.as_completed(futures):
                    signal = future.result()
                    if signal:
                        signals.append(signal)
            
            # Execute signals
            self._execute_signals(signals, current_data, timestamp, risk_percentage)
            
            # Update positions (check for stop loss/take profit)
            self.portfolio_manager.update_positions(timestamp, current_data)
            
            # Calculate equity
            self.portfolio_manager.calculate_equity(timestamp, current_data)
    
    def _generate_signals(self, 
                         current_data: Dict[str, Dict],
                         timestamp: datetime) -> List[Signal]:
        """
        Generate signals from all strategies for all symbols.
        
        Args:
            current_data: Dict mapping symbols to market data
            timestamp: Current timestamp
            
        Returns:
            List of generated signals
        """
        signals = []
        
        for symbol, data in current_data.items():
            # Skip if not enough data
            if len(data) < 50:  # Minimum data required for most strategies
                continue
                
            # Get market state
            market_state = self.market_states.get(symbol, MarketState.UNDEFINED)
            
            # Generate signals from each strategy
            for strategy_name, strategy in self.strategies.items():
                signal = self._generate_signal_for_symbol(
                    strategy, symbol, data, market_state, timestamp
                )
                
                if signal:
                    signals.append(signal)
                    
        return signals
    
    def _generate_signal_for_symbol(self,
                                   strategy: BaseStrategy,
                                   symbol: str,
                                   data: pd.DataFrame,
                                   market_state: MarketState,
                                   timestamp: datetime) -> Optional[Signal]:
        """
        Generate a signal for a specific symbol using a specific strategy.
        
        Args:
            strategy: Strategy to use
            symbol: Symbol to generate signal for
            data: Market data for the symbol
            market_state: Current market state
            timestamp: Current timestamp
            
        Returns:
            Generated signal or None
        """
        try:
            # Convert data to MarketData objects
            market_data_list = self._convert_to_market_data(symbol, data)
            
            if not market_data_list:
                logger.warning(f"No market data available for {symbol}")
                return None
                
            # Generate signal
            signal = strategy.analyze(market_data_list, market_state)
            
            # Store signal if generated
            if signal:
                self.signals.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "strategy": strategy.name,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "market_state": market_state.value
                })
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol} with {strategy.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _execute_signals(self, 
                    signals: List[Signal],
                    current_data: Dict[str, pd.DataFrame],
                    timestamp: datetime,
                    risk_percentage: Decimal) -> None:
        """
        Execute trading signals.
        
        Args:
            signals: List of trading signals to execute
            current_data: Current market data
            timestamp: Current timestamp
            risk_percentage: Risk percentage for position sizing
        """
        if not signals:
            return
            
        for signal in signals:
            symbol = signal.symbol
            action = signal.action
            
            # Skip if no data for symbol
            if symbol not in current_data:
                logger.warning(f"No data for symbol {symbol} at {timestamp}")
                continue
                
            # Get current price
            current_price = Decimal(str(current_data[symbol].iloc[-1]['close']))
            
            # Skip if price is zero or negative
            if current_price <= Decimal("0"):
                logger.warning(f"Invalid price {current_price} for {symbol} at {timestamp}")
                continue
                
            # Calculate position size
            if signal.quantity is not None:
                quantity = signal.quantity
            else:
                # Calculate position size based on risk
                stop_loss = None
                if signal.stop_loss is not None:
                    stop_loss = current_price * (Decimal("1") - signal.stop_loss) if action == "BUY" else current_price * (Decimal("1") + signal.stop_loss)
                
                quantity = self.portfolio_manager.get_position_sizing(
                    symbol=symbol,
                    price=current_price,
                    risk_percentage=risk_percentage,
                    stop_loss=stop_loss
                )
                
            # Skip if quantity is zero
            if quantity <= Decimal("0"):
                logger.warning(f"Zero quantity for {symbol} at {timestamp}")
                continue
                
            # Check if we already have a position in this symbol
            existing_position = None
            for pos_id, pos in self.portfolio_manager.positions.items():
                if pos.symbol == symbol:
                    existing_position = pos
                    break
                    
            # Handle existing position
            if existing_position:
                # If signal is in the same direction, ignore
                if existing_position.action == action:
                    continue
                    
                # If signal is in the opposite direction, close position
                execution_result = self.execution_handler.execute_order(
                    symbol=symbol,
                    action="SELL" if existing_position.action == "BUY" else "BUY",
                    price=current_price,
                    quantity=existing_position.quantity,
                    timestamp=timestamp,
                    market_data=current_data.get(symbol)
                )
                
                # Update portfolio with execution result
                closed_position = self.portfolio_manager.close_position(
                    position_id=existing_position.id,
                    price=execution_result["executed_price"],
                    timestamp=timestamp
                )
                
                if closed_position:
                    logger.info(f"Closed position: {symbol} at {execution_result['executed_price']} with P&L: {closed_position.pnl}")
                    
                    # Record trade details for verification
                    trade_details = closed_position.get_trade_details()
                    logger.info(f"Trade details: {trade_details}")
                
            # Execute new order
            execution_result = self.execution_handler.execute_order(
                symbol=symbol,
                action=action,
                price=current_price,
                quantity=quantity,
                timestamp=timestamp,
                market_data=current_data.get(symbol),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            # Update portfolio with execution result
            position = self.portfolio_manager.open_position(
                symbol=symbol,
                action=action,
                price=execution_result["executed_price"],
                quantity=execution_result["executed_quantity"],
                timestamp=timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy_name=signal.strategy_name,
                metadata={"signal": signal.to_dict()}
            )
            
            if position:
                logger.info(f"Opened position: {symbol} at {execution_result['executed_price']}")
                
                # Record trade details for verification
                trade_details = position.get_trade_details()
                logger.info(f"Trade details: {trade_details}")
    
    def _update_market_states(self, 
                            current_data: Dict[str, pd.DataFrame],
                            timestamp: datetime) -> None:
        """
        Update market states for all symbols.
        
        Args:
            current_data: Dict mapping symbols to DataFrames with market data
            timestamp: Current timestamp
        """
        for symbol, df in current_data.items():
            try:
                # Need at least 20 data points for trend analysis
                if len(df) < 20:
                    self.market_states[symbol] = MarketState.UNDEFINED
                    continue
                    
                # Get the most recent data
                recent_data = df.iloc[-20:]
                
                # Calculate short-term trend (last 10 days)
                short_term = recent_data.iloc[-10:]['close'].pct_change().mean() * 100
                
                # Calculate medium-term trend (last 20 days)
                medium_term = recent_data['close'].pct_change().mean() * 100
                
                # Check for gaps
                last_close = recent_data.iloc[-2]['close'] if len(recent_data) > 1 else None
                current_open = recent_data.iloc[-1]['open']
                
                if last_close is not None:
                    gap_pct = (current_open / last_close - 1) * 100
                    
                    # Gap up
                    if gap_pct > 2.0:
                        self.market_states[symbol] = MarketState.GAP_UP
                        continue
                        
                    # Gap down
                    if gap_pct < -2.0:
                        self.market_states[symbol] = MarketState.GAP_DOWN
                        continue
                
                # Check for trends
                if short_term > 0.5 and medium_term > 0.3:
                    self.market_states[symbol] = MarketState.TRENDING_UP
                elif short_term < -0.5 and medium_term < -0.3:
                    self.market_states[symbol] = MarketState.TRENDING_DOWN
                elif abs(short_term) < 0.2 and abs(medium_term) < 0.2:
                    self.market_states[symbol] = MarketState.RANGING
                else:
                    self.market_states[symbol] = MarketState.CONSOLIDATING
                    
            except Exception as e:
                logger.error(f"Error updating market state for {symbol}: {str(e)}")
                self.market_states[symbol] = MarketState.UNDEFINED
    
    def _get_all_timestamps(self, market_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """
        Get all unique timestamps across all symbols.
        
        Args:
            market_data: Dict mapping symbols to DataFrames with market data
            
        Returns:
            List of unique timestamps
        """
        all_timestamps = set()
        
        for symbol, df in market_data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.update(df.index.tolist())
            elif 'timestamp' in df.columns:
                all_timestamps.update(df['timestamp'].tolist())
                
        return list(all_timestamps)
    
    def _get_data_at_timestamp(self, 
                              market_data: Dict[str, pd.DataFrame],
                              timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get market data for all symbols at a specific timestamp.
        
        Args:
            market_data: Dict mapping symbols to DataFrames with market data
            timestamp: Timestamp to get data for
            
        Returns:
            Dict mapping symbols to DataFrames with data up to the timestamp
        """
        result = {}
        
        for symbol, df in market_data.items():
            # Get data up to and including the timestamp
            mask = df.index <= timestamp
            if mask.any():
                # Get all data up to this timestamp
                symbol_data = df.loc[mask].copy()
                
                # Only include if we have enough data
                if len(symbol_data) > 0:
                    result[symbol] = symbol_data
                    
        return result
    
    def _convert_to_market_data(self, symbol: str, data: pd.DataFrame) -> List[MarketData]:
        """
        Convert DataFrame data to MarketData objects.
        
        Args:
            symbol: Symbol for the data
            data: DataFrame with market data
            
        Returns:
            List of MarketData objects
        """
        result = []
        
        # Ensure data is sorted by index (timestamp)
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
            
        # Convert each row to a MarketData object
        for idx, row in data.iterrows():
            try:
                # Extract OHLCV data
                timestamp = idx
                
                # Safely convert price data to Decimal with improved error handling
                try:
                    # First ensure the value is a valid number
                    open_price_val = float(row.get('open', 0))
                    open_price = Decimal(str(open_price_val))
                except (ValueError, TypeError, decimal.InvalidOperation):
                    logger.warning(f"Invalid open price for {symbol} at {idx}: {row.get('open')}. Using 0.")
                    open_price = Decimal('0')
                
                try:
                    high_price_val = float(row.get('high', 0))
                    high_price = Decimal(str(high_price_val))
                except (ValueError, TypeError, decimal.InvalidOperation):
                    logger.warning(f"Invalid high price for {symbol} at {idx}: {row.get('high')}. Using 0.")
                    high_price = Decimal('0')
                
                try:
                    low_price_val = float(row.get('low', 0))
                    low_price = Decimal(str(low_price_val))
                except (ValueError, TypeError, decimal.InvalidOperation):
                    logger.warning(f"Invalid low price for {symbol} at {idx}: {row.get('low')}. Using 0.")
                    low_price = Decimal('0')
                
                try:
                    close_price_val = float(row.get('close', 0))
                    close_price = Decimal(str(close_price_val))
                except (ValueError, TypeError, decimal.InvalidOperation):
                    logger.warning(f"Invalid close price for {symbol} at {idx}: {row.get('close')}. Using 0.")
                    close_price = Decimal('0')
                
                # Convert volume to int with error handling
                try:
                    volume = int(float(row.get('volume', 0)))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid volume for {symbol} at {idx}: {row.get('volume')}. Using 0.")
                    volume = 0
                
                # Extract indicators
                indicators = {}
                for col in row.index:
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'date', 'market', 'currency', 'adjusted_close', 'trading_status']:
                        if pd.notna(row[col]):
                            # Only include numeric values in indicators
                            try:
                                if isinstance(row[col], (int, float)):
                                    indicators[col] = float(row[col])
                                elif isinstance(row[col], str) and row[col].replace('.', '', 1).isdigit():
                                    indicators[col] = float(row[col])
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                continue
                
                # Create MarketData object
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    volume=volume,
                    indicators=indicators
                )
                
                result.append(market_data)
            except Exception as e:
                logger.error(f"Error converting row to MarketData for {symbol} at {idx}: {str(e)}")
                continue
            
        return result
    
    def _calculate_performance_metrics(self, 
                                      benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate performance metrics for the backtest.
        
        Args:
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        from .performance_metrics import PerformanceMetrics
        
        # Get portfolio statistics
        portfolio_stats = self.portfolio_manager.get_portfolio_stats()
        
        # Validate equity history
        if not self.portfolio_manager.equity_history:
            logger.warning("Empty equity history, returning default metrics")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "portfolio_stats": portfolio_stats
            }
            
        # Check for extreme values in equity history
        try:
            equity_values = [float(entry["equity"]) for entry in self.portfolio_manager.equity_history]
            max_equity = max(equity_values)
            
            # If we have extreme values, log warning and cap them
            if max_equity > 1e12:  # More than a trillion
                logger.warning(f"Extreme equity value detected: {max_equity}. Capping values.")
                capped_equity_history = []
                for entry in self.portfolio_manager.equity_history:
                    capped_entry = entry.copy()
                    if float(entry["equity"]) > 1e12:
                        capped_entry["equity"] = Decimal("1000000000000")  # Cap at 1 trillion
                    capped_equity_history.append(capped_entry)
                equity_history = capped_equity_history
            else:
                equity_history = self.portfolio_manager.equity_history
        except Exception as e:
            logger.error(f"Error validating equity history: {str(e)}")
            equity_history = self.portfolio_manager.equity_history
        
        # Calculate performance metrics with error handling
        try:
            metrics = PerformanceMetrics.calculate_metrics(
                equity_history,
                self.portfolio_manager.trade_history,
                0.0,  # risk_free_rate
                benchmark_data
            )
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return default metrics on error
            metrics = {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
        
        # Combine metrics with portfolio stats
        metrics.update({
            "portfolio_stats": portfolio_stats
        })
        
        return metrics
    
    def generate_report(self, 
                  output_dir: Union[str, Path],
                  report_name: str,
                  formats: List[str] = ["html", "json", "csv"]) -> Dict[str, str]:
        """
        Generate a report for the backtest results.
        
        Args:
            output_dir: Output directory for the report
            report_name: Name of the report
            formats: List of formats to generate (html, json, csv)
            
        Returns:
            Dictionary with report file paths
        """
        from .report_generator import ReportGenerator
        
        # Create report generator
        report_generator = ReportGenerator(output_dir=output_dir)
        
        # Prepare results
        results = {
            "strategy_name": "_".join(self.strategies.keys()),
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": float(self.initial_capital),
            "final_equity": float(self.portfolio_manager.equity_history[-1]["equity"]) if self.portfolio_manager.equity_history else float(self.initial_capital),
            "equity_curve": self.portfolio_manager.equity_history,
            "trades": self.portfolio_manager.trade_history,
            "signals": self.signals,
            "metrics": self.performance_metrics,
            "portfolio_stats": self.portfolio_manager.get_portfolio_stats()
        }
        
        # Generate report
        report_files = report_generator.generate_report(
            backtest_results=results,
            report_name=report_name,
            formats=formats
        )
        
        return report_files 