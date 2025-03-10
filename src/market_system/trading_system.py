from typing import Dict, List, Optional
from decimal import Decimal
import logging
from datetime import datetime

from market_system.domain.models import MarketData, Signal, Position, MarketState
from market_system.data_pipeline.market_data_service import MarketDataService
from market_system.analysis.market_classifier import MarketClassifier
from market_system.strategy.base_strategy import BaseStrategy
from market_system.risk.position_manager import PositionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, total_capital: Decimal):
        """
        Initialize the trading system with its core components.
        
        Args:
            total_capital: Initial capital for trading
        """
        self._market_data_service = MarketDataService()
        self._market_classifier = MarketClassifier()
        self._position_manager = PositionManager()
        self._strategies: Dict[str, BaseStrategy] = {}
        
        # Set initial capital
        self._position_manager.set_capital(total_capital)
        
        # Track market states and data
        self._market_states: Dict[str, MarketState] = {}
        self._latest_data: Dict[str, List[MarketData]] = {}
        
        logger.info(f"Trading system initialized with capital: {total_capital}")

    def add_strategy(self, strategy: BaseStrategy):
        """Register a new trading strategy."""
        self._strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")

    def process_market_data(self, symbol: str, data: List[MarketData]):
        """
        Process new market data for a symbol.
        
        Args:
            symbol: Trading symbol
            data: List of MarketData objects
        """
        try:
            # Process and store market data
            processed_data = self._market_data_service.process_market_data(symbol, data)
            self._latest_data[symbol] = processed_data
            
            # Update positions with new market data
            market_data_dict = {symbol: processed_data[-1]} if processed_data else {}
            closed_positions = self._position_manager.update_positions(market_data_dict)
            
            for position in closed_positions:
                self._log_position_closed(position)
            
            # Classify market state
            market_state, confidence = self._market_classifier.classify_market_state(processed_data)
            self._market_states[symbol] = market_state
            
            logger.info(f"Market state for {symbol}: {market_state.name} "
                       f"(confidence: {confidence:.2f})")
            
            # Generate and process trading signals
            self._process_trading_signals(symbol)
            
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {str(e)}")
            raise

    def _process_trading_signals(self, symbol: str):
        """
        Generate and process trading signals for a symbol.
        
        Args:
            symbol: Trading symbol to analyze
        """
        if symbol not in self._latest_data or not self._latest_data[symbol]:
            return

        market_data = self._latest_data[symbol]
        market_state = self._market_states.get(symbol, MarketState.UNDEFINED)
        
        for strategy in self._strategies.values():
            try:
                signal = strategy.analyze(market_data, market_state)
                
                if signal:
                    self._handle_trading_signal(signal, market_data[-1])
                    
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {str(e)}")

    def _handle_trading_signal(self, signal: Signal, current_market_data: MarketData):
        """
        Process a trading signal and potentially open a new position.
        
        Args:
            signal: Trading signal to process
            current_market_data: Current market data for the symbol
        """
        try:
            strategy = self._strategies[signal.strategy_name]
            
            # Calculate stop loss and take profit levels
            stop_loss = Decimal(str(strategy.get_stop_loss(
                self._latest_data[signal.symbol],
                signal
            )))
            
            take_profit = Decimal(str(strategy.get_take_profit(
                self._latest_data[signal.symbol],
                signal
            )))
            
            # Attempt to open position
            position = self._position_manager.open_position(
                signal,
                current_market_data,
                stop_loss,
                take_profit
            )
            
            if position:
                self._log_position_opened(position, signal)
            else:
                logger.info(f"Signal rejected for {signal.symbol}: "
                          f"Risk parameters not met")
                
        except Exception as e:
            logger.error(f"Error handling signal for {signal.symbol}: {str(e)}")

    def _log_position_opened(self, position: Position, signal: Signal):
        """Log details of a newly opened position."""
        logger.info(
            f"Opened {signal.action} position for {position.symbol}:\n"
            f"  Quantity: {position.quantity}\n"
            f"  Entry Price: {position.entry_price}\n"
            f"  Stop Loss: {position.stop_loss}\n"
            f"  Take Profit: {position.take_profit}\n"
            f"  Signal Confidence: {signal.confidence:.2f}"
        )

    def _log_position_closed(self, position: Position):
        """Log details of a closed position."""
        pnl = (position.current_price - position.entry_price) * position.quantity
        logger.info(
            f"Closed position for {position.symbol}:\n"
            f"  Entry Price: {position.entry_price}\n"
            f"  Exit Price: {position.current_price}\n"
            f"  PnL: {pnl}"
        )

    def get_current_positions(self) -> Dict[str, Position]:
        """Get all current open positions."""
        return self._position_manager.get_positions()

    def get_risk_metrics(self):
        """Get current risk metrics for the portfolio."""
        return self._position_manager.get_risk_metrics()

    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Get the current market state for a symbol."""
        return self._market_states.get(symbol)