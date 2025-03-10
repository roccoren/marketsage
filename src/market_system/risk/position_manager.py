from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from ..domain.models import Position, Signal, RiskMetrics, MarketData
import numpy as np

class PositionManager:
    def __init__(self, config: Dict = None):
        self._config = config or {
            'max_position_size': 0.05,  # Maximum size of any single position (% of capital)
            'max_total_exposure': 0.30,  # Maximum total portfolio exposure
            'risk_per_trade': 0.01,     # Maximum risk per trade (% of capital)
            'min_risk_reward': 1.5,     # Minimum risk/reward ratio
            'position_sizing_atr_multiplier': 2.0,  # ATR multiplier for position sizing
            'max_correlated_positions': 3,  # Maximum number of correlated positions
            'rebalance_threshold': 0.1,  # Portfolio rebalance threshold (10% deviation)
        }
        self._positions: Dict[str, Position] = {}
        self._total_capital: Decimal = Decimal('0')
        self._available_capital: Decimal = Decimal('0')

    def set_capital(self, total_capital: Decimal):
        """Set or update the total capital available for trading."""
        self._total_capital = total_capital
        self._update_available_capital()

    def _update_available_capital(self):
        """Update available capital based on current positions."""
        total_exposure = sum(
            abs(pos.quantity * pos.current_price)
            for pos in self._positions.values()
        )
        self._available_capital = self._total_capital - total_exposure

    def can_take_position(self, signal: Signal, market_data: MarketData, 
                         stop_loss: Decimal) -> bool:
        """
        Check if a new position can be taken based on risk parameters.
        
        Args:
            signal: Trading signal
            market_data: Current market data
            stop_loss: Calculated stop loss level
            
        Returns:
            Boolean indicating if position can be taken
        """
        # Check if symbol already in portfolio
        if signal.symbol in self._positions:
            return False

        # Calculate potential position size
        position_size = self._calculate_position_size(
            market_data.close_price,
            stop_loss,
            signal
        )

        if position_size == Decimal('0'):
            return False

        # Check against position limits
        position_value = position_size * market_data.close_price
        
        # Individual position size limit
        if position_value > self._total_capital * Decimal(str(self._config['max_position_size'])):
            return False

        # Total exposure limit
        current_exposure = sum(
            abs(pos.quantity * pos.current_price)
            for pos in self._positions.values()
        )
        if (current_exposure + position_value) > (
            self._total_capital * Decimal(str(self._config['max_total_exposure']))
        ):
            return False

        # Check correlated positions
        if not self._check_correlation_limits(signal.symbol):
            return False

        return True

    def _calculate_position_size(self, 
                               current_price: Decimal,
                               stop_loss: Decimal,
                               signal: Signal) -> Decimal:
        """Calculate position size based on risk parameters."""
        if current_price == Decimal('0') or stop_loss == Decimal('0'):
            return Decimal('0')

        # Calculate risk per trade in absolute terms
        risk_amount = self._total_capital * Decimal(str(self._config['risk_per_trade']))
        
        # Calculate position size based on risk
        price_risk = abs(current_price - stop_loss)
        if price_risk == Decimal('0'):
            return Decimal('0')
            
        position_size = risk_amount / price_risk
        
        # Round to reasonable number of decimal places based on price
        if current_price >= Decimal('100'):
            position_size = position_size.quantize(Decimal('0.01'))
        else:
            position_size = position_size.quantize(Decimal('0.0001'))
            
        return position_size

    def _check_correlation_limits(self, new_symbol: str) -> bool:
        """
        Check if adding a new position would violate correlation limits.
        In a real implementation, this would use actual correlation data.
        """
        # Simplified correlation check based on symbol prefix
        # In a real system, this would use actual correlation calculations
        symbol_prefix = new_symbol[:3]  # Example: "EUR" from "EURUSD"
        correlated_count = sum(
            1 for symbol in self._positions.keys()
            if symbol.startswith(symbol_prefix)
        )
        
        return correlated_count < self._config['max_correlated_positions']

    def open_position(self, signal: Signal, market_data: MarketData,
                     stop_loss: Decimal, take_profit: Decimal) -> Optional[Position]:
        """
        Open a new position based on the signal and risk parameters.
        
        Args:
            signal: Trading signal
            market_data: Current market data
            stop_loss: Calculated stop loss level
            take_profit: Calculated take profit level
            
        Returns:
            New Position object if successful, None otherwise
        """
        if not self.can_take_position(signal, market_data, stop_loss):
            return None

        position_size = self._calculate_position_size(
            market_data.close_price,
            stop_loss,
            signal
        )
        
        # Adjust position size sign based on direction
        if signal.action == "SELL":
            position_size = -position_size

        position = Position(
            symbol=signal.symbol,
            quantity=position_size,
            entry_price=market_data.close_price,
            entry_time=market_data.timestamp,
            current_price=market_data.close_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self._positions[signal.symbol] = position
        self._update_available_capital()
        
        return position

    def update_positions(self, market_data_dict: Dict[str, MarketData]) -> List[Position]:
        """
        Update all positions with latest market data.
        Returns positions that hit their stop loss or take profit levels.
        """
        closed_positions = []
        
        for symbol, position in list(self._positions.items()):
            if symbol not in market_data_dict:
                continue
                
            market_data = market_data_dict[symbol]
            position.current_price = market_data.close_price
            
            # Check stop loss
            if (position.quantity > 0 and market_data.low_price <= position.stop_loss) or \
               (position.quantity < 0 and market_data.high_price >= position.stop_loss):
                closed_positions.append(position)
                del self._positions[symbol]
                
            # Check take profit
            elif (position.quantity > 0 and market_data.high_price >= position.take_profit) or \
                 (position.quantity < 0 and market_data.low_price <= position.take_profit):
                closed_positions.append(position)
                del self._positions[symbol]

        self._update_available_capital()
        return closed_positions

    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics for the portfolio."""
        total_exposure = Decimal('0')
        position_values = []
        
        for position in self._positions.values():
            position_value = abs(position.quantity * position.current_price)
            total_exposure += position_value
            position_values.append(float(position_value))

        # Calculate Value at Risk (VaR) using historical data
        # In a real implementation, this would use actual historical returns
        if position_values:
            var_95 = np.percentile(position_values, 5)
        else:
            var_95 = 0

        # Calculate simple Sharpe ratio
        # In a real implementation, this would use actual returns and risk-free rate
        sharpe_ratio = 0.0  # Placeholder
        
        return RiskMetrics(
            position_size=total_exposure,
            current_exposure=total_exposure / self._total_capital,
            max_drawdown=Decimal('0'),  # Would be calculated from historical data
            var_95=Decimal(str(var_95)),
            sharpe_ratio=sharpe_ratio,
            current_volatility=0.0  # Would be calculated from historical data
        )

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self._positions.copy()