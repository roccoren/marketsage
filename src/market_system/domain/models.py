from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from decimal import Decimal

class MarketState(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    GAP_UP = "GAP_UP"
    GAP_DOWN = "GAP_DOWN"
    CONSOLIDATING = "CONSOLIDATING"  # For mixed/conflicting signals
    UNDEFINED = "UNDEFINED"

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    indicators: Dict[str, float]

@dataclass
class Position:
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    entry_time: datetime
    current_price: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    action: str  # BUY, SELL
    confidence: float
    strategy_name: str
    market_state: MarketState
    indicators_snapshot: Dict[str, float]
    priority: int = 100  # Lower is higher priority, default is lowest
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    
    def to_dict(self):
        """Convert signal to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "action": self.action,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "market_state": self.market_state.value,
            "indicators_snapshot": self.indicators_snapshot,
            "priority": self.priority,
            "stop_loss": float(self.stop_loss) if self.stop_loss is not None else None,
            "take_profit": float(self.take_profit) if self.take_profit is not None else None,
            "quantity": float(self.quantity) if self.quantity is not None else None
        }

@dataclass
class RiskMetrics:
    position_size: Decimal
    current_exposure: Decimal
    max_drawdown: Decimal
    var_95: Decimal  # Value at Risk
    sharpe_ratio: float
    current_volatility: float

@dataclass
class StrategyPerformance:
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    profit_factor: float
    average_win: Decimal
    average_loss: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    start_date: datetime
    end_date: datetime