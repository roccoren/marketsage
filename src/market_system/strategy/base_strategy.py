from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.models import MarketData, Signal, MarketState

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self._required_indicators: List[str] = []
        
    @abstractmethod
    def analyze(self, data: List[MarketData], market_state: MarketState) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data: List of MarketData objects
            market_state: Current market state classification
            
        Returns:
            Signal object if a trading opportunity is identified, None otherwise
        """
        pass

    def validate_indicators(self, data: MarketData) -> bool:
        """Verify all required indicators are present in the data."""
        return all(
            indicator in data.indicators
            for indicator in self._required_indicators
        )

    @abstractmethod
    def get_stop_loss(self, data: List[MarketData], signal: Signal) -> float:
        """Calculate stop loss level for a given signal."""
        pass

    @abstractmethod
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """Calculate take profit level for a given signal."""
        pass

    def _calculate_confidence(self, confirmations: List[bool], weights: List[float]) -> float:
        """Calculate signal confidence based on weighted confirmation factors."""
        if len(confirmations) != len(weights) or not confirmations:
            return 0.0
            
        return sum(
            conf * weight for conf, weight in zip(confirmations, weights)
        ) / sum(weights)