from typing import List, Optional
from decimal import Decimal
from ..domain.models import MarketData, Signal, MarketState
from .base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Trend Following")
        self._required_indicators = [
            'sma_20', 'sma_50', 'rsi_14',
            'macd_12_26_9', 'macds_12_26_9',
            'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0'
        ]
        
        self._config = {
            'min_trend_strength': 0.6,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_signal_threshold': 0.0,
            'atr_multiplier': 2.0,
            'profit_factor': 1.5  # Risk:Reward ratio
        }

    def analyze(self, data: List[MarketData], market_state: MarketState) -> Optional[Signal]:
        """
        Analyze market data for trend-following opportunities.
        
        Args:
            data: List of MarketData ordered by time
            market_state: Current market state classification
            
        Returns:
            Signal object if a valid trend trading opportunity is found
        """
        if len(data) < 50:  # Need enough data for analysis
            return None
            
        current = data[-1]
        if not self.validate_indicators(current):
            return None

        # Only trade in strong trending markets
        if market_state not in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN]:
            return None

        # Gather indicator values
        sma_fast = current.indicators['sma_20']
        sma_slow = current.indicators['sma_50']
        rsi = current.indicators['rsi_14']
        macd = current.indicators['macd_12_26_9']
        macd_signal = current.indicators['macds_12_26_9']
        
        # Calculate confirmations for trend trading
        if market_state == MarketState.TRENDING_UP:
            confirmations = [
                sma_fast > sma_slow,  # Trend direction
                float(current.close_price) > sma_fast,  # Price above fast MA
                rsi < self._config['rsi_overbought'],  # Not overbought
                macd > macd_signal,  # MACD confirmation
                macd > self._config['macd_signal_threshold']  # Positive momentum
            ]
            
            weights = [0.3, 0.2, 0.15, 0.2, 0.15]
            confidence = self._calculate_confidence(confirmations, weights)
            
            if confidence >= self._config['min_trend_strength']:
                return Signal(
                    symbol=current.symbol,
                    timestamp=current.timestamp,
                    action="BUY",
                    confidence=confidence,
                    strategy_name=self.name,
                    market_state=market_state,
                    indicators_snapshot=current.indicators
                )
                
        elif market_state == MarketState.TRENDING_DOWN:
            confirmations = [
                sma_fast < sma_slow,  # Trend direction
                float(current.close_price) < sma_fast,  # Price below fast MA
                rsi > self._config['rsi_oversold'],  # Not oversold
                macd < macd_signal,  # MACD confirmation
                macd < -self._config['macd_signal_threshold']  # Negative momentum
            ]
            
            weights = [0.3, 0.2, 0.15, 0.2, 0.15]
            confidence = self._calculate_confidence(confirmations, weights)
            
            if confidence >= self._config['min_trend_strength']:
                return Signal(
                    symbol=current.symbol,
                    timestamp=current.timestamp,
                    action="SELL",
                    confidence=confidence,
                    strategy_name=self.name,
                    market_state=market_state,
                    indicators_snapshot=current.indicators
                )

        return None

    def get_stop_loss(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate stop loss level using ATR for dynamic position sizing.
        
        For long positions: Entry - (ATR * multiplier)
        For short positions: Entry + (ATR * multiplier)
        """
        if len(data) < 2:
            return 0.0
            
        current_data = data[-1]
        atr = current_data.indicators.get(f'atr_14', 0)
        entry_price = float(current_data.close_price)
        
        if signal.action == "BUY":
            return entry_price - (atr * self._config['atr_multiplier'])
        else:
            return entry_price + (atr * self._config['atr_multiplier'])

    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        """
        Calculate take profit level based on stop loss distance.
        Uses configured profit factor for risk:reward ratio.
        """
        current_data = data[-1]
        entry_price = float(current_data.close_price)
        stop_loss = self.get_stop_loss(data, signal)
        
        # Calculate take profit based on risk:reward ratio
        stop_distance = abs(entry_price - stop_loss)
        take_profit_distance = stop_distance * self._config['profit_factor']
        
        if signal.action == "BUY":
            return entry_price + take_profit_distance
        else:
            return entry_price - take_profit_distance