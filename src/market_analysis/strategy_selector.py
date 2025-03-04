"""
Strategy Selector Module
Determines appropriate trading strategies based on market conditions
"""
from enum import Enum
from typing import Dict, Any
import numpy as np
from .market_classifier import MarketCondition, MarketClassifier

class TradingStrategy(Enum):
    """Available trading strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    RANGE_TRADING = "range_trading"
    VOLATILITY_EXPANSION = "volatility_expansion"
    DEFENSIVE = "defensive"

class StrategySelector:
    """Selects appropriate trading strategies based on market conditions"""
    
    def __init__(self):
        """Initialize strategy selector with market classifier"""
        self.market_classifier = MarketClassifier()
        self.strategy_map = {
            MarketCondition.TRENDING_UP: TradingStrategy.TREND_FOLLOWING,
            MarketCondition.TRENDING_DOWN: TradingStrategy.TREND_FOLLOWING,
            MarketCondition.RANGE_BOUND: TradingStrategy.RANGE_TRADING,
            MarketCondition.BREAKOUT: TradingStrategy.BREAKOUT,
            MarketCondition.HIGH_VOLATILITY: TradingStrategy.VOLATILITY_EXPANSION,
            MarketCondition.MEAN_REVERSION: TradingStrategy.MEAN_REVERSION
        }
        
    def calculate_volume_profile(self, volume: np.ndarray) -> Dict[str, float]:
        """
        Analyze trading volume characteristics
        
        Args:
            volume: Array of trading volume data
            
        Returns:
            dict: Volume metrics including relative volume and trends
        """
        avg_volume = np.mean(volume)
        recent_volume = np.mean(volume[-5:])
        
        return {
            'relative_volume': recent_volume / avg_volume,
            'volume_trend': np.polyfit(np.arange(len(volume)), volume, 1)[0]
        }
        
    def analyze_volatility(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate volatility metrics
        
        Args:
            prices: Array of price data
            
        Returns:
            dict: Volatility metrics
        """
        returns = np.diff(prices) / prices[:-1]
        return {
            'historical_volatility': np.std(returns) * np.sqrt(252),
            'recent_volatility': np.std(returns[-20:]) * np.sqrt(252)
        }
        
    def select_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate trading strategy based on market conditions
        
        Args:
            market_data: Dictionary containing OHLCV data
            
        Returns:
            dict: Selected strategy and supporting metrics
        """
        # Get market condition
        market_condition = self.market_classifier.classify_market(market_data)
        
        # Calculate supporting metrics
        volume_metrics = self.calculate_volume_profile(np.array(market_data['volume']))
        volatility_metrics = self.analyze_volatility(np.array(market_data['close']))
        market_metrics = self.market_classifier.get_market_metrics(market_data)
        
        # Select base strategy from condition mapping
        selected_strategy = self.strategy_map[market_condition]
        
        # Adjust strategy based on additional factors
        if selected_strategy == TradingStrategy.TREND_FOLLOWING:
            if volume_metrics['relative_volume'] < 0.8:
                selected_strategy = TradingStrategy.DEFENSIVE
                
        elif selected_strategy == TradingStrategy.BREAKOUT:
            if volatility_metrics['recent_volatility'] < volatility_metrics['historical_volatility']:
                selected_strategy = TradingStrategy.RANGE_TRADING
        
        return {
            'strategy': selected_strategy,
            'market_condition': market_condition,
            'confidence': self._calculate_confidence(market_metrics, volume_metrics, volatility_metrics),
            'metrics': {
                'market': market_metrics,
                'volume': volume_metrics,
                'volatility': volatility_metrics
            }
        }
    
    def _calculate_confidence(self, 
                            market_metrics: Dict[str, Any],
                            volume_metrics: Dict[str, float],
                            volatility_metrics: Dict[str, float]) -> float:
        """
        Calculate confidence score for strategy selection
        
        Args:
            market_metrics: Market condition metrics
            volume_metrics: Volume analysis metrics
            volatility_metrics: Volatility metrics
            
        Returns:
            float: Confidence score (0-1)
        """
        # Simple confidence calculation based on metric alignment
        confidence_factors = []
        
        # Strong volume supports strategy
        if volume_metrics['relative_volume'] > 1.2:
            confidence_factors.append(0.8)
        elif volume_metrics['relative_volume'] > 0.8:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.3)
            
        # Trend strength alignment
        if market_metrics['trend_strength'] > 70 or market_metrics['trend_strength'] < 30:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.4)
            
        # Volatility alignment
        vol_ratio = (volatility_metrics['recent_volatility'] / 
                    volatility_metrics['historical_volatility'])
        if 0.8 <= vol_ratio <= 1.2:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
            
        return np.mean(confidence_factors)