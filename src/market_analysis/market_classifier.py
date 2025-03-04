"""
Market Classifier Module
Detects and classifies different market conditions using technical indicators
"""
import numpy as np
from enum import Enum

class MarketCondition(Enum):
    """Enum representing different market conditions"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    BREAKOUT = "breakout"
    HIGH_VOLATILITY = "high_volatility"
    MEAN_REVERSION = "mean_reversion"

class MarketClassifier:
    """Analyzes market data to classify current market conditions"""
    
    def __init__(self, lookback_period: int = 20):
        """
        Initialize market classifier
        
        Args:
            lookback_period (int): Period for calculating technical indicators
        """
        self.lookback_period = lookback_period
        
    def calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-self.lookback_period:])
        avg_loss = np.mean(losses[-self.lookback_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate Average Directional Index"""
        # Placeholder for ADX calculation
        # TODO: Implement full ADX logic
        tr = np.max([high - low, abs(high - close), abs(low - close)])
        return np.mean(tr)
    
    def detect_price_gaps(self, opens: np.ndarray, closes: np.ndarray) -> bool:
        """Detect significant price gaps between closes and next day opens"""
        avg_range = np.mean(np.abs(closes - opens))
        gaps = np.abs(opens[1:] - closes[:-1])
        return np.any(gaps > 2 * avg_range)
    
    def is_trending(self, prices: np.ndarray, threshold: float = 25) -> bool:
        """
        Determine if market is trending using ADX
        
        Args:
            prices: Array of price data
            threshold: ADX threshold for trend determination
        
        Returns:
            bool: True if market is trending
        """
        # Simple trend detection using price momentum
        returns = np.diff(prices) / prices[:-1]
        momentum = np.sum(returns[-self.lookback_period:])
        return abs(momentum) > threshold / 100
    
    def is_range_bound(self, prices: np.ndarray, threshold: float = 0.02) -> bool:
        """
        Detect range-bound market conditions
        
        Args:
            prices: Array of price data
            threshold: Maximum price deviation threshold
        
        Returns:
            bool: True if market is range-bound
        """
        recent_prices = prices[-self.lookback_period:]
        price_range = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
        return price_range < threshold
    
    def classify_market(self, data: dict) -> MarketCondition:
        """
        Analyze market data and return current market condition
        
        Args:
            data: Dictionary containing OHLCV data
                Required keys: 'open', 'high', 'low', 'close', 'volume'
                
        Returns:
            MarketCondition: Classified market condition
        """
        closes = np.array(data['close'])
        opens = np.array(data['open'])
        highs = np.array(data['high'])
        lows = np.array(data['low'])
        
        # Check for price gaps (potential mean reversion opportunities)
        if self.detect_price_gaps(opens, closes):
            return MarketCondition.MEAN_REVERSION
            
        # Calculate technical indicators
        rsi = self.calculate_rsi(closes)
        
        # Trend detection
        if self.is_trending(closes):
            return (MarketCondition.TRENDING_UP 
                   if closes[-1] > closes[-self.lookback_period] 
                   else MarketCondition.TRENDING_DOWN)
            
        # Range-bound detection
        if self.is_range_bound(closes):
            return MarketCondition.RANGE_BOUND
            
        # Default to high volatility if no other conditions met
        return MarketCondition.HIGH_VOLATILITY

    def get_market_metrics(self, data: dict) -> dict:
        """
        Calculate and return key market metrics
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            dict: Market metrics including volatility, trend strength, etc.
        """
        closes = np.array(data['close'])
        
        return {
            'volatility': np.std(closes[-self.lookback_period:]),
            'trend_strength': self.calculate_rsi(closes),
            'condition': self.classify_market(data)
        }