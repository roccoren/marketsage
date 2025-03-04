"""
Market Analyzer Module
Main interface for market analysis and strategy selection system
"""
from typing import Dict, Any, Optional
import numpy as np
from .market_classifier import MarketClassifier, MarketCondition
from .strategy_selector import StrategySelector, TradingStrategy

class MarketAnalyzer:
    """
    Coordinates market analysis and strategy selection
    
    This class serves as the main interface for the market analysis system,
    combining market classification and strategy selection into a cohesive workflow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market analyzer with optional configuration
        
        Args:
            config: Optional configuration dictionary
                   lookback_period: Period for technical indicators (default: 20)
                   confidence_threshold: Minimum confidence for strategy changes (default: 0.6)
        """
        self.config = config or {}
        self.lookback_period = self.config.get('lookback_period', 20)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Initialize components
        self.classifier = MarketClassifier(lookback_period=self.lookback_period)
        self.strategy_selector = StrategySelector()
        
        # Track market state
        self.current_market_state = None
        self.current_strategy = None
        
    def validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """
        Validate required market data fields are present
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            bool: True if data is valid
        """
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        return all(field in market_data for field in required_fields)
    
    def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis
        
        Args:
            market_data: Dictionary containing OHLCV data
            
        Returns:
            dict: Analysis results including market condition, selected strategy,
                 and supporting metrics
        
        Raises:
            ValueError: If market data is invalid
        """
        if not self.validate_market_data(market_data):
            raise ValueError("Invalid market data: Missing required fields")
            
        # Get market classification
        market_condition = self.classifier.classify_market(market_data)
        
        # Get strategy recommendation
        strategy_analysis = self.strategy_selector.select_strategy(market_data)
        
        # Update internal state
        self.current_market_state = market_condition
        
        # Only update strategy if confidence threshold is met
        if strategy_analysis['confidence'] >= self.confidence_threshold:
            self.current_strategy = strategy_analysis['strategy']
            
        return {
            'market_condition': market_condition,
            'selected_strategy': self.current_strategy,
            'analysis': strategy_analysis,
            'metrics': {
                'market': self.classifier.get_market_metrics(market_data),
                'confidence': strategy_analysis['confidence']
            }
        }
    
    def get_trading_parameters(self, strategy: TradingStrategy) -> Dict[str, Any]:
        """
        Get recommended trading parameters for given strategy
        
        Args:
            strategy: Selected trading strategy
            
        Returns:
            dict: Strategy-specific trading parameters
        """
        # Default parameters
        base_params = {
            'position_size': 1.0,  # Default to 100% position size
            'stop_loss': 0.02,    # 2% stop loss
            'take_profit': 0.04   # 4% take profit
        }
        
        # Strategy-specific adjustments
        if strategy == TradingStrategy.TREND_FOLLOWING:
            base_params.update({
                'stop_loss': 0.03,
                'take_profit': 0.06,
                'trailing_stop': True
            })
            
        elif strategy == TradingStrategy.MEAN_REVERSION:
            base_params.update({
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'mean_period': 20
            })
            
        elif strategy == TradingStrategy.RANGE_TRADING:
            base_params.update({
                'stop_loss': 0.01,
                'take_profit': 0.02,
                'range_bounds': 0.01
            })
            
        elif strategy == TradingStrategy.DEFENSIVE:
            base_params.update({
                'position_size': 0.5,  # Reduce position size
                'stop_loss': 0.01,     # Tighter stops
                'take_profit': 0.02
            })
            
        return base_params

    def get_summary(self) -> Dict[str, Any]:
        """
        Get current market analysis summary
        
        Returns:
            dict: Current market state and strategy information
        """
        return {
            'market_condition': self.current_market_state,
            'active_strategy': self.current_strategy,
            'trading_params': (
                self.get_trading_parameters(self.current_strategy)
                if self.current_strategy else None
            )
        }

# Example usage:
if __name__ == "__main__":
    # Sample market data
    sample_data = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100) * 1000000
    }
    
    # Initialize analyzer
    analyzer = MarketAnalyzer(config={'lookback_period': 20})
    
    # Run analysis
    results = analyzer.analyze_market(sample_data)
    print(f"Market Condition: {results['market_condition']}")
    print(f"Selected Strategy: {results['selected_strategy']}")
    print(f"Confidence: {results['analysis']['confidence']:.2f}")