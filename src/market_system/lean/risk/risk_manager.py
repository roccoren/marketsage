"""
Risk Management Module

This module provides risk management functionality for LEAN-based trading algorithms.
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

class RiskManager:
    """Risk management class for trading algorithms."""
    
    def __init__(self,
                 max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.02,
                 max_correlation: float = 0.7,
                 max_drawdown: float = 0.15,
                 max_positions: int = 5,
                 min_liquidity: float = 1000000):
        """Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as fraction of portfolio
            max_portfolio_risk: Maximum portfolio risk as fraction of portfolio
            max_correlation: Maximum correlation between positions
            max_drawdown: Maximum allowed drawdown
            max_positions: Maximum number of concurrent positions
            min_liquidity: Minimum daily dollar volume for trading
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions
        self.min_liquidity = min_liquidity
        
        self.positions = {}
        self.correlations = {}
        self.portfolio_metrics = {
            'current_risk': 0,
            'current_drawdown': 0,
            'position_count': 0
        }
    
    def calculate_position_risk(self, 
                              symbol: str, 
                              price: float, 
                              volatility: float,
                              quantity: float) -> float:
        """Calculate risk for a single position.
        
        Args:
            symbol: Trading symbol
            price: Current price
            volatility: Historical volatility
            quantity: Position quantity
            
        Returns:
            Position risk as fraction of portfolio
        """
        position_value = price * quantity
        position_risk = position_value * volatility
        return position_risk
    
    def calculate_portfolio_risk(self, new_position: Optional[Dict] = None) -> float:
        """Calculate total portfolio risk including correlations.
        
        Args:
            new_position: Optional new position to include in calculation
            
        Returns:
            Total portfolio risk as fraction of portfolio
        """
        if new_position:
            temp_positions = self.positions.copy()
            temp_positions[new_position['symbol']] = new_position
        else:
            temp_positions = self.positions
            
        # Calculate correlation-weighted portfolio risk
        total_risk = 0
        for symbol1, pos1 in temp_positions.items():
            for symbol2, pos2 in temp_positions.items():
                if symbol1 == symbol2:
                    correlation = 1
                else:
                    correlation = self.correlations.get((symbol1, symbol2), 0)
                    
                risk1 = self.calculate_position_risk(
                    pos1['symbol'], pos1['price'], pos1['volatility'], pos1['quantity']
                )
                risk2 = self.calculate_position_risk(
                    pos2['symbol'], pos2['price'], pos2['volatility'], pos2['quantity']
                )
                
                total_risk += risk1 * risk2 * correlation
                
        return np.sqrt(total_risk)
    
    def check_liquidity(self, symbol: str, volume: float, price: float) -> bool:
        """Check if symbol meets liquidity requirements.
        
        Args:
            symbol: Trading symbol
            volume: Daily volume
            price: Current price
            
        Returns:
            bool: True if liquidity requirements are met
        """
        dollar_volume = volume * price
        return dollar_volume >= self.min_liquidity
    
    def can_add_position(self, 
                        symbol: str,
                        price: float,
                        volatility: float,
                        quantity: float,
                        volume: float) -> Dict:
        """Check if a new position can be added within risk limits.
        
        Args:
            symbol: Trading symbol
            price: Current price
            volatility: Historical volatility
            quantity: Proposed position quantity
            volume: Daily volume
            
        Returns:
            Dict with approval status and reason
        """
        # Check number of positions
        if self.portfolio_metrics['position_count'] >= self.max_positions:
            return {'approved': False, 'reason': 'Maximum positions reached'}
            
        # Check position size
        position_value = price * quantity
        if position_value > self.max_position_size:
            return {'approved': False, 'reason': 'Position size too large'}
            
        # Check liquidity
        if not self.check_liquidity(symbol, volume, price):
            return {'approved': False, 'reason': 'Insufficient liquidity'}
            
        # Check portfolio risk
        new_position = {
            'symbol': symbol,
            'price': price,
            'volatility': volatility,
            'quantity': quantity
        }
        
        portfolio_risk = self.calculate_portfolio_risk(new_position)
        if portfolio_risk > self.max_portfolio_risk:
            return {'approved': False, 'reason': 'Portfolio risk too high'}
            
        # Check correlation with existing positions
        for existing_symbol in self.positions:
            correlation = self.correlations.get((symbol, existing_symbol), 0)
            if correlation > self.max_correlation:
                return {'approved': False, 'reason': f'High correlation with {existing_symbol}'}
                
        return {'approved': True, 'reason': 'Position approved'}
    
    def update_position(self,
                       symbol: str,
                       price: float,
                       volatility: float,
                       quantity: float) -> None:
        """Update or add a position.
        
        Args:
            symbol: Trading symbol
            price: Current price
            volatility: Historical volatility
            quantity: Position quantity
        """
        self.positions[symbol] = {
            'symbol': symbol,
            'price': price,
            'volatility': volatility,
            'quantity': quantity
        }
        self.portfolio_metrics['position_count'] = len(self.positions)
    
    def update_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """Update correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Correlation coefficient
        """
        self.correlations[(symbol1, symbol2)] = correlation
        self.correlations[(symbol2, symbol1)] = correlation
    
    def update_portfolio_metrics(self, portfolio_value: float, peak_value: float) -> None:
        """Update portfolio metrics.
        
        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
        """
        self.portfolio_metrics['current_drawdown'] = (portfolio_value / peak_value - 1)
        self.portfolio_metrics['current_risk'] = self.calculate_portfolio_risk()
    
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics.
        
        Returns:
            Dict containing portfolio metrics
        """
        return self.portfolio_metrics.copy() 