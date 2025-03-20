"""
Signal strategy module for interpreting prediction signals
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union, Tuple

logger = logging.getLogger(__name__)

class SignalStrengthStrategy:
    """
    Strategy for interpreting prediction signals based on their strength
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the strategy with custom thresholds
        
        Parameters
        ----------
        thresholds : dict, optional
            Dictionary of thresholds for different signal strengths
        """
        self.thresholds = thresholds or {
            'strong_buy': 0.015,
            'buy': 0.008,
            'weak_buy': 0.003,
            'neutral': 0.0,
            'weak_sell': -0.003,
            'sell': -0.008,
            'strong_sell': -0.015
        }
    
    def get_signal_strength(self, prediction: float) -> str:
        """
        Get the signal strength for a prediction
        
        Parameters
        ----------
        prediction : float
            Prediction value
            
        Returns
        -------
        str
            Signal strength category
        """
        try:
            if prediction >= self.thresholds['strong_buy']:
                return "STRONG BUY"
            elif prediction >= self.thresholds['buy']:
                return "BUY"
            elif prediction >= self.thresholds['weak_buy']:
                return "WEAK BUY"
            elif prediction > self.thresholds['weak_sell']:
                return "NEUTRAL"
            elif prediction > self.thresholds['sell']:
                return "WEAK SELL"
            elif prediction > self.thresholds['strong_sell']:
                return "SELL"
            else:
                return "STRONG SELL"
        except Exception as e:
            logger.error(f"Error determining signal strength: {e}")
            return "UNKNOWN"
    
    def get_recommendation(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Get a recommendation based on multiple predictions
        
        Parameters
        ----------
        predictions : dict
            Dictionary mapping stock symbols to prediction values
            
        Returns
        -------
        dict
            Recommendation object
        """
        try:
            if not predictions:
                return {'error': 'No predictions provided'}
            
            # Sort stocks by prediction value (descending)
            sorted_stocks = sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Get top stock
            top_stock, top_value = sorted_stocks[0]
            top_signal = self.get_signal_strength(top_value)
            
            # Get bottom stock
            bottom_stock, bottom_value = sorted_stocks[-1]
            bottom_signal = self.get_signal_strength(bottom_value)
            
            # Create recommendation
            recommendation = {
                'top_stock': top_stock,
                'top_value': top_value,
                'top_signal': top_signal,
                'bottom_stock': bottom_stock,
                'bottom_value': bottom_value,
                'bottom_signal': bottom_signal,
                'all_signals': {stock: self.get_signal_strength(value) for stock, value in predictions.items()},
                'ranked_stocks': sorted_stocks
            }
            
            # Add overall recommendation
            if top_value > self.thresholds['buy']:
                recommendation['action'] = f"BUY {top_stock}"
                recommendation['explanation'] = f"{top_stock} shows a strong positive signal ({top_value:.4f})"
            elif bottom_value < self.thresholds['sell']:
                recommendation['action'] = f"SELL {bottom_stock}"
                recommendation['explanation'] = f"{bottom_stock} shows a strong negative signal ({bottom_value:.4f})"
            else:
                recommendation['action'] = "HOLD"
                recommendation['explanation'] = "No strong signals detected"
            
            return recommendation
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {'error': str(e)}
    
    def get_portfolio_allocation(self, predictions: Dict[str, float], total_allocation: float = 100.0) -> Dict[str, float]:
        """
        Get suggested portfolio allocation based on prediction signals
        
        Parameters
        ----------
        predictions : dict
            Dictionary mapping stock symbols to prediction values
        total_allocation : float
            Total allocation percentage/amount
            
        Returns
        -------
        dict
            Suggested allocation for each stock
        """
        try:
            if not predictions:
                return {}
            
            # Filter only positive predictions
            positive_preds = {k: v for k, v in predictions.items() if v > 0}
            
            if not positive_preds:
                # No positive predictions, suggest equal allocation for all
                equal_weight = total_allocation / len(predictions)
                return {stock: equal_weight for stock in predictions}
            
            # Calculate weights based on prediction strength
            total_strength = sum(positive_preds.values())
            
            if total_strength == 0:
                # All predictions have strength 0, use equal weights
                equal_weight = total_allocation / len(positive_preds)
                allocations = {stock: equal_weight for stock in positive_preds}
            else:
                # Allocate based on relative strength
                allocations = {
                    stock: (value / total_strength) * total_allocation
                    for stock, value in positive_preds.items()
                }
            
            # Add 0 allocation for negative predictions
            for stock in predictions:
                if stock not in allocations:
                    allocations[stock] = 0.0
            
            return allocations
        except Exception as e:
            logger.error(f"Error calculating portfolio allocation: {e}")
            return {} 