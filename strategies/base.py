from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd

class Strategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """
        Analyze the provided data and return a list of signals.
        
        Args:
            data: DataFrame with historical price data
            
        Returns:
            List of dictionaries containing signal information
        """
        pass

    def calculate_signal_strength(self, 
                                macd_hist: float, 
                                volume_ratio: float, 
                                price_change: float) -> float:
        """
        Calculate signal strength score (0-1) based on multiple factors.
        
        Args:
            macd_hist: MACD histogram value
            volume_ratio: Current volume / Average volume
            price_change: Price percentage change
            
        Returns:
            Float between 0 and 1 representing signal strength
        """
        # Normalize components to 0-1 range
        macd_score = min(abs(macd_hist) / 2, 1.0)  # Normalize MACD histogram
        volume_score = min((volume_ratio - 1) / 4, 1.0)  # Normalize volume surge
        price_score = min(abs(price_change) / 10, 1.0)  # Normalize price change

        # Weight the components
        weighted_score = (
            0.4 * macd_score +    # MACD histogram (40% weight)
            0.4 * volume_score +  # Volume surge (40% weight)
            0.2 * price_score     # Price change (20% weight)
        )

        return round(weighted_score, 4)