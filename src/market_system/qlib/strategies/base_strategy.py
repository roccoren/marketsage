"""
Base strategy module for Qlib integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from qlib.strategy.base import BaseStrategy
from qlib.backtest import backtest, executor
from ..models.base_model import QlibBaseModel
from ..config import BACKTEST_CONFIG

class QlibStrategyBase(BaseStrategy):
    """Base class for all Qlib-based strategies."""

    def __init__(self, model: Optional[QlibBaseModel] = None, **kwargs):
        """Initialize strategy.
        
        Args:
            model (QlibBaseModel, optional): The model to use for predictions. Defaults to None.
            **kwargs: Additional strategy parameters
        """
        super().__init__()
        self.model = model
        self.config = {**BACKTEST_CONFIG["strategy"]["kwargs"], **kwargs}
        
    @abstractmethod
    def generate_signals(self, dataset) -> pd.Series:
        """Generate trading signals from the dataset.
        
        Args:
            dataset: The dataset to generate signals from
            
        Returns:
            pd.Series: Trading signals
        """
        raise NotImplementedError("Subclass must implement generate_signals method")

    def get_positions(self, signals: pd.Series) -> pd.Series:
        """Convert signals to position sizes.
        
        Args:
            signals (pd.Series): Trading signals
            
        Returns:
            pd.Series: Position sizes for each instrument
        """
        # Default implementation: equal weight top k stocks
        topk = self.config.get("topk", 50)
        n_drop = self.config.get("n_drop", 5)
        
        # Sort signals and get top k
        signals = signals.sort_values(ascending=False)
        topk_signals = signals.head(topk)
        
        # Drop bottom n from top k if specified
        if n_drop > 0:
            topk_signals = topk_signals.iloc[:-n_drop]
            
        # Equal weight positions
        positions = pd.Series(1.0 / len(topk_signals), index=topk_signals.index)
        return positions
    
    def generate_trade_decision(self, dataset):
        """Generate trade decisions based on the strategy.
        
        Args:
            dataset: The dataset to generate trades from
            
        Returns:
            object: Trade decision object
        """
        signals = self.generate_signals(dataset)
        positions = self.get_positions(signals)
        
        # Create trade decision
        trade_decision = pd.DataFrame({
            "instrument": positions.index,
            "weight": positions.values
        })
        return trade_decision 