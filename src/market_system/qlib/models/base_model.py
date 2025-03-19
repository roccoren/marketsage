"""
Base model module for Qlib integration.
"""

from typing import Optional, Union, Dict, Any
import pandas as pd
import numpy as np
from qlib.model.base import Model
from qlib.data.dataset import Dataset
from ..config import MODEL_CONFIG

class QlibBaseModel:
    """Base model class for Qlib integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model.
        
        Args:
            config (dict, optional): Model configuration. Defaults to None.
        """
        self.config = config or MODEL_CONFIG
        self.model = self._initialize_model()
        
    def _initialize_model(self) -> Model:
        """Initialize the Qlib model based on configuration.
        
        Returns:
            Model: Initialized Qlib model
        """
        if self.config["model_type"] == "lightgbm":
            from qlib.contrib.model.gbdt import LGBModel
            return LGBModel(**self.config["model_kwargs"])
        else:
            raise ValueError(f"Unsupported model type: {self.config['model_type']}")
        
    def train(self, dataset: Dataset) -> None:
        """Train the model.
        
        Args:
            dataset (Dataset): Training dataset
        """
        self.model.fit(dataset)
        
    def predict(self, dataset: Dataset) -> pd.Series:
        """Make predictions using the trained model.
        
        Args:
            dataset (Dataset): Dataset for prediction
            
        Returns:
            pd.Series: Predictions
        """
        return self.model.predict(dataset)
    
    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model performance.
        
        Args:
            dataset (Dataset): Dataset for evaluation
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        from qlib.contrib.evaluate import risk_analysis
        pred = self.predict(dataset)
        analysis = risk_analysis(pred)
        return {
            "IC": analysis["IC"],
            "ICIR": analysis["ICIR"],
            "rank_IC": analysis["rank_IC"],
            "rank_ICIR": analysis["rank_ICIR"],
        }
    
    def save(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path (str): Path to save the model
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load the model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        self.model.load(path) 