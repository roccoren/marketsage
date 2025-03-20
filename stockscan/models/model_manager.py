"""
ModelManager - Handles model training and management for Qlib
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import pickle
import numpy as np
import pandas as pd
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.xgboost import XGBModel

class ModelManager:
    """
    Manager class for handling models in Qlib
    """
    
    def __init__(self):
        """
        Initialize the ModelManager
        """
        self.models = {}
        self.model_configs = {}
        
    def create_model_from_config(self, 
                               model_name: str, 
                               model_config: Dict[str, Any]) -> Model:
        """
        Create a model from config
        
        Parameters
        ----------
        model_name : str
            Name of the model
        model_config : Dict[str, Any]
            Configuration for the model
            
        Returns
        -------
        Model
            The created model
        """
        model = init_instance_by_config(model_config)
        self.models[model_name] = model
        self.model_configs[model_name] = model_config
        return model
    
    def create_lgb_model(self, 
                       model_name: str, 
                       loss: str = 'mse', 
                       **kwargs) -> LGBModel:
        """
        Create a LightGBM model
        
        Parameters
        ----------
        model_name : str
            Name of the model
        loss : str
            Loss function for the model
        **kwargs : Dict[str, Any]
            Additional parameters for the model
            
        Returns
        -------
        LGBModel
            The created LightGBM model
        """
        config = {
            'class': 'LGBModel',
            'module_path': 'qlib.contrib.model.gbdt',
            'kwargs': {
                'loss': loss,
                **kwargs
            }
        }
        return self.create_model_from_config(model_name, config)
    
    def create_xgb_model(self, 
                       model_name: str, 
                       objective: str = 'reg:squarederror', 
                       **kwargs) -> XGBModel:
        """
        Create an XGBoost model
        
        Parameters
        ----------
        model_name : str
            Name of the model
        objective : str
            Objective function for the model
        **kwargs : Dict[str, Any]
            Additional parameters for the model
            
        Returns
        -------
        XGBModel
            The created XGBoost model
        """
        config = {
            'class': 'XGBModel',
            'module_path': 'qlib.contrib.model.xgboost',
            'kwargs': {
                'objective': objective,
                **kwargs
            }
        }
        return self.create_model_from_config(model_name, config)
    
    def train_model(self, 
                  model_name: str, 
                  dataset: DatasetH) -> Tuple[float, float]:
        """
        Train a model on a dataset
        
        Parameters
        ----------
        model_name : str
            Name of the model to train
        dataset : DatasetH
            Dataset to train on
            
        Returns
        -------
        Tuple[float, float]
            Tuple of (training score, validation score)
        """
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Prepare the data
        x_train, y_train = dataset.prepare("train", col_set=["feature", "label"])
        x_valid, y_valid = dataset.prepare("valid", col_set=["feature", "label"])
        
        # Train the model
        model.fit(x_train, y_train)
        
        # Evaluate the model
        train_score = model.score(x_train, y_train)
        valid_score = model.score(x_valid, y_valid)
        
        return train_score, valid_score
    
    def predict(self, 
              model_name: str, 
              dataset: DatasetH, 
              segment: str = "test") -> pd.Series:
        """
        Generate predictions using a trained model
        
        Parameters
        ----------
        model_name : str
            Name of the model to use
        dataset : DatasetH
            Dataset to generate predictions for
        segment : str
            Segment of the dataset to use
            
        Returns
        -------
        pd.Series
            Series of predictions
        """
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Prepare the data
        x_test = dataset.prepare(segment, col_set="feature")
        
        # Generate predictions
        return pd.Series(model.predict(x_test), index=x_test.index)
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a model to disk
        
        Parameters
        ----------
        model_name : str
            Name of the model to save
        filepath : str
            Path to save the model
        """
        if model_name not in self.models:
            raise KeyError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        config = self.model_configs[model_name]
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': model,
                'config': config
            }, f)
            
    def load_model(self, model_name: str, filepath: str) -> Model:
        """
        Load a model from disk
        
        Parameters
        ----------
        model_name : str
            Name to assign to the loaded model
        filepath : str
            Path to load the model from
            
        Returns
        -------
        Model
            The loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        model = model_data['model']
        config = model_data['config']
        
        self.models[model_name] = model
        self.model_configs[model_name] = config
        
        return model 