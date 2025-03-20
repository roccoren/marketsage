"""
FeatureManager - Handles feature engineering and management for Qlib
"""
from typing import List, Dict, Any, Optional, Union
import json
import os
import pandas as pd
import numpy as np
from qlib.data.dataset.utils import fetch_df_by_index
from qlib.contrib.data.handler import Alpha158

class FeatureManager:
    """
    Manager class for handling features in Qlib
    """
    
    def __init__(self):
        """
        Initialize the FeatureManager
        """
        self.features = {}
        self.custom_features = {}
        
    def load_alpha158_features(self):
        """
        Load the built-in Alpha158 feature set from Qlib
        """
        alpha158 = Alpha158()
        self.features.update(alpha158.get_feature_config())
        return self.features
        
    def add_custom_feature(self, name: str, feature_expr: str):
        """
        Add a custom feature
        
        Parameters
        ----------
        name : str
            Name of the feature
        feature_expr : str
            Expression for the feature in Qlib's feature language
        """
        self.custom_features[name] = feature_expr
        self.features[name] = feature_expr
        
    def load_features_from_file(self, filepath: str):
        """
        Load features from a file
        
        Parameters
        ----------
        filepath : str
            Path to the file containing features
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature file {filepath} not found")
            
        with open(filepath, 'r') as f:
            features = json.load(f)
            
        for name, expr in features.items():
            self.add_custom_feature(name, expr)
            
    def save_features_to_file(self, filepath: str, features_type: str = 'custom'):
        """
        Save features to a file
        
        Parameters
        ----------
        filepath : str
            Path to save the features
        features_type : str
            Type of features to save ('all' or 'custom')
        """
        if features_type == 'all':
            features_to_save = self.features
        else:
            features_to_save = self.custom_features
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(features_to_save, f, indent=4)
            
    def get_feature_list(self, features_type: str = 'all') -> List[str]:
        """
        Get a list of feature names
        
        Parameters
        ----------
        features_type : str
            Type of features to list ('all' or 'custom')
            
        Returns
        -------
        List[str]
            List of feature names
        """
        if features_type == 'all':
            return list(self.features.keys())
        else:
            return list(self.custom_features.keys())
    
    def extract_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Extract features from a DataFrame
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing raw data
        features : List[str]
            List of features to extract
            
        Returns
        -------
        pd.DataFrame
            DataFrame with extracted features
        """
        # Only extract features that are available
        available_features = [f for f in features if f in self.features]
        
        # Create an empty DataFrame for the results
        result = pd.DataFrame(index=df.index)
        
        # Extract each feature
        for feature in available_features:
            # Parse and compute the feature expression
            # This is a simplified implementation
            # In a real application, this would involve evaluating the expression
            # based on Qlib's feature language
            result[feature] = df[feature]
            
        return result 