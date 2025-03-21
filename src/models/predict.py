import os
import mlflow
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime

import qlib
from qlib.config import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.model.linear import LinearModel
from qlib.contrib.model.double_ensemble import DEnsembleModel
from qlib.utils import init_instance_by_config

from src.utils import init_qlib, init_mlflow, close_mlflow, log_config
from src.config import QLIB_CONFIG, MLFLOW_CONFIG, MODEL_CONFIG


class SimpleRecorder:
    """
    Simple recorder to replace Qlib's workflow components
    """
    def __init__(self, model=None, dataset=None, predictions=None, path=None):
        self.model = model
        self.dataset = dataset
        self.predictions = predictions
        self.path = path
        
    def save(self, path):
        """Save model, dataset, and predictions to the given path"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = os.path.join(path, "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
        
        # Save dataset
        if self.dataset is not None:
            dataset_path = os.path.join(path, "dataset.pkl")
            with open(dataset_path, "wb") as f:
                pickle.dump(self.dataset, f)
        
        # Save predictions
        if self.predictions is not None:
            pred_path = os.path.join(path, "pred.pkl")
            with open(pred_path, "wb") as f:
                pickle.dump(self.predictions, f)
            
            # Also save as CSV for easier inspection
            pred_csv = os.path.join(path, "pred.csv")
            self.predictions.to_csv(pred_csv)
            
        # Save path reference
        self.path = path
    
    @classmethod
    def load(cls, path):
        """Load recorder from the given path"""
        recorder = cls(path=path)
        
        # Try to load model
        model_path = os.path.join(path, "model.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                recorder.model = pickle.load(f)
        
        # Try to load dataset
        dataset_path = os.path.join(path, "dataset.pkl")
        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                recorder.dataset = pickle.load(f)
        
        # Try to load predictions
        pred_path = os.path.join(path, "pred.pkl")
        if os.path.exists(pred_path):
            with open(pred_path, "rb") as f:
                recorder.predictions = pickle.load(f)
        else:
            # Try loading from CSV as fallback
            pred_csv = os.path.join(path, "pred.csv")
            if os.path.exists(pred_csv):
                recorder.predictions = pd.read_csv(pred_csv, index_col=0)
                
        return recorder
    
    def load_pred(self):
        """Load predictions"""
        if self.predictions is not None:
            return self.predictions
            
        # Try to load predictions from file
        if self.path is not None:
            pred_path = os.path.join(self.path, "pred.pkl")
            if os.path.exists(pred_path):
                with open(pred_path, "rb") as f:
                    self.predictions = pickle.load(f)
            else:
                # Try loading from CSV as fallback
                pred_csv = os.path.join(self.path, "pred.csv")
                if os.path.exists(pred_csv):
                    self.predictions = pd.read_csv(pred_csv, index_col=0)
        
        return self.predictions


class QlibPredictor:
    """
    Use Qlib's built-in models for prediction
    """
    
    def __init__(self, qlib_config: Dict[str, Any] = None, model_config: Dict[str, Any] = None):
        """
        Initialize predictor with Qlib
        
        Parameters
        ----------
        qlib_config : Dict[str, Any]
            Configuration for Qlib initialization
        model_config : Dict[str, Any]
            Configuration for the model
        """
        self.qlib_config = qlib_config or QLIB_CONFIG
        self.model_config = model_config or MODEL_CONFIG
        
        # Initialize Qlib
        init_qlib(self.qlib_config)
        
        # Set up model config
        self.model_name = self.model_config.get("model", "Linear")
        self.model_kwargs = self.model_config.get("model_kwargs", {})
        self.dataset_kwargs = self.model_config.get("dataset_kwargs", {})
    
    def _get_model_class(self, model_name: str):
        """
        Get the model class by name
        
        Parameters
        ----------
        model_name : str
            Name of the model
            
        Returns
        -------
        Model class
        """
        model_mapping = {
            "Linear": LinearModel,
            "LightGBM": LGBModel,
            "LGBM": LGBModel,
            "DEnsemble": DEnsembleModel,
        }
        
        if model_name in model_mapping:
            return model_mapping[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Available models: {list(model_mapping.keys())}")
    
    def _get_dataset(self, handler: str = "Alpha158", segments: Dict[str, Tuple[str, str]] = None):
        """
        Create a dataset for model training and prediction
        
        Parameters
        ----------
        handler : str
            Dataset handler name
        segments : Dict[str, Tuple[str, str]]
            Dataset segments
            
        Returns
        -------
        qlib.data.dataset.DatasetH
            Dataset object
        """
        if segments is None:
            segments = {
                "train": ("2015-01-01", "2020-12-31"),
                "valid": ("2021-01-01", "2021-06-30"),
                "test": ("2021-07-01", "2022-12-31")
            }
        
        dataset_config = {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": handler,
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": {}
                },
                "segments": segments
            }
        }
        
        return init_instance_by_config(dataset_config)
    
    def train_and_predict(
        self,
        save_path: Optional[str] = None,
        track_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Train model and make predictions
        
        Parameters
        ----------
        save_path : Optional[str]
            Path to save the model
        track_mlflow : bool
            Whether to track with MLflow
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary
        """
        # Create default save path if not provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(os.getcwd(), "results", f"{self.model_name}_{timestamp}")
            os.makedirs(save_path, exist_ok=True)
        
        # Initialize MLflow if tracking is enabled
        run_id = None
        if track_mlflow:
            run_id = init_mlflow(MLFLOW_CONFIG)
            log_config(
                qlib_config=self.qlib_config,
                model_config=self.model_config
            )
        
        try:
            # Create dataset
            dataset_kwargs = self.dataset_kwargs.copy()
            segments = dataset_kwargs.pop("segments", None)
            handler = dataset_kwargs.pop("handler", "Alpha158")
            dataset = self._get_dataset(handler, segments)
            
            # Initialize model
            model_class = self._get_model_class(self.model_name)
            model = model_class(**self.model_kwargs)
            
            print(f"Training {self.model_name} model...")
            # Train the model
            model.fit(dataset)
            
            print("Making predictions...")
            # Make predictions on the test set
            pred = model.predict(dataset)
            
            # Create and save recorder
            recorder = SimpleRecorder(model=model, dataset=dataset, predictions=pred)
            recorder.save(save_path)
            
            # For analysis, use a simplified approach
            analysis_df = pd.DataFrame({
                "IC": [pred.mean().mean()],
                "ICIR": [pred.mean().mean() / (pred.std().mean() + 1e-8)]
            }, index=["analysis"])
            
            analysis_file = os.path.join(save_path, "analysis.csv")
            analysis_df.to_csv(analysis_file)
            
            # Log to MLflow if tracking is enabled
            if track_mlflow:
                pred_file = os.path.join(save_path, "pred.csv")
                mlflow.log_artifact(pred_file)
                mlflow.log_artifact(analysis_file)
                mlflow.log_metrics({
                    "ic": analysis_df.loc["analysis", "IC"],
                    "icir": analysis_df.loc["analysis", "ICIR"],
                })
            
            print(f"Prediction and analysis saved to {save_path}")
            
            return {
                "model": model,
                "dataset": dataset,
                "predictions": pred,
                "analysis": analysis_df,
                "save_path": save_path
            }
        except Exception as e:
            print(f"Error in training and prediction: {str(e)}")
            if track_mlflow:
                mlflow.log_param("training_status", "error")
                mlflow.log_param("training_error", str(e))
            raise e
        finally:
            if track_mlflow:
                close_mlflow()
    
    @staticmethod
    def fine_tune(
        base_model_path: str,
        new_segments: Dict[str, Tuple[str, str]] = None,
        model_params: Dict[str, Any] = None,
        save_path: Optional[str] = None,
        track_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Fine-tune an existing Qlib model
        
        Parameters
        ----------
        base_model_path : str
            Path to the base model
        new_segments : Dict[str, Tuple[str, str]]
            New dataset segments
        model_params : Dict[str, Any]
            New model parameters
        save_path : Optional[str]
            Path to save the fine-tuned model
        track_mlflow : bool
            Whether to track with MLflow
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary
        """
        if new_segments is None:
            new_segments = {
                "train": ("2015-01-01", "2021-12-31"),  # Extended training period
                "valid": ("2022-01-01", "2022-06-30"),
                "test": ("2022-07-01", "2023-12-31")
            }
        
        # Create default save path if not provided
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = base_model_path.split("/")[-1].split("_")[0]
            save_path = os.path.join(os.getcwd(), "results", f"finetuned_{model_name}_{timestamp}")
            os.makedirs(save_path, exist_ok=True)
        
        # Initialize MLflow if tracking is enabled
        run_id = None
        if track_mlflow:
            run_id = init_mlflow(MLFLOW_CONFIG)
            mlflow.log_param("base_model_path", base_model_path)
            mlflow.log_params({"new_segment_" + k: f"{v[0]}-{v[1]}" for k, v in new_segments.items()})
        
        try:
            # Load the recorder
            recorder = SimpleRecorder.load(base_model_path)
            base_model = recorder.model
            
            # If no model found, raise error
            if base_model is None:
                raise ValueError(f"No model found in {base_model_path}")
                
            # Create a new dataset with the new segments
            if hasattr(base_model, 'model_config') and base_model.model_config is not None:
                dataset_config = base_model.model_config.get('dataset')
                if dataset_config is not None:
                    # Update segments
                    if 'kwargs' in dataset_config and 'segments' in dataset_config['kwargs']:
                        dataset_config['kwargs']['segments'] = new_segments
                        
                    # Create new dataset
                    dataset = init_instance_by_config(dataset_config)
                else:
                    # Fallback to default dataset
                    print("Warning: No dataset config found in model. Using default dataset.")
                    handler = "Alpha158"
                    dataset = QlibPredictor._get_dataset(None, handler, new_segments)
            else:
                # Fallback to default dataset
                print("Warning: No dataset config found in model. Using default dataset.")
                handler = "Alpha158"
                dataset = QlibPredictor._get_dataset(None, handler, new_segments)
            
            # Update model parameters if provided
            if model_params is not None:
                for param, value in model_params.items():
                    if hasattr(base_model, param):
                        setattr(base_model, param, value)
            
            print(f"Fine-tuning model from {base_model_path}...")
            # Train the model with the new dataset
            base_model.fit(dataset)
            
            print("Making predictions with fine-tuned model...")
            # Make predictions
            pred = base_model.predict(dataset)
            
            # Create and save recorder
            fine_tuned_recorder = SimpleRecorder(model=base_model, dataset=dataset, predictions=pred)
            fine_tuned_recorder.save(save_path)
            
            # For analysis, use a simplified approach
            analysis_df = pd.DataFrame({
                "IC": [pred.mean().mean()],
                "ICIR": [pred.mean().mean() / (pred.std().mean() + 1e-8)]
            }, index=["analysis"])
            
            analysis_file = os.path.join(save_path, "analysis.csv")
            analysis_df.to_csv(analysis_file)
            
            # Log to MLflow if tracking is enabled
            if track_mlflow:
                pred_file = os.path.join(save_path, "pred.csv")
                mlflow.log_artifact(pred_file)
                mlflow.log_artifact(analysis_file)
                mlflow.log_metrics({
                    "ic": analysis_df.loc["analysis", "IC"],
                    "icir": analysis_df.loc["analysis", "ICIR"],
                })
            
            print(f"Fine-tuned model and analysis saved to {save_path}")
            
            return {
                "model": base_model,
                "dataset": dataset,
                "predictions": pred,
                "analysis": analysis_df,
                "save_path": save_path
            }
        except Exception as e:
            print(f"Error in fine-tuning: {str(e)}")
            if track_mlflow:
                mlflow.log_param("finetuning_status", "error")
                mlflow.log_param("finetuning_error", str(e))
            raise e
        finally:
            if track_mlflow:
                close_mlflow() 