#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimization module for Qlib models and strategies
"""

import os
import pandas as pd
import numpy as np
import mlflow
import optuna
from optuna.samplers import TPESampler
from optuna.integration.mlflow import MLflowCallback
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from datetime import datetime
import pickle
import joblib
from copy import deepcopy
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

import qlib
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config

from src.models.predict import QlibPredictor, SimpleRecorder
from src.strategies.backtest import SimpleBacktester


def setup_mlflow_tracking(experiment_name: str) -> str:
    """
    Set up MLflow tracking for optimization
    
    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment
        
    Returns
    -------
    str
        MLflow run ID
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    run = mlflow.start_run()
    
    return run.info.run_id


def log_optimization_results(
    study: optuna.Study,
    metric: str,
    save_path: Optional[str] = None,
    prefix: str = ""
):
    """
    Log optimization results to MLflow and save visualizations
    
    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study
    metric : str
        Name of the optimization metric
    save_path : Optional[str]
        Path to save visualization plots
    prefix : str
        Prefix for MLflow metric names
    """
    # Log best value and parameters
    mlflow.log_metric(f"{prefix}best_{metric}", study.best_value)
    mlflow.log_params({
        f"{prefix}{k}": v for k, v in study.best_params.items()
    })
    
    # Create optimization history plot
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=list(range(len(study.trials))),
        y=[t.value for t in study.trials],
        mode="markers+lines",
        name=metric
    ))
    fig_history.update_layout(
        title=f"Optimization History - {metric}",
        xaxis_title="Trial",
        yaxis_title=metric
    )
    
    # Create parameter importance plot
    importances = optuna.importance.get_param_importances(study)
    fig_importance = px.bar(
        x=list(importances.keys()),
        y=list(importances.values()),
        title="Parameter Importances"
    )
    
    # Save plots if path provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fig_history.write_html(os.path.join(save_path, f"{prefix}optimization_history.html"))
        fig_importance.write_html(os.path.join(save_path, f"{prefix}parameter_importance.html"))
        
        # Log plots to MLflow
        mlflow.log_artifact(os.path.join(save_path, f"{prefix}optimization_history.html"))
        mlflow.log_artifact(os.path.join(save_path, f"{prefix}parameter_importance.html"))
    
    # Log optimization history
    history_df = pd.DataFrame({
        "trial": range(len(study.trials)),
        metric: [t.value for t in study.trials],
        **{f"param_{k}": [t.params.get(k) for t in study.trials] 
           for k in study.best_params.keys()}
    })
    
    if save_path is not None:
        history_path = os.path.join(save_path, f"{prefix}optimization_history.csv")
        history_df.to_csv(history_path, index=False)
        mlflow.log_artifact(history_path)


class FeatureSelector:
    """
    Feature selection for Qlib models
    """
    def __init__(self, dataset, importance_method="shap"):
        """
        Initialize feature selector
        
        Parameters
        ----------
        dataset : qlib.data.dataset.DatasetH
            Dataset to use for feature selection
        importance_method : str
            Method to calculate feature importance (shap or permutation)
        """
        self.dataset = dataset
        self.importance_method = importance_method
        self.feature_importances = None
        
    def calculate_importances(self, model, top_k=None):
        """
        Calculate feature importances
        
        Parameters
        ----------
        model : Any
            Trained model
        top_k : Optional[int]
            Number of top features to select
            
        Returns
        -------
        pd.DataFrame
            Feature importances
        """
        if self.importance_method == "shap":
            importances = self._calculate_shap_importances(model)
        else:
            importances = self._calculate_permutation_importances(model)
            
        # Sort importances
        sorted_importances = importances.sort_values(ascending=False)
        
        if top_k is not None:
            sorted_importances = sorted_importances.head(top_k)
            
        self.feature_importances = sorted_importances
        return sorted_importances
        
    def _calculate_shap_importances(self, model):
        """
        Calculate SHAP-based feature importances
        
        Parameters
        ----------
        model : Any
            Trained model
            
        Returns
        -------
        pd.Series
            Feature importances
        """
        try:
            import shap
            # Get a small subset of data for explanation
            x_valid = self.dataset.prepare("valid", col_set="feature")
            
            # Depending on model type, use appropriate explainer
            if hasattr(model, "model"):
                if hasattr(model.model, "booster"):
                    # For LightGBM models
                    explainer = shap.TreeExplainer(model.model)
                    shap_values = explainer.shap_values(x_valid)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]  # Take first class for binary classification
                        
                    # Calculate mean absolute SHAP values
                    importances = pd.Series(
                        np.abs(shap_values).mean(axis=0),
                        index=x_valid.columns
                    )
                else:
                    # For other model types like Linear
                    explainer = shap.LinearExplainer(model, x_valid)
                    shap_values = explainer.shap_values(x_valid)
                    
                    # Calculate mean absolute SHAP values
                    importances = pd.Series(
                        np.abs(shap_values).mean(axis=0),
                        index=x_valid.columns
                    )
            else:
                # Fallback to model coefficient if SHAP is not applicable
                importances = pd.Series(
                    np.abs(model.coef_) if hasattr(model, "coef_") else np.ones(len(x_valid.columns)),
                    index=x_valid.columns
                )
                
            return importances
            
        except (ImportError, Exception) as e:
            print(f"Error calculating SHAP importances: {str(e)}")
            print("Falling back to permutation importance")
            return self._calculate_permutation_importances(model)
    
    def _calculate_permutation_importances(self, model):
        """
        Calculate permutation-based feature importances
        
        Parameters
        ----------
        model : Any
            Trained model
            
        Returns
        -------
        pd.Series
            Feature importances
        """
        from sklearn.inspection import permutation_importance
        
        # Get validation data
        x_valid = self.dataset.prepare("valid", col_set="feature")
        y_valid = self.dataset.prepare("valid", col_set="label")
        
        if hasattr(y_valid, "values"):
            y_valid = y_valid.values
            
        # Perform permutation importance calculation
        perm_importance = permutation_importance(
            model, x_valid, y_valid, n_repeats=10, random_state=42
        )
        
        # Create feature importance Series
        importances = pd.Series(
            perm_importance.importances_mean,
            index=x_valid.columns
        )
        
        return importances
        
    def select_features(self, top_k=50):
        """
        Select top features based on importance
        
        Parameters
        ----------
        top_k : int
            Number of top features to select
            
        Returns
        -------
        List[str]
            List of selected feature names
        """
        if self.feature_importances is None:
            raise ValueError("Feature importances not calculated. Call calculate_importances first.")
            
        selected_features = self.feature_importances.head(top_k).index.tolist()
        return selected_features
        
    def create_filtered_dataset(self, selected_features, save_path=None):
        """
        Create a new dataset with only selected features
        
        Parameters
        ----------
        selected_features : List[str]
            List of selected feature names
        save_path : Optional[str]
            Path to save the filtered dataset
            
        Returns
        -------
        qlib.data.dataset.DatasetH
            Filtered dataset
        """
        # Clone dataset config
        if hasattr(self.dataset, "config"):
            new_config = deepcopy(self.dataset.config)
            
            # Modify handler to use only selected features
            if "kwargs" in new_config and "handler" in new_config["kwargs"]:
                if "kwargs" in new_config["kwargs"]["handler"]:
                    new_config["kwargs"]["handler"]["kwargs"]["fields_group"] = {
                        "feature": selected_features
                    }
            
            # Create new dataset
            filtered_dataset = init_instance_by_config(new_config)
            
            # Save if path provided
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, "filtered_dataset.pkl"), "wb") as f:
                    pickle.dump(filtered_dataset, f)
                
                # Save feature list
                pd.Series(selected_features).to_csv(
                    os.path.join(save_path, "selected_features.csv"),
                    index=False,
                    header=["feature_name"]
                )
                
            return filtered_dataset
        else:
            print("Warning: Dataset doesn't have config attribute. Unable to create filtered dataset.")
            return None


class ModelOptimizer:
    """
    Hyperparameter optimization for Qlib models
    """
    def __init__(
        self,
        qlib_config: Dict[str, Any],
        study_name: str = "qlib_model_optimization",
        n_trials: int = 50,
        timeout: Optional[int] = None,
        base_dir: str = "results/optimize"
    ):
        """
        Initialize model optimizer
        
        Parameters
        ----------
        qlib_config : Dict[str, Any]
            Qlib configuration
        study_name : str
            Optuna study name
        n_trials : int
            Number of optimization trials
        timeout : Optional[int]
            Timeout in seconds
        base_dir : str
            Base directory for saving results
        """
        self.qlib_config = qlib_config
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.base_dir = base_dir
        self.best_params = None
        
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
    def optimize_model(
        self,
        model_name: str,
        param_space: Dict[str, Any],
        dataset_kwargs: Dict[str, Any],
        metric: str = "IC",
        resume_study: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize model hyperparameters
        
        Parameters
        ----------
        model_name : str
            Name of the model to optimize
        param_space : Dict[str, Any]
            Parameter space for optimization
        dataset_kwargs : Dict[str, Any]
            Dataset configuration
        metric : str
            Metric to optimize (IC or ICIR)
        resume_study : bool
            Whether to resume previous study if exists
            
        Returns
        -------
        Dict[str, Any]
            Best parameters
        """
        # Set up MLflow tracking
        run_id = setup_mlflow_tracking(f"model_optimization_{model_name}")
        mlflow.log_params({
            "model_name": model_name,
            "n_trials": self.n_trials,
            "timeout": self.timeout,
            "metric": metric,
            "dataset_kwargs": dataset_kwargs
        })
        
        # Create study storage directory
        storage_dir = os.path.join(self.base_dir, f"study_{model_name}")
        os.makedirs(storage_dir, exist_ok=True)
        storage_path = os.path.join(storage_dir, "study.db")
        
        try:
            # Create or load the Optuna study
            storage = f"sqlite:///{storage_path}"
            if resume_study and os.path.exists(storage_path):
                study = optuna.load_study(
                    study_name=self.study_name,
                    storage=storage
                )
                print(f"Resuming study from {storage_path}")
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    direction="maximize",
                    sampler=TPESampler(),
                    pruner=optuna.pruners.MedianPruner(),
                    storage=storage
                )
                print(f"Created new study at {storage_path}")
            
            mlflow_callback = MLflowCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                metric_name=metric
            )
            
            # Define objective function with better error handling
            def objective(trial):
                trial_dir = os.path.join(storage_dir, f"trial_{trial.number}")
                os.makedirs(trial_dir, exist_ok=True)
                
                try:
                    # Sample parameters with validation
                    model_kwargs = {}
                    for param_name, param_spec in param_space.items():
                        try:
                            if param_spec["type"] == "int":
                                model_kwargs[param_name] = trial.suggest_int(
                                    param_name, 
                                    param_spec["min"], 
                                    param_spec["max"], 
                                    step=param_spec.get("step", 1)
                                )
                            elif param_spec["type"] == "float":
                                model_kwargs[param_name] = trial.suggest_float(
                                    param_name, 
                                    param_spec["min"], 
                                    param_spec["max"], 
                                    log=param_spec.get("log", False)
                                )
                            elif param_spec["type"] == "categorical":
                                model_kwargs[param_name] = trial.suggest_categorical(
                                    param_name, 
                                    param_spec["choices"]
                                )
                        except Exception as e:
                            print(f"Error sampling parameter {param_name}: {str(e)}")
                            raise
                    
                    # Log parameters for this trial
                    mlflow.log_params({
                        f"trial_{trial.number}_{k}": v 
                        for k, v in model_kwargs.items()
                    })
                    
                    # Create model config
                    model_config = {
                        "model": model_name,
                        "model_kwargs": model_kwargs,
                        "dataset_kwargs": dataset_kwargs
                    }
                    
                    # Initialize predictor with error handling
                    try:
                        predictor = QlibPredictor(
                            qlib_config=self.qlib_config,
                            model_config=model_config
                        )
                    except Exception as e:
                        print(f"Error initializing predictor: {str(e)}")
                        raise
                    
                    # Train and evaluate model
                    try:
                        results = predictor.train_and_predict(
                            save_path=trial_dir,
                            track_mlflow=True
                        )
                        
                        # Extract and validate metric
                        analysis_df = results["analysis"]
                        if metric not in analysis_df.columns:
                            raise ValueError(f"Metric {metric} not found in analysis results")
                        
                        metric_value = analysis_df.loc["analysis", metric]
                        
                        # Log metric
                        mlflow.log_metric(f"trial_{trial.number}_{metric}", metric_value)
                        
                        # Save trial results
                        results_path = os.path.join(trial_dir, "results.pkl")
                        with open(results_path, "wb") as f:
                            pickle.dump(results, f)
                        
                        return metric_value
                        
                    except Exception as e:
                        print(f"Error in training/prediction: {str(e)}")
                        raise
                        
                except Exception as e:
                    print(f"Trial {trial.number} failed: {str(e)}")
                    mlflow.log_param(f"trial_{trial.number}_error", str(e))
                    return float("-inf")
                
                finally:
                    # Clean up temporary files if needed
                    if os.path.exists(trial_dir) and not any(f.endswith('.pkl') for f in os.listdir(trial_dir)):
                        import shutil
                        shutil.rmtree(trial_dir)
            
            # Run optimization with progress bar and error handling
            try:
                study.optimize(
                    objective, 
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    callbacks=[mlflow_callback],
                    show_progress_bar=True,
                    catch=(Exception,)
                )
                
                # Store best parameters
                self.best_params = study.best_params
                
                # Log optimization results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(storage_dir, f"results_{timestamp}")
                
                log_optimization_results(
                    study=study,
                    metric=metric,
                    save_path=save_path,
                    prefix="model_"
                )
                
                # Save study object
                study_path = os.path.join(save_path, "study.pkl")
                joblib.dump(study, study_path)
                mlflow.log_artifact(study_path)
                
                return self.best_params
                
            except Exception as e:
                print(f"Optimization failed: {str(e)}")
                mlflow.log_param("optimization_error", str(e))
                raise
                
        finally:
            mlflow.end_run()
            
    def get_best_trial_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about the best trial
        
        Parameters
        ----------
        model_name : str
            Name of the model
            
        Returns
        -------
        Dict[str, Any]
            Best trial information
        """
        storage_dir = os.path.join(self.base_dir, f"study_{model_name}")
        storage_path = os.path.join(storage_dir, "study.db")
        
        if not os.path.exists(storage_path):
            raise ValueError(f"No study found for model {model_name}")
        
        study = optuna.load_study(
            study_name=self.study_name,
            storage=f"sqlite:///{storage_path}"
        )
        
        best_trial = study.best_trial
        
        return {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "state": best_trial.state,
            "datetime_start": best_trial.datetime_start,
            "datetime_complete": best_trial.datetime_complete
        }


class StrategyOptimizer:
    """
    Optimizer for trading strategies
    """
    def __init__(
        self,
        prediction_path: str,
        n_trials: int = 30,
        study_name: str = "strategy_optimization"
    ):
        """
        Initialize strategy optimizer
        
        Parameters
        ----------
        prediction_path : str
            Path to prediction file
        n_trials : int
            Number of optimization trials
        study_name : str
            Optuna study name
        """
        self.prediction_path = prediction_path
        self.n_trials = n_trials
        self.study_name = study_name
        self.predictions = None
        self.load_predictions()
        
    def load_predictions(self):
        """
        Load predictions from file
        """
        try:
            # Try to load from pickle file
            pred_pkl = os.path.join(self.prediction_path, "pred.pkl")
            if os.path.exists(pred_pkl):
                with open(pred_pkl, "rb") as f:
                    self.predictions = pickle.load(f)
            else:
                # Try to load from CSV
                pred_csv = os.path.join(self.prediction_path, "pred.csv")
                if os.path.exists(pred_csv):
                    self.predictions = pd.read_csv(pred_csv, index_col=0)
                else:
                    # Check if path is a direct file
                    if os.path.isfile(self.prediction_path):
                        if self.prediction_path.endswith(".pkl"):
                            with open(self.prediction_path, "rb") as f:
                                self.predictions = pickle.load(f)
                        elif self.prediction_path.endswith(".csv"):
                            self.predictions = pd.read_csv(self.prediction_path, index_col=0)
                            
            if self.predictions is None:
                raise ValueError(f"No predictions found at {self.prediction_path}")
                
        except Exception as e:
            raise ValueError(f"Error loading predictions: {str(e)}")
    
    def optimize_topk_strategy(self, metric="sharpe") -> Dict[str, Any]:
        """
        Optimize TopkDropoutStrategy parameters
        
        Parameters
        ----------
        metric : str
            Metric to optimize (sharpe, total_return, annual_return, or max_drawdown)
            
        Returns
        -------
        Dict[str, Any]
            Best parameters
        """
        if self.predictions is None:
            raise ValueError("Predictions not loaded")
        
        # Set up MLflow tracking
        run_id = setup_mlflow_tracking("strategy_optimization_topk")
        mlflow.log_params({
            "strategy": "TopkDropoutStrategy",
            "n_trials": self.n_trials,
            "metric": metric,
            "prediction_path": self.prediction_path
        })
        
        try:
            # Create Optuna study with MLflow callback
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize" if metric != "max_drawdown" else "minimize",
                sampler=TPESampler()
            )
            
            mlflow_callback = MLflowCallback(
                tracking_uri=mlflow.get_tracking_uri(),
                metric_name=metric
            )
            
            # Define objective function
            def objective(trial):
                # Sample parameters
                topk = trial.suggest_int("topk", 5, 100)
                n_drop = trial.suggest_int("n_drop", 0, min(topk-1, 20))  # n_drop must be less than topk
                
                # Create backtester
                backtester = SimpleBacktester(
                    pred_df=self.predictions,
                    topk=topk,
                    n_drop=n_drop
                )
                
                # Run backtest
                results = backtester.backtest()
                
                # Extract metric
                return results[metric]
            
            # Run optimization with progress bar
            study.optimize(
                objective,
                n_trials=self.n_trials,
                callbacks=[mlflow_callback],
                show_progress_bar=True
            )
            
            # Log optimization results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join("results", "optimize", f"strategy_topk_{timestamp}")
            
            log_optimization_results(
                study=study,
                metric=metric,
                save_path=save_path,
                prefix="strategy_"
            )
            
            return study.best_params
            
        finally:
            mlflow.end_run()
    
    def optimize_multiple_strategies(self, strategies, metrics=["sharpe"]) -> Dict[str, Dict[str, Any]]:
        """
        Optimize multiple strategies
        
        Parameters
        ----------
        strategies : List[str]
            List of strategy names
        metrics : List[str]
            List of metrics to optimize
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Best parameters for each strategy and metric
        """
        results = {}
        
        for strategy in strategies:
            strategy_results = {}
            
            for metric in metrics:
                print(f"Optimizing {strategy} for {metric}...")
                
                if strategy == "TopkDropoutStrategy":
                    best_params = self.optimize_topk_strategy(metric=metric)
                    strategy_results[metric] = best_params
                # Add other strategy types here
                
            results[strategy] = strategy_results
            
        return results


def feature_importance_analysis(
    model_path: str,
    top_k: int = 50,
    save_path: Optional[str] = None,
    track_mlflow: bool = True
):
    """
    Analyze feature importance and select top features
    
    Parameters
    ----------
    model_path : str
        Path to the trained model
    top_k : int
        Number of top features to select
    save_path : Optional[str]
        Path to save the results
    track_mlflow : bool
        Whether to track with MLflow
        
    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    if track_mlflow:
        run_id = setup_mlflow_tracking("feature_importance_analysis")
        mlflow.log_params({
            "model_path": model_path,
            "top_k": top_k
        })
    
    try:
        # Load model recorder
        recorder = SimpleRecorder.load(model_path)
        
        if recorder.model is None:
            raise ValueError(f"No model found in {model_path}")
        
        if recorder.dataset is None:
            raise ValueError(f"No dataset found in {model_path}")
        
        # Initialize feature selector
        feature_selector = FeatureSelector(recorder.dataset)
        
        # Calculate feature importances
        importances = feature_selector.calculate_importances(recorder.model, top_k=top_k)
        
        # Select top features
        selected_features = feature_selector.select_features(top_k=top_k)
        
        # Create filtered dataset
        filtered_dataset = feature_selector.create_filtered_dataset(selected_features, save_path=save_path)
        
        # Create feature importance visualization
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
            # Create importance plot
            fig = px.bar(
                x=importances.index[:top_k],
                y=importances.values[:top_k],
                title="Feature Importances"
            )
            
            plot_path = os.path.join(save_path, "feature_importances.html")
            fig.write_html(plot_path)
            
            if track_mlflow:
                mlflow.log_artifact(plot_path)
                
                # Log top features and their importance scores
                for feature, importance in importances.head(top_k).items():
                    mlflow.log_metric(f"importance_{feature}", importance)
        
        return {
            "importances": importances,
            "selected_features": selected_features,
            "filtered_dataset": filtered_dataset
        }
        
    finally:
        if track_mlflow:
            mlflow.end_run()


def model_ensemble(
    model_paths: List[str],
    weights: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    track_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Create an ensemble of models
    
    Parameters
    ----------
    model_paths : List[str]
        List of paths to trained models
    weights : Optional[List[float]]
        Weights for each model. If None, equal weights will be used.
    save_path : Optional[str]
        Path to save the ensemble
    track_mlflow : bool
        Whether to track with MLflow
        
    Returns
    -------
    Dict[str, Any]
        Ensemble results
    """
    if track_mlflow:
        run_id = setup_mlflow_tracking("model_ensemble")
        mlflow.log_params({
            "num_models": len(model_paths),
            "use_custom_weights": weights is not None
        })
        
        # Log model paths
        for i, path in enumerate(model_paths):
            mlflow.log_param(f"model_{i}_path", path)
    
    try:
        # Check if weights are provided
        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            # Normalize weights
            weights = np.array(weights) / sum(weights)
        
        # Load models and their predictions
        models = []
        predictions = []
        
        for path in model_paths:
            recorder = SimpleRecorder.load(path)
            
            if recorder.model is None:
                print(f"Warning: No model found in {path}")
                continue
            
            if recorder.predictions is None:
                print(f"Warning: No predictions found in {path}")
                continue
            
            models.append(recorder.model)
            predictions.append(recorder.predictions)
        
        if not models or not predictions:
            raise ValueError("No valid models or predictions found")
        
        # Create weighted ensemble prediction
        ensemble_pred = None
        
        for i, pred in enumerate(predictions):
            if ensemble_pred is None:
                ensemble_pred = pred * weights[i]
            else:
                # Ensure indices match
                ensemble_pred = ensemble_pred.add(pred * weights[i], fill_value=0)
        
        if track_mlflow:
            # Log weights
            for i, weight in enumerate(weights):
                mlflow.log_metric(f"model_{i}_weight", weight)
            
            # Log ensemble prediction statistics
            mlflow.log_metrics({
                "ensemble_mean": ensemble_pred.mean().mean(),
                "ensemble_std": ensemble_pred.std().mean()
            })
            
            if save_path is not None:
                # Create correlation heatmap of model predictions
                corr_matrix = pd.concat(predictions, axis=1).corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Model Predictions Correlation"
                )
                
                plot_path = os.path.join(save_path, "prediction_correlations.html")
                fig.write_html(plot_path)
                mlflow.log_artifact(plot_path)
        
        return {
            "models": models,
            "predictions": predictions,
            "ensemble_pred": ensemble_pred,
            "weights": weights
        }
        
    finally:
        if track_mlflow:
            mlflow.end_run()


def run_model_optimization(
    qlib_config: Dict[str, Any],
    model_name: str,
    dataset_kwargs: Dict[str, Any],
    n_trials: int = 50,
    timeout: Optional[int] = None,
    metric: str = "IC",
    resume: bool = True,
    base_dir: str = "results/optimize"
) -> Dict[str, Any]:
    """
    Helper function to run model optimization
    
    Parameters
    ----------
    qlib_config : Dict[str, Any]
        Qlib configuration
    model_name : str
        Name of the model to optimize
    dataset_kwargs : Dict[str, Any]
        Dataset configuration
    n_trials : int
        Number of optimization trials
    timeout : Optional[int]
        Timeout in seconds
    metric : str
        Metric to optimize
    resume : bool
        Whether to resume previous study
    base_dir : str
        Base directory for results
        
    Returns
    -------
    Dict[str, Any]
        Optimization results including best parameters and trial info
    """
    # Initialize optimizer
    optimizer = ModelOptimizer(
        qlib_config=qlib_config,
        n_trials=n_trials,
        timeout=timeout,
        base_dir=base_dir
    )
    
    # Get parameter space for model
    param_spaces = {
        "Linear": {
            "alpha": {"type": "float", "min": 1e-6, "max": 1.0, "log": True},
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
            "max_iter": {"type": "int", "min": 100, "max": 10000},
            "tol": {"type": "float", "min": 1e-6, "max": 1e-2, "log": True}
        },
        "LGBM": {
            "num_leaves": {"type": "int", "min": 8, "max": 256},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.2, "log": True},
            "n_estimators": {"type": "int", "min": 50, "max": 500},
            "max_depth": {"type": "int", "min": 3, "max": 12},
            "subsample": {"type": "float", "min": 0.5, "max": 1.0},
            "colsample_bytree": {"type": "float", "min": 0.5, "max": 1.0}
        },
        "DEnsemble": {
            "num_models": {"type": "int", "min": 3, "max": 20},
            "ensemble_method": {"type": "categorical", "choices": ["average", "bagging"]},
            "base_model": {"type": "categorical", "choices": ["Linear", "LGBM"]}
        }
    }
    
    param_space = param_spaces.get(model_name)
    if param_space is None:
        raise ValueError(f"No parameter space defined for model {model_name}")
    
    try:
        # Run optimization
        print(f"Starting optimization for {model_name} model...")
        print(f"Number of trials: {n_trials}")
        print(f"Optimization metric: {metric}")
        print(f"Dataset config: {dataset_kwargs}")
        
        best_params = optimizer.optimize_model(
            model_name=model_name,
            param_space=param_space,
            dataset_kwargs=dataset_kwargs,
            metric=metric,
            resume_study=resume
        )
        
        # Get best trial info
        best_trial = optimizer.get_best_trial_info(model_name)
        
        print("\nOptimization completed successfully!")
        print(f"Best {metric}: {best_trial['value']:.4f}")
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
            
        return {
            "best_params": best_params,
            "best_trial": best_trial
        }
        
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        raise 