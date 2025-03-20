"""
Pipeline - Orchestrates the entire system workflow
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from stockscan.core.qlib_manager import QlibManager
from stockscan.core.feature_manager import FeatureManager
from stockscan.models.model_manager import ModelManager
from stockscan.strategies.strategy_manager import StrategyManager
from stockscan.evolution.rdagent_manager import RDAgentManager

class Pipeline:
    """
    Main pipeline for the self-evolving signal prediction system
    """
    
    def __init__(self, 
               provider_uri: str = '~/.qlib/qlib_data/cn_data',
               region: str = 'cn',
               output_dir: str = './output',
               log_dir: str = './logs',
               rdagent_api_key: Optional[str] = None,
               azure_endpoint: Optional[str] = None,
               azure_deployment: Optional[str] = None,
               azure_api_version: str = '2023-05-15',
               chat_model: str = 'gpt-4-turbo'):
        """
        Initialize the pipeline
        
        Parameters
        ----------
        provider_uri : str
            URI of the data provider
        region : str
            Region of the data
        output_dir : str
            Directory to store outputs
        log_dir : str
            Directory to store logs
        rdagent_api_key : Optional[str]
            API key for RDAgent
        azure_endpoint : Optional[str]
            Azure OpenAI endpoint URL (if None, will use OpenAI API)
        azure_deployment : Optional[str]
            Azure OpenAI deployment name (if None, will use OpenAI API)
        azure_api_version : str
            Azure OpenAI API version to use
        chat_model : str
            LLM model to use for RDAgent
        """
        # Set up logging
        self.setup_logging(log_dir)
        
        # Set up directories
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.qlib_manager = QlibManager(provider_uri=provider_uri, region=region)
        self.feature_manager = FeatureManager()
        self.model_manager = ModelManager()
        self.strategy_manager = StrategyManager()
        self.rdagent_manager = RDAgentManager(
            log_dir=os.path.join(log_dir, 'rdagent'),
            api_key=rdagent_api_key,
            chat_model=chat_model,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            azure_api_version=azure_api_version
        )
        
        # Initialize state
        self.instruments = None
        self.dataset = None
        self.model_name = None
        self.strategy_name = None
        self.iteration = 0
        self.max_iterations = 10
        self.stopping_threshold = 0.01  # 1% improvement threshold
        self.best_metrics = None
        
    def setup_logging(self, log_dir: str):
        """
        Set up logging
        
        Parameters
        ----------
        log_dir : str
            Directory to store logs
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('Pipeline')
        
    def initialize_qlib(self):
        """
        Initialize Qlib
        """
        self.logger.info("Initializing Qlib")
        self.qlib_manager.init()
        
    def prepare_data(self, 
                   instruments: Union[str, List[str]], 
                   start_time: str,
                   end_time: str,
                   fields: Optional[List[str]] = None):
        """
        Prepare data for the pipeline
        
        Parameters
        ----------
        instruments : Union[str, List[str]]
            Instruments to use
        start_time : str
            Start time for the data
        end_time : str
            End time for the data
        fields : Optional[List[str]]
            Fields to include
        """
        self.logger.info(f"Preparing data from {start_time} to {end_time}")
        
        # Initialize Qlib if not already
        if not self.qlib_manager.initialized:
            self.initialize_qlib()
            
        # Set instruments
        self.instruments = instruments
        
        # Load Alpha158 features if not provided
        if fields is None:
            self.logger.info("Loading Alpha158 features")
            self.feature_manager.load_alpha158_features()
            fields = self.feature_manager.get_feature_list()
            
        # Create dataset handler
        handler = self.qlib_manager.create_dataset_handler(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            fields=fields
        )
        
        # Define segments
        segments = {
            'train': (start_time, self._get_segment_date(start_time, end_time, 0.7)),
            'valid': (self._get_segment_date(start_time, end_time, 0.7), 
                     self._get_segment_date(start_time, end_time, 0.85)),
            'test': (self._get_segment_date(start_time, end_time, 0.85), end_time)
        }
        
        # Prepare dataset
        self.dataset = self.qlib_manager.prepare_dataset(handler, segments)
        self.logger.info(f"Dataset prepared with segments: {segments}")
        
    def evolve_factors(self, existing_factors: Optional[Dict[str, str]] = None):
        """
        Evolve factors using RDAgent
        
        Parameters
        ----------
        existing_factors : Optional[Dict[str, str]]
            Existing factors to start with
        """
        self.logger.info(f"Evolving factors (iteration {self.iteration})")
        
        # Set up data and output directories
        data_path = os.path.join(self.output_dir, 'data')
        output_path = os.path.join(self.output_dir, f'factors_iteration_{self.iteration}')
        
        # Get existing factors if not provided
        if existing_factors is None:
            existing_factors = self.feature_manager.custom_features
            
        # Get backtest results if available
        backtest_results = None
        if self.best_metrics is not None:
            backtest_results = self.best_metrics
            
        # Evolve factors
        evolved_factors = self.rdagent_manager.evolve_factors(
            data_path=data_path,
            output_path=output_path,
            existing_factors=existing_factors,
            backtest_results=backtest_results
        )
        
        # Update feature manager with evolved factors
        for name, expr in evolved_factors.items():
            self.feature_manager.add_custom_feature(name, expr)
            
        self.logger.info(f"Evolved {len(evolved_factors)} factors")
        return evolved_factors
        
    def train_model(self, model_type: str = 'lgb', **kwargs):
        """
        Train a model
        
        Parameters
        ----------
        model_type : str
            Type of model to train
        **kwargs : Dict[str, Any]
            Additional parameters for the model
        """
        model_name = f"{model_type}_model_iteration_{self.iteration}"
        self.logger.info(f"Training model {model_name}")
        
        # Create and train model
        if model_type.lower() == 'lgb':
            model = self.model_manager.create_lgb_model(model_name, **kwargs)
        elif model_type.lower() == 'xgb':
            model = self.model_manager.create_xgb_model(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Train the model
        train_score, valid_score = self.model_manager.train_model(model_name, self.dataset)
        
        self.logger.info(f"Model {model_name} trained with scores - Train: {train_score}, Valid: {valid_score}")
        
        # Save the model
        model_path = os.path.join(self.output_dir, f'models/{model_name}.pkl')
        self.model_manager.save_model(model_name, model_path)
        
        self.model_name = model_name
        return model_name, (train_score, valid_score)
        
    def optimize_model(self):
        """
        Optimize model using RDAgent
        """
        if self.model_name is None:
            self.logger.warning("No model to optimize")
            return None
            
        self.logger.info(f"Optimizing model {self.model_name}")
        
        # Set up data and output directories
        data_path = os.path.join(self.output_dir, 'data')
        output_path = os.path.join(self.output_dir, f'model_optimization_iteration_{self.iteration}')
        
        # Get model config
        model_config = self.model_manager.model_configs.get(self.model_name)
        
        # Get backtest results if available
        backtest_results = None
        if self.best_metrics is not None:
            backtest_results = self.best_metrics
            
        # Optimize model
        optimized_config = self.rdagent_manager.optimize_model(
            model_config=model_config,
            data_path=data_path,
            output_path=output_path,
            backtest_results=backtest_results
        )
        
        # Create new model with optimized config
        optimized_model_name = f"{self.model_name}_optimized"
        model = self.model_manager.create_model_from_config(optimized_model_name, optimized_config)
        
        # Train the optimized model
        train_score, valid_score = self.model_manager.train_model(optimized_model_name, self.dataset)
        
        self.logger.info(f"Optimized model {optimized_model_name} trained with scores - Train: {train_score}, Valid: {valid_score}")
        
        # Save the model
        model_path = os.path.join(self.output_dir, f'models/{optimized_model_name}.pkl')
        self.model_manager.save_model(optimized_model_name, model_path)
        
        self.model_name = optimized_model_name
        return optimized_model_name, (train_score, valid_score)
        
    def generate_signals(self, 
                       strategy_type: str = 'topk', 
                       segment: str = 'test',
                       **kwargs):
        """
        Generate signals using a strategy
        
        Parameters
        ----------
        strategy_type : str
            Type of strategy to use
        segment : str
            Segment to generate signals for
        **kwargs : Dict[str, Any]
            Additional parameters for the strategy
        """
        if self.model_name is None:
            self.logger.warning("No model for signal generation")
            return None
            
        strategy_name = f"{strategy_type}_strategy_iteration_{self.iteration}"
        self.logger.info(f"Generating signals with {strategy_name}")
        
        # Generate predictions
        predictions = self.model_manager.predict(self.model_name, self.dataset, segment)
        
        # Create strategy
        if strategy_type.lower() == 'topk':
            strategy = self.strategy_manager.create_topk_strategy(strategy_name, **kwargs)
        elif strategy_type.lower() == 'enhanced_indexing':
            strategy = self.strategy_manager.create_enhanced_indexing_strategy(
                strategy_name, market_view=predictions, **kwargs)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
            
        # Generate signals
        signals = self.strategy_manager.generate_signals(strategy_name, predictions)
        
        self.strategy_name = strategy_name
        return signals, predictions
        
    def run_backtest(self, 
                   predictions: pd.Series, 
                   benchmark: Optional[pd.Series] = None):
        """
        Run backtest with the current strategy
        
        Parameters
        ----------
        predictions : pd.Series
            Prediction scores
        benchmark : Optional[pd.Series]
            Benchmark for comparison
        """
        if self.strategy_name is None:
            self.logger.warning("No strategy for backtesting")
            return None
            
        self.logger.info(f"Running backtest with strategy {self.strategy_name}")
        
        # Run backtest
        backtest_result = self.strategy_manager.backtest_strategy(
            self.strategy_name, predictions, benchmark)
        
        # Evaluate results
        metrics = self.strategy_manager.evaluate_strategy(backtest_result)
        
        self.logger.info(f"Backtest results: {metrics}")
        
        # Save backtest results
        backtest_path = os.path.join(self.output_dir, 
                                   f'backtest_results/backtest_iteration_{self.iteration}.pkl')
        os.makedirs(os.path.dirname(backtest_path), exist_ok=True)
        pd.to_pickle(backtest_result, backtest_path)
        
        # Update best metrics
        if self.best_metrics is None:
            self.best_metrics = metrics
        elif metrics.get('sharpe_ratio', 0) > self.best_metrics.get('sharpe_ratio', 0):
            improvement = metrics.get('sharpe_ratio', 0) - self.best_metrics.get('sharpe_ratio', 0)
            self.best_metrics = metrics
            self.logger.info(f"New best Sharpe ratio: {metrics.get('sharpe_ratio', 0)} (improvement: {improvement:.4f})")
            
        return backtest_result, metrics
        
    def check_stopping_criteria(self, metrics: Dict[str, float]) -> bool:
        """
        Check if stopping criteria are met
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Current metrics
            
        Returns
        -------
        bool
            True if stopping criteria are met
        """
        # Check if maximum iterations reached
        if self.iteration >= self.max_iterations:
            self.logger.info(f"Maximum iterations ({self.max_iterations}) reached")
            return True
            
        # Check if no significant improvement
        if self.best_metrics is not None and self.iteration > 0:
            prev_sharpe = self.best_metrics.get('sharpe_ratio', 0)
            curr_sharpe = metrics.get('sharpe_ratio', 0)
            improvement = curr_sharpe - prev_sharpe
            
            if improvement < self.stopping_threshold:
                self.logger.info(f"No significant improvement (threshold: {self.stopping_threshold}, actual: {improvement:.4f})")
                return True
                
        return False
        
    def run_iteration(self, 
                    evolve_factors: bool = True, 
                    optimize_model: bool = True):
        """
        Run a single iteration of the pipeline
        
        Parameters
        ----------
        evolve_factors : bool
            Whether to evolve factors
        optimize_model : bool
            Whether to optimize the model
            
        Returns
        -------
        Dict[str, float]
            Metrics from the iteration
        """
        self.logger.info(f"Starting iteration {self.iteration}")
        
        # Evolve factors if requested
        if evolve_factors:
            evolved_factors = self.evolve_factors()
            
            # Update dataset with new features if factors evolved
            if evolved_factors:
                self.prepare_data(
                    instruments=self.instruments,
                    start_time=self.dataset.segments['train'][0],
                    end_time=self.dataset.segments['test'][1],
                    fields=self.feature_manager.get_feature_list()
                )
                
        # Train model
        model_name, scores = self.train_model()
        
        # Optimize model if requested
        if optimize_model:
            optimized_model_name, opt_scores = self.optimize_model()
            
        # Generate signals
        signals, predictions = self.generate_signals()
        
        # Run backtest
        backtest_result, metrics = self.run_backtest(predictions)
        
        # Increment iteration counter
        self.iteration += 1
        
        return metrics
        
    def run_pipeline(self, max_iterations: int = 10, stopping_threshold: float = 0.01):
        """
        Run the complete pipeline
        
        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations
        stopping_threshold : float
            Threshold for improvement to continue
            
        Returns
        -------
        Dict[str, Any]
            Final results
        """
        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold
        
        self.logger.info(f"Starting pipeline with max_iterations={max_iterations}, stopping_threshold={stopping_threshold}")
        
        while True:
            # Run iteration
            metrics = self.run_iteration()
            
            # Check stopping criteria
            if self.check_stopping_criteria(metrics):
                break
                
        self.logger.info(f"Pipeline completed after {self.iteration} iterations")
        self.logger.info(f"Best metrics: {self.best_metrics}")
        
        return {
            'iterations': self.iteration,
            'best_metrics': self.best_metrics,
            'best_model': self.model_name,
            'best_strategy': self.strategy_name
        }
        
    def _get_segment_date(self, start_date: str, end_date: str, ratio: float) -> str:
        """
        Get date at a specific ratio between start and end dates
        
        Parameters
        ----------
        start_date : str
            Start date
        end_date : str
            End date
        ratio : float
            Ratio between start and end (0 to 1)
            
        Returns
        -------
        str
            Date at the specified ratio
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        delta = end - start
        segment_date = start + delta * ratio
        return segment_date.strftime('%Y-%m-%d') 