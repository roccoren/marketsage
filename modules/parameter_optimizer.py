#!/usr/bin/env python3
"""
Parameter optimization module for the Qlib-based quantitative analysis project.
This module provides classes for managing and optimizing stock-specific parameters
through continuous backtesting and prediction cycles.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

from modules.storage import StorageManager
from modules.backtesting import BacktestingManager
from modules.forecasting import ForecastingManager

logger = logging.getLogger(__name__)


class StockParameterManager:
    """Manages stock-specific parameters and their optimization."""
    
    PARAM_STORE_PATH = "data/csv/stock_params"
    MAX_HISTORY_VERSIONS = 20  # Maximum number of versions to keep
    PERFORMANCE_THRESHOLD = 0.8  # 80% of best historical performance
    
    def __init__(self, storage_manager: Optional[StorageManager] = None):
        """
        Initialize the parameter manager.
        
        Args:
            storage_manager (StorageManager, optional): Storage manager for parameter persistence
        """
        self.storage_manager = storage_manager if storage_manager else StorageManager()
        self.parameter_store = {}
        self.optimization_frequency = 5  # trading days
        self.last_optimization = {}
        self.parameter_history = {}  # Store parameter history for analysis
        self.performance_history = {}  # Store performance metrics for each version
        self.best_versions = {}  # Track best performing versions per symbol
        
        # Ensure parameter history directory exists
        os.makedirs(self.PARAM_STORE_PATH, exist_ok=True)
    
    def get_stock_parameters(self, symbol: str, version: Optional[str] = None) -> Dict:
        """
        Retrieve stock-specific parameters or create default.
        
        Args:
            symbol (str): Stock symbol
            version (str, optional): Specific version of parameters to load
            
        Returns:
            Dict: Stock parameters
        """
        # Check version-specific parameters first if requested
        if version and self._has_version(symbol, version):
            return self._load_version(symbol, version)
        
        # Try to load latest parameters from storage
        if symbol not in self.parameter_store:
            stored_params = self._load_parameters(symbol)
            if stored_params:
                self.parameter_store[symbol] = stored_params
                self._add_to_history(symbol, stored_params)
            else:
                default_params = self._create_default_params()
                self.parameter_store[symbol] = default_params
                self._add_to_history(symbol, default_params)
        
        return self.parameter_store[symbol]
    
    def rollback_parameters(self, symbol: str, n_versions: int = 1) -> Dict:
        """
        Rollback parameters to a previous version.
        
        Args:
            symbol (str): Stock symbol
            n_versions (int): Number of versions to rollback
            
        Returns:
            Dict: Previous parameter version
        """
        try:
            # Get available versions
            versions = self._get_version_list(symbol)
            if not versions or len(versions) <= n_versions:
                logger.warning(f"Not enough versions to rollback {n_versions} steps for {symbol}")
                return self.get_stock_parameters(symbol)
            
            # Get target version
            target_version = versions[-(n_versions + 1)]
            previous_params = self._load_version(symbol, target_version)
            
            # Update current parameters
            self.parameter_store[symbol] = previous_params
            self._save_parameters(symbol, previous_params)
            
            logger.info(f"Rolled back parameters for {symbol} to version {target_version}")
            return previous_params
            
        except Exception as e:
            logger.error(f"Failed to rollback parameters for {symbol}: {e}")
            return self.get_stock_parameters(symbol)
    
    def update_performance(self, symbol: str, version: str, 
                         performance_metrics: Dict[str, float]) -> None:
        """
        Update performance metrics for a parameter version.
        
        Args:
            symbol (str): Stock symbol
            version (str): Parameter version
            performance_metrics (Dict[str, float]): Performance metrics
        """
        if symbol not in self.performance_history:
            self.performance_history[symbol] = {}
        
        self.performance_history[symbol][version] = {
            'metrics': performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update best version if this one performs better
        self._update_best_version(symbol, version, performance_metrics)
        
        # Save performance history
        self._save_performance_history(symbol)
    
    def should_rollback(self, symbol: str, current_performance: Dict[str, float]) -> bool:
        """
        Determine if parameters should be rolled back based on performance.
        
        Args:
            symbol (str): Stock symbol
            current_performance (Dict[str, float]): Current performance metrics
            
        Returns:
            bool: True if rollback is recommended
        """
        if symbol not in self.best_versions:
            return False
        
        best_version = self.best_versions[symbol]
        if not best_version:
            return False
        
        best_metrics = self.performance_history[symbol][best_version]['metrics']
        
        # Compare key metrics
        current_score = self._calculate_performance_score(current_performance)
        best_score = self._calculate_performance_score(best_metrics)
        
        return current_score < best_score * self.PERFORMANCE_THRESHOLD
    
    def get_best_parameters(self, symbol: str) -> Dict:
        """
        Get the best performing parameters for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Best performing parameters
        """
        if symbol in self.best_versions and self.best_versions[symbol]:
            return self._load_version(symbol, self.best_versions[symbol])
        return self.get_stock_parameters(symbol)
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score."""
        return (
            metrics.get('sharpe_ratio', 0) * 0.4 +
            metrics.get('win_rate', 0) * 0.3 +
            (1 - abs(metrics.get('max_drawdown', 1))) * 0.3
        )
    
    def _get_version_list(self, symbol: str) -> List[str]:
        """Get list of available versions for a symbol."""
        try:
            symbol_dir = os.path.join(self.PARAM_STORE_PATH, symbol)
            if not os.path.exists(symbol_dir):
                return []
            
            versions = [f[1:-5] for f in os.listdir(symbol_dir) 
                       if f.startswith('v') and f.endswith('.json')]
            return sorted(versions)
        except Exception as e:
            logger.error(f"Failed to get version list for {symbol}: {e}")
            return []
    
    def _update_best_version(self, symbol: str, version: str, metrics: Dict[str, float]):
        """Update the best performing version for a symbol."""
        if symbol not in self.best_versions or not self.best_versions[symbol]:
            self.best_versions[symbol] = version
            return
            
        current_best_metrics = self.performance_history[symbol][self.best_versions[symbol]]['metrics']
        
        current_score = self._calculate_performance_score(metrics)
        best_score = self._calculate_performance_score(current_best_metrics)
        
        if current_score > best_score:
            self.best_versions[symbol] = version
    
    def _save_performance_history(self, symbol: str) -> None:
        """Save performance history to storage."""
        try:
            history_path = os.path.join(self.PARAM_STORE_PATH, symbol, 'performance_history.json')
            with open(history_path, 'w') as f:
                json.dump({
                    'performance_history': self.performance_history.get(symbol, {}),
                    'best_version': self.best_versions.get(symbol, None)
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save performance history for {symbol}: {e}")
    
    def _create_default_params(self) -> Dict:
        """Create default parameters for a new stock."""
        return {
            'lookback_window': 252,  # Default 1 year
            'volatility_threshold': 0.2,
            'volume_threshold': 1.5,
            'momentum_window': 20,
            'prediction_horizon': 10,
            'backtest_split': 0.7,  # 70% training, 30% testing
            'optimization_metrics': {
                'accuracy': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def update_parameters(self, symbol: str, performance_metrics: Dict,
                         market_conditions: Dict) -> Dict:
        """
        Update parameters based on performance and market conditions.
        
        Args:
            symbol (str): Stock symbol
            performance_metrics (Dict): Performance metrics from recent trading
            market_conditions (Dict): Current market conditions
            
        Returns:
            Dict: Updated parameters
        """
        current_params = self.get_stock_parameters(symbol)
        
        # Check if optimization is needed
        if not self._should_optimize(symbol):
            return current_params
        
        try:
            new_params = self._optimize_parameters(
                current_params, 
                performance_metrics,
                market_conditions
            )
            
            # Update store and save
            self.parameter_store[symbol] = new_params
            self._save_parameters(symbol, new_params)
            self.last_optimization[symbol] = datetime.now()
            
            return new_params
            
        except Exception as e:
            logger.error(f"Parameter optimization failed for {symbol}: {e}")
            return current_params
    
    def _should_optimize(self, symbol: str) -> bool:
        """Check if optimization is needed based on frequency."""
        if symbol not in self.last_optimization:
            return True
        
        days_since_opt = (datetime.now() - self.last_optimization[symbol]).days
        return days_since_opt >= self.optimization_frequency
    
    def _optimize_parameters(self, current_params: Dict,
                           performance_metrics: Dict,
                           market_conditions: Dict) -> Dict:
        """
        Optimize parameters based on performance and market conditions.
        
        Args:
            current_params (Dict): Current parameters
            performance_metrics (Dict): Recent performance metrics
            market_conditions (Dict): Current market conditions
            
        Returns:
            Dict: Optimized parameters
        """
        new_params = current_params.copy()
        
        # Adjust lookback window based on volatility
        if market_conditions.get('volatility', 0) > 0.3:  # High volatility
            new_params['lookback_window'] = min(
                int(current_params['lookback_window'] * 1.2),
                756  # Max 3 years
            )
        elif market_conditions.get('volatility', 0) < 0.15:  # Low volatility
            new_params['lookback_window'] = max(
                int(current_params['lookback_window'] * 0.8),
                126  # Min 6 months
            )
        
        # Adjust prediction horizon based on trend strength
        trend_strength = abs(market_conditions.get('trend', 0))
        if trend_strength > 0.05:  # Strong trend
            new_params['prediction_horizon'] = min(
                int(current_params['prediction_horizon'] * 1.2),
                20  # Max 20 days
            )
        else:  # Weak trend
            new_params['prediction_horizon'] = max(
                int(current_params['prediction_horizon'] * 0.8),
                5  # Min 5 days
            )
        
        # Update optimization metrics
        new_params['optimization_metrics'] = {
            'accuracy': performance_metrics.get('accuracy', 0.0),
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': performance_metrics.get('max_drawdown', 0.0)
        }
        
        new_params['last_updated'] = datetime.now().isoformat()
        
        return new_params
    
    def _load_parameters(self, symbol: str) -> Optional[Dict]:
        """Load parameters from storage."""
        try:
            if self.storage_manager:
                return self.storage_manager.load_model_params(symbol, 'stock_parameters')
            return None
        except Exception as e:
            logger.warning(f"Failed to load parameters for {symbol}: {e}")
            return None
    
    def _save_parameters(self, symbol: str, params: Dict):
        """Save parameters to storage."""
        try:
            if self.storage_manager:
                # Add version information
                params['version'] = self._generate_version()
                params['timestamp'] = datetime.now().isoformat()
                
                # Save to main storage
                self.storage_manager.save_model_params(symbol, 'stock_parameters', params)
                
                # Save version to history
                self._add_to_history(symbol, params)
                
                # Save detailed version
                self._save_version(symbol, params['version'], params)
        except Exception as e:
            logger.error(f"Failed to save parameters for {symbol}: {e}")
    
    def _generate_version(self) -> str:
        """Generate version string for parameters."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _has_version(self, symbol: str, version: str) -> bool:
        """Check if a specific version exists."""
        version_path = os.path.join(self.PARAM_STORE_PATH, symbol, f"v{version}.json")
        return os.path.exists(version_path)
    
    def _load_version(self, symbol: str, version: str) -> Dict:
        """Load a specific version of parameters."""
        try:
            version_path = os.path.join(self.PARAM_STORE_PATH, symbol, f"v{version}.json")
            with open(version_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load version {version} for {symbol}: {e}")
            return self._create_default_params()
    
    def _save_version(self, symbol: str, version: str, params: Dict):
        """Save a specific version of parameters."""
        try:
            # Create symbol directory if it doesn't exist
            symbol_dir = os.path.join(self.PARAM_STORE_PATH, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save version file
            version_path = os.path.join(symbol_dir, f"v{version}.json")
            with open(version_path, 'w') as f:
                json.dump(params, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save version {version} for {symbol}: {e}")
    
    def _add_to_history(self, symbol: str, params: Dict):
        """Add parameters to history for analysis."""
        if symbol not in self.parameter_history:
            self.parameter_history[symbol] = []
        
        self.parameter_history[symbol].append({
            'timestamp': datetime.now().isoformat(),
            'params': params.copy()
        })
        
        # Keep only last 10 versions in memory
        if len(self.parameter_history[symbol]) > 10:
            self.parameter_history[symbol] = self.parameter_history[symbol][-10:]