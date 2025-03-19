"""
Parameter Storage Module for Trading Strategies.

This module provides functionality to store and retrieve strategy parameters
for individual symbols, allowing for symbol-specific parameter optimization.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/strategy_parameters.db"

class ParameterStorage:
    """
    Storage for trading strategy parameters with symbol-specific settings.
    
    This class provides methods to:
    1. Store strategy parameters for specific symbols
    2. Retrieve parameters for a symbol, falling back to defaults
    3. Update parameters based on optimization results
    4. Track performance of parameter sets
    """
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize parameter storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_database()
        
    def _ensure_db_dir(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _init_database(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create parameters table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    UNIQUE(strategy_name, symbol)
                )
                ''')
                
                # Create performance tracking table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS parameter_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    parameters_id INTEGER NOT NULL,
                    backtest_period TEXT NOT NULL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parameters_id) REFERENCES strategy_parameters(id)
                )
                ''')
                
                # Create index for faster lookups
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_strat_symbol ON strategy_parameters(strategy_name, symbol)')
                
                conn.commit()
                logger.info(f"Initialized parameter storage database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing parameter database: {str(e)}")
            raise
    
    def get_parameters(self, strategy_name: str, symbol: str, default_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get parameters for a specific strategy and symbol.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            default_params: Default parameters to return if no custom parameters exist
            
        Returns:
            Dictionary of parameters
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Query for symbol-specific parameters
                cursor.execute(
                    'SELECT parameters FROM strategy_parameters WHERE strategy_name = ? AND symbol = ? AND is_active = 1',
                    (strategy_name, symbol)
                )
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row['parameters'])
                
                # If no symbol-specific parameters, check for strategy defaults (empty symbol)
                cursor.execute(
                    'SELECT parameters FROM strategy_parameters WHERE strategy_name = ? AND symbol = "" AND is_active = 1',
                    (strategy_name,)
                )
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row['parameters'])
                    
                # Return provided defaults if no stored parameters found
                return default_params or {}
                
        except Exception as e:
            logger.error(f"Error retrieving parameters for {strategy_name}/{symbol}: {str(e)}")
            return default_params or {}
    
    def set_parameters(self, strategy_name: str, symbol: str, parameters: Dict[str, Any]) -> bool:
        """
        Store parameters for a specific strategy and symbol.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            parameters: Dictionary of parameters
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert parameters to JSON
                params_json = json.dumps(parameters)
                
                # Use UPSERT pattern (INSERT OR REPLACE)
                cursor.execute('''
                INSERT OR REPLACE INTO strategy_parameters 
                (strategy_name, symbol, parameters, updated_at) 
                VALUES (?, ?, ?, datetime('now'))
                ''', (strategy_name, symbol, params_json))
                
                conn.commit()
                logger.info(f"Stored parameters for {strategy_name}/{symbol}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing parameters for {strategy_name}/{symbol}: {str(e)}")
            return False
    
    def record_performance(self, strategy_name: str, symbol: str, backtest_period: str, 
                          metrics: Dict[str, Any]) -> bool:
        """
        Record performance for a parameter set.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            backtest_period: Period of backtest (e.g., "20240101-20240310")
            metrics: Dictionary of performance metrics
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the parameters_id
                cursor.execute(
                    'SELECT id FROM strategy_parameters WHERE strategy_name = ? AND symbol = ? AND is_active = 1',
                    (strategy_name, symbol)
                )
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"No active parameters found for {strategy_name}/{symbol}")
                    return False
                
                parameters_id = row[0]
                
                # Insert performance record
                cursor.execute('''
                INSERT INTO parameter_performance 
                (strategy_name, symbol, parameters_id, backtest_period, 
                 total_return, sharpe_ratio, max_drawdown, win_rate, total_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy_name, 
                    symbol, 
                    parameters_id,
                    backtest_period,
                    metrics.get('total_return', 0.0),
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('max_drawdown', 0.0),
                    metrics.get('win_rate', 0.0),
                    metrics.get('total_trades', 0)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error recording performance for {strategy_name}/{symbol}: {str(e)}")
            return False
    
    def get_best_parameters_by_metric(self, strategy_name: str, metric: str = 'total_return', 
                                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get best performing parameters by a specific metric.
        
        Args:
            strategy_name: Name of the strategy
            metric: Metric to rank by (total_return, sharpe_ratio, win_rate)
            limit: Number of results to return
            
        Returns:
            List of parameter sets with performance metrics
        """
        valid_metrics = {'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'}
        if metric not in valid_metrics:
            metric = 'total_return'
            
        order_direction = 'DESC' if metric != 'max_drawdown' else 'ASC'
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = f'''
                SELECT p.symbol, p.parameters, pp.{metric}, pp.total_return, 
                       pp.sharpe_ratio, pp.max_drawdown, pp.win_rate, pp.total_trades
                FROM parameter_performance pp
                JOIN strategy_parameters p ON pp.parameters_id = p.id
                WHERE pp.strategy_name = ? AND p.is_active = 1
                ORDER BY pp.{metric} {order_direction}
                LIMIT ?
                '''
                
                cursor.execute(query, (strategy_name, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'symbol': row['symbol'],
                        'parameters': json.loads(row['parameters']),
                        'total_return': row['total_return'],
                        'sharpe_ratio': row['sharpe_ratio'],
                        'max_drawdown': row['max_drawdown'],
                        'win_rate': row['win_rate'],
                        'total_trades': row['total_trades']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving best parameters for {strategy_name}: {str(e)}")
            return []
    
    def get_all_symbols_with_parameters(self, strategy_name: str) -> List[str]:
        """
        Get all symbols that have custom parameters for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            List of symbols
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    'SELECT DISTINCT symbol FROM strategy_parameters WHERE strategy_name = ? AND symbol != "" AND is_active = 1',
                    (strategy_name,)
                )
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error retrieving symbols with parameters for {strategy_name}: {str(e)}")
            return []
    
    def optimize_parameters(self, strategy_name: str, symbol: str, 
                           parameter_ranges: Dict[str, List], 
                           evaluation_func: callable) -> Dict[str, Any]:
        """
        Simple parameter optimization by grid search.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            parameter_ranges: Dictionary of parameter names to lists of possible values
            evaluation_func: Function that takes parameters and returns performance metrics
            
        Returns:
            Best parameters found
        """
        # This is just a placeholder - a real implementation would do a proper grid search
        # or use more sophisticated optimization techniques
        logger.warning("Parameter optimization is not fully implemented - returning base parameters")
        
        # Get current parameters as a starting point
        current_params = self.get_parameters(strategy_name, symbol, {})
        return current_params
        
    def bulk_import_from_json(self, json_file_path: str) -> int:
        """
        Import parameters from a JSON file.
        
        Args:
            json_file_path: Path to JSON file with parameters
            
        Returns:
            Number of parameter sets imported
        """
        try:
            if not os.path.exists(json_file_path):
                logger.error(f"JSON file not found: {json_file_path}")
                return 0
                
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.error(f"Invalid JSON format in {json_file_path}. Expected a list.")
                return 0
                
            count = 0
            for item in data:
                strategy_name = item.get('strategy_name')
                symbol = item.get('symbol')
                parameters = item.get('parameters')
                
                if not all([strategy_name, symbol, parameters]):
                    logger.warning(f"Skipping invalid entry: {item}")
                    continue
                    
                if self.set_parameters(strategy_name, symbol, parameters):
                    count += 1
                    
            return count
                
        except Exception as e:
            logger.error(f"Error importing parameters from JSON: {str(e)}")
            return 0
        
    def export_to_json(self, output_file_path: str, strategy_name: Optional[str] = None) -> bool:
        """
        Export parameters to a JSON file.
        
        Args:
            output_file_path: Path to output JSON file
            strategy_name: Optional strategy name to filter by
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if strategy_name:
                    cursor.execute(
                        'SELECT strategy_name, symbol, parameters FROM strategy_parameters WHERE strategy_name = ? AND is_active = 1',
                        (strategy_name,)
                    )
                else:
                    cursor.execute(
                        'SELECT strategy_name, symbol, parameters FROM strategy_parameters WHERE is_active = 1'
                    )
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'strategy_name': row['strategy_name'],
                        'symbol': row['symbol'],
                        'parameters': json.loads(row['parameters'])
                    })
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                with open(output_file_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Exported {len(results)} parameter sets to {output_file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting parameters to JSON: {str(e)}")
            return False 