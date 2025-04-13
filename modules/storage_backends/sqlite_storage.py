#!/usr/bin/env python3
"""
SQLite storage backend for the Qlib-based quantitative analysis project.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional

import pandas as pd

from modules.storage import StorageBackend


class SQLiteStorage(StorageBackend):
    """Storage backend using SQLite database."""
    
    def __init__(self, db_path: str = "data/sqlite/stocks.db"):
        """
        Initialize SQLite storage.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stock data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(symbol, date)
        )
        ''')
        
        # Create model parameters table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_name TEXT NOT NULL,
            params TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(symbol, model_name)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data for a symbol to SQLite.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Data to save
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Reset index to make date a column
            df = data.reset_index()
            df['symbol'] = symbol
            
            # Rename columns to match table schema
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Select only needed columns
            columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
            df = df[columns]
            
            # Convert date to string if it's a datetime
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Save to database
            df.to_sql('stock_data', conn, if_exists='replace', index=False)
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a symbol from SQLite.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date for filtering
            end_date (str, optional): End date for filtering
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query
            query = f"SELECT date, open, high, low, close, volume FROM stock_data WHERE symbol = '{symbol}'"
            
            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
                
            query += " ORDER BY date"
            
            # Load data
            df = pd.read_sql_query(query, conn, parse_dates=['date'])
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Set date as index
            df = df.set_index('date')
            
            # Rename columns to match yfinance format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_model_params(self, symbol: str, model_name: str, params: Dict[str, Any]) -> bool:
        """
        Save model parameters to SQLite.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Name of the model
            params (Dict[str, Any]): Model parameters
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert params to JSON string
            params_json = json.dumps(params, default=str)
            created_at = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert or replace
            cursor.execute('''
            INSERT OR REPLACE INTO model_params (symbol, model_name, params, created_at)
            VALUES (?, ?, ?, ?)
            ''', (symbol, model_name, params_json, created_at))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving model parameters for {symbol}/{model_name}: {e}")
            return False
    
    def load_model_params(self, symbol: str, model_name: str) -> Dict[str, Any]:
        """
        Load model parameters from SQLite.
        
        Args:
            symbol (str): Stock symbol
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Model parameters
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT params FROM model_params
            WHERE symbol = ? AND model_name = ?
            ''', (symbol, model_name))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            else:
                return {}
        except Exception as e:
            print(f"Error loading model parameters for {symbol}/{model_name}: {e}")
            return {}
    
    def list_symbols(self) -> List[str]:
        """
        List all available symbols in SQLite.
        
        Returns:
            List[str]: List of symbols
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT symbol FROM stock_data')
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
        except Exception as e:
            print(f"Error listing symbols: {e}")
            return []
    
    def list_models(self, symbol: str) -> List[str]:
        """
        List all available models for a symbol in SQLite.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            List[str]: List of model names
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT DISTINCT model_name FROM model_params
            WHERE symbol = ?
            ''', (symbol,))
            
            models = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return models
        except Exception as e:
            print(f"Error listing models for {symbol}: {e}")
            return []