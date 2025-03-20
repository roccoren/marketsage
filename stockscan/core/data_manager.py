"""
Data manager for downloading and preparing stock data
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple

logger = logging.getLogger(__name__)

def get_stock_data(ticker: str, days: int = 365, provider: str = 'yfinance') -> pd.DataFrame:
    """
    Get historical stock data for a given ticker
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    days : int
        Number of days of historical data to retrieve
    provider : str
        Data provider ('yfinance' or 'qlib')
        
    Returns
    -------
    data : pd.DataFrame
        DataFrame containing historical stock data
    """
    if provider.lower() == 'yfinance':
        return get_yfinance_data(ticker, days)
    elif provider.lower() == 'qlib':
        return get_qlib_data(ticker, days)
    else:
        logger.warning(f"Unknown data provider: {provider}")
        return get_yfinance_data(ticker, days)  # Fallback to yfinance

def get_yfinance_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Get historical stock data using yfinance
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    days : int
        Number of days of historical data to retrieve
        
    Returns
    -------
    data : pd.DataFrame
        DataFrame containing historical stock data
    """
    try:
        import yfinance as yf
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Downloading {ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Download data with auto_adjust=False to get both Close and Adj Close
        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=False,
            progress=False
        )
        
        if data.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return pd.DataFrame()
            
        logger.info(f"Retrieved {len(data)} days of data for {ticker}")
        
        # Log the actual data shape and column names
        logger.debug(f"Data shape: {data.shape}")
        logger.debug(f"Column names: {data.columns.tolist()}")
        
        # Try an alternative column name format approach
        # Handle multi-index columns from newer yfinance versions
        if isinstance(data.columns, pd.MultiIndex):
            # Create a more descriptive mapping that preserves information
            flat_columns = []
            for col in data.columns:
                # Join the levels with an underscore
                flat_columns.append(f"{col[0]}_{col[1]}" if len(col) > 1 else col[0])
            
            # Set the new column names
            data.columns = flat_columns
            logger.debug(f"Flattened column names: {data.columns.tolist()}")
        
        # Create an expanded mapping of possible column names to standardized names
        column_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 
            'Volume': 'volume', 'Adj Close': 'adj_close',
            'Open_Open': 'open', 'High_High': 'high', 'Low_Low': 'low', 
            'Close_Close': 'close', 'Volume_Volume': 'volume', 
            'Adj Close_Adj Close': 'adj_close', 'Adj_Close': 'adj_close',
            'Adj Close_adj close': 'adj_close', 'Volume_volume': 'volume'
        }
        
        # Rename columns to align with standardized format, but only for columns that exist
        new_columns = {}
        for old_col in data.columns:
            if old_col in column_map:
                new_columns[old_col] = column_map[old_col]
        
        if new_columns:
            data = data.rename(columns=new_columns)
            logger.debug(f"Renamed columns: {data.columns.tolist()}")
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns for {ticker}: {missing_columns}")
            
            # Handle missing columns with fallbacks
            for col in missing_columns:
                if col == 'close' and 'adj_close' in data.columns:
                    data['close'] = data['adj_close']
                elif col != 'volume':  # For other price columns, use close as fallback
                    if 'close' in data.columns:
                        data[col] = data['close']
                    elif 'adj_close' in data.columns:
                        data[col] = data['adj_close']
                else:  # For volume, use 0 as fallback
                    data['volume'] = 0
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Important: convert index to normal column for consistent handling
        data = data.reset_index()
        # Ensure the date column has a standardized name
        if 'Date' in data.columns and 'date' not in data.columns:
            data = data.rename(columns={'Date': 'date'})
        
        return data
    except ImportError:
        logger.warning("yfinance not installed. Install with: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error retrieving data for {ticker}: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return pd.DataFrame()

def get_qlib_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Get historical stock data using qlib
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    days : int
        Number of days of historical data to retrieve
        
    Returns
    -------
    data : pd.DataFrame
        DataFrame containing historical stock data
    """
    try:
        from qlib.data import D
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get data from qlib
        fields = ['$close', '$open', '$high', '$low', '$volume']
        data = D.features(
            [ticker],
            fields,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return pd.DataFrame()
        
        # Rename columns to standardized format
        data = data.reset_index()
        data = data.rename(columns={
            '$close': 'close',
            '$open': 'open',
            '$high': 'high',
            '$low': 'low',
            '$volume': 'volume'
        })
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        return data
    except ImportError:
        logger.warning("qlib not installed. Falling back to yfinance.")
        return get_yfinance_data(ticker, days)
    except Exception as e:
        logger.error(f"Error retrieving data for {ticker} with qlib: {e}")
        return get_yfinance_data(ticker, days)  # Fallback to yfinance

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock data
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing stock data
        
    Returns
    -------
    data : pd.DataFrame
        DataFrame with technical indicators added
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    try:
        # Ensure index is datetime if it's not already
        if isinstance(df.index, pd.DatetimeIndex):
            # If we have DatetimeIndex, ensure it's accessible as a column as well
            df = df.reset_index()
            date_col = df.columns[0]  # The first column will be the former index
        
        # Check for a date column (from a reset index or existing)
        date_cols = [col for col in df.columns if 'date' in str(col).lower() or 'time' in str(col).lower()]
        
        if date_cols:
            date_col = date_cols[0]
        else:
            # If no date column exists, but we have a DatetimeIndex, use that
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                date_col = df.columns[0]
            else:
                # If we have no date column and no DatetimeIndex, just create a date column
                date_col = 'date'
                df[date_col] = pd.date_range(start=datetime.now() - timedelta(days=len(df)), periods=len(df))
        
        # Ensure required columns exist
        required_cols = ['close', 'open', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Use close for price columns if missing
                if col in ['open', 'high', 'low'] and 'close' in df.columns:
                    df[col] = df['close']
                # Use 0 for volume if missing
                elif col == 'volume':
                    df[col] = 0
                # If close is missing but we have adj_close, use that
                elif col == 'close' and 'adj_close' in df.columns:
                    df[col] = df['adj_close']
                # Otherwise create a column with random values (only for development)
                else:
                    df[col] = np.random.rand(len(df)) * 100
                    logger.warning(f"Created synthetic {col} column because original was missing")
        
        # Moving averages
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # Average True Range (ATR)
        tr1 = abs(df['high'] - df['low'])
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        # Ensure date column is datetime type
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
            
        # Extract date features if we have a valid date column
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['month'] = df[date_col].dt.month
            df['quarter'] = df[date_col].dt.quarter
        
        # Clean up NaN values using modern methods
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)
        
        return df
    except Exception as e:
        logger.warning(f"Error adding technical indicators: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return data  # Return original data if there's an error 