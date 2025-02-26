import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_historical_data(data_or_df) -> pd.DataFrame:
    """Clean and standardize historical price data."""
    try:
        # Convert input to DataFrame
        if isinstance(data_or_df, dict):
            if 'data' in data_or_df and isinstance(data_or_df['data'], list):
                # Add debug logging
                logger.debug(f"First data item type: {type(data_or_df['data'][0])}")
                logger.debug(f"First data item: {str(data_or_df['data'][0])[:200]}")
                
                # Convert each data item to a proper dict
                records = []
                data_items = data_or_df['data']
                for item in data_items:
                    # Skip if item is not a dict
                    if not isinstance(item, dict):
                        continue
                        
                    record = {}
                    # Split string into parts
                    item_str = str(item).strip('{}').replace("'", "")
                    parts = item_str.split()
                    record = {}
                    
                    # Parse parts in pairs (key: value)
                    for i in range(0, len(parts), 2):
                        if i + 1 < len(parts):
                            key = parts[i].lower().rstrip(':')  # Remove trailing colon
                            value = parts[i + 1].rstrip(',')    # Remove trailing comma
                            if key == 'date':
                                # Special handling for date
                                record['date'] = value
                            else:
                                try:
                                    # Convert numeric values
                                    # Strip any trailing commas before conversion
                                    record[key] = float(value.rstrip(','))
                                except ValueError:
                                    logger.debug(f"Could not convert value '{value}' for key '{key}'")
                                    continue
                    
                    if record:  # Only append if we have data
                        records.append(record)
                
                if not records:
                    logger.error("No valid records found in data")
                    return None
                    
                df = pd.DataFrame(records)
            else:
                data = data_or_df.get('data', data_or_df.get('history', []))
                df = pd.DataFrame.from_records(data)
        elif isinstance(data_or_df, pd.DataFrame):
            df = data_or_df.copy()
        else:
            df = pd.DataFrame(data_or_df)

        # Convert date to datetime and set as index
        if 'date' in df.columns:
            # Try multiple date formats
            try:
                df['Date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            except ValueError:
                try:
                    df['Date'] = pd.to_datetime(df['date'])
                except ValueError:
                    logger.error(f"Could not parse dates. Sample date: {df['date'].iloc[0]}")
                    return None
            
            df = df.set_index('Date')
            if 'date' in df.columns:
                df = df.drop('date', axis=1)
            
        # Standardize column names
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        df = df.rename(columns=column_map)
        
        # Sort and clean up the data
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # Convert numeric columns and handle missing values
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return None
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Fill missing values and clean up
        df = df.ffill()
        df = df.dropna()
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning historical data: {str(e)}")
        return None
