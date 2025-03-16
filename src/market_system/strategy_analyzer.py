"""
Strategy analyzer module for evaluating trading strategies on market data.
Implements strategy definition, backtesting, and performance reporting.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
import os
import concurrent.futures
import time
import json
from math import ceil
import numpy as np
import pandas as pd
from .optimized_market_processor import OptimizedMarketProcessor

def safe_series_value(value: Any, default: float = 0.0) -> float:
    """
    Safely extract a float value from a pandas Series or scalar value.
    Handles numpy types and series aggregations.
    """
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return default
            value = value.iloc[0] if len(value) == 1 else value.mean()
        if isinstance(value, (np.generic, np.ndarray)):
            value = value.item() if hasattr(value, 'item') else float(value)
        return float(value) if pd.notnull(value) else default
    except (ValueError, TypeError, AttributeError):
        return default

# Helper function for safe float conversion of pandas series
def safe_float(value: Union[pd.Series, float, int, np.number], default: float = 0.0) -> float:
    """
    Safely convert various types to float.
    
    Args:
        value: Value to convert (can be Series, float, int, or numpy number)
        default: Default value to return if conversion fails
    
    Returns:
        float: Converted value or default
    """
    try:
        if isinstance(value, pd.Series):
            if value.empty:
                return default
            # For series, take mean for aggregated values, first value for individual metrics
            if len(value) > 1:
                return float(value.mean())
            return float(value.iloc[0])
        elif isinstance(value, (np.generic, np.ndarray)):
            return float(value)
        elif pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default

# Constants
MAX_WORKERS = max(1, os.cpu_count() - 1)  # Leave one CPU free

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyAnalyzer:
    """
    Analyzes trading strategies and generates performance reports.
    
    Implements:
    - Strategy evaluation
    - Performance metrics calculation
    - Report generation
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize the strategy analyzer.
        
        Args:
            base_path: Base directory containing market data
        """
        self.processor = OptimizedMarketProcessor(base_path)
        
    def _process_chunk_parallel(self, chunk_data: tuple) -> pd.DataFrame:
        """Process a data chunk in parallel."""
        df, start_idx = chunk_data
        return self._calculate_indicators(df)

    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate data quality and log any issues.
        
        Args:
            df: DataFrame to validate
        """
        quality_metrics = {
            'missing_rates': {
                col: df[col].isna().mean() * 100
                for col in ['price', 'volume', 'vwap', 'turnover']
            },
            'zero_rates': {
                col: (df[col] == 0).mean() * 100
                for col in ['price', 'volume']
            },
            'negative_rates': {
                col: (df[col] < 0).mean() * 100
                for col in ['price', 'volume', 'turnover']
            }
        }
        
        # Log data quality summary
        logger.info(
            "Data quality metrics:\n"
            f"Missing value rates: {json.dumps(quality_metrics['missing_rates'], indent=2)}\n"
            f"Zero value rates: {json.dumps(quality_metrics['zero_rates'], indent=2)}\n"
            f"Negative value rates: {json.dumps(quality_metrics['negative_rates'], indent=2)}"
        )
        
        # Check for concerning issues
        for metric_type, rates in quality_metrics.items():
            for col, rate in rates.items():
                if rate > 10:  # More than 10% problematic values
                    logger.warning(
                        f"High {metric_type} rate for {col}: {rate:.1f}%"
                    )

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators efficiently.
        Implements data validation and robust calculation methods.
        """
        # Run data quality validation
        self._validate_data_quality(df)
        # Only process active trading status
        df = df[df['trading_status'] == 'active'].copy()
        
        # Vectorized calculations using numpy for better performance
        price_arr = df['price'].values
        volume_arr = df['volume'].values
        vwap_arr = df['vwap'].values
        
        # Constants for data quality requirements
        MIN_RECORDS = 50
        MIN_TRADING_DAYS = 5
        MIN_DATA_RATIO = 0.7  # Minimum ratio of valid records required
        
        # Check data quality
        n_records = len(df)
        n_trading_days = len(df['timestamp'].unique())
        
        data_quality = {
            'total_records': n_records,
            'trading_days': n_trading_days,
            'symbols': len(df['symbol'].unique()),
            'missing_rate': df[['price', 'volume', 'vwap', 'turnover']].isna().mean().to_dict()
        }
        
        # Log data quality summary
        logger.info(
            f"Data quality check:\n"
            f"- Records: {n_records} (min: {MIN_RECORDS})\n"
            f"- Trading days: {n_trading_days} (min: {MIN_TRADING_DAYS})\n"
            f"- Missing rates: {json.dumps(data_quality['missing_rate'], indent=2)}"
        )
        
        # Decide whether to proceed with full or limited analysis
        if n_records < MIN_RECORDS or n_trading_days < MIN_TRADING_DAYS:
            logger.warning(
                f"Limited analysis mode: Insufficient data\n"
                f"Using shorter windows for indicators due to limited data points"
            )
            # Adjust windows based on available data
            ma_window = min(20, max(5, n_records // 4))
            vol_window = min(20, max(5, n_records // 4))
        else:
            ma_window = 20
            vol_window = 20
        
        # Calculate moving averages and volatility with consistent handling
        for symbol, group in df.groupby('symbol'):
            # Calculate SMAs
            df.loc[group.index, 'SMA_20'] = group['price'].rolling(window=20, min_periods=1).mean()
            df.loc[group.index, 'SMA_50'] = group['price'].rolling(window=50, min_periods=1).mean()
            
            # Calculate volatility using the same window
            df.loc[group.index, 'volatility'] = group['price'].rolling(window=20, min_periods=1).std()
            
        # Handle missing values with forward/backward fill
        fill_columns = ['SMA_20', 'SMA_50', 'volatility']
        df[fill_columns] = df.groupby('symbol')[fill_columns].transform(lambda x: x.ffill().bfill())
        # Calculate volatility with adjusted windows per symbol
        df['volatility'] = df.groupby('symbol')['price'].rolling(
            window=vol_window,
            min_periods=max(2, vol_window//2)
        ).std().reset_index(0, drop=True)

        # Calculate volume trends per symbol with enhanced validation
        try:
            # First, ensure volume data is numeric and handle any invalid values
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Fill missing values with median within each symbol
            df['volume'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.fillna(x.median() if not x.median() != x.median() else x.mean())
            )
            
            # If still have NaNs, forward/backward fill within each symbol
            if df['volume'].isna().any():
                df['volume'] = df.groupby('symbol')['volume'].fillna(method='ffill').fillna(method='bfill')
            
            # Calculate volume MA with more lenient validation
            df['volume_ma'] = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(
                    window=vol_window,
                    min_periods=max(2, vol_window//4)  # More lenient min_periods
                ).mean()
            )
            
            # Handle any remaining NaN values in volume_ma
            df['volume_ma'] = df.groupby('symbol')['volume_ma'].transform(
                lambda x: x.ffill().bfill().fillna(x.mean())
            )
            
            # Calculate volume ratio with additional safety checks
            df['volume_ratio'] = (
                df['volume'].div(df['volume_ma'].replace(0, np.nan))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)  # Use neutral value for any invalid ratios
            )
            
            # Log data quality metrics
            na_counts = {
                'volume': df['volume'].isna().sum(),
                'volume_ma': df['volume_ma'].isna().sum(),
                'volume_ratio': df['volume_ratio'].isna().sum()
            }
            
            if any(na_counts.values()):
                logger.warning(
                    "Volume metrics calculation summary:\n"
                    f"- Missing volume values: {na_counts['volume']}\n"
                    f"- Missing MA values: {na_counts['volume_ma']}\n"
                    f"- Missing ratio values: {na_counts['volume_ratio']}"
                )
            
        except Exception as e:
            logger.error(f"Error calculating volume metrics: {str(e)}")
            # Set neutral/fallback values
            df['volume_ma'] = df['volume']
            df['volume_ratio'] = 1.0
            logger.info("Using fallback values for volume metrics due to calculation error")
        
        # Add VWAP-based indicators with data quality check
        if df['vwap'].notna().mean() >= MIN_DATA_RATIO:
            df['price_to_vwap'] = df['price'].div(df['vwap']).replace([np.inf, -np.inf], np.nan)
        else:
            logger.warning("Using price-based approximation for VWAP indicators due to insufficient data")
            df['vwap'] = df['price']  # Use price as fallback
            df['price_to_vwap'] = 1.0  # Neutral value
            logger.info("Using price-based VWAP approximation for all records")
        
        # Log detailed indicator calculation summary
        indicators = ['SMA_20', 'SMA_50', 'volatility', 'volume_ma', 'price_to_vwap']
        indicator_stats = pd.DataFrame({
            'coverage': df[indicators].notna().mean(),
            'non_null': df[indicators].notna().sum(),
            'window_size': [ma_window, 50 if n_records >= 50 else ma_window,
                          vol_window, vol_window, 'N/A']
        }, index=indicators)

        logger.info(
            f"Indicator calculation details:\n"
            f"Window sizes used:\n"
            f"- Moving Average: {ma_window} periods\n"
            f"- Volatility/Volume: {vol_window} periods\n"
            f"\nData coverage by indicator:\n"
            f"{indicator_stats.to_string(float_format=lambda x: f'{x:.1%}' if isinstance(x, float) else str(x))}\n"
            f"\nAnalysis mode: {'Limited' if n_records < MIN_RECORDS else 'Full'}"
        )
        
        return df
    
    def apply_strategy_filters(self, df: pd.DataFrame, chunk_size: int = 100000) -> Dict[str, pd.DataFrame]:
        """
        Apply different trading strategy filters to the data with parallel processing.
        
        Args:
            df: Market data DataFrame
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of filtered DataFrames by strategy
        """
        import concurrent.futures
        from math import ceil
        
        # Split data into chunks for parallel processing
        num_chunks = ceil(len(df) / chunk_size)
        chunks = [(df[i*chunk_size:(i+1)*chunk_size].copy(), i)
                 for i in range(num_chunks)]
        
        # Process chunks in parallel with optimized worker count
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            try:
                processed_chunks = list(executor.map(self._process_chunk_parallel, chunks))
                logger.info(f"Processed {len(chunks)} chunks using {MAX_WORKERS} workers")
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                raise
            
        # Combine processed chunks
        df = pd.concat(processed_chunks, axis=0)
        
        # More lenient validation - only require essential indicators
        valid_mask = (
            df['price'].notna() &
            df['volume'].notna() &
            ((df['SMA_20'].notna() & df['SMA_50'].notna()) |  # Either need both SMAs
             (df['volatility'].notna()) |                      # OR volatility
             (df['volume_ratio'].notna() & df['price_to_vwap'].notna()))  # OR volume & price metrics
        )
        
        valid_df = df[valid_mask].copy()
        if len(valid_df) < 10:  # Reduced minimum requirement
            logger.warning("Insufficient valid data points after filtering NaN values")
            return {'moving_average_crossover': pd.DataFrame(),
                   'volatility_breakout': pd.DataFrame(),
                   'volume_spike': pd.DataFrame()}
        # Calculate more lenient dynamic thresholds with detailed logging
        vol_threshold = max(1.05, valid_df['volume_ratio'].quantile(0.6))  # Reduced from 0.7 to 0.6
        volatility_threshold = valid_df['volatility'].quantile(0.6)  # Reduced from 0.7 to 0.6
        turnover_threshold = valid_df['turnover'].mean() * 0.6  # Reduced from 0.8 to 0.6

        # Log threshold calculation details
        logger.info(
            "Threshold Calculation Details:\n"
            f"Volume Ratio Stats:\n"
            f"- Mean: {valid_df['volume_ratio'].mean():.2f}\n"
            f"- Median: {valid_df['volume_ratio'].median():.2f}\n"
            f"- Selected Threshold: {vol_threshold:.2f}\n"
            f"Volatility Stats:\n"
            f"- Mean: {valid_df['volatility'].mean():.2f}\n"
            f"- Median: {valid_df['volatility'].median():.2f}\n"
            f"- Selected Threshold: {volatility_threshold:.2f}\n"
            f"Turnover Stats:\n"
            f"- Mean: {valid_df['turnover'].mean():.2f}\n"
            f"- Selected Threshold: {turnover_threshold:.2f}"
        )

        # Adjust thresholds based on recent data characteristics
        vol_threshold = max(1.0, valid_df['volume_ratio'].quantile(0.5))  # Even more lenient
        volatility_threshold = valid_df['volatility'].quantile(0.5)  # Median volatility
        turnover_threshold = valid_df['turnover'].mean() * 0.5  # Reduced threshold

        # Strategy 1: Adapted Moving Average Crossover for recent data
        ma_crossover = valid_df[
            (valid_df['SMA_20'] > valid_df['SMA_50']) &
            (valid_df['volume_ratio'] > vol_threshold * 0.6) &  # Further reduced requirement
            (valid_df['price_to_vwap'].between(0.85, 1.15)) &  # Even wider VWAP range
            (valid_df['turnover'] > turnover_threshold * 0.5)   # Reduced turnover requirement
        ].copy()
        
        # Strategy 2: Adapted Volatility Breakout for recent data
        volatility_breakout = valid_df[
            (valid_df['volatility'] > volatility_threshold * 0.6) &  # Reduced volatility requirement
            (valid_df['volume'] > valid_df['volume'].mean() * 0.5) & # Lower volume threshold
            (valid_df['price'] > valid_df['price'].shift(1))         # Simple price increase
        ].copy()
        
        # Strategy 3: Adapted Volume Spike for recent data
        volume_spike = valid_df[
            (valid_df['volume_ratio'] > vol_threshold * 0.6) &      # Lower volume ratio requirement
            (valid_df['price'] > valid_df['price'].rolling(5).mean()) & # Price above short-term average
            (valid_df['turnover'] > turnover_threshold * 0.5)        # Lower turnover requirement
        ].copy()
        
        # Log threshold values for debugging
        logger.info(
            f"Strategy thresholds:\n"
            f"- Volume ratio threshold: {vol_threshold:.2f}\n"
            f"- Volatility threshold: {volatility_threshold:.2f}\n"
            f"- Turnover threshold: {turnover_threshold:.2f}"
        )
        
        # Log strategy results
        logger.info(
            f"Strategy results summary:\n"
            f"- Valid data points: {len(valid_df)}\n"
            f"- MA Crossover matches: {len(ma_crossover)}\n"
            f"- Volatility Breakout matches: {len(volatility_breakout)}\n"
            f"- Volume Spike matches: {len(volume_spike)}"
        )
        
        return {
            'moving_average_crossover': ma_crossover,
            'volatility_breakout': volatility_breakout,
            'volume_spike': volume_spike
        }
    
    def calculate_strategy_metrics(self, df: pd.DataFrame, chunk_size: int = 100000) -> Dict:
        """
        Calculate performance metrics for each strategy with parallel processing.
        
        Args:
            df: Market data DataFrame
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Dict: Strategy performance metrics
        """
        # Validate required columns
        required_columns = ['timestamp', 'price', 'volume', 'vwap', 'turnover', 'trading_status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Filter out invalid data with more flexible validation
        # First validate essential data (price and volume)
        valid_mask = (
            df['trading_status'].eq('active')
            & df['price'].notna()
            & df['volume'].notna()
        )

        # Handle VWAP and turnover more flexibly
        if df['vwap'].isna().all() or df['turnover'].isna().all():
            logger.warning("Using price-based approximations for VWAP and turnover")
            df.loc[valid_mask, 'vwap'] = df.loc[valid_mask, 'price']
            df.loc[valid_mask, 'turnover'] = df.loc[valid_mask, 'price'] * df.loc[valid_mask, 'volume']
        
        # Log validation results
        invalid_records = len(df) - valid_mask.sum()
        if invalid_records > 0:
            logger.warning(f"Found {invalid_records} invalid records that will be filtered out")
            logger.warning(f"Invalid records breakdown:")
            logger.warning(f"- Non-active trading status: {len(df[~df['trading_status'].eq('active')])}")
            logger.warning(f"- Missing prices: {df['price'].isna().sum()}")
            logger.warning(f"- Missing volumes: {df['volume'].isna().sum()}")
            logger.warning(f"- Missing VWAP: {df['vwap'].isna().sum()}")
            logger.warning(f"- Missing turnover: {df['turnover'].isna().sum()}")
        
        filtered_df = df[valid_mask].copy()
        total_records = len(filtered_df)

        if total_records == 0:
            raise ValueError(
                "No valid records found after filtering.\n"
                f"Original records: {len(df)}\n"
                f"Records removed due to validation: {len(df) - total_records}\n"
                "Please check data quality and validation criteria."
            )
        
        logger.info(
            f"Data validation complete:\n"
            f"- Original records: {len(df):,}\n"
            f"- Valid records: {total_records:,}\n"
            f"- Invalid records: {len(df) - total_records:,}\n"
            f"- Validation rate: {(total_records / len(df) * 100):.1f}%"
        )

        df = filtered_df

        strategy_results = self.apply_strategy_filters(df, chunk_size)

        n_symbols = len(df['symbol'].unique())
        n_trading_days = len(df['timestamp'].unique())
        
        # Initialize metrics with enhanced data quality information
        metrics = {
            'analysis_period': {
                'start': df['timestamp'].min().isoformat() if not df.empty else None,
                'end': df['timestamp'].max().isoformat() if not df.empty else None,
                'trading_days': n_trading_days,
                'calendar_days': (df['timestamp'].max() - df['timestamp'].min()).days + 1 if not df.empty else 0
            },
            'total_stocks_analyzed': total_records,
            'data_quality': {
                'valid_records': total_records,
                'invalid_records': len(df) - total_records,
                'validation_rate': float(total_records / len(df) * 100) if len(df) > 0 else 0,
                'symbols': n_symbols,
                'records_per_symbol': total_records / n_symbols if n_symbols > 0 else 0,
                'missing_rates': {
                    'price': float(df['price'].isna().mean() * 100),
                    'volume': float(df['volume'].isna().mean() * 100),
                    'vwap': float(df['vwap'].isna().mean() * 100),
                    'turnover': float(df['turnover'].isna().mean() * 100)
                }
            },
            'data_sufficiency': {
                'has_minimum_records': total_records >= 50,
                'has_minimum_trading_days': n_trading_days >= 5,
                'analysis_mode': 'Full' if total_records >= 50 else 'Limited'
            },
            'strategies': {}
        }

        for strategy_name, strategy_df in strategy_results.items():
            try:
                if strategy_df.empty:
                    logger.warning(f"No matches found for strategy: {strategy_name}")
                    metrics['strategies'][strategy_name] = {
                        'status': 'No matches',
                        'matches': 0,
                        'match_rate': 0,
                        'analysis_attempted': True,
                        'error': None
                    }
                    continue

                matched_records = len(strategy_df)
                
                # Calculate returns with safe operations
                strategy_df['next_day_return'] = strategy_df.groupby('symbol')['price'].pct_change(-1)
                strategy_df['vwap_return'] = strategy_df.groupby('symbol')['vwap'].pct_change(-1)
                
                # Calculate strategy metrics with safe handling of NaN values
                success_mask = strategy_df['next_day_return'].notna() & (strategy_df['next_day_return'] > 0)
                vwap_success_mask = strategy_df['vwap_return'].notna() & (strategy_df['vwap_return'] > 0)
                
                # Enhanced performance metrics
                recent_data = strategy_df.sort_values('timestamp').tail(10)  # Last 10 days
                
                # Calculate detailed performance metrics with safe value extraction
                performance_metrics = {
                    'success_rate': safe_series_value(success_mask.mean()),
                    'average_return': safe_series_value(strategy_df['next_day_return'].mean()),
                    'vwap_success_rate': safe_series_value(vwap_success_mask.mean()),
                    'winning_trades': int(safe_series_value(success_mask.sum())),
                    'total_trades': matched_records,
                    'recent_performance': {
                        'win_rate_10d': safe_series_value(success_mask.tail(10).mean()),
                        'avg_return_10d': safe_series_value(strategy_df['next_day_return'].tail(10).mean()),
                        'consistency_score': safe_series_value(
                            (success_mask.rolling(5).mean() > 0.5).mean()
                        )
                    }
                }
                
                # Enhanced statistics
                price_stats = strategy_df.groupby(strategy_df['timestamp'].dt.date)['price'].agg([
                    ('open', 'first'),
                    ('high', 'max'),
                    ('low', 'min'),
                    ('close', 'last')
                ]).tail(10)
                
                volume_stats = strategy_df.groupby(strategy_df['timestamp'].dt.date)['volume'].agg([
                    ('total', 'sum'),
                    ('avg', 'mean'),
                    ('std', 'std')
                ]).tail(10)
                
                # Technical indicators for recent data
                recent_stats = {
                    'price': {
                        'mean': float(strategy_df['price'].mean()),
                        'median': float(strategy_df['price'].median()),
                        'std': float(strategy_df['price'].std()),
                        'recent_range': {
                            'high': float(price_stats['high'].max()),
                            'low': float(price_stats['low'].min()),
                            'amplitude': float(
                                ((price_stats['high'] - price_stats['low']) /
                                 price_stats['low']).mean() * 100  # Daily amplitude %
                            )
                        }
                    },
                    'volume': {
                        'mean': float(strategy_df['volume'].mean()),
                        'median': float(strategy_df['volume'].median()),
                        'recent_profile': {
                            'avg_daily': float(volume_stats['total'].mean()),
                            'std_daily': float(volume_stats['std'].mean()),
                            'trend': float(
                                volume_stats['total'].pct_change().mean() * 100  # Volume trend %
                            )
                        }
                    },
                    'vwap': {
                        'mean': float(strategy_df['vwap'].mean()),
                        'price_to_vwap_ratio': float((strategy_df['price'] / strategy_df['vwap']).mean()),
                        'vwap_trend': float(
                            (strategy_df['price'] > strategy_df['vwap']).rolling(5).mean().tail(1).iloc[0]
                        )
                    }
                }
                
                # Industry metrics if available
                industry_metrics = {}
                if 'industry' in strategy_df.columns:
                    industry_stats = strategy_df.groupby('industry').agg({
                        'price': ['mean', 'std'],
                        'volume': 'mean',
                        'next_day_return': 'mean'
                    })
                    industry_metrics = {
                        'sector_performance': {
                            'avg_return': float(industry_stats['next_day_return']['mean'].mean()),
                            'volatility': float(industry_stats['price']['std'].mean())
                        }
                    }
                
                metrics['strategies'][strategy_name] = {
                    'status': 'Success',
                    'matches': matched_records,
                    'match_rate': matched_records / total_records,
                    'analysis_attempted': True,
                    'error': None,
                    'performance': performance_metrics,
                    'stats': recent_stats,
                    'industry_analysis': industry_metrics if industry_metrics else None,
                    'risk_metrics': {
                        'volatility': safe_series_value(
                            strategy_df['price'].pct_change().std() * np.sqrt(252)  # Annualized
                        ),
                        'drawdown': safe_series_value(
                            (strategy_df['price'].cummax() - strategy_df['price']) /
                            strategy_df['price'].cummax().replace(0, np.nan),  # Avoid division by zero
                            'max'
                        ),
                        'sharpe_ratio': safe_series_value(
                            strategy_df['next_day_return'].mean() / strategy_df['next_day_return'].std()
                            if not strategy_df['next_day_return'].empty and strategy_df['next_day_return'].std() != 0
                            else 0
                        )
                    }
                }
            except Exception as e:
                logger.error(f"Error calculating metrics for strategy {strategy_name}: {str(e)}")
                metrics['strategies'][strategy_name] = {
                    'status': 'Error',
                    'matches': 0,
                    'match_rate': 0,
                    'analysis_attempted': True,
                    'error': str(e)
                }
        
        return metrics
    
    def generate_report(self, market_name: str, chunk_size: int = 100000, start_date: Optional[datetime] = None) -> Dict:
        """
        Generate a comprehensive strategy analysis report with enhanced metrics.
        
        Args:
            market_name: Name of the market to analyze
            chunk_size: Size of chunks for parallel processing
            start_date: Optional start date for filtering data
            
        Returns:
            Dict: Comprehensive strategy report
        """
        import time
        import os

        start_time = time.time()
        try:
            logger.info(f"Generating strategy report for {market_name}")
            
            # Process market data with parallel processing
            # Note: start_date is optional, when None it processes all available data
            df = self.processor.process_market_data(
                market_name,
                parallel=True,
                start_date=start_date
            )

            # Initial validation for empty dataset
            if df is None or df.empty:
                raise ValueError(f"No market data found for {market_name}")

            # Data validation with more informative messages
            required_columns = {
                'timestamp': 'Date and time information',
                'price': 'Price data',
                'volume': 'Trading volume',
                'vwap': 'Volume Weighted Average Price',
                'turnover': 'Trading turnover',
                'trading_status': 'Market trading status'
            }
            
            missing_columns = []
            for col, description in required_columns.items():
                if col not in df.columns:
                    missing_columns.append(f"{col} ({description})")
            
            if missing_columns:
                raise ValueError(
                    f"Required columns missing in market data:\n"
                    f"{chr(10).join('- ' + col for col in missing_columns)}"
                )
            
            # Log data range information with timezone awareness
            data_start = df['timestamp'].min()
            data_end = df['timestamp'].max()
            date_filter_info = (
                f" (filtered from {start_date.date()})" if start_date else ""
            )
            
            logger.info(
                f"Processing market data for {market_name}:\n"
                f"- Date Range: {data_start.date()} to {data_end.date()}{date_filter_info}\n"
                f"- Trading Days: {len(df['timestamp'].unique())}\n"
                f"- Symbols: {len(df['symbol'].unique())}"
            )
            
            # Calculate strategy metrics with chunked processing
            metrics = self.calculate_strategy_metrics(df, chunk_size)
            
            # Ensure metrics is a dictionary
            if not isinstance(metrics, dict):
                metrics = {}
            
            try:
                # Filter active trading data
                active_data = df[df['trading_status'] == 'active']
                
                # Define aggregation function with safe operations
                def safe_agg(group):
                    """Safely aggregate group data using safe_series_value for all calculations."""
                    if len(group) == 0:
                        return pd.Series({
                            'volume_sum': 0, 'volume_mean': 0,
                            'price_min': 0, 'price_max': 0, 'price_mean': 0,
                            'price_std': 0, 'volume_std': 0,
                            'vwap_mean': 0, 'turnover_sum': 0
                        })
                    try:
                        return pd.Series({
                            'volume_sum': safe_series_value(group['volume'].sum()),
                            'volume_mean': safe_series_value(group['volume'].mean()),
                            'price_min': safe_series_value(group['price'].min()),
                            'price_max': safe_series_value(group['price'].max()),
                            'price_mean': safe_series_value(group['price'].mean()),
                            'price_std': safe_series_value(group['price'].std()),
                            'volume_std': safe_series_value(group['volume'].std()),
                            'vwap_mean': safe_series_value(group['vwap'].mean()),
                            'turnover_sum': safe_series_value(group['turnover'].sum())
                        })
                    except Exception as e:
                        logger.warning(f"Error in safe_agg: {str(e)}, returning zeros")
                        return pd.Series({
                            'volume_sum': 0, 'volume_mean': 0,
                            'price_min': 0, 'price_max': 0, 'price_mean': 0,
                            'price_std': 0, 'volume_std': 0,
                            'vwap_mean': 0, 'turnover_sum': 0
                        })

                # Calculate daily statistics with safe aggregation
                daily_stats = active_data.groupby(active_data['timestamp'].dt.date).apply(safe_agg)

                # Calculate market summary with active trading data
                metrics['market_summary'] = {
                    'total_trading_days': len(df['timestamp'].dt.date.unique()),
                    'average_daily_volume': safe_float(daily_stats['volume_sum']),
                    'price_range': {
                        'min': safe_float(daily_stats['price_min']),
                        'max': safe_float(daily_stats['price_max']),
                        'mean': safe_float(daily_stats['price_mean'])
                    }
                }

                # Add detailed market metrics
                # Calculate market detail metrics
                metrics['market_detail'] = {
                    'trading_activity': {
                        'active_days': len(daily_stats.index),
                        'total_days': len(df['timestamp'].dt.date.unique()),
                        'active_records': len(active_data),
                        'total_records': len(df),
                        'activity_rate': float(len(active_data) / len(df)) if len(df) > 0 else 0
                    },
                    'volume_analysis': {
                        'total_volume': safe_float(df['volume'].sum()),
                        'daily_stats': {
                            'mean': safe_float(daily_stats['volume_mean']),
                            'std': safe_float(daily_stats['volume_std']),
                            'max': safe_float(daily_stats['volume_sum'].max())
                        }
                    },
                    'price_analysis': {
                        'volatility': safe_float(df['price'].pct_change().std()),
                        'daily_stats': {
                            'std': safe_float(daily_stats['price_std']),
                            'mean_range': safe_float(
                                (daily_stats['price_max'] - daily_stats['price_min']).mean()
                            )
                        }
                    },
                    'turnover_analysis': {
                        'total': safe_float(active_data['turnover'].sum()),
                        'daily_stats': {
                            'mean': safe_float(daily_stats['turnover_sum'].mean()),
                            'max': safe_float(daily_stats['turnover_sum'].max())
                        }
                    }
                }

            except Exception as e:
                logger.error(f"Error calculating market summary: {str(e)}")
                metrics['market_summary'] = {
                    'total_trading_days': 0,
                    'average_daily_volume': 0,
                    'price_range': {'min': 0, 'max': 0, 'mean': 0}
                }
                # Set empty market detail with proper structure
                metrics['market_detail'] = {
                    'trading_activity': {
                        'active_days': 0,
                        'total_days': len(df['timestamp'].dt.date.unique()) if not df.empty else 0,
                        'active_records': 0,
                        'total_records': len(df),
                        'activity_rate': 0.0
                    },
                    'volume_analysis': {
                        'total_volume': 0.0,
                        'daily_stats': {'mean': 0.0, 'std': 0.0, 'max': 0.0}
                    },
                    'price_analysis': {
                        'volatility': 0.0,
                        'daily_stats': {'std': 0.0, 'mean_range': 0.0}
                    },
                    'turnover_analysis': {
                        'total': 0.0,
                        'daily_stats': {'mean': 0.0, 'max': 0.0}
                    }
                }

            # Add metadata
            metrics['report_generated'] = datetime.now().isoformat()
            metrics['report_version'] = '2.0.0'
            
            # Add performance metrics
            execution_time = time.time() - start_time
            memory_metrics = self.processor.get_performance_metrics()
            
            metrics['execution_metrics'] = {
                'timing': {
                    'total_seconds': round(execution_time, 2),
                    'records_per_second': round(len(df) / execution_time, 2) if execution_time > 0 else 0
                },
                'resources': {
                    'memory_usage_mb': memory_metrics.get('peak_memory_mb', 0),
                    'worker_count': MAX_WORKERS,
                    'chunk_size': chunk_size
                },
                'data_processed': {
                    'total_records': len(df),
                    'chunks': ceil(len(df) / chunk_size) if chunk_size > 0 else 1
                }
            }
            
            # Add report metadata
            metrics['report_metadata'] = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'version': '2.0.0',
                    'market': market_name
                },
                'data_range': {
                    'start': df['timestamp'].min().isoformat() if not df.empty else None,
                    'end': df['timestamp'].max().isoformat() if not df.empty else None,
                    'filter_applied': bool(start_date),
                    'filter_date': start_date.isoformat() if start_date else None
                },
                'data_summary': {
                    'records': len(df),
                    'symbols': len(df['symbol'].unique()),
                    'trading_days': len(df['timestamp'].dt.date.unique()),
                    'valid_ratio': metrics['data_quality']['validation_rate'] / 100
                        if 'data_quality' in metrics else 0
                }
            }
            
            logger.info(f"Successfully generated report for {market_name} in {execution_time:.2f} seconds")
            return metrics
            
        except ValueError as ve:
            logger.error(f"Validation error in report generation for {market_name}: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Error generating report for {market_name}: {str(e)}")
            logger.exception("Detailed error traceback:")
            raise