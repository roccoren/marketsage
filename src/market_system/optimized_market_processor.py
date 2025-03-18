"""
Optimized market data processor with performance benchmarking.

Time Complexity Analysis:
- File Reading: O(n) where n is total size of all JSON files
- Data Processing: O(n log n) due to sorting operations
- Memory Optimization: O(k) where k is chunk size for streaming processing
- Parallel Processing: Reduces time by factor of p (number of processes)

Space Complexity Analysis:
- Main Memory: O(k) where k is chunk size
- Temporary Storage: O(n) for complete dataset
- Index Structures: O(m) where m is number of unique timestamps
"""

from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Union, Generator
import pandas as pd
import numpy as np
from jsonschema import validate, ValidationError
import concurrent.futures
from functools import partial
import time
import os
from datetime import datetime

from .core.base_processor import BaseProcessor, MarketDataError
from .core.utils import ProcessingStats, check_memory_usage, log_system_stats, chunked_dataframe

# Configure logging
logger = logging.getLogger(__name__)

# Constants for optimization
CHUNK_SIZE = 10000  # Records per chunk
MAX_WORKERS = max(1, os.cpu_count() - 1)  # Leave one CPU free
MEMORY_THRESHOLD = 0.75  # Maximum memory usage threshold (75%)

class PerformanceMetrics:
    """Tracks and records performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'file_read_time': 0,
            'processing_time': 0,
            'memory_usage': [],
            'total_records': 0
        }
    
    def record_memory(self):
        """Record current memory usage."""
        memory_info = log_system_stats()
        self.metrics['memory_usage'].append(memory_info['memory_used_percent'])
    
    def get_summary(self) -> Dict:
        """Get performance metrics summary."""
        if not self.metrics['memory_usage']:
            self.record_memory()
            
        return {
            'file_read_time_seconds': round(self.metrics['file_read_time'], 3),
            'processing_time_seconds': round(self.metrics['processing_time'], 3),
            'total_execution_time': round(self.metrics['file_read_time'] + self.metrics['processing_time'], 3),
            'records_per_second': round(self.metrics['total_records'] / 
                                      (self.metrics['processing_time'] + 0.001), 1),
            'peak_memory_percent': round(max(self.metrics['memory_usage']), 1),
            'avg_memory_percent': round(sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']), 1),
            'total_records': self.metrics['total_records']
        }

class OptimizedMarketProcessor(BaseProcessor):
    """
    Memory-optimized market data processor.
    
    Features:
    - Memory-efficient processing through chunking
    - Parallel processing for improved performance
    - Performance monitoring and benchmarking
    """
    
    def __init__(self, base_path: Union[str, Path]):
        """Initialize the optimized processor."""
        super().__init__(base_path)
        self.performance = PerformanceMetrics()
    
    def _check_memory_usage(self) -> bool:
        """
        Check if memory usage is below threshold.
        
        Returns:
            bool: True if memory usage is below threshold
        """
        return check_memory_usage(MEMORY_THRESHOLD)
    
    def _process_chunk(self, chunk: Dict, file_name: str) -> pd.DataFrame:
        """
        Process a chunk of market data.
        
        Args:
            chunk: JSON data chunk
            file_name: Source file name for reference
            
        Returns:
            pd.DataFrame: Processed data chunk
        """
        # Start timing
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            if isinstance(chunk, dict):
                # Single record
                df = pd.DataFrame([chunk])
            else:
                # Multiple records
                df = pd.DataFrame(chunk)
            
            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close', 'date', 'symbol', 'market']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            # Extract metadata from file name if needed
            if missing_columns and file_name:
                parts = Path(file_name).stem.split('_')
                if len(parts) >= 2:
                    if 'market' not in df.columns:
                        df['market'] = parts[0]
                    if 'symbol' not in df.columns:
                        df['symbol'] = parts[1]
            
            # Convert date column if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Track performance
            self.performance.metrics['processing_time'] += time.time() - start_time
            self.performance.metrics['total_records'] += len(df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            # Return empty DataFrame with expected schema
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'adjusted_close', 
                                        'date', 'symbol', 'market'])
    
    def process_file(self, file_path: Path) -> pd.DataFrame:
        """
        Process a single market data file.
        
        Args:
            file_path: Path to the market data file
            
        Returns:
            pd.DataFrame: Processed data
        """
        # Start timing
        start_time = time.time()
        
        try:
            # Check memory usage
            if not self._check_memory_usage():
                logger.warning("Memory usage high, waiting before processing...")
                time.sleep(1)  # Wait before continuing
            
            # Process file in chunks
            all_data = []
            for chunk in self._read_json_chunk(file_path, CHUNK_SIZE):
                chunk_df = self._process_chunk(chunk, file_path.name)
                all_data.append(chunk_df)
            
            # Combine chunks
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
            else:
                df = pd.DataFrame()
            
            # Update metrics
            self.performance.metrics['file_read_time'] += time.time() - start_time
            
            return df
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return pd.DataFrame()
    
    def process_market_data(self, market_name: str, parallel: bool = True, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Process all market data for a specific market with memory optimization.
        
        Args:
            market_name: Name of the market
            parallel: Whether to use parallel processing
            start_date: Optional start date filter
            
        Returns:
            pd.DataFrame: Processed market data
        """
        # Reset performance metrics
        self.performance = PerformanceMetrics()
        
        # Start timing
        start_time = time.time()
        self.performance.record_memory()
        
        market_dir = self.base_path / market_name
        if not market_dir.exists():
            raise MarketDataError(f"Market directory does not exist: {market_dir}")
        
        # Define processing function for symbols
        def process_symbol_files(symbol_dir: Path) -> pd.DataFrame:
            """Process all files for a symbol."""
            if not symbol_dir.is_dir():
                return pd.DataFrame()
                
            symbol_name = symbol_dir.name
            symbol_dfs = []
            
            try:
                for file_path in symbol_dir.glob('*.json'):
                    file_df = self.process_file(file_path)
                    
                    # Add symbol if not present
                    if 'symbol' not in file_df.columns:
                        file_df['symbol'] = symbol_name
                        
                    # Add market if not present
                    if 'market' not in file_df.columns:
                        file_df['market'] = market_name
                    
                    # Filter by start date if specified
                    if start_date and 'date' in file_df.columns:
                        file_df = file_df[file_df['date'] >= start_date]
                        
                    symbol_dfs.append(file_df)
                    
                # Record memory usage periodically
                self.performance.record_memory()
                
                if symbol_dfs:
                    return pd.concat(symbol_dfs, ignore_index=True)
                else:
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error processing symbol {symbol_name}: {e}")
                return pd.DataFrame()
        
        # Process all symbols for the market
        symbols = self.get_market_symbols(market_name)
        symbol_dirs = [market_dir / symbol for symbol in symbols]
        all_data = []
        
        # Log the number of symbols
        logger.info(f"Processing {len(symbol_dirs)} symbols for market {market_name}")
        
        if parallel and len(symbol_dirs) > 1:
            # Use parallel processing for multiple symbols
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Process symbols in parallel
                results = list(executor.map(process_symbol_files, symbol_dirs))
                all_data = [df for df in results if not df.empty]
        else:
            # Process sequentially
            for symbol_dir in symbol_dirs:
                symbol_df = process_symbol_files(symbol_dir)
                if not symbol_df.empty:
                    all_data.append(symbol_df)
        
        # Combine all symbol data
        if not all_data:
            raise MarketDataError(f"No valid data found for market: {market_name}")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert date to datetime if needed
        if 'date' in combined_df.columns and not pd.api.types.is_datetime64_any_dtype(combined_df['date']):
            combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        # Set index to date and symbol
        if 'date' in combined_df.columns and 'symbol' in combined_df.columns:
            combined_df = combined_df.set_index(['date', 'symbol']).sort_index()
        
        # Record final memory and timing
        self.performance.record_memory()
        self.performance.metrics['processing_time'] += time.time() - start_time
        
        return combined_df
    
    def get_performance_metrics(self) -> Dict:
        """Get processor performance metrics."""
        return self.performance.get_summary()

def benchmark_processor(data_path: Path, market_name: str) -> Dict:
    """
    Benchmark the optimized processor against a market.
    
    Args:
        data_path: Path to market data
        market_name: Name of the market to process
        
    Returns:
        Dict: Benchmark results
    """
    processor = OptimizedMarketProcessor(data_path)
    
    # Process with and without parallelization
    results = {}
    
    # Parallel processing benchmark
    start = time.time()
    df_parallel = processor.process_market_data(market_name, parallel=True)
    parallel_time = time.time() - start
    results['parallel'] = {
        'time_seconds': round(parallel_time, 3),
        'record_count': len(df_parallel),
        'performance_metrics': processor.get_performance_metrics()
    }
    
    # Sequential processing benchmark
    start = time.time()
    df_sequential = processor.process_market_data(market_name, parallel=False)
    sequential_time = time.time() - start
    results['sequential'] = {
        'time_seconds': round(sequential_time, 3),
        'record_count': len(df_sequential),
        'performance_metrics': processor.get_performance_metrics()
    }
    
    # Calculate speedup
    if sequential_time > 0:
        results['speedup_factor'] = round(sequential_time / parallel_time, 2)
    else:
        results['speedup_factor'] = 1.0
    
    return results

# Example usage and benchmarking:
"""
# Initialize processor
processor = OptimizedMarketProcessor('/path/to/data')

# Process market data with parallel processing
df = processor.process_market_data('market_name', parallel=True)

# Get performance metrics
metrics = processor.get_performance_metrics()
print(f"Processed {metrics['records_per_second']} records/second")
print(f"Peak memory usage: {metrics['peak_memory_percent']}%")

# Run benchmark
results = benchmark_processor('/path/to/data', 'market_name')
print("Benchmark results:", json.dumps(results, indent=2))
"""