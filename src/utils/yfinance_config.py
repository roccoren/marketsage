"""YFinance configuration module to set up global settings before imports"""
import urllib3
import logging
import os
import sys
import tempfile
from pathlib import Path

# Configure deeper urllib3 settings
urllib3.util.timeout.DEFAULT_TIMEOUT = 60.0  # Increased timeout for reliability
urllib3.connectionpool.DEFAULT_POOLSIZE = 50  # Reduced pool size
urllib3.connectionpool.DEFAULT_RETRIES = 5  # Match our retry strategy

# Configure urllib3 connection pooling
urllib3.poolmanager.PoolManager(
    maxsize=50,
    retries=urllib3.Retry(5, backoff_factor=2.0)
)

# Set up module-level logging
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("yfinance").setLevel(logging.INFO)  # Enable debug logging for yfinance

# Ensure cache directories exist with proper permissions
try:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    CACHE_DIR = os.path.join(ROOT_DIR, "data", "yfinance_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Make cache dir writable
    os.chmod(CACHE_DIR, 0o755)
except Exception as e:
    # If we can't create the directory, use a temporary one
    logging.warning(f"Failed to create custom cache directory: {e}")
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "yfinance_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

# Default configuration settings
DEFAULT_CONFIG = {
    # Data retrieval settings
    'start_date': '2020-01-01',  # Default start date for historical data
    'data_directory': 'data',    # Base directory for storing stock data
    
    # API settings
    'rate_limit_ms': 5000,       # Milliseconds between API calls (matches REQUEST_DELAY)
    'max_retries': 5,            # Maximum retry attempts for failed requests
    'batch_size': 3,             # Number of symbols to process in parallel
    
    # Yahoo Finance specific settings
    'yahoo_finance': {
        'use_proxy': False,      # Whether to use proxy servers
        'proxies': [],           # List of proxy servers to cycle through
        'user_agent_rotation': True,  # Rotate user agents to avoid blocks
        'cache_directory': CACHE_DIR
    },
    
    # Update settings
    'update_schedule': {
        'daily_update_hour': 18,  # Hour to perform daily updates (24-hour format)
        'daily_update_minute': 0, # Minute to perform daily updates
        'timezone': 'America/New_York'  # Timezone for updates (NYSE/NASDAQ trading)
    },
    
    # Error handling
    'max_consecutive_failures': 5,  # Maximum failures before circuit breaker trips
    'circuit_breaker_reset_time': 300,  # Seconds before circuit breaker resets
    
    # Logging
    'log_directory': 'logs',
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    
    # Data format
    'output_format': {
        'date_format': '%Y-%m-%d',
        'price_decimal_places': 4,
        'volume_decimal_places': 0,
        'include_adjusted_close': True,
        'compress_output': True  # Enable gzip compression for JSON files
    },
    
    # Symbol validation
    'symbol_validation': {
        'skip_special_securities': False,  # Skip warrants, preferred stocks, etc.
        'max_symbol_length': 20           # Maximum length of ticker symbols to process
    }
}

# Ensure required directories exist
for directory in [DEFAULT_CONFIG['data_directory'], DEFAULT_CONFIG['log_directory']]:
    os.makedirs(directory, exist_ok=True)

# Create empty file to allow importing the module
open(os.path.join(DEFAULT_CONFIG['data_directory'], '__init__.py'), 'a').close()