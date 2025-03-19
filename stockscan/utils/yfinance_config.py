"""YFinance configuration module to set up global settings before imports"""
import urllib3
import logging

# Configure deeper urllib3 settings
urllib3.util.timeout.DEFAULT_TIMEOUT = 20.0  # Increased timeout for reliability
urllib3.connectionpool.DEFAULT_POOLSIZE = 100  # Match aiohttp connection pool
urllib3.connectionpool.DEFAULT_RETRIES = 5  # More retries for resilience

# Configure urllib3 connection pooling
urllib3.poolmanager.PoolManager(
    maxsize=100,
    retries=urllib3.Retry(5, backoff_factor=0.5)
)

# Set up module-level logging
logging.getLogger("urllib3").setLevel(logging.INFO)
logging.getLogger("yfinance").setLevel(logging.INFO)  # Enable debug logging for yfinance