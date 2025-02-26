"""YFinance configuration module to set up global settings before imports"""
import urllib3
import logging

# Configure deeper urllib3 settings
urllib3.util.timeout.DEFAULT_TIMEOUT = 15.0  # Increase default timeout
urllib3.connectionpool.DEFAULT_POOLSIZE = 50  # Increase default pool size
urllib3.connectionpool.DEFAULT_RETRIES = 3

# Set up module-level logging
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.ERROR)  # Enable debug logging for yfinance