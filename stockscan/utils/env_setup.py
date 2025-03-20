"""
Environment setup utilities
"""
import os
import logging
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import importlib.util
import pandas as pd
import numpy as np
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EnvSetup')

def setup_environment():
    """Set up the environment for stockscan"""
    # Check and set up OpenAI API key
    check_openai_api_key()
    
    # Check for RDAgent
    check_rdagent()
    
    # Check for Qlib data
    check_qlib_data('cn')  # Chinese stocks (default Qlib data)
    check_qlib_data('us')  # US stocks

def check_openai_api_key():
    """Check if OpenAI API key is set and warn if not"""
    if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY'].strip():
        logger.warning("OPENAI_API_KEY environment variable not set. RDAgent will not work properly.")
        return False
    
    # Check if using Azure OpenAI
    is_azure = ('OPENAI_API_TYPE' in os.environ and os.environ['OPENAI_API_TYPE'] == 'azure')
    
    if is_azure:
        # Check for required Azure OpenAI environment variables
        missing_vars = []
        if 'OPENAI_API_BASE' not in os.environ or not os.environ['OPENAI_API_BASE'].strip():
            missing_vars.append('OPENAI_API_BASE')
        
        if 'OPENAI_API_VERSION' not in os.environ or not os.environ['OPENAI_API_VERSION'].strip():
            missing_vars.append('OPENAI_API_VERSION')
            
        if 'OPENAI_DEPLOYMENT_NAME' not in os.environ or not os.environ['OPENAI_DEPLOYMENT_NAME'].strip():
            missing_vars.append('OPENAI_DEPLOYMENT_NAME')
            
        if missing_vars:
            logger.warning(f"Using Azure OpenAI but missing required environment variables: {', '.join(missing_vars)}")
            logger.warning("You can set these using command line arguments or environment variables.")
            return False
        else:
            logger.info(f"Azure OpenAI configuration detected with deployment: {os.environ['OPENAI_DEPLOYMENT_NAME']}")
    else:
        logger.info("OpenAI API key found. RDAgent features will be available.")
    
    return True

def check_rdagent():
    """Check if RDAgent is installed"""
    try:
        import rdagent
        return True
    except ImportError:
        logger.warning("RDAgent is not available. Please install it with 'pip install rdagent'.")
        return False

def check_qlib_data(region='cn'):
    """Check if Qlib data is available for the specified region
    
    Args:
        region (str): The region to check data for: 'cn' or 'us'
        
    Returns:
        bool: True if data is available, False otherwise
    """
    # Define data paths based on region
    if region == 'cn':
        data_path = os.path.expanduser('~/.qlib/qlib_data/cn_data')
    else:  # 'us'
        data_path = os.path.expanduser('~/.qlib/qlib_data/us_data')
    
    # Check if directory exists
    if not os.path.exists(data_path):
        logger.warning(f"Qlib data not found at {data_path}.")
        print_download_instructions(region)
        return False
    
    # Check for key files that should exist in a valid Qlib dataset
    instruments_file = os.path.join(data_path, 'instruments.pkl')
    calendar_file = os.path.join(data_path, 'calendar.pkl')
    
    if not os.path.exists(instruments_file) or not os.path.exists(calendar_file):
        logger.warning(f"Qlib data at {data_path} appears incomplete (instruments.pkl or calendar.pkl not found).")
        logger.info("To ensure the data is properly downloaded, run:")
        if region == 'us':
            logger.info("For US-Stock data: python scripts/download_us_data.py")
        else:
            logger.info("For China-Stock data: python scripts/download_cn_data.py")
        return False
        
    # Additional check: verify if raw data directory contains stock data
    raw_dir = os.path.join(data_path, 'raw')
    if os.path.exists(raw_dir):
        # Check if there are any stock directories inside raw/
        stock_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        if not stock_dirs:
            logger.warning(f"No stock data found in {raw_dir}")
            print_download_instructions(region)
            return False
    else:
        logger.warning(f"Raw data directory {raw_dir} not found")
        print_download_instructions(region)
        return False
    
    # Data appears valid
    logger.info(f"Qlib data for {region} region found at {data_path}")
    return True

def print_download_instructions(region):
    """Print instructions for downloading Qlib data"""
    logger.info("To download Qlib data, run one of the following commands:")
    if region == 'us':
        logger.info("For US-Stock data: python scripts/download_us_data.py")
    else:
        logger.info("For China-Stock data: python scripts/download_cn_data.py")

def install_qlib_data(region='us'):
    """Install Qlib data for the specified region
    
    Args:
        region (str): The region to install data for: 'cn' or 'us'
        
    Returns:
        bool: True if installation successful, False otherwise
    """
    # Only support US region in this version
    if region != 'us':
        logger.error(f"Region {region} is not supported for automatic installation.")
        logger.info("Only 'us' region is supported at this time.")
        return False
    
    try:
        logger.info(f"Installing Qlib data for {region} region...")
        import subprocess
        
        # Run the appropriate download script
        if region == 'us':
            subprocess.run(['python', 'scripts/download_us_data.py'], check=True)
        
        # Check if data is now available
        return check_qlib_data(region)
    except Exception as e:
        logger.error(f"Error installing Qlib data: {e}")
        return False 