import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default values
DEFAULT_INFLUXDB_URL = "http://128.85.51.203:8086"
DEFAULT_INFLUXDB_TOKEN = "your-token-here"
DEFAULT_INFLUXDB_ORG = "stockscan"
DEFAULT_INFLUXDB_BUCKET = "stocks"

def load_config(config_file: str = "config/config.json") -> dict:
    """Load configuration from a JSON file.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_file} not found, using default values")
        return {
            "influxdb": {
                "url": DEFAULT_INFLUXDB_URL,
                "token": DEFAULT_INFLUXDB_TOKEN,
                "org": DEFAULT_INFLUXDB_ORG,
                "bucket": DEFAULT_INFLUXDB_BUCKET
            }
        }
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {
            "influxdb": {
                "url": DEFAULT_INFLUXDB_URL,
                "token": DEFAULT_INFLUXDB_TOKEN,
                "org": DEFAULT_INFLUXDB_ORG,
                "bucket": DEFAULT_INFLUXDB_BUCKET
            }
        } 