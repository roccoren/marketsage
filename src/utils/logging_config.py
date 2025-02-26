import logging
import os
import json
from typing import Optional

def configure_logging(
    logger_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    debug_mode: bool = False
) -> logging.Logger:
    """
    Configure a logger with the specified settings.
    
    Args:
        logger_name: Name of the logger to configure
        log_level: Logging level (default: INFO)
        log_file: Optional file path for logging
        debug_mode: Whether to enable debug mode (default: False)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set base level to allow all logs in debug mode
    logger.setLevel(logging.DEBUG if debug_mode else log_level.upper())
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    if debug_mode:
        # In debug mode, add detailed file handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Add detailed console handler for debug mode
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
    else:
        # In production mode, only add console handler for INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Optional file logging for production
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger

def load_debug_config() -> dict:
    """Load debug configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config.get('debug', {})
    except Exception:
        return {}