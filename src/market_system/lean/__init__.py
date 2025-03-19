"""
LEAN Integration Package

This package provides integration with the QuantConnect LEAN algorithmic trading framework.
"""

import logging
import os
from pathlib import Path

# Set up module-level logger
logger = logging.getLogger(__name__)

# Log import info
try:
    from clr import AddReference
    
    # Try to import LEAN namespaces
    AddReference("QuantConnect.Common")
    AddReference("QuantConnect.Algorithm")
    AddReference("QuantConnect.Indicators")
    
    from QuantConnect import *
    
    LEAN_AVAILABLE = True
    logger.info("QuantConnect LEAN framework successfully imported")
except Exception as e:
    LEAN_AVAILABLE = False
    logger.warning(f"QuantConnect LEAN framework import failed: {e}")
    logger.warning("Some features may not be available")

# Create necessary directories
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "lean")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "lean")

Path(DEFAULT_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(DEFAULT_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Package version
__version__ = "1.0.0" 