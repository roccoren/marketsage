#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the absolute path of the project root directory
project_root = Path(__file__).parent.resolve()

# Add the project root to Python path so we can import from src
sys.path.append(str(project_root))

# Change working directory to project root
os.chdir(project_root)

# Now we can import our modules
from src.backtest_multi_indicator import main

if __name__ == "__main__":
    main()