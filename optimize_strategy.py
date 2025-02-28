#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent / 'src')
sys.path.insert(0, src_path)

# Now we can import our modules
from run_optimization import main

if __name__ == "__main__":
    main()