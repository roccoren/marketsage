# MarketSage DataHub Examples

This directory contains example scripts demonstrating how to use the MarketSage DataHub component.

## Available Examples

### Basic Usage

`basic_usage.py` - Demonstrates the basic functionality of the DataHub component:
- Initializing the components
- Downloading stock data
- Preprocessing the data
- Accessing the processed data

To run this example:

```bash
python examples/basic_usage.py
```

## Creating Your Own Scripts

You can use these examples as a starting point for creating your own scripts. Here's a template:

```python
#!/usr/bin/env python3
"""
Example script for [your purpose here].
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import DataHub components
from marketsage_datahub.acquisition import DataAcquisition
from marketsage_datahub.symbols import SymbolManager
from marketsage_datahub.storage import StorageManager
from marketsage_datahub.preprocessing import DataPreprocessor

def main():
    """Main function."""
    # Initialize components
    symbol_manager = SymbolManager()
    storage_manager = StorageManager()
    data_acquisition = DataAcquisition(storage_manager=storage_manager)
    
    # Your code here
    
if __name__ == "__main__":
    main()
```

## Integration with Other MarketSage Components

These examples focus on the DataHub component in isolation. For examples of integrating with other MarketSage components, see the main MarketSage documentation.