# StockScan Directory Structure Reorganization Plan

Following Qlib's best practices and user requirements, here is the recommended directory structure:

```
stockscan/
├── config/                       # Configuration files directory
│   ├── default_config.yaml       # Default configuration
│   ├── logging_config.yaml       # Logging configuration
│   ├── exchanges/                # Exchange-specific configurations
│   │   ├── nyse.yaml
│   │   ├── nasdaq.yaml
│   │   ├── asx.yaml
│   │   └── hkex.yaml
│   └── strategies/               # Strategy parameter configurations
│       ├── macd_config.yaml
│       └── multi_indicator_config.yaml
│
├── data/                         # Data directory (organized according to Qlib practices)
│   ├── raw/                      # Raw data
│   │   ├── nyse/
│   │   ├── nasdaq/
│   │   ├── asx/
│   │   └── hkex/
│   ├── processed/                # Preprocessed data
│   │   ├── nyse/
│   │   ├── nasdaq/
│   │   ├── asx/
│   │   └── hkex/
│   ├── features/                 # Feature data
│   ├── labels/                   # Label data
│   └── cache/                    # Cached data
│
├── stockscan/                    # Main code directory (replacing src)
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── data_processor.py
│   │   └── market.py
│   ├── strategies/               # Strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py               # Base strategy class
│   │   ├── backtester.py         # Backtester
│   │   ├── optimizer.py          # Optimizer
│   │   ├── macd.py               # MACD strategy
│   │   └── multi_indicator.py    # Multi-indicator strategy
│   ├── models/                   # Machine learning models
│   │   ├── __init__.py
│   │   └── linear_model.py
│   ├── utils/                    # Utility classes
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── backtest_formatter.py
│   │   └── exchange_data_loader.py
│   ├── scanner/                  # Scanner related
│   │   ├── __init__.py
│   │   ├── base_scanner.py
│   │   └── stock_scanner.py
│   └── __init__.py
│
├── scripts/                      # Utility scripts
│   ├── download_data.py
│   ├── update_data.py
│   └── generate_features.py
│
├── examples/                     # Example code
│   ├── backtest_examples/
│   │   ├── simple_macd.py
│   │   └── multi_indicator.py
│   ├── optimization_examples/
│   │   └── parameter_tuning.py
│   └── scanner_examples/
│       └── scan_nyse.py
│
├── tests/                        # Test code
│   ├── unit/                     # Unit tests
│   │   ├── test_indicators.py
│   │   ├── test_strategies.py
│   │   └── test_utils.py
│   ├── integration/              # Integration tests
│   │   ├── test_backtester.py
│   │   └── test_scanner.py
│   └── data/                     # Test data
│       └── sample_stock_data.csv
│
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── strategy_evaluation.ipynb
│
├── docs/                         # Documentation
│   ├── api/                      # API documentation
│   ├── getting_started.md
│   ├── strategies.md
│   └── backtesting.md
│
├── output/                       # Output directory (backtest results, charts, etc.)
│   ├── backtest/
│   ├── optimization/
│   └── scan/
│
├── requirements/                 # Dependency management
│   ├── base.txt                  # Base dependencies
│   ├── dev.txt                   # Development dependencies
│   └── docs.txt                  # Documentation dependencies
│
├── .gitignore                    # Git ignore file
├── README.md                     # Project description
├── setup.py                      # Installation script
├── LICENSE                       # License
├── run_backtest.py               # Main entry script
└── pyproject.toml                # Project configuration
```

## Directory Descriptions

### config/
Contains all configuration files, including default configurations, logging configurations, exchange-specific configurations, and strategy parameter configurations. YAML format is used for easy reading and modification.

### data/
Data directory organized according to Qlib's best practices:
- `raw/`: Stores raw market data
- `processed/`: Preprocessed data
- `features/`: Extracted feature data
- `labels/`: Label data
- `cache/`: Cached data

### stockscan/
Main code directory (replacing the original src), organized by functional modules:
- `core/`: Core functionality implementation
- `strategies/`: Strategy implementations
- `models/`: Machine learning models
- `utils/`: Utility classes and helper functions
- `scanner/`: Scanner-related implementations

### scripts/
Utility scripts for data downloading, updating, feature generation, etc.

### examples/
Example code showing how to use various features of the system.

### tests/
Test code, including unit tests and integration tests.

### notebooks/
Jupyter notebooks for exploratory analysis and strategy evaluation.

### docs/
Project documentation, including API documentation and user guides.

### output/
Output directory for storing backtesting results, optimization results, and scanning results.

### requirements/
Dependency management, divided into base dependencies, development dependencies, and documentation dependencies.

## Implementation Recommendations

1. Create the new directory structure
2. Migrate existing code to the new directory structure
3. Update import paths
4. Update configuration files
5. Adjust test code 