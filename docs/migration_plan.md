# StockScan Code Migration Plan

This document provides a detailed plan for migrating the existing StockScan code to the new directory structure.

## 1. Preparation

1. Create a backup
   ```bash
   cp -r /path/to/stockscan /path/to/stockscan_backup
   ```

2. Install necessary tools
   ```bash
   pip install isort black pyyaml
   ```

3. Create the new directory structure
   ```bash
   mkdir -p config/{exchanges,strategies}
   mkdir -p data/{raw,processed,features,labels,cache}/{nyse,nasdaq,asx,hkex}
   mkdir -p stockscan/{core,strategies,models,utils,scanner}
   mkdir -p scripts
   mkdir -p examples/{backtest_examples,optimization_examples,scanner_examples}
   mkdir -p tests/{unit,integration,data}
   mkdir -p notebooks
   mkdir -p docs/api
   mkdir -p output/{backtest,optimization,scan}
   mkdir -p requirements
   ```

## 2. Code Migration Plan

### Configuration File Migration

1. Create default configuration file
   ```bash
   # Create default configuration
   cat > config/default_config.yaml << EOF
   # StockScan Default Configuration

   # Data settings
   data:
     base_dir: "data"
     default_exchange: "NYSE"
     supported_exchanges: ["NYSE", "NASDAQ", "ASX", "HKEX"]
   
   # Backtesting settings
   backtest:
     default_initial_capital: 100000.0
     default_commission: 0.0
     default_slippage: 0.0
   
   # Output settings
   output:
     base_dir: "output"
     save_backtest_results: true
     save_plots: true
   
   # Logging settings
   logging:
     level: "INFO"
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     config_file: "config/logging_config.yaml"
   EOF
   ```

2. Create logging configuration
   ```bash
   cat > config/logging_config.yaml << EOF
   version: 1
   formatters:
     standard:
       format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   handlers:
     console:
       class: logging.StreamHandler
       level: INFO
       formatter: standard
       stream: ext://sys.stdout
     file:
       class: logging.FileHandler
       level: INFO
       formatter: standard
       filename: output/stockscan.log
   loggers:
     stockscan:
       level: INFO
       handlers: [console, file]
       propagate: false
   root:
     level: INFO
     handlers: [console]
     propagate: false
   EOF
   ```

3. Create strategy configuration example
   ```bash
   cat > config/strategies/macd_config.yaml << EOF
   # MACD strategy parameter configuration
   params:
     fast_period: 12
     slow_period: 26
     signal_period: 9
     rsi_period: 14
     volume_ma_period: 20
   EOF
   
   cat > config/strategies/multi_indicator_config.yaml << EOF
   # Multi-indicator strategy parameter configuration
   params:
     sma_short_period: 20
     sma_long_period: 50
     ema_short_period: 12
     ema_long_period: 26
     rsi_period: 14
     bb_period: 20
     bb_std: 2.0
     macd_fast: 12
     macd_slow: 26
     macd_signal: 9
   EOF
   ```

### Code file migration

#### 1. Core module migration

1. Create initialization file
   ```bash
   echo '"""StockScan core module."""' > stockscan/core/__init__.py
   echo '"""StockScan package."""' > stockscan/__init__.py
   ```

2. Migrate ExchangeDataLoader to core module
   ```bash
   cp src/utils/exchange_data_loader.py stockscan/core/data_loader.py
   # Edit file, update import path and class name
   # Rename ExchangeDataLoader class to DataLoader
   ```

#### 2. Strategy module migration

1. Create initialization file
   ```bash
   echo '"""StockScan strategies module."""' > stockscan/strategies/__init__.py
   ```

2. Migrate base strategy class
   ```bash
   cp src/strategies/base.py stockscan/strategies/base.py
   # Edit file, update import path
   # from src.utils.technical_indicators import TechnicalIndicators
   # Change to
   # from stockscan.utils.technical_indicators import TechnicalIndicators
   ```

3. Migrate backtester
   ```bash
   cp src/strategies/backtester.py stockscan/strategies/backtester.py
   # Edit file, update import path
   # from .base import Strategy
   # Change to
   # from stockscan.strategies.base import Strategy
   ```

4. Migrate optimizer
   ```bash
   cp src/strategies/optimizer.py stockscan/strategies/optimizer.py
   # Edit file, update import path
   # from src.strategies.backtester import Backtester
   # Change to
   # from stockscan.strategies.backtester import Backtester
   ```

5. Migrate strategy implementation
   ```bash
   # Move MACD strategy implementation to stockscan/strategies/macd.py
   cp src/strategies/macd.py stockscan/strategies/macd.py
   # Edit file, update import path
   
   # Move multi-indicator strategy implementation to stockscan/strategies/multi_indicator.py
   cp src/strategies/multi_indicator_strategy.py stockscan/strategies/multi_indicator.py
   # Edit file, update import path
   ```

#### 3. Tool module migration

1. Create initialization file
   ```bash
   echo '"""StockScan utilities module."""' > stockscan/utils/__init__.py
   ```

2. Migrate technical indicators tool
   ```bash
   cp src/utils/technical_indicators.py stockscan/utils/technical_indicators.py
   # No need to modify import path
   ```

3. Migrate backtest formatter tool
   ```bash
   cp src/utils/backtest_formatter.py stockscan/utils/backtest_formatter.py
   # No need to modify import path
   ```

4. Create visualization tool
   ```bash
   # Extract visualization-related code from backtest_formatter.py to create visualization.py
   touch stockscan/utils/visualization.py
   # Edit file, add visualization-related functionality
   ```

5. Create metrics tool
   ```bash
   touch stockscan/utils/metrics.py
   # Edit file, add performance metrics calculation functionality
   ```

#### 4. Scanner module migration

1. Create initialization file
   ```bash
   echo '"""StockScan scanner module."""' > stockscan/scanner/__init__.py
   ```

2. Migrate scanner implementation
   ```bash
   cp src/scanner.py stockscan/scanner/stock_scanner.py
   # Edit file, update import path
   ```

3. Create base scanner class
   ```bash
   touch stockscan/scanner/base_scanner.py
   # Edit file, extract common scanner functionality
   ```

#### 5. Model module migration

1. Create initialization file
   ```bash
   echo '"""StockScan models module."""' > stockscan/models/__init__.py
   ```

2. Create linear model example
   ```bash
   touch stockscan/models/linear_model.py
   # Edit file, add linear model implementation
   ```

#### 6. Script migration

1. Create data download script
   ```bash
   touch scripts/download_data.py
   # Edit file, implement data download functionality
   ```

2. Create data update script
   ```bash
   touch scripts/update_data.py
   # Edit file, implement data update functionality
   ```

3. Create feature generation script
   ```bash
   touch scripts/generate_features.py
   # Edit file, implement feature generation functionality
   ```

#### 7. Example code migration

1. Create backtest example
   ```bash
   touch examples/backtest_examples/simple_macd.py
   touch examples/backtest_examples/multi_indicator.py
   # Edit file, implement example code
   ```

2. Create optimization example
   ```bash
   touch examples/optimization_examples/parameter_tuning.py
   # Edit file, implement example code
   ```

3. Create scan example
   ```bash
   touch examples/scanner_examples/scan_nyse.py
   # Edit file, implement example code
   ```

#### 8. Test code migration

1. Create unit test
   ```bash
   touch tests/unit/test_indicators.py
   touch tests/unit/test_strategies.py
   touch tests/unit/test_utils.py
   # Edit file, implement unit test
   ```

2. Create integration test
   ```bash
   touch tests/integration/test_backtester.py
   touch tests/integration/test_scanner.py
   # Edit file, implement integration test
   ```

3. Prepare test data
   ```bash
   # Create example test data
   touch tests/data/sample_stock_data.csv
   # Add test data
   ```

#### 9. Main entry script migration

1. Migrate run_backtest.py
   ```bash
   cp run_backtest.py ./
   # Edit file, update import path
   # from src.utils.exchange_data_loader import ExchangeDataLoader
   # from src.strategies.backtester import Backtester
   # from src.utils.backtest_formatter import display_results
   # Change to
   # from stockscan.core.data_loader import DataLoader
   # from stockscan.strategies.backtester import Backtester
   # from stockscan.utils.backtest_formatter import display_results
   ```

#### 10. Set script and requirement files

1. Create setup.py
   ```bash
   cat > setup.py << EOF
   from setuptools import setup, find_packages

   setup(
       name="stockscan",
       version="0.1.0",
       packages=find_packages(),
       include_package_data=True,
       install_requires=[
           "pandas>=1.3.0",
           "numpy>=1.20.0",
           "matplotlib>=3.4.0",
           "scipy>=1.7.0",
           "scikit-learn>=1.0.0",
       ],
       entry_points={
           'console_scripts': [
               'stockscan-backtest=run_backtest:main',
           ],
       },
   )
   EOF
   ```

2. Separate requirement files
   ```bash
   # Base dependency
   cat > requirements/base.txt << EOF
   pandas>=1.3.0
   numpy>=1.20.0
   matplotlib>=3.4.0
   scipy>=1.7.0
   scikit-learn>=1.0.0
   pyyaml>=6.0
   requests>=2.26.0
   yfinance>=0.1.70
   pyarrow>=5.0.0
   EOF
   
   # Development dependency
   cat > requirements/dev.txt << EOF
   -r base.txt
   pytest>=6.2.5
   pytest-cov>=2.12.1
   flake8>=3.9.2
   black>=21.9b0
   isort>=5.9.3
   scikit-optimize>=0.9.0
   EOF
   
   # Documentation dependency
   cat > requirements/docs.txt << EOF
   -r base.txt
   sphinx>=4.0.0
   sphinx-rtd-theme>=1.0.0
   EOF
   ```

3. Create pyproject.toml
   ```bash
   cat > pyproject.toml << EOF
   [build-system]
   requires = ["setuptools>=42", "wheel"]
   build-backend = "setuptools.build_meta"

   [tool.black]
   line-length = 88
   include = '\.pyi?$'
   exclude = '''
   /(
       \.git
     | \.hg
     | \.mypy_cache
     | \.tox
     | \.venv
     | _build
     | buck-out
     | build
     | dist
   )/
   '''

   [tool.isort]
   profile = "black"
   line_length = 88
   EOF
   ```

## 3. Import path update

Use the following script to find and replace all import paths:

```bash
#!/bin/bash
# update_imports.sh
# Usage: ./update_imports.sh

# Update src.* imports to stockscan.*
find stockscan -type f -name "*.py" | xargs sed -i 's/from src\./from stockscan./g'
find stockscan -type f -name "*.py" | xargs sed -i 's/import src\./import stockscan./g'

# Update relative imports
find stockscan -type f -name "*.py" | xargs sed -i 's/from \.utils/from stockscan.utils/g'
find stockscan -type f -name "*.py" | xargs sed -i 's/from \.strategies/from stockscan.strategies/g'
find stockscan -type f -name "*.py" | xargs sed -i 's/from \.core/from stockscan.core/g'
find stockscan -type f -name "*.py" | xargs sed -i 's/from \.scanner/from stockscan.scanner/g'
find stockscan -type f -name "*.py" | xargs sed -i 's/from \.models/from stockscan.models/g'

# Update imports in scripts
find scripts -type f -name "*.py" | xargs sed -i 's/from src\./from stockscan./g'
find scripts -type f -name "*.py" | xargs sed -i 's/import src\./import stockscan./g'

# Update imports in examples
find examples -type f -name "*.py" | xargs sed -i 's/from src\./from stockscan./g'
find examples -type f -name "*.py" | xargs sed -i 's/import src\./import stockscan./g'

# Update imports in tests
find tests -type f -name "*.py" | xargs sed -i 's/from src\./from stockscan./g'
find tests -type f -name "*.py" | xargs sed -i 's/import src\./import stockscan./g'

echo "Import path update completed"
```

## 4. Code formatting

Use the following script to format all Python code:

```bash
#!/bin/bash
# format_code.sh
# Usage: ./format_code.sh

# Run isort
isort stockscan scripts examples tests

# Run black
black stockscan scripts examples tests

echo "Code formatting completed"
```

## 5. Documentation update

1. Update README.md to reflect new directory structure and usage method

2. Create basic documentation
   ```bash
   cat > docs/getting_started.md << EOF
   # Getting Started Guide
   
   This document introduces how to install and start using StockScan for stock analysis and backtesting.
   
   ## Installation
   
   ```bash
   git clone https://github.com/yourusername/stockscan.git
   cd stockscan
   pip install -e .
   ```
   
   ## Basic Usage
   
   ### Run Backtest
   
   ```bash
   python run_backtest.py backtest --strategy macd --exchange NYSE --symbol AAPL
   ```
   
   ### More Examples
   
   Refer to example code in \`examples\` directory.
   EOF
   ```

## 6. Verification migration

After migration is complete, run the following command to verify:

1. Install package (development mode)
   ```bash
   pip install -e .
   ```

2. Run tests
   ```bash
   pytest tests/
   ```

3. Run examples
   ```bash
   python examples/backtest_examples/simple_macd.py
   ```

4. Verify command line tool
   ```bash
   python run_backtest.py backtest --strategy macd --exchange NYSE --symbol AAPL
   ```

## 7. Follow-up work

1. Create new branch and commit code
   ```bash
   git checkout -b restructure
   git add .
   git commit -m "Restructure project according to Qlib best practices"
   ```

2. Update documentation and complete examples

3. Add new features and adjust structure as needed

4. Consider adding continuous integration (CI) configuration 