# StockScan Code Organization Structure

This document provides a detailed explanation of the code organization structure of the StockScan project, including the responsibilities and organization of each module.

## Overall Structure

The StockScan project adopts a modular design, breaking down different functions into independent modules to improve code maintainability and scalability.

```
stockscan/                       # Main code directory
├── core/                        # Core functionality
│   ├── __init__.py
│   ├── data/                    # Data processing core
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loader
│   │   ├── processor.py         # Data processor
│   │   └── store.py             # Data storage
│   ├── market/                  # Market related
│   │   ├── __init__.py
│   │   ├── exchange.py          # Exchange definitions
│   │   └── calendar.py          # Trading calendar
│   └── types/                   # Type definitions
│       ├── __init__.py
│       └── data_types.py        # Data type definitions
├── strategies/                  # Strategy module
│   ├── __init__.py
│   ├── base.py                  # Strategy base class
│   ├── technical/               # Technical analysis strategies
│   │   ├── __init__.py
│   │   ├── ma_crossover.py      # Moving average crossover strategy
│   │   ├── macd.py              # MACD strategy
│   │   └── multi_indicator.py   # Multi-indicator strategy
│   ├── statistical/             # Statistical strategies
│   │   ├── __init__.py
│   │   ├── mean_reversion.py    # Mean reversion strategy
│   │   └── momentum.py          # Momentum strategy
│   └── ml/                      # Machine learning strategies
│       ├── __init__.py
│       └── linear_model.py      # Linear model-based strategy
├── models/                      # Machine learning models
│   ├── __init__.py
│   ├── linear/                  # Linear models
│   │   ├── __init__.py
│   │   └── linear_regression.py # Linear regression model
│   └── tree/                    # Tree models
│       ├── __init__.py
│       └── decision_tree.py     # Decision tree model
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── technical_indicators.py  # Technical indicator calculations
│   ├── metrics.py               # Performance metrics
│   ├── visualization.py         # Visualization tools
│   ├── backtest_formatter.py    # Backtest result formatter
│   └── config.py                # Configuration utilities
├── scanner/                     # Market scanner
│   ├── __init__.py
│   ├── base_scanner.py          # Base scanner class
│   └── stock_scanner.py         # Stock market scanner
└── __init__.py                  # Package initialization
```

## Module Responsibilities

### Core Module

The core module provides the fundamental infrastructure for the entire project, including data handling, market definitions, and type systems.

#### Data Submodule

- **loader.py**: Handles loading data from various sources (APIs, CSV files, databases)
- **processor.py**: Processes raw data to prepare it for analysis (handling missing values, normalization, etc.)
- **store.py**: Manages data storage and retrieval

#### Market Submodule

- **exchange.py**: Contains exchange definitions and characteristics
- **calendar.py**: Provides trading calendar functionality (trading days, holidays, etc.)

#### Types Submodule

- **data_types.py**: Defines data types and structures used across the project

### Strategies Module

The strategies module contains various trading strategy implementations.

- **base.py**: Defines the base strategy class that all strategies inherit from
- **technical/**: Technical analysis-based strategies
  - **ma_crossover.py**: Moving average crossover strategy
  - **macd.py**: MACD-based strategy
  - **multi_indicator.py**: Strategy combining multiple technical indicators
- **statistical/**: Statistical method-based strategies
  - **mean_reversion.py**: Mean reversion strategy
  - **momentum.py**: Momentum strategy
- **ml/**: Machine learning-based strategies
  - **linear_model.py**: Strategy using linear models for prediction

### Models Module

The models module contains machine learning model implementations.

- **linear/**: Linear model implementations
  - **linear_regression.py**: Linear regression model
- **tree/**: Tree-based model implementations
  - **decision_tree.py**: Decision tree model

### Utils Module

The utils module provides utility functions and tools used across the project.

- **technical_indicators.py**: Calculations for technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **metrics.py**: Performance metric calculations (Sharpe ratio, drawdown, etc.)
- **visualization.py**: Data visualization tools
- **backtest_formatter.py**: Formats and presents backtest results
- **config.py**: Configuration management utilities

### Scanner Module

The scanner module provides market scanning functionality to find trading opportunities.

- **base_scanner.py**: Base class for all scanner implementations
- **stock_scanner.py**: Stock market scanner implementation

## Class Hierarchies

### Strategy Classes

```
Strategy (Abstract Base Class)
├── TechnicalStrategy
│   ├── MACrossoverStrategy
│   ├── MACDStrategy
│   └── MultiIndicatorStrategy
├── StatisticalStrategy
│   ├── MeanReversionStrategy
│   └── MomentumStrategy
└── MLStrategy
    └── LinearModelStrategy
```

### Scanner Classes

```
BaseScanner (Abstract Base Class)
└── StockScanner
```

## Common Design Patterns

### Factory Pattern

Used in data loading and strategy creation to abstract the creation logic:

```python
class StrategyFactory:
    @staticmethod
    def create_strategy(strategy_type, **kwargs):
        # Create and return appropriate strategy instance
```

### Observer Pattern

Used in backtesting to notify listeners about trading events:

```python
# Observer interface
class TradeObserver:
    def update(self, trade_event):
        pass

# Subject
class BacktestEngine:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify_observers(self, trade_event):
        for observer in self.observers:
            observer.update(trade_event)
```

### Strategy Pattern

Used to allow interchangeable strategy implementations:

```python
# Strategy interface
class Strategy:
    def analyze(self, data):
        pass
    
    def generate_signals(self, data):
        pass

# Concrete strategies
class MACrossoverStrategy(Strategy):
    def analyze(self, data):
        # Implementation
```

## Code Conventions

### Naming Conventions

- Classes: CamelCase (e.g., `DataLoader`, `MACrossoverStrategy`)
- Functions/Methods: snake_case (e.g., `load_data`, `calculate_rsi`)
- Variables: snake_case (e.g., `price_data`, `moving_average`)
- Constants: UPPER_CASE (e.g., `DEFAULT_PERIOD`, `CONFIG_PATH`)

### Documentation Standards

- Docstrings: Google style
- Example:
```python
def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Args:
        prices (numpy.ndarray): Array of price data
        period (int, optional): RSI period. Defaults to 14.
    
    Returns:
        numpy.ndarray: RSI values
    """
```

### Error Handling

- Use explicit exception handling
- Example:
```python
try:
    data = loader.load_data(filepath)
except FileNotFoundError:
    logger.error(f"File not found: {filepath}")
    # Handle the error
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    # Handle general errors
```

## Integration Points

### Data Sources

- YFinance API
- Alpha Vantage API
- CSV files
- SQLite/PostgreSQL databases

### External Libraries

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Technical Analysis**: ta-lib (or custom implementations)
- **Backtesting**: custom implementation

## Extension Guidelines

### Adding New Strategies

1. Create a new strategy class in the appropriate subdirectory (technical, statistical, or ml)
2. Inherit from the corresponding base class
3. Implement required methods (analyze, generate_signals, etc.)
4. Register the strategy in the factory if applicable
5. Add appropriate tests

### Adding New Data Sources

1. Implement a new loader class or extend the existing DataLoader
2. Ensure output is compatible with the existing data structures
3. Add appropriate data conversion and normalization
4. Update documentation
5. Add tests for the new data source

## Testing Guidelines

- Unit tests for each component
- Integration tests for combined functionality
- Use pytest as the testing framework
- Include test data in `tests/data/`
- Mock external API calls in tests 