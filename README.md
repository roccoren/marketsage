# StockScan

StockScan is a stock analysis and backtesting system for developing, testing, and evaluating quantitative trading strategies. The project follows modular design principles, organized based on Qlib's best practices to improve code maintainability and scalability.

## Features

- **Data Management**: Support for multi-exchange data loading, processing, and storage
- **Technical Indicators**: Rich technical analysis indicator calculation tools
- **Strategy Development**: Flexible strategy development framework supporting various types of strategies
- **Backtesting System**: Precise backtesting functionality to evaluate strategy performance
- **Parameter Optimization**: Automatic optimization of strategy parameters to improve effectiveness
- **Market Scanning**: Scan the market based on strategy signals to find trading opportunities
- **Visualization**: Intuitive display of backtesting results and technical indicators

## Directory Structure

```
stockscan/
├── config/                       # Configuration files directory
│   ├── default_config.yaml       # Default configuration
│   ├── logging_config.yaml       # Logging configuration
│   ├── exchanges/                # Exchange-specific configurations
│   └── strategies/               # Strategy parameter configurations
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   ├── processed/                # Preprocessed data
│   ├── features/                 # Feature data
│   ├── labels/                   # Label data
│   └── cache/                    # Cached data
│
├── stockscan/                    # Main code directory
│   ├── core/                     # Core functionality
│   │   ├── data/                 # Data processing core
│   │   ├── market/               # Market related
│   │   └── types/                # Type definitions
│   ├── strategies/               # Strategy implementations
│   │   ├── technical/            # Technical analysis strategies
│   │   ├── statistical/          # Statistical strategies
│   │   └── ml/                   # Machine learning strategies
│   ├── models/                   # Machine learning models
│   │   ├── linear/               # Linear models
│   │   └── tree/                 # Tree models
│   ├── utils/                    # Utility classes
│   └── scanner/                  # Scanner related
│
├── scripts/                      # Utility scripts
├── examples/                     # Example code
├── tests/                        # Test code
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── output/                       # Output directory
│
├── requirements/                 # Dependency management
├── .gitignore                    # Git ignore file
├── README.md                     # Project description
├── setup.py                      # Installation script
└── run_backtest.py               # Main entry script
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Installation Steps

1. Clone the repository

```bash
git clone https://github.com/yourusername/stockscan.git
cd stockscan
```

2. Install dependencies

```bash
pip install -r requirements/dev.txt  # Install development dependencies
```

3. Development mode installation

```bash
pip install -e .  # Install in development mode
```

## Usage

### Data Download

```bash
python scripts/download_data.py --exchange NYSE --symbols AAPL,MSFT,GOOGL --start-date 2020-01-01
```

### Run Backtesting

```bash
python run_backtest.py backtest --strategy macd --exchange NYSE --symbol AAPL
```

### Parameter Optimization

```bash
python run_backtest.py optimize --strategy macd --exchange NYSE --symbol AAPL
```

### Market Scanning

```bash
python run_backtest.py scan --strategy macd --exchange NYSE --date 2023-07-15
```

## Examples

Example code is located in the `examples` directory:

- `examples/backtest_examples/`: Backtesting examples
- `examples/optimization_examples/`: Optimization examples
- `examples/scanner_examples/`: Scanning examples

## Documentation

Detailed documentation is located in the `docs` directory:

- `docs/getting_started.md`: Getting started guide
- `docs/strategies.md`: Strategy development guide
- `docs/backtesting.md`: Backtesting system description
- `docs/api/`: API documentation

## Development Guide

### Adding New Strategies

1. Create a new strategy class in `stockscan/strategies`, inheriting from the `Strategy` base class
2. Implement the `analyze` method to generate trading signals
3. Add strategy parameter configuration in `config/strategies/`

### Running Tests

```bash
pytest tests/  # Run all tests
pytest tests/unit/  # Run unit tests only
```

### Code Formatting

```bash
# Use automated script to format code
python scripts/format_code.py
```

## Project Reorganization

This project has recently undergone code restructuring, adopting a more modular and maintainable structure. If you are using an older version of the code, please use the migration script to update:

```bash
python scripts/migrate_code.py --backup
```

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details