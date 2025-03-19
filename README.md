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
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   ├── processed/                # Preprocessed data
│   ├── features/                 # Feature data
│   ├── labels/                   # Label data
│   └── cache/                    # Cached data
│
├── stockscan/                    # Main code directory
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Main application entry point
│   ├── py.typed                  # Type hint marker file
│   ├── core/                     # Core functionality
│   │   ├── __init__.py           # Package initialization
│   │   ├── data_loader.py        # Data loading functionality
│   │   ├── manager.py            # Core management functions
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
├── tests/                        # Test code
├── output/                       # Output directory
├── logs/                         # Log files directory
│
├── .gitignore                    # Git ignore file
├── .cursorignore                 # Cursor editor ignore file
├── .dockerignore                 # Docker ignore file
├── Dockerfile                    # Docker container definition
├── pyproject.toml                # Python project configuration
├── README.md                     # Project description
├── setup.py                      # Installation script
├── run.py                        # Unified CLI entry point
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
pip install -e .  # Install in development mode
```

## Usage

### Unified Command Line Interface

StockScan provides a unified command line interface for all operations:

```bash
# Run backtesting
python run.py backtest --strategy macd --symbols AAPL MSFT --market nasdaq

# Run optimization
python run.py optimize --strategy macd --symbols AAPL --market nyse --n-trials 100

# Download market data
python run.py download --symbols-file data/symbols/nasdaq.txt --start-date 2020-01-01

# Scan stocks with a strategy
python run.py scan --strategy macd --market nasdaq
```

### Running the Stock Scanner

```bash
python -m stockscan.main --exchanges HKEX --strategies rsi_stoch,atr_ma,adx_macd_cci,obv
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

## Development Guide

### Docker Usage

You can run StockScan in a Docker container:

```bash
# Build the Docker image
docker build -t stockscan .

# Run the container
docker run -it stockscan backtest --strategy macd --exchange NYSE --symbol AAPL

# Run with volume mounts for data persistence
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  stockscan scan --strategy macd --exchange NYSE --date 2023-07-15
```

### Adding New Strategies

1. Create a new strategy class in `stockscan/strategies`, inheriting from the `Strategy` base class
2. Implement the `analyze` method to generate trading signals

### Running Tests

```bash
pytest tests/  # Run all tests
pytest tests/unit/  # Run unit tests only
```

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details