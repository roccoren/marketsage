# Market Analysis System

A comprehensive stock market analytics solution that classifies market conditions and applies appropriate trading strategies. The system uses technical indicators and multi-factor analysis to detect market conditions and recommend suitable trading approaches.

## Features

- **Market Condition Classification**
  - Trending market detection using momentum indicators
  - Range-bound market identification
  - Breakout signal detection
  - Mean reversion opportunity detection
  - Volatility analysis

- **Dynamic Strategy Selection**
  - Real-time market state monitoring
  - Volume profile analysis
  - Adaptive strategy switching
  - Confidence-based decision making

- **Modular Architecture**
  - Extensible classification system
  - Pluggable strategy framework
  - Configurable parameters

- **High-Performance Data Acquisition**
  - Parallel stock data downloading
  - Batch processing for large datasets
  - Organized data storage structure
  - Cross-market data collection

## Components

### Market Classifier (`src/market_analysis/market_classifier.py`)
Detects and classifies different market conditions using technical indicators:
- RSI (Relative Strength Index)
- ADX (Average Directional Index)
- Price gaps and momentum
- Volatility measures

### Strategy Selector (`src/market_analysis/strategy_selector.py`)
Determines appropriate trading strategies based on market conditions:
- Trend following
- Mean reversion
- Breakout trading
- Range trading
- Defensive positioning

### Market Analyzer (`src/market_analysis/market_analyzer.py`)
Main interface coordinating market analysis and strategy selection:
- Comprehensive market analysis
- Strategy parameter optimization
- State tracking and monitoring

### Data Downloader (`src/market_analysis/data_downloader.py`)
High-performance stock data acquisition tool:
- Multi-threaded parallel downloading
- Batch processing for rate limiting
- Organized ticker-specific storage
- Support for multiple markets (NYSE, NASDAQ, ASX, HKEX)

## Usage

### Market Analysis

```python
from market_analysis.market_analyzer import MarketAnalyzer

# Initialize analyzer with configuration
analyzer = MarketAnalyzer(config={
    'lookback_period': 20,
    'confidence_threshold': 0.6
})

# Analyze market data
results = analyzer.analyze_market({
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volume_data
})

# Get analysis results
print(f"Market Condition: {results['market_condition']}")
print(f"Selected Strategy: {results['selected_strategy']}")
print(f"Confidence: {results['analysis']['confidence']}")

# Get trading parameters
params = analyzer.get_trading_parameters(results['selected_strategy'])
```

### Stock Data Downloading

```python
from market_analysis.data_downloader import StockDataDownloader

# Initialize downloader
downloader = StockDataDownloader()

# Download a single stock
start_date = "2023-01-01"
data = downloader.download_stock_data("AAPL", "NASDAQ", start_date)
print(f"Downloaded {len(data['history'])} days of data")

# Download multiple stocks in parallel
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
results = downloader.download_multiple_stocks(
    tickers, 
    "NASDAQ", 
    start_date,
    parallel=True,        # Enable parallel downloading
    batch_size=10,        # Download in batches of 10 stocks
    save_daily=True       # Save individual daily files
)

# Download all stocks from a market listing
market_data = downloader.download_market_list(
    "NYSE",
    start_date="2023-01-01",
    parallel=True,
    batch_size=20
)

# Download from multiple markets at once
multiple_markets = downloader.batch_download_markets(
    ["NYSE", "NASDAQ"], 
    start_date="2023-01-01",
    parallel=True,
    batch_size=10
)
```

### Command Line Usage
The data downloader can be used from the command line with the following options:

```bash
# Download data for all markets (NYSE and NASDAQ)
python examples/market_data_download.py

# Download data for a specific market
python examples/market_data_download.py -m NYSE
python examples/market_data_download.py -m NASDAQ

# Specify a start date (default is 30 days ago)
python examples/market_data_download.py -s 2023-01-01

# Control parallel processing
python examples/market_data_download.py --no-parallel  # Disable parallel processing
python examples/market_data_download.py -b 20          # Set batch size to 20 stocks
python examples/market_data_download.py -w 8           # Set maximum worker threads to 8

# Disable daily file saving (only saves history.json)
python examples/market_data_download.py --no-daily

# Full example with all options
python examples/market_data_download.py -m NASDAQ -s 2023-01-01 -b 20 -w 8 --no-daily
```

Available command-line arguments:
- `-m`, `--market`: Market to download (NYSE, NASDAQ, or ALL) [default: ALL]
- `-s`, `--start`: Start date in YYYY-MM-DD format [default: 30 days ago]
- `-p`, `--parallel`: Use parallel downloading [default: True]
- `-np`, `--no-parallel`: Disable parallel downloading
- `-b`, `--batch-size`: Number of stocks to download in each batch [default: 10]
- `-w`, `--max-workers`: Maximum number of worker threads for parallel downloading
- `-nd`, `--no-daily`: Do not save individual daily files

## Example

See `examples/market_analysis_demo.py` for a complete demonstration of the system using real market data.

```bash
# Run the demo
python examples/market_analysis_demo.py
```

## Data Organization

The stock data is organized in the following structure:

```
data/
├── NASDAQ/              # Market directory
│   ├── AAPL/           # Ticker-specific subdirectory
│   │   ├── history.json          # Complete history for the ticker
│   │   ├── 2023-01-01.json       # Individual daily data
│   │   └── ...
│   └── MSFT/
│       ├── history.json
│       └── ...
├── NYSE/
│   └── ...
└── ASX/
    └── ...
```

Each ticker has its own subdirectory containing:
- `history.json` - Complete historical data for the ticker
- Individual daily files (e.g., `2023-01-01.json`) with data for specific dates

## Configuration

The system can be configured through the MarketAnalyzer constructor:

```python
config = {
    'lookback_period': 20,     # Period for technical indicators
    'confidence_threshold': 0.6 # Minimum confidence for strategy changes
}
analyzer = MarketAnalyzer(config=config)
```

### Data Downloader Configuration

```python
# Custom data directory and worker threads
downloader = StockDataDownloader(
    base_data_dir="/path/to/custom/data/dir",
    max_workers=8  # Control parallelism level
)
```

## Market Conditions

The system detects the following market conditions:
- `TRENDING_UP`: Strong upward price movement
- `TRENDING_DOWN`: Strong downward price movement
- `RANGE_BOUND`: Price moving within a defined range
- `BREAKOUT`: Price breaking out of established range
- `HIGH_VOLATILITY`: Increased price volatility
- `MEAN_REVERSION`: Opportunity for price to return to mean

## Trading Strategies

Available trading strategies:
- `TREND_FOLLOWING`: Follow established price trends
- `MEAN_REVERSION`: Trade reversals to mean price
- `BREAKOUT`: Capitalize on range breakouts
- `RANGE_TRADING`: Trade within price ranges
- `VOLATILITY_EXPANSION`: Adapt to high volatility
- `DEFENSIVE`: Conservative positioning

## Future Enhancements

- Additional technical indicators
- Machine learning-based classification
- Real-time market data integration
- Performance analytics and reporting
- Risk management overlay
- Portfolio-level strategy allocation

## Requirements

- Python 3.7+
- NumPy
- yfinance
- pandas
- concurrent.futures (standard library)

## Installation

```bash
git clone [repository-url]
cd market-analysis-system
pip install -r requirements.txt
```

## Performance

The parallel downloading feature significantly improves performance when downloading multiple stocks. Benchmarks show up to 5x faster downloads compared to sequential processing, depending on system capabilities and network conditions.

## License

MIT License