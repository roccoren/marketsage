# StockScan Backtesting System

A comprehensive backtesting framework for testing trading strategies across multiple markets.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Management](#data-management)
  - [Fetching Data Periodically](#fetching-data-periodically)
  - [Data Structure](#data-structure)
- [Running Backtests](#running-backtests)
  - [Single Symbol Backtest](#single-symbol-backtest)
  - [Market-wide Backtests](#market-wide-backtests)
  - [Supported Markets](#supported-markets)
- [Strategies](#strategies)
  - [Built-in Strategies](#built-in-strategies)
  - [Adding Custom Strategies](#adding-custom-strategies)
  - [Strategy Filtering](#strategy-filtering)
- [Performance Analysis](#performance-analysis)
- [Configuration](#configuration)
- [Examples](#examples)

## Overview

StockScan is a powerful backtesting system designed to evaluate trading strategies across multiple markets. It supports various timeframes, parallel execution, and comprehensive performance reporting.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stockscan.git
cd stockscan

# Install dependencies
pip install -r requirements.txt
```

## Data Management

### Fetching Data Periodically

The system supports periodic data fetching to keep your market data up-to-date. You can set up automated data fetching using the following approaches:

#### 1. Using the Data Fetcher Script

```bash
# Fetch data for specific symbols
python data_fetcher.py --market hkex --symbols 0700.HK 9988.HK --start-date 2023-01-01

# Fetch data for an entire market
python data_fetcher.py --market nasdaq --all-symbols --start-date 2023-01-01
```

#### 2. Setting Up Scheduled Data Updates

You can set up a cron job (Linux/Mac) or Task Scheduler (Windows) to run the data fetcher periodically:

**Linux/Mac (crontab):**
```
# Run daily at 6 PM
0 18 * * * cd /path/to/stockscan && python data_fetcher.py --market hkex --all-symbols --days 1
```

**Windows (Task Scheduler):**
Create a batch file `update_data.bat`:
```batch
cd C:\path\to\stockscan
python data_fetcher.py --market hkex --all-symbols --days 1
```

Then schedule this batch file to run daily using Task Scheduler.

#### 3. Programmatic Data Updates

You can also update data programmatically:

```python
from src.market_system.data.data_fetcher import DataFetcher

# Initialize data fetcher
fetcher = DataFetcher(base_path="data")

# Update data for specific market and symbols
fetcher.update_data(
    market="hkex",
    symbols=["0700.HK", "9988.HK"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    timeframe="1d"
)
```

### Data Structure

The system stores market data in a structured format:

```
data/
├── hkex/
│   ├── 0700.HK/
│   │   ├── 2023-01-01.json
│   │   ├── 2023-01-02.json
│   │   └── ...
│   ├── 9988.HK/
│   │   └── ...
├── nasdaq/
│   └── ...
├── nyse/
│   └── ...
└── asx/
    └── ...
```

Each JSON file contains OHLCV data and additional market information for a specific date.

## Running Backtests

### Single Symbol Backtest

To run a backtest for a single symbol:

```bash
python market_backtest.py --symbols 0700.HK --start-date 2023-01-01 --end-date 2023-12-31 --strategies trend_following mean_reversion --generate-reports
```

### Market-wide Backtests

To run a backtest across an entire market:

```bash
# Run backtest for all HKEX symbols
python market_backtest.py --all-symbols --market hkex --start-date 2023-01-01 --end-date 2023-12-31 --strategies trend_following --execution-mode parallel --workers 4 --generate-reports

# Run backtest for specific symbols
python market_backtest.py --symbols 0700.HK 9988.HK --start-date 2023-01-01 --end-date 2023-12-31 --strategies trend_following mean_reversion --execution-mode parallel --workers 2 --generate-reports

# Run backtest with symbols from a file
python market_backtest.py --symbol-file symbols.txt --start-date 2023-01-01 --end-date 2023-12-31 --strategies trend_following
```

#### Execution Modes

The system supports two execution modes:

1. **Parallel**: Uses multiple processes to run backtests in parallel
   ```bash
   python market_backtest.py --symbols 0700.HK 9988.HK --execution-mode parallel --workers 2
   ```

2. **Async**: Uses asynchronous execution for improved performance with many symbols
   ```bash
   python market_backtest.py --all-symbols --market nasdaq --execution-mode async --max-concurrent 20
   ```

### Supported Markets

The system currently supports the following markets:

- HKEX (Hong Kong Exchange)
- NASDAQ
- NYSE (New York Stock Exchange)
- ASX (Australian Securities Exchange)

To specify a market:

```bash
python market_backtest.py --all-symbols --market nasdaq --start-date 2023-01-01 --end-date 2023-12-31
```

## Strategies

### Built-in Strategies

The system comes with several built-in strategies:

1. **Trend Following**: Based on moving average crossovers. Generates buy signals when the fast moving average crosses above the slow moving average, and sell signals when the fast moving average crosses below the slow moving average.
   ```bash
   python market_backtest.py --symbols 0700.HK --strategies trend_following --fast-period 20 --slow-period 50
   ```

2. **Mean Reversion**: Based on RSI (Relative Strength Index). Generates buy signals when the RSI falls below the oversold threshold, and sell signals when the RSI rises above the overbought threshold.
   ```bash
   python market_backtest.py --symbols 0700.HK --strategies mean_reversion --rsi-period 14 --oversold-threshold 30 --overbought-threshold 70
   ```

3. **Moving Average Crossover**: Identifies opportunities when a faster moving average crosses above a slower moving average, combined with volume and price-to-VWAP filters.
   ```bash
   python market_backtest.py --symbols 0700.HK --strategies moving_average_crossover
   ```

4. **Volatility Breakout**: Identifies breakout opportunities based on increased volatility and volume, targeting stocks that are breaking out of consolidation patterns.
   ```bash
   python market_backtest.py --symbols 0700.HK --strategies volatility_breakout
   ```

5. **Volume Spike**: Identifies trading opportunities based on significant increases in trading volume, which often precede price movements.
   ```bash
   python market_backtest.py --symbols 0700.HK --strategies volume_spike
   ```

### Adding Custom Strategies

To add a custom strategy:

1. Create a new strategy file in `src/market_system/backtesting/strategies/`
2. Implement your strategy by extending the `BaseStrategy` class
3. Register your strategy in `src/market_system/backtesting/strategies/__init__.py`

Example of a custom strategy:

```python
# src/market_system/backtesting/strategies/my_strategy.py
from src.market_system.strategy.base_strategy import BaseStrategy
from src.market_system.domain.models import MarketData, Signal, MarketState
from typing import List, Optional

class MyStrategy(BaseStrategy):
    def __init__(self, name="MyStrategy", param1=10, param2=20):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
        self._required_indicators = ['indicator1', 'indicator2']
    
    def analyze(self, data: List[MarketData], market_state: MarketState) -> Optional[Signal]:
        # Implement your strategy logic here
        # Return a Signal object if a trading opportunity is identified
        # Return None otherwise
        pass
    
    def get_stop_loss(self, data: List[MarketData], signal: Signal) -> float:
        # Implement stop loss calculation
        pass
    
    def get_take_profit(self, data: List[MarketData], signal: Signal) -> float:
        # Implement take profit calculation
        pass
```

Then register your strategy in `__init__.py`:

```python
# src/market_system/backtesting/strategies/__init__.py
from src.market_system.backtesting.strategies.trend_following import TrendFollowingStrategy
from src.market_system.backtesting.strategies.mean_reversion import MeanReversionStrategy
from src.market_system.backtesting.strategies.my_strategy import MyStrategy

__all__ = [
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MyStrategy'
]
```

### Strategy Filtering

The system includes a strategy analyzer that can filter symbols based on predefined criteria before running a full backtest. This can significantly improve performance by focusing only on promising candidates.

#### Is Manual Strategy Filtering Necessary?

No, manual strategy filtering is not necessary before running a backtest. The backtesting engine automatically applies appropriate filters based on the strategies being used. However, you can run a separate strategy filtering step if you want to:

1. Pre-screen a large universe of symbols to identify the most promising candidates
2. Apply more complex filtering criteria than what's built into the strategies
3. Generate a list of filtered symbols for further analysis

#### Running Strategy Filtering

```bash
# Filter symbols based on strategy criteria
python strategy_filter.py --market hkex --start-date 2023-01-01 --end-date 2023-12-31 --min-volume 1000000 --output filtered_symbols.txt

# Then run backtest with filtered symbols
python market_backtest.py --symbol-file filtered_symbols.txt --start-date 2023-01-01 --end-date 2023-12-31 --strategies trend_following
```

## Performance Analysis

After running a backtest, the system generates comprehensive performance reports:

```bash
# Generate reports for a previous backtest
python report_generator.py --input-file results/backtest_results_20230101_20231231.json --formats html,json,csv
```

Reports are saved in the `reports/backtest/` directory and include:

- Equity curve charts
- Trade statistics
- Performance metrics (Sharpe ratio, max drawdown, etc.)
- Trade distribution analysis

## Configuration

You can customize the backtest configuration using command-line arguments or a configuration file:

```bash
# Using command-line arguments
python market_backtest.py --symbols 0700.HK --start-date 2023-01-01 --end-date 2023-12-31 --initial-capital 100000 --position-size 0.1 --stop-loss 0.05 --take-profit 0.1

# Using a configuration file
python market_backtest.py --config config.json
```

Example `config.json`:
```json
{
  "symbols": ["0700.HK", "9988.HK"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "strategies": ["trend_following", "mean_reversion"],
  "initial_capital": 100000,
  "position_size": 0.1,
  "stop_loss": 0.05,
  "take_profit": 0.1,
  "execution_mode": "parallel",
  "workers": 4,
  "generate_reports": true
}
```

## Examples

### Example 1: Running a Simple Backtest

```bash
python market_backtest.py --symbols 0700.HK --start-date 2023-01-01 --end-date 2023-01-31 --strategies trend_following
```

### Example 2: Market-wide Backtest with Multiple Strategies

```bash
python market_backtest.py --all-symbols --market hkex --start-date 2023-01-01 --end-date 2023-01-31 --strategies trend_following mean_reversion --execution-mode parallel --workers 4 --generate-reports
```

### Example 3: Customizing Strategy Parameters

```bash
python market_backtest.py --symbols 0700.HK 9988.HK --start-date 2023-01-01 --end-date 2023-01-31 --strategies trend_following --fast-period 10 --slow-period 30 --position-size 0.2 --stop-loss 0.03 --take-profit 0.09
```

### Example 4: Running Backtests with Different Timeframes

```bash
# Daily timeframe
python market_backtest.py --symbols 0700.HK --start-date 2023-01-01 --end-date 2023-01-31 --timeframe 1d --strategies trend_following

# Hourly timeframe
python market_backtest.py --symbols 0700.HK --start-date 2023-01-01 --end-date 2023-01-31 --timeframe 1h --strategies mean_reversion
```