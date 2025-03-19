# StockScan

StockScan is a comprehensive market analysis and backtesting platform designed for algorithmic trading strategies. It provides tools for data collection, analysis, strategy development, and backtesting.

## Features

* **Market Data Collection**: Automatically fetch data from various sources including Yahoo Finance, Alpha Vantage, and local data files.
* **Technical Indicators**: Over 30 built-in technical indicators including Moving Averages, RSI, MACD, Bollinger Bands, and more.
* **Strategy Development**: Framework for creating and testing custom trading strategies.
* **Backtesting System**: Powered by QuantConnect LEAN framework for professional-grade backtesting.
* **Performance Metrics**: Comprehensive performance analysis including returns, Sharpe ratio, drawdowns, and more.
* **Visualization**: Charts and plots for visual analysis of strategy performance.

## Installation

### Prerequisites

* Python 3.8+
* .NET Core 6.0+ (for LEAN framework)
* Market data (sample data included)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stockscan.git
   cd stockscan
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install QuantConnect LEAN:
   ```bash
   # Clone the LEAN repository
   git clone https://github.com/QuantConnect/Lean.git
   
   # Follow LEAN installation instructions
   cd Lean
   ./build.sh
   ```

4. Configure the system:
   ```bash
   cp config.example.json config.json
   # Edit config.json to set your API keys and preferences
   ```

## Usage

### Basic Usage

Run a simple backtest using the command-line interface:

```bash
python main.py backtest --algorithm TrendFollowingAlgorithm --symbols AAPL MSFT --start-date 2022-01-01 --end-date 2022-12-31
```

### Advanced Options

```bash
python main.py backtest --algorithm TrendFollowingAlgorithm --symbols AAPL MSFT --start-date 2022-01-01 --end-date 2022-12-31 --initial-capital 100000 --resolution Daily --parameters '{"FastPeriod": 10, "SlowPeriod": 30}'
```

### Available LEAN Algorithms

The system comes with several pre-built algorithms:

1. **TrendFollowingAlgorithm**: Follows market trends using moving averages.
   ```bash
   python main.py backtest --algorithm TrendFollowingAlgorithm --symbols AAPL --start-date 2022-01-01 --end-date 2022-12-31 --parameters '{"FastPeriod": 20, "SlowPeriod": 50}'
   ```

2. **MeanReversionAlgorithm**: Based on RSI (Relative Strength Index).
   ```bash
   python main.py backtest --algorithm MeanReversionAlgorithm --symbols AAPL --start-date 2022-01-01 --end-date 2022-12-31 --parameters '{"RsiPeriod": 14, "OversoldThreshold": 30, "OverboughtThreshold": 70}'
   ```

3. **MovingAverageCrossoverAlgorithm**: Uses moving average crossovers with volume filters.
   ```bash
   python main.py backtest --algorithm MovingAverageCrossoverAlgorithm --symbols AAPL --start-date 2022-01-01 --end-date 2022-12-31
   ```

4. **VolatilityBreakoutAlgorithm**: Identifies breakout opportunities based on volatility.
   ```bash
   python main.py backtest --algorithm VolatilityBreakoutAlgorithm --symbols AAPL --start-date 2022-01-01 --end-date 2022-12-31
   ```

5. **VolumeSpikeAlgorithm**: Identifies trading opportunities based on volume increases.
   ```bash
   python main.py backtest --algorithm VolumeSpikeAlgorithm --symbols AAPL --start-date 2022-01-01 --end-date 2022-12-31
   ```

### Adding Custom LEAN Algorithms

To add a custom algorithm:

1. Create a new algorithm file in `src/market_system/lean/algorithms/`
2. Implement your algorithm by extending the `QCAlgorithm` class
3. The algorithm will be automatically discovered by the system

Example of a custom LEAN algorithm:

```python
# src/market_system/lean/algorithms/my_algorithm.py
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Indicators import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)
        
        self.symbol = self.AddEquity("AAPL", Resolution.Daily).Symbol
        
        # Define parameters with default values
        self.fast_period = 10
        self.slow_period = 30
        
        # Initialize indicators
        self.fast_ma = self.SMA(self.symbol, self.fast_period, Resolution.Daily)
        self.slow_ma = self.SMA(self.symbol, self.slow_period, Resolution.Daily)
        
        self.previous_state = None
    
    def OnData(self, data):
        # Wait for indicators to be ready
        if not self.fast_ma.IsReady or not self.slow_ma.IsReady:
            return
        
        # Trading logic
        current_price = data[self.symbol].Close
        holdings = self.Portfolio[self.symbol].Quantity
        
        # Long when fast MA crosses above slow MA
        if self.fast_ma.Current.Value > self.slow_ma.Current.Value and holdings <= 0:
            self.SetHoldings(self.symbol, 1.0)
        
        # Short when fast MA crosses below slow MA
        elif self.fast_ma.Current.Value < self.slow_ma.Current.Value and holdings >= 0:
            self.SetHoldings(self.symbol, -1.0)
```

### Converting Data for LEAN

LEAN requires data in a specific format. Use the built-in converter:

```bash
python main.py convert-data --source-file data/raw/AAPL.csv --symbol AAPL
```

### Listing Available Algorithms

To view all available algorithms:

```bash
python main.py list-algorithms
```

## Performance Analysis

After running a backtest, the system generates comprehensive performance reports:

1. **Equity Curve**: Shows the growth of your portfolio over time
2. **Drawdown Analysis**: Identifies periods of losses and their severity
3. **Return Metrics**: Includes annualized return, Sharpe ratio, Sortino ratio
4. **Trade Statistics**: Win rate, average win/loss, profit factor
5. **Risk Metrics**: Beta, alpha, maximum drawdown

Reports are saved to the `results` directory by default.

## API Reference

The system can also be used programmatically:

```python
from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode

# Initialize adapter
adapter = LeanAdapter(
    config_path="config.json",
    mode=BacktestMode.LEAN,
    data_folder="data",
    results_folder="results"
)

# Run a backtest
results = adapter.run_backtest(
    algorithm_name="TrendFollowingAlgorithm",
    symbols=["AAPL", "MSFT"],
    start_date="2022-01-01",
    end_date="2022-12-31",
    initial_capital=100000,
    resolution="Daily",
    parameters={"FastPeriod": 10, "SlowPeriod": 30}
)

# Access results
print(f"Total return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Maximum drawdown: {results['metrics']['max_drawdown']:.2%}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* QuantConnect for the LEAN framework
* Yahoo Finance and Alpha Vantage for market data
* The open-source community for various libraries used in this project