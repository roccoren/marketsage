# Backtesting Tutorial

This tutorial explains how to use the backtesting system with Qlib integration in our stock analysis platform.

## Prerequisites

Before running a backtest, make sure you have the required dependencies installed:

```bash
pip install pyqlib pandas numpy matplotlib yfinance
```

## Quick Start

The simplest way to run a backtest is to use the provided example script:

```bash
python scripts/run_backtest_example.py
```

This script will:
1. Download historical data for a sample list of stocks
2. Calculate technical indicators
3. Generate predictions based on these indicators
4. Run a backtesting simulation
5. Save the results and metrics

## Running a Custom Backtest

To run a custom backtest, you can use the `run_backtest.py` script with your own configuration:

```bash
python scripts/run_backtest.py config/stocks/your_stock_list.txt \
  --historical-start 2020-01-01 \
  --historical-end 2023-01-01 \
  --backtest-start 2022-01-01 \
  --backtest-end 2023-01-01 \
  --benchmark SPY \
  --strategy topk \
  --topk 10 \
  --n-drop 3
```

### Parameters Explained

- `config/stocks/your_stock_list.txt`: File containing a list of stock symbols, one per line
- `--historical-start`: Start date for historical data (for training/analysis)
- `--historical-end`: End date for historical data
- `--backtest-start`: Start date for the backtesting period
- `--backtest-end`: End date for the backtesting period
- `--benchmark`: Benchmark symbol (default: SPY)
- `--strategy`: Strategy type (options: topk, enhanced_indexing)
- `--topk`: Number of top stocks to include in the portfolio
- `--n-drop`: Number of stocks to drop during rebalancing

## Understanding the Results

After running a backtest, the system will generate several outputs:

1. **Metrics File**: A JSON file containing performance metrics such as:
   - Annualized return
   - Sharpe ratio
   - Maximum drawdown
   - Win rate
   - Turnover

2. **Portfolio CSV**: A CSV file with daily portfolio values, positions, and returns

3. **Visualization**: A plot showing portfolio value and daily returns over time

The results are saved in the `output/backtest/` directory with a timestamp.

## Customizing the Backtest

### Using a Configuration File

You can customize the backtest by modifying the `config/backtest_config.yaml` file:

```yaml
# Strategy settings
strategy:
  type: "topk"
  params:
    topk: 50
    n_drop: 5

# Backtest parameters
backtest:
  account: 100000000.0
  exchange:
    limit_threshold: 0.095
    deal_price: "close"
    open_cost: 0.0005
    close_cost: 0.0015
    min_cost: 5
```

### Customizing Technical Indicators

You can modify the prediction weights in the configuration file:

```yaml
# Technical indicators weights for prediction
prediction_weights:
  rsi: 0.2
  macd: 0.2
  stochastic: 0.1
  volatility: 0.1
  volume: 0.1
  momentum: 0.1
  trend: 0.1
```

## Advanced Usage: Programmatic API

You can also use the backtesting system programmatically:

```python
from src.backtest.qlib_backtest import QlibBacktest

# Initialize backtesting system
backtest_system = QlibBacktest(
    provider_uri="path/to/qlib_data",
    region='us',
    start_time="2022-01-01",
    end_time="2023-01-01",
    instruments=symbols,
    benchmark='SPY',
    strategy_type='topk',
    strategy_params={
        'topk': 10,
        'n_drop': 3
    }
)

# Run backtest
metrics, backtest_result = backtest_system.run_backtest(predictions)

# Save results
backtest_system.save_results(backtest_result, metrics, "output/my_backtest")

# Access metrics
print(f"Sharpe ratio: {metrics['sharpe_ratio']}")
print(f"Annualized return: {metrics['annualized_return']}")
```

## Troubleshooting

If you encounter any issues when running a backtest:

1. **Missing Qlib data**: Ensure you have qlib data available or use the dummy data approach in the example script
2. **Import errors**: Make sure all dependencies are installed
3. **Missing stock data**: Verify that your stocks are available in your data source

For more details, refer to the Qlib documentation: https://github.com/microsoft/qlib 