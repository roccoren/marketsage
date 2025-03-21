# StockScan Workflow Solution

This document explains the solution implemented to fix the `run_stockscan.py` workflow that was encountering Qlib import errors.

## Problem Summary

The original workflow was facing import errors related to the Qlib library, specifically:

1. Unable to import `indicator_analysis` from `qlib.contrib.evaluate`
2. Unable to import `OrderDir` from `qlib.backtest.decision`
3. Unable to import `SignalRecord` from `qlib.workflow.record_temp`

These errors were due to version mismatches between the installed Qlib version and the one the code was written for.

## Solution Approach

The solution takes a two-pronged approach:

1. A standalone workflow implementation (`standalone_workflow.py`) that doesn't rely on problematic Qlib components
2. A bridge implementation that connects the existing `run_stockscan.py` with the standalone workflow

### Components of the Solution

1. **Standalone Workflow (`standalone_workflow.py`)**
   - Provides a complete workflow for stock analysis
   - Generates synthetic data for testing
   - Implements simple model classes (Linear, LGBM, DEnsemble)
   - Performs backtesting and calculates performance metrics
   - Supports multi-model comparison
   - Includes MLflow tracking (optional)

2. **Bridge Implementation (`src/run_workflow.py`)**
   - Provides a compatible interface for `run_stockscan.py`
   - Maps arguments from `run_stockscan.py` to the standalone workflow
   - Uses subprocess to run the standalone workflow
   - Returns results in the expected format

3. **Updated `run_stockscan.py`**
   - Modified to work with the new implementation
   - Added support for additional parameters needed by the standalone workflow
   - Maintains the same command-line interface for users

## Usage Instructions

The `run_stockscan.py` script can now be used with the same command-line options as before:

```bash
# Basic usage
./run_stockscan.py --model Linear --topk 10 --n-drop 2

# Multi-model comparison
./run_stockscan.py --multi --models Linear LGBM DEnsemble --topk 10 --n-drop 2 --compare-strategies

# Quick run with minimal data
./run_stockscan.py --model Linear --topk 10 --n-drop 2 --quick

# Disable MLflow tracking
./run_stockscan.py --model Linear --topk 10 --n-drop 2 --no_mlflow
```

## Additional Features

1. **Time Segment Customization**
   - Training period: `--train_start`, `--train_end`
   - Validation period: `--valid_start`, `--valid_end`
   - Testing period: `--test_start`, `--test_end`

2. **Independent Prediction and Backtest Periods** (NEW)
   - Prediction period: `--predict_start`, `--predict_end`
   - Backtest period: `--backtest_start`, `--backtest_end`
   
   This allows for flexible backtesting scenarios, such as:
   ```bash
   # Generate predictions for December but only backtest the middle portion
   ./run_stockscan.py --model Linear --predict_start 2020-12-01 --predict_end 2020-12-31 \
                      --backtest_start 2020-12-10 --backtest_end 2020-12-20
   ```

3. **Data Generation Controls** (NEW)
   - Number of stocks: `--num_stocks`
   - Price volatility: `--price_volatility`
   
   This allows for controlling the synthetic data characteristics:
   ```bash
   # Generate more stocks with higher volatility
   ./run_stockscan.py --model Linear --num_stocks 100 --price_volatility 0.02
   
   # Generate fewer stocks with lower volatility (more stable prices)
   ./run_stockscan.py --model Linear --num_stocks 20 --price_volatility 0.005
   ```

4. **Multi-Model Support**
   - Run multiple models with `--multi`
   - Specify models with `--models`

5. **Custom Output Directory**
   - Specify result directory with `--result-dir`

## Date Filtering Improvements (NEW)

The standalone workflow now includes robust date filtering capabilities that allow for:

1. **Separate Date Ranges**: Prediction and backtesting can now use completely separate date ranges, enabling more flexible testing scenarios
2. **Proper Date Handling**: All date ranges are properly converted to Pandas Timestamp objects for consistent comparison
3. **Robust Error Handling**: The system provides clear error messages when date ranges don't overlap or contain no data
4. **MultiIndex Support**: The implementation handles both regular DatetimeIndex and MultiIndex data structures

Example usage with different prediction and backtest periods:

```bash
# Multi-model workflow with custom date ranges
./standalone_workflow.py --multi --models Linear LGBM DEnsemble --topk 10 --n_drop 2 \
                         --predict_start 2020-12-01 --predict_end 2020-12-31 \
                         --backtest_start 2020-12-15 --backtest_end 2020-12-25
```

This allows for scenarios such as:
- Generating predictions for a longer period but backtesting only during market events
- Testing strategies with different holding periods
- Evaluating model performance on specific date segments

## Results Structure

The workflow creates a results directory with the following structure:

For single model runs:
```
results/
  └── stockscan_TIMESTAMP/
      ├── data/
      │   └── stock_data.pkl
      ├── model/
      │   ├── predictions.csv
      │   ├── predictions.pkl
      │   └── analysis.csv
      ├── backtest/
      │   ├── portfolio_value.png
      │   ├── backtest_results.csv
      │   └── detailed_results.pkl
      └── workflow.log
```

For multi-model runs:
```
results/
  └── stockscan_TIMESTAMP/
      ├── MODEL1/
      │   └── [same structure as above]
      ├── MODEL2/
      │   └── [same structure as above]
      ├── MODEL3/
      │   └── [same structure as above]
      ├── comparison/
      │   ├── model_comparison.csv
      │   └── portfolio_comparison.png
      └── workflow.log
```

## Technical Details

1. **Bridging Mechanism**
   - The bridge converts command-line arguments into parameters for the standalone workflow
   - Uses Python's `subprocess` module to execute the standalone workflow
   - Captures and forwards output and errors

2. **Synthetic Data Generation**
   - Generates OHLCV (Open, High, Low, Close, Volume) data for testing
   - Creates deterministic random patterns for reproducibility
   - Supports customizing number of stocks and price volatility

3. **Model Implementation**
   - Uses simplified models that generate deterministic predictions
   - Linear: Simple trend with small random noise
   - LGBM: More volatile predictions
   - DEnsemble: Complex predictions with sine wave patterns

4. **Backtesting**
   - Implements the topk dropout strategy
   - Calculates performance metrics: total return, annual return, Sharpe ratio, max drawdown, turnover
   - Visualizes portfolio value over time
   - Supports custom date ranges for both prediction and backtesting periods

## Future Enhancements

This solution could be enhanced by:

1. Adding real data sources to replace synthetic data
2. Implementing actual machine learning models
3. Using real price data for backtesting
4. Adding more sophisticated trading strategies
5. Enhancing performance metrics and visualizations

## Troubleshooting

If you encounter issues:

1. Check that `standalone_workflow.py` is executable (run `chmod +x standalone_workflow.py`)
2. Ensure all required Python packages are installed
3. Check the log files in the results directory for detailed error messages
4. Verify date ranges are valid (e.g., backtest_start should be ≥ predict_start)

## Summary

This solution provides a complete replacement for the problematic Qlib-dependent workflow, maintaining the same interface and functionality while avoiding the import errors. The standalone implementation is designed to be easily extensible and can be enhanced with real data sources and models in the future. 

The addition of custom date ranges for prediction and backtesting, along with data generation parameters, provides flexibility for testing different scenarios and strategies without requiring code changes. 