# Standalone Stock Analysis Workflow

This standalone workflow implements key functionality for stock analysis without relying on problematic Qlib components. It provides a simplified but complete solution for generating synthetic data, training models, making predictions, and backtesting strategies.

## Features

- Generate synthetic stock data for testing and demonstration
- Implement simple models for prediction (Linear, LGBM, DEnsemble)
- Backtest strategies based on the top-k approach
- Calculate and visualize performance metrics
- Multi-model execution and comparison
- MLflow tracking integration (optional)
- Independent prediction and backtest date ranges
- Customizable data generation parameters

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- mlflow (optional)

## Usage

### Basic Usage

```bash
# Run a single model workflow
./standalone_workflow.py --model Linear --topk 10 --n_drop 2

# Disable MLflow tracking
./standalone_workflow.py --model Linear --topk 10 --n_drop 2 --no_mlflow

# Specify custom date ranges
./standalone_workflow.py --model Linear --train_start 2020-01-01 --train_end 2020-06-30 --test_start 2020-10-01 --test_end 2020-12-31

# Specify output directory
./standalone_workflow.py --model Linear --output_dir ./my_results

# Control data generation
./standalone_workflow.py --model Linear --num_stocks 100 --price_volatility 0.02
```

### Multi-Model Workflow

```bash
# Run and compare multiple models
./standalone_workflow.py --multi --models Linear LGBM DEnsemble --topk 10 --n_drop 2

# Run a subset of models
./standalone_workflow.py --multi --models Linear LGBM --topk 10 --n_drop 2

# Custom prediction and backtest date ranges
./standalone_workflow.py --multi --models Linear LGBM DEnsemble --predict_start 2020-12-01 --predict_end 2020-12-31 --backtest_start 2020-12-15 --backtest_end 2020-12-25
```

## Command-Line Arguments

- `--model`: Model to use (Linear, LGBM, DEnsemble) [default: Linear]
- `--multi`: Run multiple models and compare
- `--models`: List of models to run when using --multi [default: Linear LGBM DEnsemble]
- `--topk`: Number of top stocks to select [default: 10]
- `--n_drop`: Number of stocks to drop at each rebalance [default: 2]
- `--output_dir`: Directory to save results
- `--num_stocks`: Number of stocks to simulate [default: 50]
- `--price_volatility`: Volatility parameter for synthetic data [default: 0.01]
- `--no_mlflow`: Disable MLflow tracking

### Time-related Arguments

- `--train_start`: Training start date (YYYY-MM-DD) [default: 2020-01-01]
- `--train_end`: Training end date (YYYY-MM-DD) [default: 2020-06-30]
- `--valid_start`: Validation start date (YYYY-MM-DD) [default: 2020-07-01]
- `--valid_end`: Validation end date (YYYY-MM-DD) [default: 2020-09-30]
- `--test_start`: Test start date (YYYY-MM-DD) [default: 2020-10-01]
- `--test_end`: Test end date (YYYY-MM-DD) [default: 2020-12-31]
- `--predict_start`: Prediction start date (YYYY-MM-DD) [default: same as test_start]
- `--predict_end`: Prediction end date (YYYY-MM-DD) [default: same as test_end]
- `--backtest_start`: Backtest start date (YYYY-MM-DD) [default: same as predict_start]
- `--backtest_end`: Backtest end date (YYYY-MM-DD) [default: same as predict_end]

## Date Range Flexibility

The workflow supports separate date ranges for prediction and backtesting, which allows you to:

1. Generate predictions for a longer period but backtest only a specific segment
2. Test different holding periods by varying the backtest date range
3. Evaluate model performance during specific market events
4. Compare model performance across different time windows

Examples:

```bash
# Generate predictions for December, but only backtest the middle portion
./standalone_workflow.py --model Linear --predict_start 2020-12-01 --predict_end 2020-12-31 --backtest_start 2020-12-10 --backtest_end 2020-12-20

# Generate predictions for Q4 2020, but only backtest December
./standalone_workflow.py --model Linear --predict_start 2020-10-01 --predict_end 2020-12-31 --backtest_start 2020-12-01 --backtest_end 2020-12-31
```

## Output Structure

The workflow creates a directory structure for results:

```
results/
  └── MODEL_TIMESTAMP/
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
  └── multi_model_TIMESTAMP/
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

## Implementation Details

### Data Generation

The workflow generates synthetic stock data using the `generate_stock_data` function. It creates price data for the specified number of stocks over the given date range. The data generation can be customized with:

- `num_stocks`: Number of stocks to generate [default: 50]
- `price_volatility`: Volatility parameter for price generation [default: 0.01]

### Models

The `SimpleModel` class provides three model types:

1. **Linear**: Simple linear trend with small random noise
2. **LGBM**: More volatile predictions with sine wave patterns
3. **DEnsemble**: Complex predictions combining sine and cosine waves

### Backtesting

The `SimpleBacktester` class implements a simple backtesting approach:

1. Select top-k stocks based on prediction scores
2. Calculate turnover rate
3. Generate random daily returns (in a real implementation, you would use actual price data)
4. Calculate performance metrics (total return, annual return, Sharpe ratio, max drawdown)

### MLflow Tracking

When MLflow tracking is enabled, the workflow logs:

- Parameters (model type, topk, n_drop, etc.)
- Metrics (total return, annual return, Sharpe ratio, max drawdown, turnover)
- Artifacts (predictions, portfolio value plot, backtest results)

## Extending the Implementation

To extend this implementation:

1. **Add real data sources**: Modify the `generate_stock_data` function to fetch real stock data from APIs or CSV files
2. **Implement real models**: Enhance the `SimpleModel` class to include actual machine learning models
3. **Improve backtesting**: Update the `SimpleBacktester` class to use real price data and more sophisticated trading strategies
4. **Add more metrics**: Extend the analysis to include additional performance metrics

## Limitations

This standalone implementation has some limitations:

1. It uses synthetic data instead of real stock data
2. The models generate synthetic predictions instead of using actual machine learning
3. The backtesting simulates returns based on prediction scores
4. It lacks some advanced features of Qlib (factor analysis, complex strategies)

Despite these limitations, it provides a complete framework that you can extend with real data sources and models. 