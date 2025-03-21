# StockScan - Quantitative Trading with Qlib and MLflow

A quantitative trading analytics system that integrates Qlib with MLflow for tracking experiments and results.

## Features

- Fetch historical stock data up to today
- Analyze data with Qlib's built-in strategies
- Make predictions using Qlib's built-in models
- Fine-tune models for improved performance
- Track all experiments with MLflow

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/stockscan.git
cd stockscan

# Using conda
conda env create -f environment.yml
conda activate stockscan
```

## StockScan Workflow

StockScan provides a complete stock analysis workflow that includes:

1. **Data Fetching**: Download stock market data from Yahoo Finance
2. **Model Prediction**: Train and test ML models on the historical data
3. **Backtesting**: Evaluate trading strategies on the predictions
4. **Analysis**: Compare and analyze multiple strategies

### Quick Start

Run the workflow with default settings:

```bash
# Run the complete workflow for Chinese market
./run_stockscan.py --region cn --market csi300

# Run a quick test with minimal data
./run_stockscan.py --quick

# Run the full workflow with all features
./run_stockscan.py --full
```

### Running Specific Stages

You can run specific parts of the workflow:

```bash
# Only fetch and prepare data
./run_stockscan.py --data-only

# Only run prediction (assumes data is already prepared)
./run_stockscan.py --predict-only

# Only run backtesting (assumes prediction is already done)
./run_stockscan.py --backtest-only

# Compare multiple trading strategies
./run_stockscan.py --compare-strategies
```

### Configuration Options

```bash
# Specify region and market
./run_stockscan.py --region us --market sp500

# Custom date range
./run_stockscan.py --start-date 2020-01-01 --end-date 2022-12-31

# Choose different model
./run_stockscan.py --model LGBM

# Fine-tune the model with additional training data
./run_stockscan.py --fine-tune

# Specify custom strategy parameters
./run_stockscan.py --strategy TopkDropoutStrategy --topk 20 --n-drop 5

# Save results to a specific directory
./run_stockscan.py --result-dir ./my_results
```

### Workflow Output

The workflow generates the following outputs:

- **Data**: Prepared in Qlib format for ML modeling
- **Model Predictions**: CSV files with stock price predictions
- **Backtest Results**: Performance metrics, positions, and risk analysis
- **Strategy Comparison**: Charts and tables comparing multiple strategies
- **MLflow Tracking**: All experiments are logged in MLflow for tracking and visualization

To view the MLflow tracking UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Directory Structure

```
stockscan/
├── data/                 # Data directory
├── results/              # Results directory
├── mlruns/               # MLflow runs
├── src/
│   ├── config.py         # Configuration files
│   ├── fetch_data.py     # Data fetching script
│   ├── run_workflow.py   # Main workflow script
│   ├── data/             # Data processing modules
│   ├── models/           # Prediction models
│   ├── strategies/       # Trading strategies
│   └── utils/            # Utility functions
└── README.md             # This file
```

## Configuration

Edit `src/config.py` to modify:

- Data sources and regions
- Model parameters
- Strategy parameters
- Backtesting parameters

## License

[MIT License](LICENSE) 