# StockScan Examples

This directory contains example scripts demonstrating how to use the StockScan system for stock prediction.

## Prerequisites

Before running these examples, make sure you have:

1. Installed the StockScan package:
   ```bash
   pip install -e .
   ```

2. Downloaded the required Qlib data:
   ```bash
   # For US stock data (required for these examples)
   python -m qlib.data.data_collector_us
   
   # For China stock data (optional, for other examples)
   python -m qlib.data.data_collector_cn
   ```

3. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   CHAT_MODEL=gpt-4-turbo
   ```

4. Docker is installed and running (for RDAgent evolution features)

## Available Examples

### 1. Trio Stocks Prediction (`trio_stocks_prediction.py`)

A focused prediction script for AAPL, MSFT, and NVDA stocks. This example:
- Uses the most recent 3 years of data
- Trains a LightGBM model
- Generates predictions for all three stocks
- Displays rankings and recommendations
- Creates a visual trend plot

**Run this example:**
```bash
python examples/trio_stocks_prediction.py
```

**Output:**
- Console output with predictions and recommendations
- Prediction trends plot in `./output/trio_stocks/plots/prediction_trends.png`

See [expected_output.md](expected_output.md) for a sample of what the output should look like.

### 2. Tech Stocks Prediction (`tech_stocks_prediction.py`)

A more comprehensive example that analyzes a larger set of tech stocks: AAPL, MSFT, NVDA, GOOGL, and AMZN. This example:
- Uses data from 2018-2023
- Trains a LightGBM model with more parameters
- Performs backtesting
- Provides detailed stock-specific predictions
- Creates cumulative returns plots

**Run this example:**
```bash
python examples/tech_stocks_prediction.py
```

**Output:**
- Console output with predictions and backtest metrics
- Cumulative returns plot in `./output/plots/cumulative_returns.png`

### 3. Simple Pipeline (`simple_pipeline.py`)

A general example that demonstrates the core features of the StockScan pipeline:
- Basic model training
- Signal generation
- Factor evolution (if Docker is available)
- Model optimization (if Docker is available)

**Run this example:**
```bash
python examples/simple_pipeline.py
```

## Interpreting Results

The prediction scores represent the expected returns of the stocks. The interpretation is generally:

- **Positive scores**: Buy signals (higher scores indicate stronger buy signals)
- **Negative scores**: Sell signals (lower scores indicate stronger sell signals)

In the trio stocks example, a more detailed interpretation is provided with signal strength categories (STRONG BUY, BUY, WEAK BUY, etc.).

## Troubleshooting

If you encounter issues:

1. **Data not found**: Make sure you've downloaded the required Qlib data
2. **Docker not available**: RDAgent evolution features require Docker
3. **API key errors**: Check that your `.env` file has the correct API key
4. **Import errors**: Ensure you've installed the package with `pip install -e .`

## Quick Start with the Examples Runner

For convenience, you can use the included shell script to run any of the examples:

```bash
# Make sure the script is executable
chmod +x examples/run_examples.sh

# Run the examples menu
./examples/run_examples.sh
```

This script will:
1. Check if StockScan is installed
2. Verify your environment setup
3. Check for required data
4. Present a menu of examples to run

It's the easiest way to get started with stock predictions! 