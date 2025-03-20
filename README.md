# StockScan

A Python package for stock analysis and prediction using machine learning models.

## Features

- Stock data retrieval from multiple sources (yfinance, Qlib)
- Technical indicator calculation (MA, RSI, MACD, etc.)
- Multiple prediction models (LightGBM, XGBoost)
- Visualization of prediction trends
- Signal strength analysis for trading decisions
- LLM integration for feature evolution and optimization with RDAgent (OpenAI and Azure OpenAI)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stockscan.git
cd stockscan
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Install optional dependencies:
```bash
pip install lightgbm xgboost yfinance qlib matplotlib rdagent
```

## Usage

### Basic Example

```python
from stockscan.core.data_manager import get_stock_data
from stockscan.core.model_factory import create_model
from stockscan.core.prediction import predict_stock_movement
from stockscan.utils.env_setup import setup_environment

# Setup environment
setup_environment()

# Get stock data
data = get_stock_data('AAPL', days=365, provider='yfinance')

# Create model
model = create_model('LightGBM')

# Predict stock movement
prediction = predict_stock_movement(model, data)
print(f"Prediction: {prediction}")
```

### Running the Trio Stocks Example

The package includes an example script that predicts movements for AAPL, MSFT, and NVDA:

```bash
python examples/trio_stocks_prediction.py
```

This will:
1. Download historical data for the three stocks
2. Train prediction models
3. Generate predictions for each stock
4. Create a visualization of prediction trends
5. Provide trading recommendations

## LLM Integration with RDAgent

StockScan uses RDAgent to leverage large language models (LLMs) for feature evolution and model optimization.

### OpenAI Integration

```bash
# Set OpenAI API Key
export OPENAI_API_KEY=your-openai-api-key

# Run with RDAgent enabled
python examples/trio_stocks_prediction.py
```

### Azure OpenAI Integration

StockScan supports using Azure OpenAI for feature evolution and model optimization with RDAgent:

```bash
python examples/azure_openai_example.py \
  --azure_endpoint https://your-resource-name.openai.azure.com/ \
  --azure_deployment your-gpt4-deployment-name \
  --symbols AAPL,MSFT,NVDA \
  --start_date 2022-01-01 \
  --end_date 2023-01-01
```

Make sure to set your `OPENAI_API_KEY` environment variable with your Azure OpenAI API key:

```bash
export OPENAI_API_KEY=your-azure-openai-api-key
```

For more details on Azure OpenAI integration, see [Azure OpenAI Integration Guide](docs/azure_openai_integration.md).

## Data Sources

StockScan supports data from:

- **yfinance**: Yahoo Finance data (default, requires internet)
- **Qlib**: Microsoft's Qlib library (requires download of data files)

### Setting up Qlib Data

To use Qlib data:

```bash
python scripts/download_us_data.py
```

## Configuration

Environment variables:

- `OPENAI_API_KEY`: Required for RDAgent integration (works with both OpenAI and Azure OpenAI)

## License

MIT