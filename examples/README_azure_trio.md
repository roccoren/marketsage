# Azure OpenAI Trio Stocks Test

This example demonstrates how to use Azure OpenAI with RDAgent for predicting the movements of three major tech stocks: AAPL, MSFT, and NVDA.

## Prerequisites

1. Python 3.7+ environment with StockScan installed
2. Azure OpenAI API access
3. Internet access for downloading stock data

## Configuration Options

### Quick Setup Helper

The easiest way to configure Azure OpenAI is to use the setup helper:

```bash
python examples/setup_azure_env.py
```

This interactive script will:
1. Prompt you for your Azure OpenAI credentials
2. Create or update a .env file with the required settings
3. Guide you through the setup process

### 1. Using Environment Variables (.env file)

Copy the template and fill in your details:

```bash
cp examples/.env.template .env
# Edit .env with your Azure OpenAI details
```

Required variables:
- `OPENAI_API_KEY`: Your Azure OpenAI API key
- `OPENAI_API_BASE`: Your Azure OpenAI endpoint URL
- `OPENAI_DEPLOYMENT_NAME`: Your Azure OpenAI deployment name

### 2. Using Configuration File

Copy the template and fill in your details:

```bash
cp examples/azure_openai_config.json.template azure_config.json
# Edit azure_config.json with your Azure OpenAI details
```

## Data Sources

The example supports two data sources:

### 1. yfinance (Default)

Uses Yahoo Finance API to download stock data. This is the default option and requires no additional setup.

- Fixed: Now correctly uses the `days` parameter instead of date range
- Default: Calculates days from 2019-01-01 to today
- Multiple fallback mechanisms for data retrieval
- Automatic generation of sample data if data fetching fails

### 2. Qlib

Uses Microsoft's Qlib library for quantitative investment. Requires downloading Qlib data first.

To use Qlib:
```bash
# Download Qlib US data and use it for analysis
python examples/azure_openai_trio_test.py --data_source qlib --download_qlib_data
```

## Running the Example

### Basic Usage

```bash
python examples/azure_openai_trio_test.py
```

This will:
1. Load Azure OpenAI settings from environment variables
2. Download data for AAPL, MSFT, and NVDA from yfinance (starting from 2019-01-01)
3. Create and train models for prediction
4. Use RDAgent for model optimization (if available)
5. Generate a report with performance metrics and predictions

### Advanced Usage

```bash
python examples/azure_openai_trio_test.py \
  --config_file azure_config.json \
  --start_date 2022-01-01 \
  --end_date 2023-01-01 \
  --max_iterations 5 \
  --output_dir ./output/my_trio_test \
  --data_source qlib \
  --download_qlib_data \
  --install_rdagent \
  --rdagent_ui
```

Command line arguments:
- `--config_file`: Path to Azure OpenAI configuration file (optional)
- `--start_date`: Start date for historical data (default: 2019-01-01)
- `--end_date`: End date for historical data (default: today)
- `--max_iterations`: Maximum iterations for self-evolution (default: 3)
- `--output_dir`: Directory for output files (default: ./output/azure_trio_test)
- `--data_source`: Data source to use: 'yfinance' or 'qlib' (default: yfinance)
- `--download_qlib_data`: Download Qlib US data if needed (optional)
- `--install_rdagent`: Attempt to install RDAgent if not available (optional)
- `--rdagent_ui`: Start the RDAgent UI (optional)

## Output

The script generates:
1. A JSON file with performance metrics and predictions
2. A Markdown report summarizing the analysis and predictions
3. Model and strategy files saved in the output directory
4. CSV files of the downloaded stock data (when using yfinance)

## Stock Predictions

The report includes directional predictions (UP/DOWN) for each stock with a confidence score, which can be used to inform trading decisions.

## Fault Tolerance

This example is designed to be robust against common issues:

- Automatically tries multiple methods to retrieve stock data
- Falls back to generating synthetic data if real data can't be fetched
- Handles missing RDAgent gracefully
- Provides reasonable defaults when predictions can't be made
- Can attempt to automatically install RDAgent with `--install_rdagent`

## Visualization

If you use the `--rdagent_ui` flag, a web interface will be started at http://localhost:19899 showing real-time evolution progress.

## Troubleshooting

If you encounter errors:

1. Make sure your Azure OpenAI API key is correctly set
2. Verify that your deployment has the necessary permissions
3. If using Qlib and encountering data errors:
   - Try the `--download_qlib_data` flag to get fresh data
   - Or switch to yfinance with `--data_source yfinance`
4. For yfinance connectivity issues:
   - Check your internet connection
   - Try a different date range with `--start_date` and `--end_date`
   - The script will automatically calculate the days parameter for you
5. For RDAgent issues:
   - Use `--install_rdagent` to attempt automatic installation
   - Or manually install with `pip install rdagent` 