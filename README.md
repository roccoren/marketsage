# MarketSage

A comprehensive quantitative analysis workflow based on Microsoft Qlib, providing advanced features for stock data acquisition, forecasting, backtesting, and visualization.

## Features

- **Data Acquisition**: Fetch historical stock data from Yahoo Finance with retry mechanisms and error handling.
- **Storage Abstraction**: Unified storage interface with multiple backends:
  - CSV storage for simple file-based storage
  - SQLite storage for relational database storage
  - InfluxDB storage for time-series data
  - Mock InfluxDB for testing
- **Forecasting Models**: Hybrid forecasting approach with multiple models:
  - Linear regression models
  - Random forest models
  - Ensemble models with weighted combinations
  - Parameter optimization for model tuning
- **Backtesting**: Sophisticated backtesting framework with:
  - Multiple trading strategies (momentum, mean reversion, forecast-based)
  - Strategy parameter optimization
  - Performance metrics (returns, Sharpe ratio, drawdown)
- **Technical Indicators**: Comprehensive set of technical indicators:
  - Moving averages (SMA, EMA)
  - Relative Strength Index (RSI)
  - Bollinger Bands
  - MACD
  - Stochastic Oscillator
  - Average True Range (ATR)
  - On-Balance Volume (OBV)
- **Risk Management**: Advanced risk metrics:
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Maximum drawdown
  - Sharpe, Sortino, and Calmar ratios
- **MLflow Integration**: Track experiments, parameters, metrics, and artifacts.
- **Configuration Management**: Centralized configuration using YAML/JSON files and environment variables.
- **Visualization Dashboard**: Interactive Dash application for data visualization and model management.
- **Event-Driven Architecture**: Lightweight event dispatcher for inter-module communication.
- **Structured Logging**: Advanced logging with Loguru, including remote logging capabilities.
- **Scheduled Updates**: Daily data updates using APScheduler.
- **Workflow Orchestration**: Flexible workflow management with step-based execution.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/marketsage.git
cd marketsage
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script with default settings:

```bash
python main.py
```

### Advanced Usage

Run with specific stock symbols and date range:

```bash
python main.py --symbols AAPL,MSFT,GOOG --start-date 2021-01-01
```

Use a specific storage backend:

```bash
python main.py --storage-backend sqlite
```

Enable additional features:

```bash
python main.py --indicators --risk-metrics
```

Run with scheduler for periodic updates:

```bash
python main.py --run-scheduler
```

### Workflow Example

Run the workflow example:

```bash
python examples/workflow_example.py --calculate-indicators --optimize-models
```

## Configuration

The project uses a centralized configuration system. You can modify the default configuration in `config/default_config.yaml` or provide your own configuration file:

```bash
python main.py --config my_config.yaml
```

You can also override configuration with environment variables:

```bash
MARKETSAGE_STORAGE_BACKEND=sqlite python main.py
```

### Connection Configuration

Database and service connection strings are centralized in the configuration file under the `connections` section:

```yaml
connections:
  # InfluxDB configuration
  influxdb:
    url: http://localhost:8086
    token: marketsagetoken
    org: marketsage
    bucket: stock_data
  
  # MLflow configuration
  mlflow:
    tracking_uri: http://localhost:5000
    experiment_name: marketsage
    artifact_dir: mlruns
```

These connection settings can be accessed programmatically using the `connection_config` module:

```python
from modules.connection_config import get_influxdb_config, get_mlflow_config

influxdb_config = get_influxdb_config()
mlflow_config = get_mlflow_config()
```

## Project Structure

```
marketsage/
├── config/                  # Configuration files
├── data/                    # Data storage directory
├── docker/                  # Docker configuration
├── docs/                    # Documentation
├── examples/                # Example scripts
├── modules/                 # Core modules
│   ├── storage_backends/    # Storage backend implementations
│   ├── backtesting.py       # Backtesting module
│   ├── config_manager.py    # Configuration management
│   ├── connection_config.py # Connection configuration utility
│   ├── data_acquisition.py  # Data acquisition module
│   ├── event_dispatcher.py  # Event dispatcher
│   ├── forecasting.py       # Forecasting module
│   ├── indicators.py        # Technical indicators
│   ├── logging_manager.py   # Logging management
│   ├── mlflow_integration.py # MLflow integration
│   ├── scheduler.py         # Scheduler for periodic tasks
│   ├── stock_manager.py     # Stock management
│   ├── storage.py           # Storage abstraction
│   └── workflow_manager.py  # Workflow orchestration
├── scripts/                 # Deployment scripts
├── assets/                  # Dashboard assets
├── dash_app.py              # Dash visualization application
├── migrate_to_influxdb.py   # Data migration script
├── main.py                  # Main entry point
└── requirements.txt         # Dependencies
```

## MLflow Integration

The project integrates with MLflow for experiment tracking. To view the MLflow UI:

```bash
mlflow ui
```

Then open your browser at http://localhost:5000

## Visualization Dashboard

The project includes an interactive Dash application for data visualization and model management:

```bash
python dash_app.py
```

Then open your browser at http://localhost:8050

## Deployment Options

### Docker Deployment

You can deploy the middleware components (InfluxDB, PostgreSQL, MLflow) and the Dash application separately:

```bash
# Deploy middleware components
./scripts/deploy_middleware_docker.sh

# Deploy Dash application
./scripts/deploy_dash_docker.sh
```

### Standalone Deployment

You can deploy the Dash application as a standalone Python application:

```bash
./scripts/deploy_dash_standalone.sh
```

### Systemd Service Deployment

You can deploy the Dash application as a systemd service on Linux:

```bash
sudo ./scripts/install_systemd_service.sh
```

### Cloud Deployment

You can deploy the Dash application to Azure:

```bash
# Prepare deployment package
./scripts/cloud/azure/prepare_deployment.sh

# Deploy to Azure App Service
./scripts/cloud/azure/deploy_app_service.sh

# Or deploy to Azure Container Apps
./scripts/cloud/azure/deploy_container_app.sh
```

See the [scripts/README.md](scripts/README.md) file for more details on deployment options.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.