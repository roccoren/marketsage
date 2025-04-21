# MarketSage DataHub

DataHub is a component of the MarketSage ecosystem responsible for data acquisition, management, and preprocessing of financial market data.

## Overview

DataHub handles the following key responsibilities:

- Stock data downloading and acquisition
- Symbol management
- Data cleaning and preprocessing
- Time-series data storage and retrieval

## Architecture

DataHub is designed as a standalone service that communicates with other MarketSage components through:

1. **gRPC** - For efficient time-series data operations via the Time Series Data Service
2. **RabbitMQ** - For asynchronous task processing and event notifications

## Directory Structure

```
marketsage-datahub/
├── src/
│   ├── acquisition/    # Data acquisition functionality
│   ├── symbols/        # Symbol management
│   ├── storage/        # Storage interfaces
│   └── preprocessing/  # Data cleaning and preprocessing
├── tests/              # Unit and integration tests
├── config/             # Configuration files
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Access to InfluxDB
- RabbitMQ instance

### Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Configuration

Create a configuration file in the `config` directory:

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` to set up your data sources and storage options.

## Usage

```python
from marketsage_datahub.acquisition import DataAcquisition
from marketsage_datahub.symbols import SymbolManager

# Initialize components
symbol_manager = SymbolManager()
data_acquisition = DataAcquisition()

# Get available symbols
symbols = symbol_manager.get_symbols()

# Download data for symbols
data_acquisition.download_data(symbols, start_date="2023-01-01")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Docker Image

```bash
docker build -t marketsage-datahub .
```

## License

[MIT License](LICENSE)