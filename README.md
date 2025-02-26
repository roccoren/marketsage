# Stock Scanner

A Python-based stock analysis system that supports multiple trading strategies and exchange data sources.

## Project Structure

```
stockscan/
├── config/           # Configuration files
│   ├── config.json   # Main configuration
│   └── requirements.txt
├── data/            # Stock data storage
├── dist/            # Build output directory
├── docs/            # Documentation
│   ├── plan.md
│   └── strategy.md
├── output/          # Analysis reports and results
├── src/             # Source code
│   ├── strategies/  # Trading strategies
│   ├── utils/      # Utility functions
│   ├── main.py     # Application entry point
│   ├── scanner.py  # Core scanner implementation
│   └── ...
└── tests/           # Test files
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd stockscan
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install package in development mode:
```bash
pip install -e .
```

This will install the package with all dependencies from `config/requirements.txt`.

### Docker Installation

1. Build the Docker image:
```bash
docker build -t stockscanpy .
```

2. Run with default settings:
```bash
docker run \
  --name stockscan \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  stockscanpy
```

3. Run with specific trading strategies:
```bash
docker run \
  --name stockscan \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  stockscanpy --strategies rsi_stoch,atr_ma
```

4. Run with specific exchanges:
```bash
docker run \
  --name stockscan \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  stockscanpy --exchanges HKEX NYSE
```

The container uses volume mounts to persist:
- `/app/data`: Stock data and cache
- `/app/output`: Analysis reports and results
- `/app/logs`: Application logs

### Docker Environment Variables

You can configure the following environment variables:
```bash
docker run \
  --name stockscan \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  -e SCAN_DAYS=30 \
  stockscanpy
```

Available environment variables:
- `SCAN_DAYS`: Number of days of historical data to analyze (default: 60)
- `DATA_DIR`: Override data directory location
- `OUTPUT_DIR`: Override output directory location
- `LOG_DIR`: Override logs directory location
- `PYTHONUNBUFFERED`: Control Python output buffering
- `PYTHONDONTWRITEBYTECODE`: Control Python bytecode generation

### Docker Troubleshooting

1. Permissions issues:
```bash
# Fix permission issues with mounted volumes
sudo chown -R $USER:$USER data output logs
```

2. Clean restart:
```bash
# Remove existing container
docker rm stockscan

# Remove and rebuild image
docker rmi stockscanpy
docker build -t stockscanpy .
```

## Usage

After installation, you can run the stock scanner in several ways:

### As an installed package:
```bash
# From anywhere after installing the package
stockscan --exchanges HKEX NYSE
```

### From source directory:
```bash
# From the project root directory
python -m src.main --exchanges HKEX NYSE
```

### With specific strategies:
```bash
stockscan --strategies rsi_stoch,atr_ma,adx_macd_cci,obv
```

### Available Strategies

- RSI/Stochastic (rsi_stoch)
- ATR/Moving Average (atr_ma)
- ADX/MACD/CCI (adx_macd_cci)
- On-Balance Volume (obv)

## Configuration

The main configuration file is located at `config/config.json`. This file contains:
- Data directory settings
- Output directory configuration
- Exchange configurations
- Provider settings

Example configuration:
```json
{
  "data_directory": "data",
  "output_directory": "output",
  "exchanges": [
    {"name": "HKEX"},
    {"name": "NYSE"}
  ]
}
```

## Directory Details

- `/config`: Contains configuration files for the application
- `/data`: Stores downloaded stock data and intermediate processing files
- `/dist`: Contains build artifacts and deployment files
- `/docs`: Project documentation and technical specifications
- `/output`: Analysis reports, trading signals, and program output
- `/src`: Source code organized by module
- `/tests`: Test files and test data

## Development

To run tests:
```bash
python -m pytest tests/
```

The project uses a src-layout pattern for better package organization and distribution. All source code is under the `src/` directory, with proper package imports within the codebase.