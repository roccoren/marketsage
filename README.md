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

Build and run using Docker:
```bash
docker build -t stockscan .
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output stockscan
```

## Usage

After installation, you can run the stock scanner in several ways:

### As an installed package:
```bash
# From anywhere after installing the package
stockscan -exchanges HKEX NYSE
```

### From source directory:
```bash
# From the project root directory
python -m src.main -exchanges HKEX NYSE
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