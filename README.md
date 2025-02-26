# Python Stock Scanner

A sophisticated Python-based stock market scanner that analyzes multiple exchanges using various technical indicators to identify trading opportunities. The system supports asynchronous data collection, multiple trading strategies, and robust error handling.

## Features

- **Multi-Exchange Support**
  - NASDAQ, NYSE, ASX (Australian Securities Exchange), HKEX (Hong Kong Exchange)
  - Exchange-specific timezone handling
  - Market hours validation

- **Trading Strategies**
  - MACD with volume confirmation
  - ADX-MACD-CCI combined strategy
  - ATR Moving Average strategy
  - RSI-Stochastic strategy
  - On-Balance Volume (OBV) analysis
  - Support for custom strategy implementation

- **Data Management**
  - Asynchronous data collection and updates
  - Efficient data caching and storage
  - Automatic data cleaning and validation
  - Progress tracking for long-running updates
  - Circuit breaker pattern for API protection

- **System Features**
  - Configurable batch processing
  - Comprehensive logging system
  - Multiple output formats (JSON, CSV)
  - Rate limiting and request throttling
  - Error handling and recovery mechanisms

## Requirements

- Python 3.8 or higher
- Required packages:
  ```bash
  pandas>=1.3.0
  numpy>=1.20.0
  pytz>=2021.1
  yfinance>=0.1.63
  aiofiles>=0.8.0
  ```

Full dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stockscanpy
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Docker Support

The application can be run as a containerized service using Docker.

### Building the Image

```bash
docker build -t stockscanpy .
```

### Running the Container

Basic usage:
```bash
docker run -d \
  --name stockscanner \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  stockscanpy
```

With custom strategies:
```bash
docker run -d \
  --name stockscanner \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  stockscanpy --strategies macd,rsi_stoch
```

### Docker Features

- **Multi-stage Build**: Optimized image size using a builder stage for compilation
- **Data Persistence**: Mounted volumes for data, output, and logs
- **Security**:
  - Non-root user `stockscanner`
  - Minimal base image (python:3.11-slim)
  - No unnecessary build dependencies in final image
- **Environment Variables**:
  - `DATA_DIR`: Data storage location (/app/data)
  - `OUTPUT_DIR`: Results output location (/app/output)
  - `LOG_DIR`: Log files location (/app/logs)
- **Health Check**: Built-in health monitoring (30s interval)
- **Port**: Exposes port 8080 for future API endpoints

### Container Management

View logs:
```bash
docker logs stockscanner
```

Stop container:
```bash
docker stop stockscanner
```

Remove container:
```bash
docker rm stockscanner
```

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "exchanges": [
    {
      "name": "NASDAQ",
      "provider": "yahoo",
      "timezone": "America/New_York",
      "marketOpen": "09:30",
      "marketClose": "16:00",
      "validation_symbol": "SPY"
    },
    {
      "name": "ASX",
      "provider": "yahoo",
      "timezone": "Australia/Sydney",
      "marketOpen": "10:00",
      "marketClose": "16:00",
      "validation_symbol": "STW.AX"
    }
  ],
  "providers": {
    "yahooFinance": {
      "rateLimit": 2000,
      "batchSize": 3
    }
  },
  "dataDirectory": "data",
  "outputDirectory": "output",
  "logDirectory": "logs"
}
```

### Configuration Parameters

- **exchanges**: Array of exchange configurations
  - name: Exchange identifier (NASDAQ, NYSE, ASX, HKEX)
  - timezone: Exchange timezone
  - marketOpen/marketClose: Trading hours
  - validation_symbol: Reference symbol for market status checks

- **providers**: API provider configurations
  - rateLimit: Milliseconds between requests
  - batchSize: Number of symbols to process simultaneously

- **directories**: Data storage locations
  - dataDirectory: Historical data storage
  - outputDirectory: Scan results output
  - logDirectory: Log file location

## Usage

### Basic Usage

Run the scanner with default configuration:

```bash
python main.py
```

### Advanced Usage

1. Update stock data:
```bash
python main.py --update-data
```

2. Run specific strategies:
```bash
# Available strategy commands:
python main.py --strategies macd              # MACD strategy with volume confirmation
python main.py --strategies adx_macd_cci     # Combined ADX, MACD, and CCI strategy
python main.py --strategies atr_ma           # ATR Moving Average strategy
python main.py --strategies obv              # On-Balance Volume strategy
python main.py --strategies rsi_stoch        # RSI with Stochastic strategy
python main.py --strategies combined         # Combined signal strategy with MACD and Bollinger Bands
python main.py --strategies crossover        # Price crossover strategy

# Run multiple strategies at once:
python main.py --strategies macd,rsi_stoch,obv
```

3. Specify custom configuration:
```bash
python main.py --config custom_config.json
```

### Implementing Custom Strategies

Create a new strategy by extending the base Strategy class:

```python
from strategies.base import Strategy

class CustomStrategy(Strategy):
    def __init__(self):
        super().__init__("CustomStrategy")
        
    def analyze(self, data):
        # Implement your analysis logic here
        return signals
```

## Output Formats

### JSON Output
Signals are saved in the following format:
```json
{
  "Symbol": "AAPL",
  "Exchange": "NASDAQ",
  "Timestamp": "2024-02-26T10:30:00Z",
  "Strategy": "MACD",
  "Signal": "BUY",
  "Confidence": 0.85
}
```

### CSV Output
Includes columns for:
- Symbol
- Exchange
- Strategy
- Signal Type
- Timestamp
- Technical Indicators
- Price Data
- Volume Data

## Error Handling

The system implements several error handling mechanisms:

1. **Circuit Breaker**
   - Prevents API overload
   - Automatic retry with backoff
   - Configurable thresholds

2. **Data Validation**
   - Market hours verification
   - Data integrity checks
   - Symbol validation

3. **Progress Tracking**
   - Resumable operations
   - Failed symbol tracking
   - Progress persistence

## Directory Structure

```
stockscanpy/
├── main.py              # Main entry point
├── scanner.py          # Core scanner implementation
├── config.json         # Configuration file
├── requirements.txt    # Dependencies
├── strategies/         # Trading strategies
│   ├── base.py        # Base strategy class
│   ├── macd.py        # MACD implementation
│   └── ...            # Other strategies
├── utils/             # Utility functions
│   ├── async_utils.py    # Async helpers
│   ├── data_cleaner.py  # Data processing
│   └── ...
├── data/              # Historical data
├── output/            # Scan results
└── logs/              # Log files
```

## Logging

Logs are stored in the configured log directory with the following information:
- Data update progress
- Strategy execution details
- Error and warning messages
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with:
   - Clear description of changes
   - Updated tests if applicable
   - Documentation updates

## License

© 2025 All Rights Reserved

This software is provided for evaluation and personal use only. Commercial use, modification, redistribution, or use in production environments is strictly prohibited without explicit written permission.

### Restrictions

- No commercial use
- No redistribution
- No modification or derivative works
- No use in production environments

For licensing inquiries or commercial use permission, please contact the repository owner.