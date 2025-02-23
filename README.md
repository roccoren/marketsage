# Python Stock Scanner

A Python implementation of a stock market scanner that analyzes multiple exchanges using technical indicators like MACD to identify trading opportunities.

## Features

- Supports multiple exchanges (NASDAQ, NYSE, ASX)
- MACD strategy implementation with volume confirmation
- Configurable parameters via JSON configuration file
- CSV and JSON output formats
- Comprehensive logging
- Batch processing capability

## Requirements

- Python 3.8 or higher
- Required packages listed in requirements.txt

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `config.json` file based on the example provided:

```json
{
  "exchanges": [
    {
      "name": "NASDAQ",
      "provider": "yahoo",
      "timezone": "America/New_York",
      "marketOpen": "09:30",
      "marketClose": "16:00"
    }
  ],
  "providers": {
    "yahooFinance": {
      "rateLimit": 1000,
      "batchSize": 20
    }
  },
  "dataDirectory": "data",
  "outputDirectory": "output",
  "logDirectory": "logs"
}
```

## Usage

Run the scanner with default configuration:

```bash
python main.py
```

Specify a custom configuration file:

```bash
python main.py --config path/to/config.json
```

## Output

The scanner generates two types of output files in the specified output directory:

1. CSV file (`macd_signals_TIMESTAMP.csv`)
2. JSON file (`macd_signals_TIMESTAMP.json`)

Both files contain:
- Stock Symbol
- Current Price
- Price Changes (1-day and 5-day)
- Volume Information
- MACD Components
- Signal Strength Score
- Timestamp

## Strategy Details

### MACD Strategy
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Signal Line: 9 periods
- Volume confirmation: 5% above 5-day average

Buy signals are generated when:
1. MACD line is above zero
2. MACD crosses above signal line
3. Volume shows significant increase

## Directory Structure

```
python_stockscan/
├── main.py
├── scanner.py
├── requirements.txt
├── strategies/
│   ├── __init__.py
│   ├── base.py
│   └── macd.py
├── data/
├── output/
└── logs/
```

## Error Handling

The application includes comprehensive error handling for:
- Configuration validation
- File I/O operations
- Data processing
- Strategy execution

All errors are logged to the configured log directory.