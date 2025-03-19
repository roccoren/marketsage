# LEAN Data Directory Structure

## Overview
LEAN expects a specific directory structure for storing market data. This document outlines the expected structure and file formats.

## Base Directory Structure
```
data/
├── equity/                  # Stock market data
│   ├── usa/                # US market data
│   │   ├── daily/         # Daily resolution data
│   │   ├── minute/        # Minute resolution data
│   │   └── second/        # Second resolution data
│   ├── nasdaq/             # NASDAQ specific data
│   ├── nyse/              # NYSE specific data
│   ├── asx/               # Australian Securities Exchange data
│   ├── hkex/              # Hong Kong Exchange data
│   ├── tsx/               # Toronto Stock Exchange data
│   └── sgx/               # Singapore Exchange data
├── forex/                  # Foreign exchange data
├── crypto/                 # Cryptocurrency data
└── alternative/            # Alternative data sources

## File Formats

### Stock Data Files
Stock data files are stored in ZIP format with the following naming convention:
```
{SYMBOL}_{RESOLUTION}_{YYYYMMDD}.zip
```
Example: `AAPL_daily_20240319.zip`

Each ZIP file contains CSV data with the following columns:
- Time: Timestamp in format YYYYMMDD HH:mm:ss
- Open: Opening price
- High: Highest price
- Low: Lowest price
- Close: Closing price
- Volume: Trading volume

### Auxiliary Data Files
Located in the auxiliary data directory for each market:

1. **Factor Files** (`*.csv`):
   - Contains split and dividend adjustment factors
   - Used for calculating adjusted prices
   - Format: Date, Price Factor, Split Factor, Dividend Factor

2. **Symbol Properties** (`symbol-properties/*.csv`):
   - Contains metadata about each symbol
   - Includes: Symbol, Description, Exchange, Currency

3. **Market Hours** (`market-hours/*.json`):
   - Defines trading hours and holidays for each market
   - Specifies regular and extended trading sessions

## Example Directory Structure
```
data/
├── equity/
│   ├── usa/
│   │   ├── daily/
│   │   │   ├── AAPL_daily_20240319.zip
│   │   │   └── MSFT_daily_20240319.zip
│   │   └── minute/
│   │       ├── AAPL/
│   │       └── MSFT/
│   └── asx/
│       └── daily/
│           └── BHP_daily_20240319.zip
├── auxiliary/
│   ├── factor_files/
│   │   ├── usa/
│   │   │   ├── AAPL.csv
│   │   │   └── MSFT.csv
│   │   └── asx/
│   │       └── BHP.csv
│   ├── symbol-properties/
│   │   ├── usa.csv
│   │   └── asx.csv
│   └── market-hours/
│       ├── usa.json
│       └── asx.json
```

## Data Requirements

1. **Minimum Data Requirements**:
   - Daily price data files for each symbol
   - Factor files for adjusting prices
   - Market hours configuration
   - Symbol properties

2. **Optional Data**:
   - Minute/second resolution data
   - Extended market hours data
   - Additional metadata

## Data Sources
LEAN can work with data from various sources:
1. QuantConnect data (requires subscription)
2. Custom data providers
3. Self-collected data (must follow the format specifications)

## Common Issues and Solutions

1. **Missing Data Files**:
   - Ensure all required ZIP files are present
   - Check file naming conventions
   - Verify file permissions

2. **Invalid Data Format**:
   - Verify CSV column order
   - Check timestamp formats
   - Ensure numeric values are properly formatted

3. **Market Hours Configuration**:
   - Verify JSON format
   - Check timezone settings
   - Confirm holiday calendar accuracy 