# StockScan Data Directory Structure

This document details the organization of the data directory in the StockScan project, designed based on Qlib's best practices.

## Overall Structure

```
data/
├── raw/                      # Raw data
│   ├── nyse/                 # New York Stock Exchange data
│   ├── nasdaq/               # NASDAQ data
│   ├── asx/                  # Australian Securities Exchange data
│   └── hkex/                 # Hong Kong Exchange data
├── processed/                # Preprocessed data
│   ├── nyse/
│   ├── nasdaq/
│   ├── asx/
│   └── hkex/
├── features/                 # Feature data
│   ├── basic/                # Basic features
│   ├── technical/            # Technical indicator features
│   └── derived/              # Derived features
├── labels/                   # Label data
│   ├── price_label/          # Price-related labels
│   └── trend_label/          # Trend-related labels
└── cache/                    # Cached data
    ├── backtest/             # Backtesting cache
    └── indicators/           # Indicator calculation cache
```

## Detailed Description

### 1. Raw Data (raw)

The raw data directory stores unprocessed stock data from different exchanges. Data is divided by exchange and time, using a hierarchical structure for easy management and access.

#### Data Source Division
- `nyse/`: New York Stock Exchange stock data
- `nasdaq/`: NASDAQ stock data
- `asx/`: Australian Securities Exchange stock data
- `hkex/`: Hong Kong Exchange stock data

#### Data File Format
Data under each exchange directory is organized as follows:
```
nyse/
├── daily/                    # Daily data
│   ├── 2020/                 # Organized by year
│   │   ├── AAPL.csv          # Individual stock data files
│   │   ├── MSFT.csv
│   │   └── ...
│   ├── 2021/
│   └── ...
├── intraday/                 # Minute-level data
│   ├── 2020/
│   │   ├── AAPL/             # Divided by stock
│   │   │   ├── 01.csv        # Files divided by month
│   │   │   ├── 02.csv
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── metadata/                 # Metadata
    ├── symbols.csv           # Stock list
    ├── trading_calendar.csv  # Trading calendar
    └── sectors.csv           # Industry classification
```

#### CSV File Structure
Daily data files (such as `AAPL.csv`) include the following columns:
- Date: Trading date (YYYY-MM-DD)
- Open: Opening price
- High: Highest price
- Low: Lowest price
- Close: Closing price
- Volume: Trading volume
- Adj Close: Adjusted closing price

### 2. Preprocessed Data (processed)

The preprocessed data directory stores data that has been cleaned, normalized, and preprocessed. The structure is similar to raw data, but the data has undergone the following processing:
- Missing value handling
- Outlier detection and processing
- Data normalization
- Adjustment processing
- Trading calendar alignment

```
processed/
├── nyse/
│   ├── daily/
│   │   ├── 2020-2022/        # Merged multi-year data
│   │   │   ├── AAPL.parquet  # Using Parquet format for efficiency
│   │   │   └── ...
│   │   └── ...
│   └── calendar_aligned/     # Calendar-aligned data
│       └── ...
├── nasdaq/
│   └── ...
└── ...
```

### 3. Feature Data (features)

The feature data directory stores various features extracted from raw and preprocessed data, used for model training and strategy development.

```
features/
├── basic/                    # Basic features
│   ├── price_volume/         # Price and volume derived features
│   │   ├── nyse/
│   │   │   ├── 2020-2022/
│   │   │   │   ├── AAPL.parquet
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── financial/            # Fundamental features
│       └── ...
├── technical/                # Technical indicator features
│   ├── momentum/             # Momentum indicators
│   │   ├── rsi/              # RSI indicator
│   │   ├── macd/             # MACD indicator
│   │   └── ...
│   ├── volatility/           # Volatility indicators
│   │   ├── atr/              # Average True Range
│   │   ├── bollinger/        # Bollinger Bands
│   │   └── ...
│   └── ...
└── derived/                  # Advanced derived features
    ├── trend_strength/       # Trend strength features
    ├── pattern_recognition/  # Pattern recognition features
    └── cross_sectional/      # Cross-sectional features
```

Feature files are in Parquet format and include the following information:
- Feature ID
- Timestamp
- Feature value
- Feature metadata (such as calculation parameters)

### 4. Label Data (labels)

The label data directory stores label data used for supervised learning, such as future price changes, trend directions, etc.

```
labels/
├── price_label/              # Price-related labels
│   ├── return/               # Return labels
│   │   ├── 1d/               # 1-day return
│   │   ├── 5d/               # 5-day return
│   │   └── ...
│   └── volatility/           # Volatility labels
│       └── ...
└── trend_label/              # Trend-related labels
    ├── direction/            # Direction labels (up/down)
    └── strength/             # Trend strength labels
```

### 5. Cached Data (cache)

The cache data directory stores temporary calculation results to avoid repeated calculations and improve performance.

```
cache/
├── backtest/                 # Backtesting cache
│   ├── strategy_results/     # Strategy backtesting results cache
│   └── performance_metrics/  # Performance metrics cache
└── indicators/               # Indicator calculation cache
    ├── daily/                # Daily indicator cache
    └── intraday/             # Minute indicator cache
```

## Data Flow

1. **Data Acquisition**: Obtain raw stock data through APIs or data providers, save to the `raw` directory
2. **Data Preprocessing**: Clean and normalize raw data, save to the `processed` directory
3. **Feature Extraction**: Calculate various features from preprocessed data, save to the `features` directory
4. **Label Generation**: Generate labels based on historical data, save to the `labels` directory
5. **Model Training and Backtesting**: Train models and backtest strategies using feature and label data, cache results in the `cache` directory

## Data Version Control

To track data changes and ensure experiment reproducibility, we use the following version control methods:

1. Add timestamps to each data update
2. Save configuration files and parameters for each data processing step in the `metadata` subdirectory
3. Use a data manifest (`manifest.json`) to record dataset characteristics, sources, and processing history

## Data Access

For efficient data access, StockScan provides the following interfaces:

1. `DataLoader` class: For loading and processing data at different levels
2. `FeatureStore` class: For managing and accessing feature data
3. `DatasetBuilder` class: For building datasets needed for model training and backtesting

## Data Management Scripts

The following scripts are used to manage the data directory:

1. `scripts/download_data.py`: Download raw data
2. `scripts/update_data.py`: Update existing data
3. `scripts/generate_features.py`: Generate feature data
4. `scripts/clean_cache.py`: Clean cached data 