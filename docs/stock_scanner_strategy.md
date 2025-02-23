# NASDAQ Stock Scanner Strategy Documentation

## Overview
This document outlines the logic and strategies implemented in the NASDAQ Stock Scanner, a tool designed to identify trading opportunities in US stocks using technical analysis with the LongPort API.

## Stock Universe
- Focuses on NASDAQ stocks filtered by market capitalization
- Uses pre-filtered list from 'nasdaq-500M-50B-listings.json'
- Targets companies with market cap between $500M and $50B
- Processes stocks in batches of up to 100 symbols

## Technical Strategy Components

### 1. MACD (Moving Average Convergence Divergence)
The strategy uses MACD with the following parameters:
- Fast EMA: 12 periods
- Slow EMA: 26 periods
- Signal Line (DEA): 9 periods

MACD Components calculation:
```python
DIF = EMA(12) - EMA(26)
DEA = EMA(9) of DIF
MACD = 2 * (DIF - DEA)
```

### 2. Volume Analysis
Volume confirmation is used to validate signals with:
- 5-day lookback period for volume comparison
- 5% increase threshold above the average volume
- Real-time volume comparison against 5-day average volume

## Signal Generation Criteria

A buy signal is generated when ALL of the following conditions are met:
1. DIF line (MACD) is above zero (indicating bullish momentum)
2. DIF crosses above DEA line (golden cross)
3. Volume shows significant increase (>5% above 5-day average)

### Signal Validation Process:
```
IF (DIF > 0) AND                           # Bullish momentum
   (Previous DIF ≤ Previous DEA) AND       # Line was below/at signal
   (Current DIF > Current DEA) AND         # Line crossed above signal
   (Current Volume > 1.05 * 5-day Avg Vol) # Volume confirmation
THEN Generate Buy Signal
```

### Signal Strength Calculation
Signals are ranked based on a composite strength score (0-1) considering:
- MACD histogram value (40% weight)
- Volume surge magnitude (40% weight)
- Price change (20% weight)

## Data Management

### Historical Data Collection
- Uses LongPort API for real-time and historical data
- Fetches additional 60 days of historical data for calculations
- Implements batch processing (max 100 symbols per batch)
- Includes retry mechanism with exponential backoff

### Error Handling
- Maximum 3 retry attempts for all API operations
- 2-second base delay between retries
- Comprehensive error logging system
- Graceful batch failure handling

## Implementation Features

### Scheduling
- Runs daily during market hours
- Uses AsyncIOScheduler for automation
- Includes misfire grace time of 1 hour
- Coalesces missed executions

### Performance Optimization
- Batch processing of symbols
- Efficient pandas DataFrame operations
- Smart subscription management
- Progressive batch processing with delays

### Results Processing
- Generates unique filenames with strategy name and timestamp
- Sorts signals by strength score
- Formats numeric values for readability
- Saves comprehensive CSV reports

## Output Format
Signal results include:
- Stock Symbol
- Current Price
- Price Change (1-day and 5-day)
- Current and Average Volume
- Volume Ratio
- MACD Components (MACD, Signal, Histogram)
- Signal Strength Score
- Timestamp

## Time Periods
- Default analysis period: Last 7 days
- Historical data fetch: Current - 60 days
- Volume comparison: 5-day lookback