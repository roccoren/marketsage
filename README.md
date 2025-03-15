# Market Analysis System

A high-performance market data analysis system designed for processing and analyzing large-scale financial market data. The system implements advanced trading strategies, data optimization techniques, and comprehensive reporting capabilities.

## Core Components

### 1. Market Data Processing
- `OptimizedMarketProcessor`: Handles efficient market data processing with memory optimization and parallel execution capabilities
- Built-in data validation and quality checks
- Support for chunked processing of large datasets
- Performance monitoring and benchmarking

### 2. Strategy Analysis
- Multiple trading strategies including:
  - Moving Average Crossover
  - Volatility Breakout
  - Volume Spike Analysis
- Advanced technical indicators with VWAP integration
- Risk metrics calculation
- Performance analytics and backtesting

### 3. Report Generation
- Comprehensive strategy analysis reports
- Real-time performance metrics
- Market sentiment analysis
- Technical signal tracking
- Action recommendations with confidence levels

## Features

### Performance Optimization
- Parallel processing support
- Memory-efficient chunked data handling
- Optimized data structures using pandas categorical types
- Automatic resource management and monitoring

### Data Quality
- Robust validation for market data integrity
- Handling of missing or incomplete data
- Flexible date range filtering
- Comprehensive error logging and reporting

### Technical Analysis
- VWAP-based indicators
- Volume profile analysis
- Price trend detection
- Market sentiment scoring
- Multi-timeframe analysis support

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd stockscan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Generate a strategy analysis report:
```bash
python -m src.market_system.generate_strategy_report [data_path] [market_name] [options]

Options:
  --output-dir DIR     Directory to save reports (default: reports)
  --start-date DATE    Start date for analysis (YYYY-MM-DD format)
  --chunk-size SIZE    Size of data chunks for processing (default: 100000)
```

### Python API

```python
from market_system.strategy_analyzer import StrategyAnalyzer
from market_system.optimized_market_processor import OptimizedMarketProcessor

# Initialize analyzer
analyzer = StrategyAnalyzer('/path/to/data')

# Generate report
report = analyzer.generate_report(
    market_name='ASX',
    chunk_size=100000,
    start_date='2025-01-01'
)
```

## Data Format

The system expects market data in JSON format with the following structure:
```json
{
    "date": "2025-03-15",
    "symbol": "BHP.AX",
    "market": "ASX",
    "close": 44.55,
    "volume": 1234567,
    "trading_status": "active",
    "currency": "AUD",
    "metrics": {
        "indicators": {
            "vwap": 44.32,
            "turnover": 54876543
        },
        "returns": {
            "daily": 0.015,
            "intraday_range": 0.025
        }
    }
}
```

## Performance Considerations

- Memory Usage: The system implements chunked processing to handle large datasets efficiently
- CPU Utilization: Parallel processing is automatically managed based on available cores
- I/O Optimization: Efficient file reading with streaming support for large files
- Data Validation: Robust error handling and data quality checks without compromising performance

## Project Structure

```
stockscan/
├── src/
│   └── market_system/
│       ├── strategy_analyzer.py      # Strategy analysis implementation
│       ├── optimized_market_processor.py  # Data processing engine
│       └── generate_strategy_report.py    # Report generation
├── data/           # Market data directory
│   └── asx/        # Market-specific data
├── reports/        # Generated analysis reports
└── README.md
```

## Testing

### Running Tests
The project uses pytest for unit testing and pytest-cov for coverage reporting.

1. Install test dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

3. Run specific test file:
```bash
python -m pytest tests/test_strategy_analyzer.py
```

### Test Organization
```
tests/
├── __init__.py              # Test configuration and utilities
├── test_strategy_analyzer.py       # Strategy analysis tests
├── test_optimized_market_processor.py  # Data processing tests
└── test_generate_strategy_report.py    # Report generation tests
```

### Coverage Requirements
- Minimum coverage: 80% for all modules
- Critical components require 90% coverage:
  - Strategy analysis logic
  - Data processing pipeline
  - Report generation
- All error handling paths must be tested
- Edge cases must be covered

## Development

### Code Quality
- Follow PEP 8 style guidelines (use flake8)
- Run formatter: `black src/ tests/`
- Run type checking: `mypy src/`
- Implement comprehensive error handling
- Maintain test coverage above required thresholds
- Use type hints for better code maintainability

### Memory Management
- Monitor memory usage with `PerformanceMetrics`
- Implement chunked processing for large datasets
- Use efficient data structures (categorical types, etc.)
- Clean up temporary data properly

## Error Handling

The system implements comprehensive error handling:
- Data validation errors
- Memory allocation issues
- File I/O errors
- Processing errors

Each error is logged with detailed context for debugging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

[Specify License]