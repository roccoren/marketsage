# Refactoring Guide

## Overview

This document details the major refactoring performed on the market system codebase in version 1.0.0. The refactoring aimed to improve code organization, reduce redundancy, and implement best practices for maintainability and extensibility.

## Core Architecture Changes

### Base Classes

The refactoring introduced a hierarchy of base classes that provide common functionality:

```
BaseProcessor
├── MarketDataProcessor
└── OptimizedMarketProcessor

BaseProcessor
└── MarketDataNormalizer
```

These base classes contain shared functionality for file handling, validation, and normalization, which reduces code duplication and standardizes behavior across the system.

### Class Responsibilities

| Class | Responsibility |
|-------|----------------|
| `BaseProcessor` | Common file handling and data validation |
| `MarketDataProcessor` | Basic market data processing and analysis |
| `OptimizedMarketProcessor` | Memory-efficient, parallel processing for large datasets |
| `MarketDataNormalizer` | Data normalization and standardization |

## Directory Structure

The codebase has been reorganized into the following structure:

```
src/
├── market_system/
│   ├── __init__.py         # Public API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_processor.py
│   │   ├── normalizer.py
│   │   ├── test_utils.py
│   │   └── utils.py
│   ├── market_data_processor.py
│   ├── optimized_market_processor.py
│   ├── normalize_market_data.py
│   ├── optimize_market_data.py
│   └── [other market system modules]
├── tools/
│   ├── batch_processor.py
│   ├── load_to_influxdb.py
│   └── [other utility scripts]
```

This structure separates core functionality from utilities and scripts, making the codebase more maintainable and easier to navigate.

## Key Improvements

### Memory Management

The refactored code includes significant improvements in memory management:

1. **Chunked Processing**: Data is now processed in configurable chunks to reduce memory footprint
2. **Memory Monitoring**: System memory usage is tracked during processing
3. **Adaptive Processing**: Processing pauses or adjusts when memory usage is high

Example:
```python
def _check_memory_usage(self) -> bool:
    """Check if memory usage is below threshold."""
    return check_memory_usage(MEMORY_THRESHOLD)

# Usage in processing
if not self._check_memory_usage():
    logger.warning("Memory usage high, waiting before processing...")
    time.sleep(1)  # Wait before continuing
```

### Parallel Processing

The OptimizedMarketProcessor now includes robust parallel processing capabilities:

1. **Process Pool Executor**: Efficiently distributes work across available CPU cores
2. **Configurable Workers**: Number of worker processes can be adjusted based on system resources
3. **Benchmarking**: Tools to compare parallel vs. sequential performance

```python
# Process symbols in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(process_symbol_files, symbol_dirs))
```

### Standardized API

The codebase now provides a clean, standardized API through the `__init__.py` file:

```python
# Factory functions for creating components
def create_processor(base_path: str, optimized: bool = True) -> BaseProcessor:
    """Create a market data processor."""
    path = Path(base_path)
    if optimized:
        return OptimizedMarketProcessor(path)
    else:
        return MarketDataProcessor(path)
```

This provides a consistent interface for users and hides implementation details.

## Testing Improvements

The refactoring added comprehensive testing utilities:

### Test Fixtures

```python
@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary directory with test market data."""
    # Create temporary directory with test data
    temp_dir = Path(tempfile.mkdtemp())
    try:
        create_test_market_files(temp_dir)
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
```

### Data Generation Utilities

```python
def generate_test_market_data(num_days: int = 30, symbols: List[str] = None) -> pd.DataFrame:
    """Generate test market data for unit tests."""
    # Implementation...
```

## Migration Strategy

### For Script Users

If you were using the standalone scripts directly:

1. Use the command-line interfaces provided by the refactored scripts
2. Update script paths to the new locations
3. Review argument changes in the script help documentation

### For API Users

If you were importing and using the classes:

1. Use the factory functions from the main package
2. Update import statements to reflect the new package structure
3. Review method signature changes

### Examples

**Before:**
```python
from src.market_data_processor import MarketDataProcessor

processor = MarketDataProcessor('/path/to/data')
data = processor.process_market_data('nyse')
```

**After:**
```python
from src.market_system import create_processor

processor = create_processor('/path/to/data')
data = processor.process_market_data('nyse')
```

## Best Practices Implemented

1. **Consistent Error Handling**: Custom exceptions are used to provide clear error information
2. **Comprehensive Logging**: Detailed logging at appropriate levels
3. **Type Hints**: Throughout the codebase for better IDE support and documentation
4. **Documentation**: Comprehensive docstrings in all modules, classes, and functions
5. **Memory Efficiency**: Careful management of memory usage for large datasets
6. **Performance Optimization**: Strategic use of parallel processing and chunking

## Conclusion

This refactoring significantly improves the maintainability, performance, and organization of the codebase. While it introduces some breaking changes, the migration path is straightforward, and the benefits in terms of code quality and maintainability are substantial. 