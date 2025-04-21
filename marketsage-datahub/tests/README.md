# MarketSage DataHub Tests

This directory contains unit tests for the MarketSage DataHub component.

## Running Tests

You can run the tests using pytest:

```bash
# Run all tests
pytest

# Run a specific test file
pytest test_acquisition.py

# Run a specific test case
pytest test_acquisition.py::TestDataAcquisition::test_download_data_success

# Run tests with coverage report
pytest --cov=marketsage_datahub
```

## Test Organization

The tests are organized by module:

- `test_acquisition.py`: Tests for the data acquisition module
- `test_symbols.py`: Tests for the symbol management module
- `test_storage.py`: Tests for the storage module
- `test_preprocessing.py`: Tests for the data preprocessing module
- `test_cli.py`: Tests for the command-line interface

## Writing New Tests

When adding new functionality, please add corresponding tests. Follow these guidelines:

1. Create test classes that inherit from `unittest.TestCase`
2. Use descriptive test method names that explain what is being tested
3. Use mocks for external dependencies
4. Include both success and failure test cases
5. Aim for high test coverage

## Test Dependencies

The tests use the following libraries:

- pytest: Test runner
- unittest.mock: For mocking dependencies
- pytest-cov: For coverage reporting