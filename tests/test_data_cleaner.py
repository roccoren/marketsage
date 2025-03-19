import logging

import pandas as pd

from utils.data_cleaner import clean_historical_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def test_dict_data():
    print("\nTesting dictionary data format:")
    # Test with dictionary format
    test_dict = {
        "data": [
            {
                "date": "2024-01-01",
                "open": "100.5",
                "high": "102.0",
                "low": "99.0",
                "close": "101.0",
                "volume": "1000000",
            },
            {
                "date": "2024-01-02",
                "open": "101.0",
                "high": "103.0",
                "low": "100.0",
                "close": "102.5",
                "volume": "1200000",
            },
        ]
    }

    result = clean_historical_data(test_dict)
    if result is not None:
        print("Dictionary data cleaned successfully:")
        print(result)
    else:
        print("Failed to clean dictionary data")


def test_dataframe():
    print("\nTesting DataFrame format:")
    # Test with DataFrame format
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100.5, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.5],
            "volume": [1000000, 1200000],
        }
    )

    result = clean_historical_data(df)
    if result is not None:
        print("DataFrame cleaned successfully:")
        print(result)
    else:
        print("Failed to clean DataFrame")


def test_malformed_data():
    print("\nTesting malformed data:")
    # Test with malformed data
    test_dict = {
        "data": [
            "malformed data string",
            {"date": "invalid-date", "open": "not-a-number"},
        ]
    }

    result = clean_historical_data(test_dict)
    if result is None:
        print("Successfully handled malformed data (returned None as expected)")
    else:
        print("Unexpected result with malformed data")


if __name__ == "__main__":
    test_dict_data()
    test_dataframe()
    test_malformed_data()
