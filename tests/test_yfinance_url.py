from functools import partial

import requests
import yfinance as yf

# Store original get function
original_get = requests.get


def intercepted_get(url, *args, **kwargs):
    print(f"\nRequesting URL: {url}")
    return original_get(url, *args, **kwargs)


# Monkey patch requests.get
requests.get = intercepted_get

# Test with a sample stock
ticker = yf.Ticker("AAPL")

print("Testing info endpoint...")
info = ticker.get_info()

print("\nTesting history endpoint...")
history = ticker.history(period="1d")
