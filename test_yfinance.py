import yfinance as yf
from utils.yfinance_utils import get_cached_session

# Test fetching data for AAPL and MSFT
symbols = ['AAPL', 'MSFT']
session = get_cached_session()

for symbol in symbols:
    ticker = yf.Ticker(symbol, session=session)
    data = ticker.history(period="5d")
    print(f"Data for {symbol}:\n{data}\n")