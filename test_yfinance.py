import yfinance as yf

# Test fetching data for AAPL and MSFT
symbols = ['AAPL', 'MSFT']

for symbol in symbols:
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="5d")
    print(f"Data for {symbol}:\n{data}\n")