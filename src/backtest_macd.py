import pandas as pd
from datetime import datetime
from strategies.macd import MACDStrategy
from strategies.backtester import Backtester
from utils.exchange_data_loader import ExchangeDataLoader
import json
import os
from utils.backtest_formatter import display_results

def load_symbols():
    """Load symbols from exchange listing files."""
    exchanges = {
        'NYSE': 'data/nyse_filtered_listings.json',
        'NASDAQ': 'data/nasdaq_filtered_listings.json',
        'ASX': 'data/asx_filtered_listings.json',
        'HKEX': 'data/hkex_filtered_listings.json'
    }
    
    symbols = {}
    for exchange, file_path in exchanges.items():
        try:
            with open(file_path, 'r') as f:
                symbols[exchange] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find listings file for {exchange}")
            symbols[exchange] = []
    
    return symbols

def run_backtest(exchange: str, symbol: str, data_loader: ExchangeDataLoader):
    """Run MACD strategy backtest for a single symbol."""
    data = data_loader.load_stock_data(exchange, symbol)
    
    if data is None:
        print(f"Could not load data for {symbol} ({exchange})")
        return
    if len(data) < 50:
        print(f"Insufficient data for {symbol} ({exchange})")
        return
    
    try:
        strategy = MACDStrategy()
        backtester = Backtester(strategy)
        result = backtester.run(data)
        
        display_results(symbol, exchange, result)
        
        return result
    except Exception as e:
        print(f"Error during backtest of {symbol} ({exchange}): {str(e)}")
        return None

def save_results(results: dict, output_dir: str = "output"):
    """Save backtest results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
    
    serializable_results = {}
    for exchange, exchange_results in results.items():
        serializable_results[exchange] = {}
        for symbol, result in exchange_results.items():
            if result and result.metrics:
                serializable_results[exchange][symbol] = result.metrics
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def main():
    data_loader = ExchangeDataLoader()
    exchange_symbols = load_symbols()
    results = {}
    for exchange, symbols in exchange_symbols.items():
        print(f"\nProcessing {exchange} symbols...")
        print("=" * 50)
        results[exchange] = {}
        for symbol in symbols[:5]:
            try:
                print(f"\nAnalyzing {symbol}...")
                result = run_backtest(exchange, symbol, data_loader)
                if result:
                    results[exchange][symbol] = result
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
    save_results(results)

if __name__ == "__main__":
    main()