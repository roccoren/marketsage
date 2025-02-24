import pandas as pd
from datetime import datetime
from strategies.macd import MACDStrategy
from strategies.backtester import Backtester
from utils.exchange_data_loader import ExchangeDataLoader
import json
import os

def format_metrics(metrics):
    """Format backtest metrics for display."""
    return f"""
Backtest Results:
----------------
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2%}
Total P&L: ${metrics['total_pnl']:.2f}
Average Return: {metrics['avg_return']:.2f}%
Average Win: {metrics['avg_winning_return']:.2f}%
Average Loss: {metrics['avg_losing_return']:.2f}%
Max Drawdown: ${metrics['max_drawdown']:.2f}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Profit Factor: {metrics['profit_factor']:.2f}
    """

def load_symbols():
    """Load symbols from exchange listing files."""
    exchanges = {
        'NYSE': 'data/nyse_valid_listings.json',
        'NASDAQ': 'data/nasdaq_valid_listings.json',
        'ASX': 'data/asx_valid_listings.json',
        'HKEX': 'data/hk_stocks.json'
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

def run_backtest(exchange: str, symbol: str, data_loader: ExchangeDataLoader) -> None:
    """Run MACD strategy backtest for a single symbol."""
    # Load historical data
    data = data_loader.load_stock_data(exchange, symbol)
    
    if data is None:
        print(f"Could not load data for {symbol} ({exchange})")
        return
        
    if len(data) < 50:  # Minimum data requirement
        print(f"Insufficient data for {symbol} ({exchange})")
        return
    
    try:
        # Initialize strategy and backtester
        strategy = MACDStrategy()
        backtester = Backtester(strategy)
        
        # Run backtest
        result = backtester.run(data)
        
        # Print results
        print(f"\nBacktest Results for {symbol} ({exchange})")
        print("=" * 40)
        print(format_metrics(result.metrics))
        
        # Print trade details
        print("\nTrade History:")
        print("-" * 40)
        for trade in result.trades:
            print(f"""
Trade Type: {trade.signal_type}
Entry Date: {trade.entry_date.date()}
Entry Price: ${trade.entry_price:.2f}
Exit Date: {trade.exit_date.date()}
Exit Price: ${trade.exit_price:.2f}
P&L: ${trade.pnl:.2f} ({trade.return_pct:.2f}%)
            """)
            
        return result
        
    except Exception as e:
        print(f"Error during backtest of {symbol} ({exchange}): {str(e)}")
        return None

def save_results(results: dict, output_dir: str = "output"):
    """Save backtest results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
    
    # Convert results to serializable format
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
    # Initialize data loader
    data_loader = ExchangeDataLoader()
    
    # Load symbols for each exchange
    exchange_symbols = load_symbols()
    
    # Store results for all backtests
    results = {}
    
    # Process each exchange
    for exchange, symbols in exchange_symbols.items():
        print(f"\nProcessing {exchange} symbols...")
        print("=" * 50)
        
        results[exchange] = {}
        
        # Take first 5 symbols for testing (remove this limit for full analysis)
        for symbol in symbols[:5]:
            try:
                print(f"\nAnalyzing {symbol}...")
                result = run_backtest(exchange, symbol, data_loader)
                if result:
                    results[exchange][symbol] = result
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
    
    # Save results to file
    save_results(results)

if __name__ == "__main__":
    main()