import pandas as pd
import json
import gzip
from pathlib import Path
import logging
from strategies.strategy_optimizer import AdvancedStrategyOptimizer
from utils.technical_indicators import TechnicalIndicators
from stock_data_fetcher.yahoo_finance_client import YahooFinanceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_market_data(data_dir: str) -> dict:
    """Load market data from multiple exchanges"""
    market_data = {}
    
    # Walk through the data directory
    for market_dir in Path(data_dir).iterdir():
        if market_dir.is_dir():
            market_name = market_dir.name
            logger.info(f"Loading data for market: {market_name}")
            
            # Load all symbol data files in this market
            for data_file in market_dir.glob("*.json.gz"):
                try:
                    symbol = data_file.stem.split('.')[0]  # Remove .json.gz extension
                    logger.info(f"Processing {symbol}")
                    
                    # Read gzipped JSON file
                    with gzip.open(data_file, 'rt') as f:
                        data = json.load(f)['history']
                        
                    # Extract OHLCV data
                    df = pd.DataFrame(data)
                    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    
                    missing_cols = set(required_columns) - set(df.columns)
                    if missing_cols:
                        logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                        continue
                    
                    # Set index to datetime
                    df['Date'] = pd.to_datetime(df['date'])
                    df.set_index('Date', inplace=True)
                    df.drop('date', axis=1, inplace=True)
                    
                    # Rename columns to match expected format
                    df.columns = df.columns.str.title()
                    
                    # Remove any rows with missing values
                    df = df.dropna()
                    df.sort_index(inplace=True)
                    
                    # Basic data validation
                    if len(df) > 0:
                        market_data[f"{symbol}_{market_name}"] = df
                        logger.info(f"Loaded {len(df)} rows for {symbol}")
                    else:
                        logger.warning(f"Skipping {symbol}: No valid data")
                        
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {str(e)}")
                    continue
    
    return market_data

def main():
    """Run the comprehensive technical analysis optimization"""
    logger.info("Starting technical analysis optimization process...")
    
    # Load market data
    data_dir = "test_data"
    market_data = load_market_data(data_dir)
    
    if not market_data:
        logger.error("No valid market data found!")
        return
        
    logger.info(f"Loaded data for {len(market_data)} symbols")
    
    try:
        # Initialize optimizer with the loaded data
        optimizer = AdvancedStrategyOptimizer(
            data_dict=market_data,
            train_size=0.8,
            window_size=252,  # 1 year
            step_size=63,     # ~3 months
            min_window_size=126  # 6 months minimum
        )
        
        # Run optimization
        results = optimizer.optimize_strategy(initial_capital=100000)
        
        # Generate and save detailed report
        report = optimizer.get_optimization_report(results)
        
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed report
        with open(output_dir / "optimization_report.txt", "w") as f:
            f.write(report)
        
        # Save raw results as JSON
        with open(output_dir / "optimization_results.json", "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {
                'market_performances': {
                    k: {
                        'avg_sharpe': float(v['avg_sharpe']),
                        'avg_profit_factor': float(v['avg_profit_factor']),
                        'avg_win_rate': float(v['avg_win_rate']),
                        'best_regime_params': v['best_regime_params']
                    }
                    for k, v in results['market_performances'].items()
                },
                'param_stability': {
                    k: {
                        metric: float(value)
                        for metric, value in v.items()
                    }
                    for k, v in results['param_stability'].items()
                },
                'regime_specific_params': results['regime_specific_params']
            }
            
            json.dump(json_results, f, indent=2)
            
        logger.info("Optimization completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary statistics
        print("\n=== Optimization Summary ===")
        print("\nBest Parameters by Market Regime:")
        for regime, params in results['regime_specific_params'].items():
            print(f"\n{regime}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        print("\nMarket Performance Summary:")
        for market, perf in results['market_performances'].items():
            print(f"\n{market}:")
            print(f"  Sharpe Ratio: {perf['avg_sharpe']:.2f}")
            print(f"  Profit Factor: {perf['avg_profit_factor']:.2f}")
            print(f"  Win Rate: {perf['avg_win_rate']:.2%}")
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
