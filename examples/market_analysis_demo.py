"""
Market Analysis System Demo
Demonstrates usage of the market analysis components with sample data
"""
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.market_analysis.market_analyzer import MarketAnalyzer
from src.market_analysis.market_classifier import MarketCondition
from src.market_analysis.strategy_selector import TradingStrategy

def load_sample_data(symbol: str = "AAPL") -> dict:
    """
    Load market data from NASDAQ dataset
    
    Args:
        symbol: Stock symbol to load
        
    Returns:
        dict: Market data in OHLCV format
    """
    data_path = Path(__file__).parent.parent / "data" / "NASDAQ" / f"{symbol}.json"
    
    if not data_path.exists():
        raise ValueError(f"No data found for symbol: {symbol}")
        
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Check if data is in the history array format
    if "history" in raw_data:
        # Extract data from history array
        history = raw_data["history"]
        
        # Initialize arrays for OHLCV data
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        # Extract data from each history item
        for item in history:
            if all(key in item for key in ['open', 'high', 'low', 'close', 'volume']):
                opens.append(float(item['open']))
                highs.append(float(item['high']))
                lows.append(float(item['low']))
                closes.append(float(item['close']))
                volumes.append(float(item['volume']))
        
        # Convert to OHLCV format
        data = {
            'open': np.array(opens, dtype=np.float64),
            'high': np.array(highs, dtype=np.float64),
            'low': np.array(lows, dtype=np.float64),
            'close': np.array(closes, dtype=np.float64),
            'volume': np.array(volumes, dtype=np.float64)
        }
    else:
        # Original format - direct access to OHLCV keys
        data = {
            'open': np.array(raw_data.get('open', []), dtype=np.float64),
            'high': np.array(raw_data.get('high', []), dtype=np.float64),
            'low': np.array(raw_data.get('low', []), dtype=np.float64), 
            'close': np.array(raw_data.get('close', []), dtype=np.float64),
            'volume': np.array(raw_data.get('volume', []), dtype=np.float64)
        }
    
    # Validate data is not empty and contains valid values
    for key, value in data.items():
        if len(value) == 0:
            raise ValueError(f"Empty data array for {key} in symbol: {symbol}")
        
        # Check for NaN or infinite values
        if np.isnan(value).any() or np.isinf(value).any():
            raise ValueError(f"Invalid values (NaN or infinite) found in {key} for symbol: {symbol}")
            
    return data

def print_analysis_results(results: dict) -> None:
    """Print formatted analysis results"""
    print("\n=== Market Analysis Results ===")
    print(f"Market Condition: {results['market_condition'].value}")
    print(f"Selected Strategy: {results['selected_strategy'].value}")
    print(f"\nConfidence Score: {results['analysis']['confidence']:.2f}")
    
    print("\n--- Market Metrics ---")
    metrics = results['metrics']['market']
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Trend Strength: {metrics['trend_strength']:.2f}")
    
    print("\n--- Trading Parameters ---")
    trading_params = results['analysis'].get('metrics', {})
    if 'volume' in trading_params:
        print(f"Relative Volume: {trading_params['volume']['relative_volume']:.2f}")
    if 'volatility' in trading_params:
        print(f"Historical Volatility: {trading_params['volatility']['historical_volatility']:.4f}")
        print(f"Recent Volatility: {trading_params['volatility']['recent_volatility']:.4f}")

def main():
    # Initialize market analyzer
    analyzer = MarketAnalyzer(config={
        'lookback_period': 20,
        'confidence_threshold': 0.6
    })
    
    # List of symbols to try in order if one fails
    symbols = ["AAL"]
    
    for symbol in symbols:
        try:
            # Load sample data
            print(f"Loading market data for {symbol}...")
            market_data = load_sample_data(symbol)
            
            # Print data shape for debugging
            print("Data loaded successfully with the following dimensions:")
            for key, value in market_data.items():
                print(f"  {key}: {value.shape}, range: {np.min(value):.2f} to {np.max(value):.2f}")
            
            # Analyze market conditions
            print("Analyzing market conditions...")
            results = analyzer.analyze_market(market_data)
            
            # Print results
            print_analysis_results(results)
            
            # Get trading parameters for current strategy
            if results['selected_strategy']:
                print("\n--- Recommended Trading Parameters ---")
                params = analyzer.get_trading_parameters(results['selected_strategy'])
                for param, value in params.items():
                    print(f"{param}: {value}")
            
            # Print system summary
            print("\n=== System Summary ===")
            summary = analyzer.get_summary()
            print(f"Current Market Condition: {summary['market_condition'].value}")
            print(f"Active Strategy: {summary['active_strategy'].value}")
            
            # If successful, exit the loop
            break
                
        except Exception as e:
            print(f"Error analyzing market data for {symbol}: {str(e)}")
            # Debug info on exception
            import traceback
            print(f"Error details:\n{traceback.format_exc()}")
            
            if symbol == symbols[-1]:
                print("All symbol attempts failed. Please check data integrity.")
                return
            print(f"Trying next symbol...")
    
if __name__ == "__main__":
    main()