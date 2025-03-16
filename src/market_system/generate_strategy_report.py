"""
Main script for generating market strategy analysis reports.
Provides a command-line interface for running strategy analysis.
"""

import argparse
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from market_system.strategy_analyzer import StrategyAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_empty_signals() -> Dict[str, List[Dict]]:
    """Create an empty signals dictionary with proper structure."""
    return {
        'moving_average_crossover': [],
        'volatility_breakout': [],
        'volume_spike': []
    }

def calculate_signal_strength(group: pd.DataFrame) -> Optional[Dict]:
    """Calculate signal strength and create signal record for a symbol group."""
    try:
        # Calculate components with safety checks
        volume_ratio = group['volume_ratio'].clip(0, 10)
        momentum = abs(((group['price'] - group['SMA_20']) /
                       group['SMA_20']).clip(-0.2, 0.2))
        vwap_align = abs(group['price_to_vwap'] - 1)
        
        # Combined signal strength (0-100 scale)
        strength = (volume_ratio * 0.4 +
                   momentum * 0.3 +
                   vwap_align * 0.3) * 100
        
        return {
            'symbol': group['symbol'].iloc[0],
            'trigger_dates': group['timestamp'].dt.date.tolist(),
            'signal_strength': round(float(strength.mean()), 2),
            'average_volume_ratio': round(float(volume_ratio.mean()), 2),
            'price_to_vwap': round(float(group['price_to_vwap'].mean()), 3)
        }
    except Exception:
        return None
        
def get_recent_strategy_signals(df: pd.DataFrame, days: int = 10,
                              analyzer: Optional[StrategyAnalyzer] = None) -> Dict[str, List[Dict]]:
    """Process market data and detect recent trading signals."""
    signals = create_empty_signals()
    
    try:
        # Input validation
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning(f"Invalid input data: {'empty' if df is None or df.empty else type(df)}")
            return signals
        
        if 'timestamp' not in df.columns:
            logger.error("Required column 'timestamp' not found")
            return signals
            
        if analyzer is None:
            analyzer = StrategyAnalyzer(Path('.'))
            logger.info("Created temporary analyzer instance")
            
        # Extract recent data
        all_dates = sorted(df['timestamp'].dt.date.unique())
        if len(all_dates) < days:
            logger.warning(f"Limited data: {len(all_dates)} days available (requested {days})")
            days = len(all_dates)
            
        recent_dates = all_dates[-days:]
        recent_mask = df['timestamp'].dt.date.isin(recent_dates)
        recent_data = df[recent_mask].copy()
        
        if recent_data.empty:
            logger.warning("No data available in specified date range")
            return signals
            
        # Process data and calculate indicators
        logger.info(f"Processing {len(recent_data):,} records")
        processed_data = analyzer._calculate_indicators(recent_data)
        
        # Verify indicators
        required = ['SMA_20', 'SMA_50', 'volatility', 'volume_ratio', 'price_to_vwap']
        missing = [ind for ind in required if ind not in processed_data.columns]
        if missing:
            logger.error(f"Missing indicators: {missing}")
            return signals
            
        # Get strategy signals
        strategy_results = analyzer.apply_strategy_filters(processed_data)
        total_signals = 0
        
        # Process each strategy's signals
        for strategy, strategy_df in strategy_results.items():
            if strategy_df.empty:
                continue
                
            strategy_signals = []
            for symbol, group in strategy_df.groupby('symbol'):
                try:
                    signal = calculate_signal_strength(group)
                    if signal:
                        strategy_signals.append(signal)
                        total_signals += 1
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {str(e)}")
                    continue
            
            if strategy_signals:
                signals[strategy] = sorted(
                    strategy_signals,
                    key=lambda x: x['signal_strength'],
                    reverse=True
                )
                logger.debug(f"Found {len(strategy_signals)} signals for {strategy}")
        
        if total_signals > 0:
            logger.info(f"Found {total_signals} signals across {len(signals)} strategies")
        else:
            logger.warning("No valid signals detected")
            
    except Exception as e:
        logger.error(f"Error in signal analysis: {str(e)}")
        logger.debug("Error details:", exc_info=True)
    
    return signals

def format_recent_signals_report(signals: Dict) -> str:
    """
    Format the recent strategy signals report in a clear, readable format.
    
    Args:
        signals: Dictionary containing strategy signals data
        
    Returns:
        str: Formatted report string
    """
    output = []
    output.append("=== Recent Strategy Signals Analysis Report ===\n")
    output.append(f"Analysis Period: Last 10 Trading Days\n")
    output.append("=" * 80 + "\n")
    
    if not any(signals.values()):
        output.append("No strategy signals detected in the recent period.")
        return "\n".join(output)
    
    for strategy_name, stocks in signals.items():
        output.append(f"\n【{strategy_name.replace('_', ' ').title()}】")
        output.append("-" * 40)
        output.append(f"Total Triggered Stocks: {len(stocks)}")
        
        if stocks:
            # Display stock details
            for stock in stocks:
                output.append(f"\nStock: {stock['symbol']}")
                output.append(f"  Signal Strength: {stock['signal_strength']}%")
                output.append(f"  Volume Ratio: {stock['average_volume_ratio']}x average")
                output.append(f"  Price/VWAP Ratio: {stock['price_to_vwap']}")
                output.append("  Trigger Dates:")
                for date in stock['trigger_dates']:
                    output.append(f"    - {date.strftime('%Y-%m-%d')}")
        else:
            output.append("\nNo stocks triggered this strategy.")
        output.append("")
    
    return "\n".join(output)

def save_report(report: Dict, signals: Dict, output_dir: Path, market_name: str):
    """
    Save the report in both JSON and text formats.
    
    Args:
        report: Strategy analysis report
        signals: Recent strategy signals data
        output_dir: Directory to save reports
        market_name: Name of the market being analyzed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{market_name}_strategy_report_{timestamp}"
    
    # Save JSON format
    json_path = output_dir / f"{base_filename}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'full_report': report,
            'recent_signals': signals
        }, f, indent=2, default=str)
    logger.info(f"Saved JSON report to {json_path}")
    
    # Save formatted text
    txt_path = output_dir / f"{base_filename}.txt"
    with open(txt_path, 'w') as f:
        f.write(format_recent_signals_report(signals))
    logger.info(f"Saved formatted report to {txt_path}")

def validate_paths(data_path: str, market_name: str) -> Path:
    """
    Validate data and market paths exist.
    
    Args:
        data_path: Base data directory path
        market_name: Name of the market to analyze
        
    Returns:
        Path: Validated Path object for data directory
        
    Raises:
        ValueError: If paths are invalid
    """
    base_path = Path(data_path)
    if not base_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")
        
    market_path = base_path / market_name
    if not market_path.exists():
        raise ValueError(f"Market directory not found: {market_path}")
        
    return base_path

def main():
    """Main entry point for generating strategy reports."""
    parser = argparse.ArgumentParser(description='Generate market strategy analysis report')
    parser.add_argument('data_path', type=str, help='Path to market data directory')
    parser.add_argument('market_name', type=str, help='Name of the market to analyze')
    parser.add_argument('--output-dir', type=str, default='reports',
                      help='Directory to save reports (default: reports)')
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for analysis (YYYY-MM-DD format)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                      help='Size of data chunks for processing (default: 100000)')
    
    args = parser.parse_args()
    
    try:
        # Validate paths first
        logger.info(f"Validating paths for market: {args.market_name}")
        data_path = validate_paths(args.data_path, args.market_name)
        
        # Initialize analyzer with logging
        logger.info(f"Initializing strategy analyzer for {args.market_name}")
        analyzer = StrategyAnalyzer(data_path)

        # Parse start date if provided
        start_date = None
        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
                logger.info(f"Filtering data from {start_date.date()}")
            except ValueError:
                raise ValueError("Invalid start date format. Please use YYYY-MM-DD")

        # Process market data and calculate indicators
        logger.info("Processing market data and calculating indicators...")
        market_data = analyzer.processor.process_market_data(
            args.market_name,
            parallel=True,
            start_date=start_date
        )
        
        if market_data is None or market_data.empty:
            raise ValueError(f"No market data found for {args.market_name}")
        
        logger.info(f"Processing {len(market_data):,} records for analysis")
        
        # Generate full report
        logger.info("Calculating strategy metrics...")
        report = analyzer.calculate_strategy_metrics(market_data, args.chunk_size)
        
        # Extract recent signals using the same analyzer instance
        logger.info("Analyzing recent signals...")
        signals = get_recent_strategy_signals(market_data, analyzer=analyzer)
        
        # Save and display reports
        logger.info("Saving reports...")
        save_report(report, signals, args.output_dir, args.market_name)
        
        # Print formatted signals report to console
        print("\n" + format_recent_signals_report(signals))
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    main()