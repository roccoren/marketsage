"""
Main script for generating market strategy analysis reports.
Provides a command-line interface for running strategy analysis.
"""

import argparse
import json
from pathlib import Path
import logging
from typing import Dict
from datetime import datetime
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from market_system.strategy_analyzer import StrategyAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_report(report: Dict) -> str:
    """
    Format the strategy report in a human-readable format with enhanced sections.
    
    Args:
        report: Strategy analysis report dictionary
        
    Returns:
        str: Formatted report string
    """
    # Convert datetime objects to strings if necessary
    if isinstance(report['analysis_period']['start'], datetime):
        report['analysis_period']['start'] = report['analysis_period']['start'].isoformat()
    if isinstance(report['analysis_period']['end'], datetime):
        report['analysis_period']['end'] = report['analysis_period']['end'].isoformat()
    
    output = []
    output.append("=== Investment Target Screening & Analysis Report ===\n")
    
    # A. Basic Screening List
    output.append("A. Basic Screening List")
    output.append("-" * 80)
    
    # Strategy Matched Stocks
    output.append("\n1. Strategy-Matched Stocks")
    for strategy_name, metrics in report['strategies'].items():
        if metrics.get('matches', 0) > 0:
            output.append(f"\n【{strategy_name.replace('_', ' ').title()}】")
            output.append(f"  Total Matches: {metrics.get('matches', 0):,}")
            output.append(f"  Match Rate: {metrics.get('match_rate', 0)*100:.1f}%")
            
            if 'performance' in metrics:
                perf = metrics['performance']
                output.append(f"  Historical Success Rate: {perf.get('success_rate', 0)*100:.1f}%")
                output.append(f"  Average Return: {perf.get('average_return', 0)*100:.2f}%")
    
    # Fundamental Data
    output.append("\n2. Fundamental Analysis")
    if 'market_detail' in report:
        detail = report['market_detail']
        output.append("  Market Metrics:")
        output.append(f"    Trading Activity Rate: {detail['trading_activity']['activity_rate']*100:.1f}%")
        output.append(f"    Average Daily Turnover: ${detail['turnover_analysis']['daily_stats']['mean']:,.2f}")
        output.append(f"    Market Volatility: {detail['price_analysis']['volatility']*100:.1f}%")
    
    # B. Technical Analysis (Last 10 Trading Days)
    output.append("\nB. Technical Analysis (Recent 10 Days)")
    output.append("-" * 80)
    
    # Price Trends
    output.append("\n1. Price Movement Analysis")
    output.append("  Daily Trading Range:")
    output.append(f"    Average Range: ${report['market_detail']['price_analysis']['daily_stats']['mean_range']:,.2f}")
    output.append(f"    Volatility (Std): ${report['market_detail']['price_analysis']['daily_stats']['std']:,.2f}")
    
    # Volume Analysis
    output.append("\n2. Volume Analysis")
    vol_stats = report['market_detail']['volume_analysis']['daily_stats']
    output.append(f"  Average Daily Volume: {vol_stats['mean']:,.0f}")
    output.append(f"  Volume Volatility: {vol_stats['std']:,.0f}")
    output.append(f"  Peak Volume: {vol_stats['max']:,.0f}")
    
    # C. Trading Signal Tracking
    output.append("\nC. Trading Signal Analysis")
    output.append("-" * 80)
    
    # Technical Indicators
    output.append("\n1. Technical Signals")
    for strategy_name, metrics in report['strategies'].items():
        if metrics.get('matches', 0) > 0 and 'stats' in metrics:
            output.append(f"\n【{strategy_name.replace('_', ' ').title()}】")
            stats = metrics['stats']
            if 'vwap' in stats:
                output.append(f"  Price/VWAP Ratio: {stats['vwap'].get('price_to_vwap_ratio', 1):.3f}")
            if 'volume' in stats:
                output.append(f"  Volume Profile: {stats['volume'].get('mean', 0):,.0f}")
    
    # Signal Strength
    output.append("\n2. Signal Strength Assessment")
    for strategy_name, metrics in report['strategies'].items():
        if metrics.get('matches', 0) > 0 and 'performance' in metrics:
            perf = metrics['performance']
            output.append(f"\n【{strategy_name.replace('_', ' ').title()}】")
            output.append(f"  Win Rate: {perf.get('success_rate', 0)*100:.1f}%")
            output.append(f"  VWAP Success Rate: {perf.get('vwap_success_rate', 0)*100:.1f}%")
            output.append(f"  Winning Trades: {perf.get('winning_trades', 0):,}")
            output.append(f"  Total Signals: {perf.get('total_trades', 0):,}")
    
    # D. Comprehensive Scoring
    output.append("\nD. Comprehensive Evaluation")
    output.append("-" * 80)
    
    # Calculate weighted scores if available
    if 'market_detail' in report and 'strategies' in report:
        output.append("\n1. Quantitative Scoring")
        output.append("  Technical Score (40%):")
        for strategy_name, metrics in report['strategies'].items():
            if 'performance' in metrics:
                perf = metrics['performance']
                tech_score = (
                    perf.get('success_rate', 0) * 0.6 +
                    perf.get('vwap_success_rate', 0) * 0.4
                ) * 40
                output.append(f"    {strategy_name.replace('_', ' ').title()}: {tech_score:.1f}/40")
        
        # Market sentiment score
        detail = report['market_detail']
        volume_ratio = 0.0
        if detail['volume_analysis']['daily_stats']['max'] > 0:
            volume_ratio = (detail['volume_analysis']['daily_stats']['mean'] /
                          detail['volume_analysis']['daily_stats']['max'])

        sentiment_score = (
            min(detail['trading_activity']['activity_rate'] * 100, 30) +
            min(volume_ratio * 30, 30)
        )
        output.append(f"  Market Sentiment Score (30%): {sentiment_score:.1f}/30")
    
    # Trading Recommendations
    output.append("\n2. Action Recommendations")
    # Add specific recommendations based on scores and signals
    for strategy_name, metrics in report['strategies'].items():
        if metrics.get('matches', 0) > 0 and 'performance' in metrics:
            perf = metrics['performance']
            output.append(f"\n【{strategy_name.replace('_', ' ').title()}】")
            success_rate = perf.get('success_rate', 0)
            if success_rate > 0.6:
                output.append("  ⭐ High Confidence Signal")
                output.append(f"  - Suggested Position: {min(success_rate * 100, 30):.0f}% of Available Capital")
                output.append(f"  - Stop Loss: {abs(perf.get('average_return', 0)) * 2:.1f}%")
            elif success_rate > 0.4:
                output.append("  ⚠️ Moderate Confidence Signal")
                output.append("  - Consider Small Position with Tight Stops")
            else:
                output.append("  ❌ Low Confidence Signal")
                output.append("  - Wait for Stronger Confirmation")
    
    output.append(f"\nReport Generated: {report['report_generated']}")
    
    return "\n".join(output)

def save_report(report: Dict, output_dir: Path, market_name: str):
    """
    Save the report in both JSON and text formats.
    
    Args:
        report: Strategy analysis report
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
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved JSON report to {json_path}")
    
    # Save formatted text
    txt_path = output_dir / f"{base_filename}.txt"
    with open(txt_path, 'w') as f:
        f.write(format_report(report))
    logger.info(f"Saved formatted report to {txt_path}")

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
        # Validate start date format if provided
        start_date = None
        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Invalid start date format. Use YYYY-MM-DD")
        
        logger.info(f"Initializing analysis from {start_date if start_date else 'beginning'}")
        
        # Initialize analyzer and generate report
        analyzer = StrategyAnalyzer(args.data_path)
        report = analyzer.generate_report(
            args.market_name,
            chunk_size=args.chunk_size,
            start_date=start_date
        )
        
        # Save and display report
        save_report(report, args.output_dir, args.market_name)
        
        # Print formatted report to console
        print(format_report(report))
        
        # Log performance summary
        perf_metrics = report.get('execution_metrics', {})
        if perf_metrics:
            logger.info(
                f"Performance Summary:\n"
                f"- Processing Time: {perf_metrics.get('processing_time_seconds', 0):.2f}s\n"
                f"- Records/Second: {perf_metrics.get('records_per_second', 0):.0f}\n"
                f"- Peak Memory: {perf_metrics.get('memory_usage', {}).get('peak_memory_mb', 0):.1f}MB"
            )
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main()