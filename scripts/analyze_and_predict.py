#!/usr/bin/env python
"""
Stock data analysis and prediction script
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.config.settings import (
    DEFAULT_SYMBOLS,
    OUTPUT_DIR,
    PLOT_SETTINGS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_stock_list(file_path: str) -> List[str]:
    """Read stock symbols from a file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def analyze_historical_data(symbols: List[str], start_date: datetime, end_date: datetime):
    """
    Analyze historical data for the given symbols and time period
    """
    logger.info(f"Analyzing historical data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Load and process data
    loader = DataLoader(symbols, start_date, end_date)
    data = loader.load_data()
    
    # Filter for valid symbols
    valid_symbols = [symbol for symbol in symbols if symbol in data and not data[symbol].empty]
    if not valid_symbols:
        logger.warning("No valid data found for any symbols")
        return None
    
    logger.info(f"Found valid data for {len(valid_symbols)} out of {len(symbols)} symbols")
    
    # Calculate performance metrics for each stock
    performance = {}
    for symbol in valid_symbols:
        df = data[symbol]
        
        # Calculate returns
        total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
        
        # Calculate volatility (annualized)
        daily_returns = df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        cumulative = (1 + df['returns']).cumprod()
        drawdown = 1 - cumulative / cumulative.cummax()
        max_drawdown = drawdown.max()
        
        # Calculate technical indicators strength
        rsi = df['rsi'].iloc[-1]
        rsi_trend = df['rsi'].diff(10).iloc[-1]  # 10-day RSI trend
        
        macd = df['macd'].iloc[-1]
        macd_signal = df['signal'].iloc[-1]
        macd_hist = macd - macd_signal
        
        # Calculate trend strength
        ma_short = df['ma10'].iloc[-1]
        ma_medium = df['ma20'].iloc[-1]
        ma_long = df['ma50'].iloc[-1]
        
        trend_strength = (ma_short / ma_medium - 1) * 100  # Percentage difference
        
        # Store performance data
        performance[symbol] = {
            'total_return': float(total_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'rsi': float(rsi),
            'rsi_trend': float(rsi_trend),
            'macd': float(macd),
            'macd_hist': float(macd_hist),
            'trend_strength': float(trend_strength),
            'price': float(df['Close'].iloc[-1]),
            'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df else 0
        }
    
    # Rank stocks based on momentum score
    for symbol in performance:
        # Calculate momentum score (higher is better)
        momentum_score = (
            performance[symbol]['total_return'] * 0.3 +
            performance[symbol]['sharpe_ratio'] * 0.2 +
            (1 - performance[symbol]['max_drawdown']) * 0.1 +
            (performance[symbol]['trend_strength'] / 10) * 0.15 +
            (performance[symbol]['macd_hist']) * 10 * 0.15 +
            (50 - abs(performance[symbol]['rsi'] - 50)) / 50 * 0.1  # RSI closer to 50 is better for mean reversion
        )
        performance[symbol]['momentum_score'] = float(momentum_score)
    
    # Sort stocks by momentum score
    ranked_stocks = sorted(
        valid_symbols,
        key=lambda x: performance[x]['momentum_score'],
        reverse=True
    )
    
    # Generate summary report
    summary = {
        'analysis_period': {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        },
        'top_momentum_stocks': ranked_stocks[:10],
        'bottom_momentum_stocks': ranked_stocks[-10:],
        'performance_data': performance
    }
    
    return summary

def generate_predictions(symbols: List[str], historical_start: datetime, historical_end: datetime,
                        prediction_start: datetime, prediction_end: datetime):
    """
    Generate predictions for the given symbols and time period
    """
    logger.info(f"Generating predictions for {len(symbols)} symbols")
    logger.info(f"Historical period: {historical_start} to {historical_end}")
    logger.info(f"Prediction period: {prediction_start} to {prediction_end}")
    
    # Load historical data
    loader = DataLoader(symbols, historical_start, historical_end)
    data = loader.load_data()
    
    # Filter for valid symbols
    valid_symbols = [symbol for symbol in symbols if symbol in data and not data[symbol].empty]
    if not valid_symbols:
        logger.warning("No valid data found for any symbols")
        return None
    
    logger.info(f"Found valid data for {len(valid_symbols)} out of {len(symbols)} symbols")
    
    # Initialize strategy
    strategy = MomentumStrategy(valid_symbols, historical_start, historical_end)
    
    # Generate signals based on historical data
    signals = strategy.generate_signals(data)
    
    # Generate predictions
    predictions = strategy.predict_signals(data)
    
    # Calculate prediction metrics
    prediction_results = {}
    for symbol in valid_symbols:
        if symbol not in predictions or symbol not in data:
            continue
            
        df = data[symbol]
        pred = predictions[symbol]
        
        # Extract the latest prediction
        latest_signal = pred.iloc[-1] if not pred.empty else 0
        
        # Calculate predicted move direction and size
        direction = "up" if latest_signal > 0 else "down" if latest_signal < 0 else "sideways"
        strength = abs(latest_signal)
        
        # Calculate confidence score based on signal consistency
        signal_consistency = pred.iloc[-5:].abs().mean() if len(pred) >= 5 else 0
        confidence = min(signal_consistency * 2, 1.0)  # Scale to [0, 1]
        
        # Get recent price and indicators
        close_price = df['Close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        macd_hist = df['macd'].iloc[-1] - df['signal'].iloc[-1]
        
        # Estimate price targets
        volatility = df['returns'].std() * np.sqrt(20)  # 20-day volatility
        predicted_move = close_price * volatility * latest_signal * 2  # Scale by signal strength
        price_target = close_price + predicted_move
        
        # Estimate support and resistance
        ma50 = df['ma50'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        
        support = min(ma50, bb_lower)
        resistance = max(ma50, bb_upper)
        
        # Store prediction results
        prediction_results[symbol] = {
            'direction': direction,
            'strength': float(strength),
            'confidence': float(confidence),
            'current_price': float(close_price),
            'price_target': float(price_target),
            'support': float(support),
            'resistance': float(resistance),
            'predicted_return': float(predicted_move / close_price),
            'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df and not pd.isna(df['rsi'].iloc[-1]) else 50.0,
            'macd_histogram': float(df['macd'].iloc[-1] - df['signal'].iloc[-1]) if 'macd' in df and 'signal' in df and not pd.isna(df['macd'].iloc[-1]) and not pd.isna(df['signal'].iloc[-1]) else 0.0,
            'stoch_k': float(df['stoch_k'].iloc[-1]) if 'stoch_k' in df and not pd.isna(df['stoch_k'].iloc[-1]) else 50.0,
            'stoch_d': float(df['stoch_d'].iloc[-1]) if 'stoch_d' in df and not pd.isna(df['stoch_d'].iloc[-1]) else 50.0,
            'atr': float(df['atr'].iloc[-1]) if 'atr' in df and not pd.isna(df['atr'].iloc[-1]) else df['Close'].iloc[-1] * 0.02,  # Default to 2% of price
            'bb_width': float(df['bb_width'].iloc[-1]) if 'bb_width' in df and not pd.isna(df['bb_width'].iloc[-1]) else 0.05,  # Default to 5%
            'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df and not pd.isna(df['volume_ratio'].iloc[-1]) else 1.0,
            'momentum': float(
                np.clip(df['Close'].pct_change(10).iloc[-1] * 100, -10, 10)  # 10-day momentum as percentage, clipped to [-10%, 10%]
            ) if not pd.isna(df['Close'].pct_change(10).iloc[-1]) else 0.0,
            'trend_direction': float(
                (1 if df['ma5'].iloc[-1] > df['ma10'].iloc[-1] else -1) * 0.4 +
                (1 if df['ma10'].iloc[-1] > df['ma20'].iloc[-1] else -1) * 0.4 +
                (1 if df['ma20'].iloc[-1] > df['ma50'].iloc[-1] else -1) * 0.2
            ) if all(col in df for col in ['ma5', 'ma10', 'ma20', 'ma50']) else 0.0,
            'trend_strength': float(
                min(
                    abs(df['ma10'].iloc[-1] / df['ma20'].iloc[-1] - 1) * 100 * 0.5 +  # Price trend component
                    np.clip(abs(df['volume_ratio'].iloc[-1] - 1) * 100, 0, 20) * 0.3 +  # Volume trend component
                    np.clip(abs(df['Close'].pct_change(10).iloc[-1]) * 100, 0, 10) * 0.2,  # Momentum component
                    100.0  # Cap at 100%
                )
            ) if all(col in df for col in ['ma10', 'ma20', 'volume_ratio']) else 0.0
        }
    
    # Rank predictions by expected return
    ranked_predictions = sorted(
        valid_symbols,
        key=lambda x: prediction_results[x]['predicted_return'] if x in prediction_results else 0,
        reverse=True
    )
    
    # Generate prediction summary
    summary = {
        'historical_period': {
            'start_date': historical_start.strftime('%Y-%m-%d'),
            'end_date': historical_end.strftime('%Y-%m-%d')
        },
        'prediction_period': {
            'start_date': prediction_start.strftime('%Y-%m-%d'),
            'end_date': prediction_end.strftime('%Y-%m-%d')
        },
        'top_predicted_stocks': ranked_predictions[:10],
        'bottom_predicted_stocks': ranked_predictions[-10:],
        'prediction_data': prediction_results
    }
    
    return summary

def plot_top_predictions(predictions: Dict, top_n: int = 5):
    """Plot price targets for top predicted stocks"""
    if not predictions or 'prediction_data' not in predictions:
        logger.warning("No prediction data available for plotting")
        return
    
    # Get top predicted stocks
    top_stocks = predictions['top_predicted_stocks'][:top_n]
    
    plt.style.use(PLOT_SETTINGS['style'])
    fig, ax = plt.subplots(figsize=PLOT_SETTINGS['figsize'], dpi=PLOT_SETTINGS['dpi'])
    
    # Plot data
    stocks = []
    current_prices = []
    targets = []
    predicted_returns = []
    
    for symbol in top_stocks:
        data = predictions['prediction_data'][symbol]
        stocks.append(symbol)
        current_prices.append(data['current_price'])
        targets.append(data['price_target'])
        predicted_returns.append(data['predicted_return'] * 100)  # As percentage
    
    # Sort by predicted return
    sort_idx = np.argsort(predicted_returns)[::-1]
    stocks = [stocks[i] for i in sort_idx]
    current_prices = [current_prices[i] for i in sort_idx]
    targets = [targets[i] for i in sort_idx]
    predicted_returns = [predicted_returns[i] for i in sort_idx]
    
    # Create bar chart
    x = np.arange(len(stocks))
    width = 0.35
    
    ax.bar(x - width/2, current_prices, width, label='Current Price', color='blue')
    ax.bar(x + width/2, targets, width, label='Price Target', color='green')
    
    # Add text labels for predicted returns
    for i, (pr, cp, tp) in enumerate(zip(predicted_returns, current_prices, targets)):
        color = 'green' if pr > 0 else 'red'
        ax.text(i, max(cp, tp) + 5, f"{pr:.1f}%", 
                ha='center', va='bottom', color=color, fontweight='bold')
    
    # Customize chart
    ax.set_xlabel('Stock')
    ax.set_ylabel('Price ($)')
    ax.set_title('Top Stock Predictions: Current Price vs Price Target')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_predictions.png", format=PLOT_SETTINGS['save_format'])
    plt.close()

def save_results(analysis_results: Dict, prediction_results: Dict, output_dir: Path):
    """Save analysis and prediction results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis results
    if analysis_results:
        with open(output_dir / 'historical_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=4)
    
    # Save prediction results
    if prediction_results:
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(prediction_results, f, indent=4)
        
        # Generate prediction report
        with open(output_dir / 'prediction_report.txt', 'w') as f:
            f.write("Stock Market Prediction Report\n")
            f.write("============================\n\n")
            
            f.write(f"Historical Analysis Period: {prediction_results['historical_period']['start_date']} to {prediction_results['historical_period']['end_date']}\n")
            f.write(f"Prediction Period: {prediction_results['prediction_period']['start_date']} to {prediction_results['prediction_period']['end_date']}\n\n")
            
            f.write("Top 10 Stocks with Positive Outlook:\n")
            for i, symbol in enumerate(prediction_results['top_predicted_stocks'][:10], 1):
                data = prediction_results['prediction_data'][symbol]
                f.write(f"{i}. {symbol}: {data['direction'].upper()} with {data['confidence']*100:.1f}% confidence\n")
                f.write(f"   Current: ${data['current_price']:.2f}, Target: ${data['price_target']:.2f} ({data['predicted_return']*100:.1f}%)\n")
                f.write(f"   Support: ${data['support']:.2f}, Resistance: ${data['resistance']:.2f}\n")
                f.write(f"   Technical Analysis Summary:")
                f.write(f"     RSI: {data['rsi']:.1f}")
                f.write(f"       • Overbought (>70): {'Yes' if data['rsi'] > 70 else 'No'}")
                f.write(f"       • Oversold (<30): {'Yes' if data['rsi'] < 30 else 'No'}")
                f.write(f"       • Trend: {'Bullish' if data['rsi'] > 50 else 'Bearish'}")
                f.write(f"     MACD: {data['macd_histogram']:.4f}")
                f.write(f"       • Histogram: {'Positive (Bullish)' if data['macd_histogram'] > 0 else 'Negative (Bearish)'}")
                f.write(f"       • Strength: {'Strong' if abs(data['macd_histogram']) > 0.5 else 'Weak'}")
                f.write(f"     Stochastic Oscillator:")
                f.write(f"       • K Line: {data['stoch_k']:.1f}")
                f.write(f"       • D Line: {data['stoch_d']:.1f}")
                f.write(f"       • Signal: {'Overbought (>80)' if data['stoch_k'] > 80 else 'Oversold (<20)' if data['stoch_k'] < 20 else 'Neutral'}")
                f.write(f"     Volatility Analysis:")
                f.write(f"       • ATR: ${data['atr']:.2f}")
                f.write(f"       • BB Width: {data['bb_width']:.3f}")
                f.write(f"       • Volatility Level: {'High' if data['bb_width'] > 0.1 else 'Medium' if data['bb_width'] > 0.05 else 'Low'}")
                f.write(f"     Volume Analysis:")
                f.write(f"       • Volume Ratio: {data['volume_ratio']:.2f}x")
                f.write(f"       • Volume Trend: {'Above Average' if data['volume_ratio'] > 1.2 else 'Average' if data['volume_ratio'] > 0.8 else 'Below Average'}")
                f.write(f"     Momentum & Trend:")
                f.write(f"       • 10-day Momentum: {data['momentum']*100:.1f}%")
                f.write(f"       • Trend Direction: {'Bullish' if data['trend_direction'] > 0 else 'Bearish'}")
                f.write(f"       • Trend Strength: {data['trend_strength']:.1f}%")
                f.write(f"       • Trend Quality: {'Strong' if data['trend_strength'] > 2 else 'Moderate' if data['trend_strength'] > 1 else 'Weak'}\n")
            
            f.write("\nBottom 10 Stocks with Negative Outlook:\n")
            for i, symbol in enumerate(prediction_results['bottom_predicted_stocks'][:10], 1):
                data = prediction_results['prediction_data'][symbol]
                f.write(f"{i}. {symbol}: {data['direction'].upper()} with {data['confidence']*100:.1f}% confidence\n")
                f.write(f"   Current: ${data['current_price']:.2f}, Target: ${data['price_target']:.2f} ({data['predicted_return']*100:.1f}%)\n")
                f.write(f"   Support: ${data['support']:.2f}, Resistance: ${data['resistance']:.2f}\n")
                f.write(f"   Technical Analysis Summary:")
                f.write(f"     RSI: {data['rsi']:.1f}")
                f.write(f"       • Overbought (>70): {'Yes' if data['rsi'] > 70 else 'No'}")
                f.write(f"       • Oversold (<30): {'Yes' if data['rsi'] < 30 else 'No'}")
                f.write(f"       • Trend: {'Bullish' if data['rsi'] > 50 else 'Bearish'}")
                f.write(f"     MACD: {data['macd_histogram']:.4f}")
                f.write(f"       • Histogram: {'Positive (Bullish)' if data['macd_histogram'] > 0 else 'Negative (Bearish)'}")
                f.write(f"       • Strength: {'Strong' if abs(data['macd_histogram']) > 0.5 else 'Weak'}")
                f.write(f"     Stochastic Oscillator:")
                f.write(f"       • K Line: {data['stoch_k']:.1f}")
                f.write(f"       • D Line: {data['stoch_d']:.1f}")
                f.write(f"       • Signal: {'Overbought (>80)' if data['stoch_k'] > 80 else 'Oversold (<20)' if data['stoch_k'] < 20 else 'Neutral'}")
                f.write(f"     Volatility Analysis:")
                f.write(f"       • ATR: ${data['atr']:.2f}")
                f.write(f"       • BB Width: {data['bb_width']:.3f}")
                f.write(f"       • Volatility Level: {'High' if data['bb_width'] > 0.1 else 'Medium' if data['bb_width'] > 0.05 else 'Low'}")
                f.write(f"     Volume Analysis:")
                f.write(f"       • Volume Ratio: {data['volume_ratio']:.2f}x")
                f.write(f"       • Volume Trend: {'Above Average' if data['volume_ratio'] > 1.2 else 'Average' if data['volume_ratio'] > 0.8 else 'Below Average'}")
                f.write(f"     Momentum & Trend:")
                f.write(f"       • 10-day Momentum: {data['momentum']*100:.1f}%")
                f.write(f"       • Trend Direction: {'Bullish' if data['trend_direction'] > 0 else 'Bearish'}")
                f.write(f"       • Trend Strength: {data['trend_strength']:.1f}%")
                f.write(f"       • Trend Quality: {'Strong' if data['trend_strength'] > 2 else 'Moderate' if data['trend_strength'] > 1 else 'Weak'}\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Stock data analysis and prediction script")
    parser.add_argument("stock_file", help="File containing list of stock symbols")
    parser.add_argument("--historical-start", type=str, required=True,
                      help="Historical data start date in YYYY-MM-DD format")
    parser.add_argument("--historical-end", type=str, required=True,
                      help="Historical data end date in YYYY-MM-DD format")
    parser.add_argument("--prediction-start", type=str, required=True,
                      help="Prediction period start date in YYYY-MM-DD format")
    parser.add_argument("--prediction-end", type=str, required=True,
                      help="Prediction period end date in YYYY-MM-DD format")
    
    args = parser.parse_args()
    
    # Read stock list
    symbols = read_stock_list(args.stock_file)
    logger.info(f"Read {len(symbols)} symbols from {args.stock_file}")
    
    # Convert date strings to datetime objects
    historical_start = datetime.strptime(args.historical_start, "%Y-%m-%d")
    historical_end = datetime.strptime(args.historical_end, "%Y-%m-%d")
    prediction_start = datetime.strptime(args.prediction_start, "%Y-%m-%d")
    prediction_end = datetime.strptime(args.prediction_end, "%Y-%m-%d")
    
    # Create output directory
    prediction_output_dir = OUTPUT_DIR / "predictions"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze historical data
    analysis_results = analyze_historical_data(symbols, historical_start, historical_end)
    
    # Generate predictions
    prediction_results = generate_predictions(
        symbols, historical_start, historical_end, prediction_start, prediction_end
    )
    
    # Plot top predictions
    if prediction_results:
        plot_top_predictions(prediction_results)
    
    # Save results
    save_results(analysis_results, prediction_results, prediction_output_dir)
    
    # Print summary with buy/sell signals
    if prediction_results:
        logger.info("\nTop 5 Stock Predictions with Signals:")
        for i, symbol in enumerate(prediction_results['top_predicted_stocks'][:5], 1):
            data = prediction_results['prediction_data'][symbol]
            
            # Skip stocks with invalid predictions
            if pd.isna(data['price_target']) or pd.isna(data['predicted_return']):
                continue
                
            # Determine signal strength
            signal_strength = "STRONG BUY" if data['predicted_return'] > 0.02 else \
                            "BUY" if data['predicted_return'] > 0.01 else \
                            "WEAK BUY" if data['predicted_return'] > 0.005 else \
                            "NEUTRAL" if data['predicted_return'] > -0.005 else \
                            "WEAK SELL" if data['predicted_return'] > -0.01 else \
                            "SELL" if data['predicted_return'] > -0.02 else \
                            "STRONG SELL"
            
            # Determine confidence level
            confidence = "Very High" if data['confidence'] > 0.8 else \
                        "High" if data['confidence'] > 0.6 else \
                        "Medium" if data['confidence'] > 0.4 else \
                        "Low"
            
            logger.info(f"{i}. {symbol}:")
            logger.info(f"   Signal: {signal_strength} ({confidence} confidence)")
            logger.info(f"   Current: ${data['current_price']:.2f}")
            logger.info(f"   Target: ${data['price_target']:.2f} ({data['predicted_return']*100:.1f}%)")
            logger.info(f"   Support: ${data['support']:.2f}, Resistance: ${data['resistance']:.2f}")
            logger.info(f"   Technical Analysis Summary:")
            logger.info(f"     RSI: {data['rsi']:.1f}")
            logger.info(f"       • Overbought (>70): {'Yes' if data['rsi'] > 70 else 'No'}")
            logger.info(f"       • Oversold (<30): {'Yes' if data['rsi'] < 30 else 'No'}")
            logger.info(f"       • Trend: {'Bullish' if data['rsi'] > 50 else 'Bearish'}")
            logger.info(f"     MACD: {data['macd_histogram']:.4f}")
            logger.info(f"       • Histogram: {'Positive (Bullish)' if data['macd_histogram'] > 0 else 'Negative (Bearish)'}")
            logger.info(f"       • Strength: {'Strong' if abs(data['macd_histogram']) > 0.5 else 'Weak'}")
            logger.info(f"     Stochastic Oscillator:")
            logger.info(f"       • K Line: {data['stoch_k']:.1f}")
            logger.info(f"       • D Line: {data['stoch_d']:.1f}")
            logger.info(f"       • Signal: {'Overbought (>80)' if data['stoch_k'] > 80 else 'Oversold (<20)' if data['stoch_k'] < 20 else 'Neutral'}")
            logger.info(f"     Volatility Analysis:")
            logger.info(f"       • ATR: ${data['atr']:.2f}")
            logger.info(f"       • BB Width: {data['bb_width']:.3f}")
            logger.info(f"       • Volatility Level: {'High' if data['bb_width'] > 0.1 else 'Medium' if data['bb_width'] > 0.05 else 'Low'}")
            logger.info(f"     Volume Analysis:")
            logger.info(f"       • Volume Ratio: {data['volume_ratio']:.2f}x")
            logger.info(f"       • Volume Trend: {'Above Average' if data['volume_ratio'] > 1.2 else 'Average' if data['volume_ratio'] > 0.8 else 'Below Average'}")
            logger.info(f"     Momentum & Trend:")
            logger.info(f"       • 10-day Momentum: {data['momentum']*100:.1f}%")
            logger.info(f"       • Trend Direction: {'Bullish' if data['trend_direction'] > 0 else 'Bearish'}")
            logger.info(f"       • Trend Strength: {data['trend_strength']:.1f}%")
            logger.info(f"       • Trend Quality: {'Strong' if data['trend_strength'] > 2 else 'Moderate' if data['trend_strength'] > 1 else 'Weak'}\n")
        
        # Print bottom 5 stocks with negative signals
        logger.info("\nBottom 5 Stocks with Negative Signals:")
        for i, symbol in enumerate(prediction_results['bottom_predicted_stocks'][:5], 1):
            data = prediction_results['prediction_data'][symbol]
            
            # Skip stocks with invalid predictions
            if pd.isna(data['price_target']) or pd.isna(data['predicted_return']):
                continue
                
            # Determine signal strength
            signal_strength = "STRONG SELL" if data['predicted_return'] < -0.02 else \
                            "SELL" if data['predicted_return'] < -0.01 else \
                            "WEAK SELL" if data['predicted_return'] < -0.005 else \
                            "NEUTRAL" if data['predicted_return'] > -0.005 else \
                            "WEAK BUY" if data['predicted_return'] > -0.01 else \
                            "BUY" if data['predicted_return'] > -0.02 else \
                            "STRONG BUY"
            
            # Determine confidence level
            confidence = "Very High" if data['confidence'] > 0.8 else \
                        "High" if data['confidence'] > 0.6 else \
                        "Medium" if data['confidence'] > 0.4 else \
                        "Low"
            
            logger.info(f"{i}. {symbol}:")
            logger.info(f"   Signal: {signal_strength} ({confidence} confidence)")
            logger.info(f"   Current: ${data['current_price']:.2f}")
            logger.info(f"   Target: ${data['price_target']:.2f} ({data['predicted_return']*100:.1f}%)")
            logger.info(f"   Support: ${data['support']:.2f}, Resistance: ${data['resistance']:.2f}")
            logger.info(f"   Technical Analysis Summary:")
            logger.info(f"     RSI: {data['rsi']:.1f}")
            logger.info(f"       • Overbought (>70): {'Yes' if data['rsi'] > 70 else 'No'}")
            logger.info(f"       • Oversold (<30): {'Yes' if data['rsi'] < 30 else 'No'}")
            logger.info(f"       • Trend: {'Bullish' if data['rsi'] > 50 else 'Bearish'}")
            logger.info(f"     MACD: {data['macd_histogram']:.4f}")
            logger.info(f"       • Histogram: {'Positive (Bullish)' if data['macd_histogram'] > 0 else 'Negative (Bearish)'}")
            logger.info(f"       • Strength: {'Strong' if abs(data['macd_histogram']) > 0.5 else 'Weak'}")
            logger.info(f"     Stochastic Oscillator:")
            logger.info(f"       • K Line: {data['stoch_k']:.1f}")
            logger.info(f"       • D Line: {data['stoch_d']:.1f}")
            logger.info(f"       • Signal: {'Overbought (>80)' if data['stoch_k'] > 80 else 'Oversold (<20)' if data['stoch_k'] < 20 else 'Neutral'}")
            logger.info(f"     Volatility Analysis:")
            logger.info(f"       • ATR: ${data['atr']:.2f}")
            logger.info(f"       • BB Width: {data['bb_width']:.3f}")
            logger.info(f"       • Volatility Level: {'High' if data['bb_width'] > 0.1 else 'Medium' if data['bb_width'] > 0.05 else 'Low'}")
            logger.info(f"     Volume Analysis:")
            logger.info(f"       • Volume Ratio: {data['volume_ratio']:.2f}x")
            logger.info(f"       • Volume Trend: {'Above Average' if data['volume_ratio'] > 1.2 else 'Average' if data['volume_ratio'] > 0.8 else 'Below Average'}")
            logger.info(f"     Momentum & Trend:")
            logger.info(f"       • 10-day Momentum: {data['momentum']*100:.1f}%")
            logger.info(f"       • Trend Direction: {'Bullish' if data['trend_direction'] > 0 else 'Bearish'}")
            logger.info(f"       • Trend Strength: {data['trend_strength']:.1f}%")
            logger.info(f"       • Trend Quality: {'Strong' if data['trend_strength'] > 2 else 'Moderate' if data['trend_strength'] > 1 else 'Weak'}\n")

if __name__ == "__main__":
    main() 