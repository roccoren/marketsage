#!/usr/bin/env python
"""
Example script to run a backtest using our Qlib integration
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtest.qlib_backtest import QlibBacktest
from src.config.settings import OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_stock_list(file_path):
    """Read stock symbols from a file"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def download_data(symbols, start_date, end_date):
    """Download historical data for the given symbols"""
    logger.info(f"Downloading data for {len(symbols)} symbols...")
    
    all_data = {}
    for symbol in symbols:
        try:
            # Download data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                continue
                
            # Calculate technical indicators
            # Moving averages
            data['ma10'] = data['Close'].rolling(window=10).mean()
            data['ma20'] = data['Close'].rolling(window=20).mean()
            data['ma50'] = data['Close'].rolling(window=50).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['ema26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['macd'] = data['ema12'] - data['ema26']
            data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            
            # Stochastic Oscillator
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            data['stoch_k'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
            
            # Bollinger Bands
            data['bb_middle'] = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (std * 2)
            data['bb_lower'] = data['bb_middle'] - (std * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            
            # Volume indicators
            data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            # Momentum
            data['momentum'] = data['Close'].pct_change(periods=10) * 100
            
            # Trend
            data['trend_direction'] = np.where(data['ma10'] > data['ma20'], 1, -1)
            data['trend_strength'] = abs(data['ma10'] / data['ma20'] - 1) * 100
            
            # Clean up data (remove NaN values)
            data = data.dropna()
            
            # Store data
            all_data[symbol] = data
            logger.info(f"Downloaded and processed data for {symbol}: {len(data)} rows")
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
    
    return all_data

def generate_predictions(data, symbols, prediction_date):
    """Generate prediction scores for the given date"""
    predictions = {}
    
    for symbol in symbols:
        if symbol not in data or data[symbol].empty:
            continue
        
        # Get data up to the prediction date
        df = data[symbol]
        df = df[df.index <= prediction_date]
        
        if df.empty:
            continue
        
        # Calculate prediction score based on technical indicators
        last_row = df.iloc[-1]
        
        rsi = last_row['rsi']
        macd_hist = last_row['macd'] - last_row['signal']
        stoch_k = last_row['stoch_k']
        bb_width = last_row['bb_width']
        volume_ratio = last_row['volume_ratio']
        momentum = last_row['momentum']
        trend_direction = last_row['trend_direction']
        
        # RSI component (centered around 50)
        rsi_score = 1.0 - abs(rsi - 50) / 50
        
        # MACD component
        macd_score = np.tanh(macd_hist)
        
        # Stochastic component
        stoch_score = 1.0 if (stoch_k < 80 and stoch_k > 20) else 0.0
        
        # Bollinger width component (inverse - narrower is better)
        bb_score = 1.0 - min(bb_width, 0.1) / 0.1
        
        # Volume component
        volume_score = min(volume_ratio, 3.0) / 3.0
        
        # Momentum component
        mom_score = np.tanh(momentum / 5.0)
        
        # Trend component
        trend_score = 0.5 + (trend_direction * 0.5)
        
        # Combined score
        score = (
            rsi_score * 0.20 +
            macd_score * 0.25 +
            stoch_score * 0.10 +
            bb_score * 0.10 +
            volume_score * 0.10 +
            mom_score * 0.15 +
            trend_score * 0.10
        )
        
        predictions[symbol] = score
    
    return pd.Series(predictions)

def main():
    # Read stock list
    stock_file = 'config/stocks/backtest_sample.txt'
    symbols = read_stock_list(stock_file)
    logger.info(f"Read {len(symbols)} symbols from {stock_file}")
    
    # Define date ranges
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of data
    
    backtest_start = end_date - timedelta(days=365)  # 1 year of backtest
    backtest_end = end_date
    
    # Download historical data
    data = download_data(symbols, start_date, end_date)
    
    # Generate predictions for the backtest start date
    predictions = generate_predictions(data, symbols, backtest_start)
    logger.info(f"Generated predictions for {len(predictions)} symbols at {backtest_start}")
    
    # Sort predictions and print the top and bottom 5
    top_predictions = predictions.nlargest(5)
    bottom_predictions = predictions.nsmallest(5)
    
    logger.info("\nTop 5 predictions:")
    for symbol, score in top_predictions.items():
        logger.info(f"{symbol}: {score:.4f}")
    
    logger.info("\nBottom 5 predictions:")
    for symbol, score in bottom_predictions.items():
        logger.info(f"{symbol}: {score:.4f}")
    
    # Create temporary Qlib data directory
    qlib_data_dir = OUTPUT_DIR / "qlib_data"
    qlib_data_dir.mkdir(exist_ok=True)
    
    # Initialize backtesting system
    logger.info("Initializing backtesting system...")
    backtest_system = QlibBacktest(
        provider_uri=str(qlib_data_dir),
        region='us',
        start_time=backtest_start.strftime('%Y-%m-%d'),
        end_time=backtest_end.strftime('%Y-%m-%d'),
        instruments=symbols,
        benchmark='SPY',
        strategy_type='topk',
        strategy_params={
            'topk': 5,
            'n_drop': 2
        }
    )
    
    # Run backtest
    logger.info("Running backtest...")
    try:
        metrics, backtest_result = backtest_system.run_backtest(predictions)
        
        # Save results
        output_dir = OUTPUT_DIR / "backtest" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backtest_system.save_results(backtest_result, metrics, str(output_dir))
        
        # Print metrics
        logger.info("\nBacktest Results:")
        logger.info("=================")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")
                
        logger.info(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        logger.error("Note: You may need to install the Qlib package using 'pip install pyqlib'")
        logger.error("For more information, visit: https://github.com/microsoft/qlib")

if __name__ == "__main__":
    main() 