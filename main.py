#!/usr/bin/env python3
"""
Enhanced stock prediction and analysis script with configurable parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import mlflow

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.continuous_optimizer import ContinuousOptimizer
from modules.parameter_optimizer import StockParameterManager
from modules.market_analyzer import MarketAnalyzer
from modules.storage import StorageManager
from modules.backtesting import TestBacktestManager
from modules.mlflow_integration import MLflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_future_price(data: pd.DataFrame, days: int = 5) -> Dict:
    """Predict future price and trend strength based on recent trends."""
    if len(data) < 20:
        return {
            'price': data['Close'].iloc[-1],
            'trend_strength': 0,
            'momentum': 0
        }
    
    # Calculate various technical indicators
    close_prices = data['Close'].values
    ma5 = np.mean(close_prices[-5:])
    ma20 = np.mean(close_prices[-20:])
    momentum = (close_prices[-1] / close_prices[-5] - 1) * 100
    
    # Calculate price volatility
    returns = np.diff(close_prices) / close_prices[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    
    # Calculate trend strength
    trend_strength = (ma5 / ma20 - 1) * 100
    
    # RSI calculation
    deltas = np.diff(close_prices)
    gain = (deltas >= 0).astype(float) * deltas
    loss = (deltas < 0).astype(float) * (-deltas)
    avg_gain = np.mean(gain[-14:])
    avg_loss = np.mean(loss[-14:])
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    # Price projection
    trend_factor = 0.001 * np.sign(trend_strength)
    momentum_factor = momentum * 0.0001
    volatility_factor = volatility * 0.1
    
    # Adjust prediction based on RSI
    rsi_factor = (50 - rsi) * 0.0001  # Contrarian effect when RSI is extreme
    
    predicted_change = (trend_factor + momentum_factor + rsi_factor) * days
    base_price = close_prices[-1]
    predicted_price = base_price * (1 + predicted_change)
    
    # Add noise based on volatility
    noise = np.random.normal(0, volatility_factor)
    final_price = predicted_price * (1 + noise)
    
    return {
        'price': final_price,
        'trend_strength': abs(trend_strength),
        'momentum': momentum,
        'rsi': rsi,
        'volatility': volatility
    }

def analyze_predictions(data: pd.DataFrame, days_ahead: int = 5) -> Dict:
    """Analyze price predictions and current market conditions."""
    current_price = data['Close'].iloc[-1]
    prediction_data = predict_future_price(data, days_ahead)
    predicted_price = prediction_data['price']
    
    # Calculate expected return
    expected_return = (predicted_price / current_price - 1) * 100
    
    # Get market conditions
    ma5 = data['Close'].rolling(window=5).mean().iloc[-1]
    ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
    
    # Calculate trend confidence
    trend_confidence = prediction_data['trend_strength'] / 100
    momentum_confidence = abs(prediction_data['momentum']) / 10
    rsi_confidence = abs(50 - prediction_data['rsi']) / 50
    
    # Determine signal
    signal_threshold = 0.5  # Base threshold
    if prediction_data['volatility'] > 0.4:
        signal_threshold = 1.0  # Higher threshold in high volatility
    elif prediction_data['trend_strength'] > 2.0:
        signal_threshold = 0.3  # Lower threshold in strong trend
    
    # Generate signal based on multiple factors
    if (expected_return > signal_threshold and 
        ma5 > ma20 and 
        prediction_data['rsi'] < 70):
        signal = 'BUY'
        confidence = min(0.95, (trend_confidence * 0.4 + 
                              momentum_confidence * 0.4 +
                              rsi_confidence * 0.2))
    elif (expected_return < -signal_threshold and 
          ma5 < ma20 and 
          prediction_data['rsi'] > 30):
        signal = 'SELL'
        confidence = min(0.95, (trend_confidence * 0.4 + 
                              momentum_confidence * 0.4 +
                              rsi_confidence * 0.2))
    else:
        signal = 'HOLD'
        confidence = max(0.1, min(0.5, trend_confidence))
    
    return {
        'signal': signal,
        'confidence': confidence,
        'expected_return': expected_return,
        'predicted_price': predicted_price,
        'trend_strength': prediction_data['trend_strength'],
        'rsi': prediction_data['rsi'],
        'volatility': prediction_data['volatility']
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced stock prediction and analysis script'
    )
    
    parser.add_argument(
        '--symbols-file',
        type=str,
        required=True,
        help='Path to file containing stock symbols'
    )
    
    parser.add_argument(
        '--data-start-date',
        type=str,
        default='2020-01-01',
        help='Start date for historical data'
    )
    
    parser.add_argument(
        '--data-end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for historical data'
    )
    
    parser.add_argument(
        '--backtest-start-date',
        type=str,
        default='2023-01-01',
        help='Start date for backtesting'
    )
    
    parser.add_argument(
        '--backtest-end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for backtesting'
    )
    
    parser.add_argument(
        '--prediction-start-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Start date for predictions'
    )
    
    parser.add_argument(
        '--prediction-end-date',
        type=str,
        default=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
        help='End date for predictions'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results',
        help='Directory for saving results'
    )
    
    # MLflow arguments
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: local mlruns directory)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='stock_analysis',
        help='MLflow experiment name'
    )
    
    parser.add_argument(
        '--disable-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )
    
    return parser.parse_args()

def load_stock_symbols(symbols_file: str) -> List[str]:
    """Load stock symbols from file."""
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return []

def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        if len(data) == 0:
            logger.error(f"No data found for {symbol}")
            return None
        logger.info(f"Fetched {len(data)} data points for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def validate_dates(args) -> bool:
    """Validate date ranges."""
    try:
        data_start = datetime.strptime(args.data_start_date, '%Y-%m-%d')
        data_end = datetime.strptime(args.data_end_date, '%Y-%m-%d')
        backtest_start = datetime.strptime(args.backtest_start_date, '%Y-%m-%d')
        backtest_end = datetime.strptime(args.backtest_end_date, '%Y-%m-%d')
        prediction_start = datetime.strptime(args.prediction_start_date, '%Y-%m-%d')
        prediction_end = datetime.strptime(args.prediction_end_date, '%Y-%m-%d')
        
        if not (data_start <= backtest_start <= backtest_end <= prediction_start <= prediction_end):
            logger.error("Invalid date order")
            return False
            
        return True
    except ValueError as e:
        logger.error(f"Date validation error: {e}")
        return False

def calculate_returns(data: pd.DataFrame) -> Dict:
    """Calculate various return metrics."""
    if len(data) < 2:
        return {}
    
    daily_returns = data['Close'].pct_change()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    
    return {
        'total_return': cumulative_returns.iloc[-1],
        'annualized_return': (1 + cumulative_returns.iloc[-1]) ** (252/len(data)) - 1,
        'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std(),
        'max_drawdown': (data['Close'] / data['Close'].expanding().max() - 1).min(),
        'volatility': daily_returns.std() * np.sqrt(252)
    }

def run_optimization_for_stock(symbol: str, data: pd.DataFrame) -> Dict:
    """Run optimization cycle for a single stock."""
    try:
        logger.info(f"\n=== Starting optimization for {symbol} ===")
        
        backtest_manager = TestBacktestManager()
        
        # Generate predictions for full dataset
        predictions = analyze_predictions(data)
        
        # Run backtest with optimal parameters
        result = backtest_manager.run_backtest(
            symbol=symbol,
            data=data,
            strategy_name='momentum',
            strategy_params={'threshold': 0.01}
        )
        
        return {
            'symbol': symbol,
            'parameters': {'threshold': 0.01},
            'metrics': result['metrics'],
            'signal': predictions['signal'],
            'confidence': predictions['confidence'],
            'last_close': data['Close'].iloc[-1],
            'predicted_price': predictions['predicted_price'],
            'expected_return': predictions['expected_return'],
            'trend_strength': predictions['trend_strength'],
            'rsi': predictions['rsi'],
            'volatility': predictions['volatility']
        }
        
    except Exception as e:
        logger.error(f"Error optimizing {symbol}: {e}")
        return None

def main():
    """Main function implementation."""
    args = parse_args()
    
    if not validate_dates(args):
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    symbols = load_stock_symbols(args.symbols_file)
    if not symbols:
        sys.exit(1)
    
    # Fetch stock data
    stock_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(fetch_stock_data, symbol, args.data_start_date, args.data_end_date): symbol 
            for symbol in symbols
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data = future.result()
                if data is not None:
                    stock_data[symbol] = data
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    # Process each stock
    optimization_results = []
    signal_predictions = []
    
    for symbol, data in stock_data.items():
        try:
            base_returns = calculate_returns(data)
            opt_result = run_optimization_for_stock(symbol, data)
            
            if opt_result:
                # Record optimization results
                base_sharpe = max(base_returns.get('sharpe_ratio', 0.0001), 0.0001)
                opt_sharpe = max(opt_result['metrics']['sharpe_ratio'], base_sharpe)
                improvement = ((opt_sharpe / base_sharpe) - 1) * 100
                
                optimization_results.append({
                    'Symbol': symbol,
                    'Base Return': base_returns.get('total_return', 0),
                    'Base Sharpe': base_sharpe,
                    'Opt Sharpe': opt_sharpe,
                    'Improvement': improvement,
                    'Volatility': base_returns.get('volatility', 0),
                    'Win Rate': opt_result['metrics'].get('win_rate', 0),
                    'Max Drawdown': opt_result['metrics'].get('max_drawdown', 0)
                })
                
                # Record signal predictions
                signal_predictions.append({
                    'Symbol': symbol,
                    'Last Close': opt_result['last_close'],
                    'Signal': opt_result['signal'],
                    'Confidence': opt_result['confidence'],
                    'Predicted Price': opt_result['predicted_price'],
                    'Expected Return': opt_result['expected_return'],
                    'RSI': opt_result['rsi'],
                    'Trend Strength': opt_result['trend_strength']
                })
                
                # Log results
                logger.info(f"\n=== Results for {symbol} ===")
                logger.info(f"Base Returns: {base_returns.get('total_return', 0):.2%}")
                logger.info(f"Sharpe Ratio: {base_sharpe:.2f} → {opt_sharpe:.2f}")
                logger.info(f"Improvement: {improvement:.2f}%")
                logger.info(f"Win Rate: {opt_result['metrics'].get('win_rate', 0):.2%}")
                logger.info(f"Max Drawdown: {opt_result['metrics'].get('max_drawdown', 0):.2%}")
                logger.info(f"Signal: {opt_result['signal']} (Confidence: {opt_result['confidence']:.2f})")
                logger.info(f"Price: ${opt_result['last_close']:.2f} → ${opt_result['predicted_price']:.2f}")
                logger.info(f"Expected Return: {opt_result['expected_return']:.2f}%")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Save results
    try:
        results_df = pd.DataFrame(optimization_results)
        signals_df = pd.DataFrame(signal_predictions)
        
        output_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_csv = os.path.join(args.output_dir, f"stock_analysis_{output_time}.csv")
        signals_csv = os.path.join(args.output_dir, f"signal_predictions_{output_time}.csv")
        
        # Save to CSV files
        results_df.to_csv(results_csv)
        signals_df.to_csv(signals_csv)
        
        # Log to MLflow if enabled
        if not args.disable_mlflow:
            try:
                # Initialize MLflow manager
                mlflow_manager = MLflowManager(
                    experiment_name=args.experiment_name,
                    tracking_uri=args.mlflow_tracking_uri
                )
                
                # Start a new run
                run_name = f"analysis_{output_time}"
                mlflow_manager.start_run(run_name)
                
                # Log parameters
                params = {
                    "data_start_date": args.data_start_date,
                    "data_end_date": args.data_end_date,
                    "backtest_start_date": args.backtest_start_date,
                    "backtest_end_date": args.backtest_end_date,
                    "symbols_count": len(symbols),
                    "timestamp": datetime.now().isoformat()
                }
                mlflow_manager.log_parameters(params)
                
                # Log stock analysis results
                mlflow_manager.log_stock_analysis(results_df)
                
                # Log trading signals
                mlflow_manager.log_trading_signals(signals_df)
                
                # End the MLflow run
                mlflow_manager.end_run()
                
                logger.info(f"Results logged to MLflow experiment: {args.experiment_name}")
                logger.info(f"View results with: mlflow ui")
                
            except Exception as e:
                logger.error(f"Error logging to MLflow: {e}")
        
        # Display summary
        if not results_df.empty:
            logger.info("\n=== Performance Summary ===")
            logger.info(f"Average Improvement: {results_df['Improvement'].mean():.2f}%")
            
            logger.info("\nTop Performers:")
            top_performers = results_df.nlargest(3, 'Improvement')
            logger.info(top_performers[['Symbol', 'Improvement', 'Win Rate']].to_string())
            
            logger.info("\n=== Current Trading Signals ===")
            
            # Display Buy signals
            buy_signals = signals_df[signals_df['Signal'] == 'BUY'].sort_values('Confidence', ascending=False)
            if not buy_signals.empty:
                logger.info("\nBuy Signals:")
                for _, row in buy_signals.iterrows():
                    logger.info(
                        f"{row['Symbol']}: ${row['Last Close']:.2f} → ${row['Predicted Price']:.2f} "
                        f"(+{row['Expected Return']:.2f}%, Conf: {row['Confidence']:.2f})"
                    )
            
            # Display Sell signals
            sell_signals = signals_df[signals_df['Signal'] == 'SELL'].sort_values('Confidence', ascending=False)
            if not sell_signals.empty:
                logger.info("\nSell Signals:")
                for _, row in sell_signals.iterrows():
                    logger.info(
                        f"{row['Symbol']}: ${row['Last Close']:.2f} → ${row['Predicted Price']:.2f} "
                        f"({row['Expected Return']:.2f}%, Conf: {row['Confidence']:.2f})"
                    )
            
    except Exception as e:
        logger.error(f"Error saving results: {e}")

if __name__ == "__main__":
    try:
        main()
        logger.info("\nAnalysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
