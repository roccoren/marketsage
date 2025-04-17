#!/usr/bin/env python3
"""
Example script demonstrating the use of the deep learning model for stock prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import in the correct order to avoid circular imports
from modules.forecasting import ForecastingManager
from modules.storage import StorageManager
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if len(data) == 0:
            logger.error(f"No data found for {symbol}")
            return None
            
        # Ensure index is tz-naive
        if hasattr(data.index, "tz_convert") and data.index.tz is not None:
            data.index = data.index.tz_convert(None)
        data.index = pd.to_datetime(data.index).normalize()
        
        # Sort data by index to ensure consistent order
        data = data.sort_index()
        
        logger.info(f"Fetched {len(data)} data points for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def prepare_features(data):
    """Prepare features for the deep learning model."""
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 30]:
        df[f'close_ma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
    
    # Price momentum and volatility
    for window in [5, 10, 20]:
        df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
        df[f'volatility_{window}d'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)
    
    # Technical indicators
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def train_and_evaluate_model(symbol='AAPL', model_type='lstm'):
    """Train and evaluate the deep learning model."""
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # About 3 years of data
    
    data = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data is None:
        return
    
    # Prepare features
    features_df = prepare_features(data)
    
    # Split data into training and test sets (80% train, 20% test)
    train_size = int(len(features_df) * 0.8)
    train_data = features_df.iloc[:train_size]
    test_data = features_df.iloc[train_size:]
    
    logger.info(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
    
    # Create and train the model
    model_params = {
        'model_type': model_type,  # 'lstm' or 'transformer'
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,  # Reduced for example
        'sequence_length': 20,
        'attention_heads': 4,  # For transformer
        'early_stopping_patience': 5
    }
    
    # Use ForecastingManager for consistent interface
    forecasting_manager = ForecastingManager(forecast_horizon=10)
    
    # Train the model
    logger.info(f"Training {model_type.upper()} model for {symbol}...")
    model = forecasting_manager.train_model(symbol, train_data, model_type='deep', save=False)
    
    # Generate forecast using the freshly trained model, not stored model
    X_test, y_test = forecasting_manager.prepare_data(test_data)
    forecast_result = forecasting_manager.generate_forecast(symbol, test_data, model_type='deep', use_stored_model=False)
    
    # Display results
    logger.info(f"Forecast Result: {forecast_result['overall_action']}")
    logger.info(f"Daily Actions: {forecast_result['daily_actions']}")
    
    # Get feature importance
    if hasattr(model, 'get_feature_importance'):
        feature_importance = model.get_feature_importance()
        logger.info("Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {feature}: {importance:.4f}")
    
    # Plot feature importance
    if hasattr(model, 'visualize_attention'):
        fig = model.visualize_attention()
        if fig:
            plt.title(f"{symbol} Feature Importance Heatmap")
            plt.tight_layout()
            plt.savefig(f"{symbol}_feature_importance.png")
            logger.info(f"Feature importance heatmap saved to {symbol}_feature_importance.png")
            plt.close(fig)
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))
    
    # Get last close price from training data
    last_train_price = train_data['Close'].iloc[-1]
    
    # Plot actual prices
    actual_prices = test_data['Close']
    plt.plot(actual_prices.index, actual_prices.values, label='Actual', color='blue')
    
    # Plot predicted prices (starting from last training price)
    predicted_changes = np.array(forecast_result['daily_changes']) / 100  # Convert percentage to decimal
    predicted_prices = [last_train_price]
    for change in predicted_changes:
        predicted_prices.append(predicted_prices[-1] * (1 + change))
    
    # Create dates for predictions (starting from last training date)
    pred_dates = [train_data.index[-1]]
    for i in range(len(predicted_changes)):
        next_date = pred_dates[-1] + timedelta(days=1)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        pred_dates.append(next_date)
    
    plt.plot(pred_dates, predicted_prices, label='Predicted', color='red', linestyle='--')
    
    plt.title(f"{symbol} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{symbol}_price_prediction.png")
    logger.info(f"Price prediction chart saved to {symbol}_price_prediction.png")
    plt.close()
    
    return forecast_result

def compare_models(symbol='AAPL'):
    """Compare LSTM and Transformer models."""
    logger.info("=== Comparing LSTM and Transformer models ===")
    
    # Train and evaluate LSTM model
    logger.info("\n=== LSTM Model ===")
    lstm_result = train_and_evaluate_model(symbol, model_type='lstm')
    
    # Train and evaluate Transformer model
    logger.info("\n=== Transformer Model ===")
    transformer_result = train_and_evaluate_model(symbol, model_type='transformer')
    
    # Compare results
    logger.info("\n=== Model Comparison ===")
    logger.info(f"LSTM Model Action: {lstm_result['overall_action']}")
    logger.info(f"Transformer Model Action: {transformer_result['overall_action']}")
    
    logger.info(f"LSTM Model Volatility: {lstm_result['volatility']:.4f}")
    logger.info(f"Transformer Model Volatility: {transformer_result['volatility']:.4f}")
    
    logger.info(f"LSTM Model Trend: {lstm_result['trend']:.4f}")
    logger.info(f"Transformer Model Trend: {transformer_result['trend']:.4f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Learning Model Example')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer', 'compare'], 
                      default='lstm', help='Model type or comparison')
    
    args = parser.parse_args()
    
    if args.model == 'compare':
        compare_models(args.symbol)
    else:
        train_and_evaluate_model(args.symbol, args.model)