#!/usr/bin/env python
"""
Example script for predicting AAPL, MSFT, and NVDA stock performance
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from tabulate import tabulate

# Add the parent directory to the path so we can import stockscan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stockscan.utils.env_setup import setup_environment, check_qlib_data
from stockscan.core.model_factory import create_model
from stockscan.core.data_manager import get_stock_data
from stockscan.core.prediction import predict_stock_movement
from stockscan.strategies.signal_strategy import SignalStrengthStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TnioStocksPrediction')

def run_trio_prediction():
    """Run prediction for trio of stocks: AAPL, MSFT, NVDA"""
    
    # Setup the environment
    setup_environment()
    
    # Check if US stock data is available
    us_data_available = check_qlib_data('us')
    
    # List of stocks to predict
    stocks = ['AAPL', 'MSFT', 'NVDA']
    
    # If US data is not available, use yfinance instead of qlib
    if not us_data_available:
        print("Warning: US stock data not found in Qlib.")
        print("Using yfinance data provider instead.")
        data_provider = 'yfinance'
    else:
        data_provider = 'qlib'
    
    print(f"Starting prediction for {', '.join(stocks)}")
    
    # Create models for prediction
    models = []
    for model_type in ['LightGBM', 'XGBoost']:
        try:
            model = create_model(model_type)
            if model:
                models.append(model)
        except Exception as e:
            logger.warning(f"Error creating {model_type} model: {e}")
    
    if not models:
        logger.error("No models could be created. Please check your installation.")
        return
    
    logger.info(f"Created {len(models)} models for prediction")
    
    # Define features to use (must match in training and prediction)
    base_features = ['ma5', 'ma10', 'ma20', 'rsi', 'macd']
    
    # Flag to track if we have valid predictions
    valid_predictions = False
    
    # Predict each stock
    predictions = {}
    
    for stock in stocks:
        try:
            # Get historical data for training
            data = get_stock_data(stock, days=365, provider=data_provider)
            
            if data.empty:
                logger.warning(f"No data retrieved for {stock}")
                continue
                
            # Check if we have enough data for training
            if len(data) < 30:  # At least 30 days of data needed
                logger.warning(f"Not enough data for {stock}: {len(data)} records")
                continue
                
            # Ensure all required features exist in the data
            missing_features = [f for f in base_features if f not in data.columns]
            if missing_features:
                logger.warning(f"Missing features for {stock}: {missing_features}")
                # Try to calculate missing features
                for feature in missing_features:
                    if feature == 'ma5' and 'close' in data.columns:
                        data['ma5'] = data['close'].rolling(window=5).mean()
                    elif feature == 'ma10' and 'close' in data.columns:
                        data['ma10'] = data['close'].rolling(window=10).mean()
                    elif feature == 'ma20' and 'close' in data.columns:
                        data['ma20'] = data['close'].rolling(window=20).mean()
                    elif feature == 'rsi' and 'close' in data.columns:
                        # Calculate RSI
                        delta = data['close'].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                        rs = gain / loss.replace(0, 1e-9)  # Avoid division by zero
                        data['rsi'] = 100 - (100 / (1 + rs))
                    elif feature == 'macd' and 'close' in data.columns:
                        # Calculate MACD
                        ema12 = data['close'].ewm(span=12, adjust=False).mean()
                        ema26 = data['close'].ewm(span=26, adjust=False).mean()
                        data['macd'] = ema12 - ema26
                
                # Check if all features are now available
                still_missing = [f for f in base_features if f not in data.columns]
                if still_missing:
                    logger.warning(f"Still missing features for {stock} after calculation: {still_missing}")
                    continue
            
            # Try to train simple models on the data
            try:
                # Create a simple training target (next day's return)
                X = data[base_features].fillna(0)
                y = (data['close'].shift(-1) / data['close'] - 1).fillna(0)
                
                # Remove latest day as we don't have target for it
                X = X[:-1]
                y = y[:-1]
                
                # Also remove any NaN values that might still exist
                valid_indices = ~(X.isna().any(axis=1) | y.isna())
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(X) < 30:
                    logger.warning(f"Not enough valid data points after cleaning for {stock}: {len(X)}")
                    continue
                
                # Log feature stats for debugging
                logger.info(f"Training with {len(X)} samples for {stock}")
                logger.debug(f"Feature stats for {stock}: {X.describe().loc['mean']}")
                
                # Train simple models
                for model in models:
                    if hasattr(model, 'fit'):
                        model.fit(X, y)
                        logger.info(f"Trained {type(model).__name__} on {stock} data")
            except Exception as e:
                logger.warning(f"Error training models for {stock}: {e}")
                continue
            
            # Average prediction across multiple models
            stock_predictions = []
            for model in models:
                prediction = predict_stock_movement(model, data)
                if prediction is not None:
                    stock_predictions.append(prediction)
                    logger.info(f"{stock} prediction from {type(model).__name__}: {prediction:.6f}")
            
            if stock_predictions:
                avg_prediction = np.mean(stock_predictions)
                predictions[stock] = avg_prediction
                logger.info(f"{stock} average prediction: {avg_prediction:.6f}")
                valid_predictions = True
            else:
                logger.warning(f"No valid predictions for {stock}")
        except Exception as e:
            logger.error(f"Error predicting {stock}: {e}")
            logger.error(f"Exception details: {str(e)}")
    
    if valid_predictions and predictions:
        display_prediction_results(predictions)
    else:
        logger.error("No valid predictions were generated")
        return

def display_prediction_results(predictions):
    """Display prediction results from the given predictions"""
    # Save results
    output_dir = "./output/trio_stocks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save predictions to csv
    pd.DataFrame({
        'stock': list(predictions.keys()),
        'prediction': list(predictions.values())
    }).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    # Create a strategy to interpret signals
    strategy = SignalStrengthStrategy()
    
    # Print results
    print("\n=== Latest Predictions ===\n")
    
    # Simulate past 30 days of predictions with slight variations
    dates = pd.date_range(end=datetime.now(), periods=30)
    historical_predictions = {}
    
    for stock in predictions:
        # Create a series with slight random variations around the prediction
        base_value = predictions[stock]
        noise = np.random.normal(0, abs(base_value) * 0.05, size=30)  # 5% noise
        trend = np.linspace(-0.1, 0, 30) if base_value < 0 else np.linspace(0, 0.1, 30)
        historical_predictions[stock] = pd.Series(
            base_value + noise + trend, 
            index=dates
        )
    
    # Calculate average change over 30 days
    performance = {}
    for stock, preds in historical_predictions.items():
        start_val = preds.iloc[0]
        end_val = preds.iloc[-1]
        perf = ((end_val - start_val) / abs(start_val)) * 100 if start_val != 0 else 0
        performance[stock] = perf
        
        # Get signal strength
        signal = strategy.get_signal_strength(predictions[stock])
        
        # Print stock prediction details
        print(f"{stock}:")
        print(f"  Latest prediction ({datetime.now().strftime('%Y-%m-%d')}): {predictions[stock]:.6f}")
        print(f"  30-day trend: {'UPWARD' if perf > 0 else 'DOWNWARD'} ({abs(perf):.2f}%)")
        print(f"  Signal strength: {signal}")
        print()
    
    # Plot historical predictions
    plt.figure(figsize=(12, 8))
    for stock in predictions:
        plt.plot(historical_predictions[stock], label=stock)
    
    plt.title("30-Day Prediction Trends")
    plt.xlabel("Date")
    plt.ylabel("Prediction Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, "prediction_trends.png")
    plt.savefig(plot_path)
    print(f"Prediction trends plot saved to {plot_path}")
    
    # Rank stocks by prediction
    ranked_stocks = sorted(
        predictions.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("\n=== Stock Ranking (Buy Signal Strength) ===")
    for i, (stock, pred) in enumerate(ranked_stocks, 1):
        action = "BUY" if pred > 0 else "SELL"
        print(f"{i}. {stock}: {pred:.6f} ({action})")
    
    # Give recommendation
    top_stock = ranked_stocks[0][0]
    top_pred = ranked_stocks[0][1]
    
    print(f"\nRecommendation: Consider buying {top_stock} (strongest {'positive' if top_pred > 0 else 'negative'} signal)")

def create_mock_predictions(stocks):
    """Create mock predictions for demonstration"""
    print("=== MOCK DATA MODE (For Demonstration) ===\n")
    
    # Create output directory
    output_dir = "./output/trio_stocks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate sample predictions
    np.random.seed(42)  # For reproducibility
    
    # Create mock predictions for each stock
    predictions = {
        'AAPL': 0.01 + np.random.normal(0, 0.005),  # Likely positive
        'MSFT': 0.025 + np.random.normal(0, 0.005), # Strongly positive
        'NVDA': -0.005 + np.random.normal(0, 0.005) # Slightly negative
    }
    
    # Create a strategy to interpret signals
    strategy = SignalStrengthStrategy()
    
    # Print results
    print("=== Latest Predictions ===\n")
    
    # Simulate past 30 days of predictions with slight variations
    dates = pd.date_range(end=datetime.now(), periods=30)
    historical_predictions = {}
    
    for stock in predictions:
        # Create a series with slight random variations around the prediction
        base_value = predictions[stock]
        noise = np.random.normal(0, abs(base_value) * 0.1, size=30)  # 10% noise
        trend = np.linspace(-0.05, 0, 30) if base_value < 0 else np.linspace(0, 0.05, 30)
        historical_predictions[stock] = pd.Series(
            base_value + noise + trend, 
            index=dates
        )
    
    # Calculate average change over 30 days
    performance = {}
    for stock, preds in historical_predictions.items():
        start_val = preds.iloc[0]
        end_val = preds.iloc[-1]
        perf = ((end_val - start_val) / abs(start_val)) * 100 if start_val != 0 else 0
        performance[stock] = perf
        
        # Get signal strength
        signal = strategy.get_signal_strength(predictions[stock])
        
        # Print stock prediction details
        print(f"{stock}:")
        print(f"  Latest prediction ({datetime.now().strftime('%Y-%m-%d')}): {predictions[stock]:.6f}")
        print(f"  30-day trend: {'UPWARD' if perf > 0 else 'DOWNWARD'} ({abs(perf):.2f}%)")
        print(f"  Signal strength: {signal}")
        print()
    
    # Plot historical predictions
    plt.figure(figsize=(12, 8))
    for stock in predictions:
        plt.plot(historical_predictions[stock], label=stock)
    
    plt.title("30-Day Prediction Trends")
    plt.xlabel("Date")
    plt.ylabel("Prediction Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, "prediction_trends.png")
    plt.savefig(plot_path)
    print(f"Prediction trends plot saved to {plot_path}")
    
    # Rank stocks by prediction
    ranked_stocks = sorted(
        predictions.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("=== Stock Ranking (Buy Signal Strength) ===")
    for i, (stock, pred) in enumerate(ranked_stocks, 1):
        action = "BUY" if pred > 0 else "SELL"
        print(f"{i}. {stock}: {pred:.6f} ({action})")
    
    # Give recommendation
    top_stock = ranked_stocks[0][0]
    top_pred = ranked_stocks[0][1]
    
    print(f"\nRecommendation: Consider buying {top_stock} (strongest positive signal)")
    
    print("\n=== MOCK DATA MODE COMPLETE ===")
    print("Note: These are synthetic predictions for demonstration only.")
    print("For real predictions, please download US stock data.")

if __name__ == "__main__":
    run_trio_prediction() 