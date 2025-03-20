"""
Tech Stocks Prediction Example - Shows how to run predictions on specific US tech stocks
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime

from stockscan.pipeline import Pipeline
from stockscan.utils.env_setup import setup_environment, install_qlib_data, check_qlib_data
from stockscan.core.qlib_manager import QlibManager
from stockscan.core.feature_manager import FeatureManager
from stockscan.models.model_manager import ModelManager
from stockscan.strategies.strategy_manager import StrategyManager

# Target stocks
TARGET_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']

# Time periods
TRAIN_START = '2018-01-01'
TRAIN_END = '2021-12-31'
PREDICT_START = '2022-01-01'
PREDICT_END = '2023-12-31'

def setup():
    """Setup environment and check/download required data"""
    print("Setting up environment...")
    env_status = setup_environment()
    
    # Check for US stock data
    us_data_path = '~/.qlib/qlib_data/us_data'
    has_data = check_qlib_data(us_data_path)
    
    if not has_data:
        print("US stock data not found. Downloading data (this may take a while)...")
        install_qlib_data('us')
    else:
        print("US stock data found.")
    
    return us_data_path

def run_prediction_pipeline():
    """Run the prediction pipeline on tech stocks"""
    # Setup environment and get data path
    us_data_path = setup()
    
    # Load environment variables for RDAgent
    load_dotenv()
    rdagent_api_key = os.environ.get('OPENAI_API_KEY')
    
    print(f"Creating pipeline for stocks: {', '.join(TARGET_STOCKS)}")
    
    # Create pipeline with US data
    pipeline = Pipeline(
        provider_uri=us_data_path,
        region='us',
        output_dir='./output',
        log_dir='./logs',
        rdagent_api_key=rdagent_api_key
    )
    
    # Initialize Qlib
    pipeline.initialize_qlib()
    
    # Prepare data for target stocks
    print(f"Preparing data from {TRAIN_START} to {PREDICT_END}")
    pipeline.prepare_data(
        instruments=TARGET_STOCKS,  
        start_time=TRAIN_START,
        end_time=PREDICT_END
    )
    
    # Train a LightGBM model
    print("Training model...")
    model_name, scores = pipeline.train_model(
        model_type='lgb',
        loss='mse',
        learning_rate=0.05,
        max_depth=8,
        num_leaves=60,
        n_estimators=200
    )
    
    print(f"Model trained with scores:")
    print(f"  Train score: {scores[0]:.4f}")
    print(f"  Valid score: {scores[1]:.4f}")
    
    # Generate predictions
    print("Generating predictions...")
    signals, predictions = pipeline.generate_signals(
        strategy_type='topk',
        segment='test',  # Use test segment for predictions
        topk=len(TARGET_STOCKS),  # Keep all stocks in the portfolio
        n_drop=0  # Don't drop any stocks
    )
    
    # Run backtest
    print("Running backtest...")
    backtest_result, metrics = pipeline.run_backtest(predictions)
    
    print("Backtest results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Extract and display stock-specific predictions
    display_stock_predictions(pipeline, predictions)
    
    # Plot cumulative returns
    plot_cumulative_returns(backtest_result)
    
    return pipeline, predictions, backtest_result

def display_stock_predictions(pipeline, predictions):
    """Display predictions for each target stock"""
    print("\nStock-specific predictions:")
    
    # Get the most recent prediction dates
    prediction_df = predictions.reset_index()
    
    # Convert predictions to a more readable DataFrame
    stocks_data = {}
    
    for stock in TARGET_STOCKS:
        stock_preds = prediction_df[prediction_df['instrument'] == stock].copy()
        if not stock_preds.empty:
            stock_preds = stock_preds.sort_values('datetime')
            
            # Get the most recent predictions
            recent = stock_preds.tail(5)
            
            # Format the output
            stocks_data[stock] = {
                'recent_predictions': recent[['datetime', 0]].rename(columns={0: 'prediction'}).values.tolist(),
                'mean_prediction': stock_preds[0].mean(),
                'max_prediction': stock_preds[0].max(),
                'min_prediction': stock_preds[0].min(),
                'latest_prediction': stock_preds[0].iloc[-1] if not stock_preds.empty else None
            }
    
    # Display stock predictions
    for stock, data in stocks_data.items():
        print(f"\n{stock}:")
        print(f"  Latest prediction: {data['latest_prediction']:.6f}")
        print(f"  Mean prediction: {data['mean_prediction']:.6f}")
        print(f"  Range: [{data['min_prediction']:.6f} to {data['max_prediction']:.6f}]")
        print("  Recent predictions (date, value):")
        for date, pred in data['recent_predictions']:
            print(f"    {date.strftime('%Y-%m-%d')}: {pred:.6f}")
    
    return stocks_data

def plot_cumulative_returns(backtest_result):
    """Plot cumulative returns from backtest results"""
    if 'portfolio' not in backtest_result:
        print("No portfolio data available for plotting")
        return
    
    portfolio = backtest_result['portfolio']
    
    # Create a simple plot of cumulative returns
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio['return']).cumprod() - 1
    
    # Plot
    plt.plot(cumulative_returns.index, cumulative_returns.values * 100)
    plt.title('Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('./output/plots', exist_ok=True)
    plt.savefig('./output/plots/cumulative_returns.png')
    print("\nPlot saved to ./output/plots/cumulative_returns.png")
    
    # Show plot
    plt.show()

def rank_stocks_by_prediction(predictions):
    """Rank stocks by prediction score (higher is better)"""
    # Get the most recent prediction for each stock
    latest_predictions = {}
    
    prediction_df = predictions.reset_index()
    max_date = prediction_df['datetime'].max()
    latest_preds = prediction_df[prediction_df['datetime'] == max_date]
    
    # Sort by prediction score (descending)
    ranked_stocks = latest_preds.sort_values(0, ascending=False)
    
    print("\nStocks Ranked by Latest Prediction (Buy Signal Strength):")
    for i, (index, row) in enumerate(ranked_stocks.iterrows(), 1):
        print(f"{i}. {row['instrument']}: {row[0]:.6f}")
    
    return ranked_stocks

if __name__ == "__main__":
    pipeline, predictions, backtest_result = run_prediction_pipeline()
    
    # Rank stocks by prediction score
    ranked_stocks = rank_stocks_by_prediction(predictions)
    
    print("\nAnalysis complete! Use the ranking above to determine which stocks have the strongest buy signals.") 