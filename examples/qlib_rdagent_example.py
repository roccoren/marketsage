"""
Example script for Qlib + RDAgent integration for stock prediction.

This script demonstrates how to use the QlibRDPredictor to train and generate 
buy/sell signals for stocks.
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

from stockscan.models.qlib_rdagent import QlibRDPredictor, prepare_qlib_data, create_qlib_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qlib + RDAgent stock prediction example")
    
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOG,AMZN,META",
                       help="Comma-separated list of stock symbols to analyze")
    parser.add_argument("--start-date", type=str, default="2020-01-01",
                       help="Start date for data in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=None,
                       help="End date for data in YYYY-MM-DD format. If None, will use today.")
    parser.add_argument("--qlib-model", type=str, default="LSTM", choices=["LSTM", "GRU"],
                       help="Qlib model type to use")
    parser.add_argument("--use-rdagent", action="store_true", default=True,
                       help="Whether to use RDAgent for signal refinement")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs for RDAgent training")
    parser.add_argument("--output-dir", type=str, default="output/qlib_rdagent",
                       help="Directory to save model and results")
    parser.add_argument("--no-plot", action="store_true",
                       help="Do not generate plots")
    
    return parser.parse_args()

def main():
    """Main function to run the example."""
    args = parse_args()
    
    # Process arguments
    symbols = args.symbols.split(",")
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Running Qlib + RDAgent example for symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data using Qlib
    logger.info("Preparing data...")
    data = prepare_qlib_data(
        symbols=symbols,
        start_time=start_date,
        end_time=end_date
    )
    
    # Create datasets for training
    logger.info("Creating training, validation, and test datasets...")
    train_data, val_data, test_data = create_qlib_dataset(data)
    
    # Initialize and train model
    logger.info(f"Initializing QlibRDPredictor with {args.qlib_model} and RDAgent={args.use_rdagent}...")
    predictor = QlibRDPredictor(
        qlib_model_type=args.qlib_model,
        use_rdagent=args.use_rdagent
    )
    
    logger.info("Training model...")
    training_results = predictor.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs
    )
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = predictor.predict(test_data)
    
    # Save model
    model_dir = os.path.join(args.output_dir, f"{args.qlib_model}_RDAgent_{args.use_rdagent}")
    logger.info(f"Saving model to {model_dir}...")
    predictor.save(model_dir)
    
    # Save predictions to CSV
    test_dates = test_data.index.get_level_values('datetime').unique()
    test_symbols = test_data.index.get_level_values('instrument').unique()
    
    # Create a DataFrame to store predictions
    pred_df = pd.DataFrame()
    idx = 0
    
    for date in test_dates:
        for symbol in test_symbols:
            if idx < len(predictions):
                pred_df = pd.concat([pred_df, pd.DataFrame({
                    'date': date,
                    'symbol': symbol,
                    'prediction': predictions[idx]
                }, index=[idx])])
                idx += 1
    
    # Save predictions
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")
    
    # Generate visualization if enabled
    if not args.no_plot:
        logger.info("Generating visualization...")
        visualize_predictions(pred_df, args.output_dir)
    
    logger.info("Example completed successfully!")
    return 0

def visualize_predictions(pred_df, output_dir):
    """Generate visualization of predictions."""
    # Group by symbol
    symbols = pred_df['symbol'].unique()
    
    plt.figure(figsize=(12, 8))
    
    for symbol in symbols:
        symbol_data = pred_df[pred_df['symbol'] == symbol].sort_values('date')
        
        plt.plot(
            pd.to_datetime(symbol_data['date']),
            symbol_data['prediction'],
            label=symbol,
            marker='o',
            linestyle='-',
            alpha=0.7
        )
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='Buy threshold')
    plt.axhline(y=-1, color='r', linestyle='--', alpha=0.3, label='Sell threshold')
    
    plt.title('Stock Trading Signals from Qlib + RDAgent')
    plt.xlabel('Date')
    plt.ylabel('Signal (1=Buy, 0=Hold, -1=Sell)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signals_visualization.png'), dpi=300)
    plt.close()
    
    # Generate individual plots for each symbol
    for symbol in symbols:
        plt.figure(figsize=(10, 6))
        
        symbol_data = pred_df[pred_df['symbol'] == symbol].sort_values('date')
        dates = pd.to_datetime(symbol_data['date'])
        signals = symbol_data['prediction']
        
        # Plot signals
        plt.plot(dates, signals, label=f'{symbol} Signal', marker='o', linewidth=2)
        
        # Add colored regions for buy/sell/hold
        for i in range(len(dates) - 1):
            if signals.iloc[i] >= 0.5:  # Buy
                plt.axvspan(dates.iloc[i], dates.iloc[i+1], alpha=0.2, color='green')
            elif signals.iloc[i] <= -0.5:  # Sell
                plt.axvspan(dates.iloc[i], dates.iloc[i+1], alpha=0.2, color='red')
            else:  # Hold
                plt.axvspan(dates.iloc[i], dates.iloc[i+1], alpha=0.1, color='yellow')
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='Buy threshold')
        plt.axhline(y=-1, color='r', linestyle='--', alpha=0.3, label='Sell threshold')
        
        plt.title(f'{symbol} Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Signal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{symbol}_signals.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    main() 