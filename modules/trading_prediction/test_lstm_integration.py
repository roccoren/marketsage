#!/usr/bin/env python3
"""
Simple test script for the LSTM trading prediction integration.
"""

import logging
from datetime import datetime, timedelta
from modules.trading_prediction.data_acquisition import get_data_manager
from modules.trading_prediction.integration import get_trading_prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the LSTM trading prediction model."""
    # Define test symbol
    symbol = "AAPL"
    
    # Fetch historical data
    data_manager = get_data_manager()
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year ago
    end_date = datetime.now().strftime('%Y-%m-%d')  # Today
    
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    data = data_manager.fetch_data_multi_source(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    
    if data.empty:
        logger.error(f"Failed to fetch data for {symbol}")
        return
    
    # Generate prediction
    logger.info("Generating prediction using LSTM model...")
    prediction = get_trading_prediction(symbol, data)
    
    # Display results
    logger.info("\n=== LSTM Model Prediction Results ===")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Last Close: ${data['Close'].iloc[-1]:.2f}")
    logger.info(f"Signal: {prediction['overall_action']}")
    logger.info(f"Confidence: {prediction['confidence']:.2f}")
    logger.info(f"Expected Change: {prediction['forecast_change_percent']:.2f}%")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()