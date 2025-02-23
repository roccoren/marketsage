import logging
from utils.yfinance_utils import download_with_session
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add file handler
file_handler = logging.FileHandler('problem_symbols_debug.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def test_symbol(symbol):
    logger.info(f"\n{'='*50}\nTesting symbol: {symbol}\n{'='*50}")
    
    # Test current data
    end_date = datetime.today()
    start_date = end_date - timedelta(days=7)
    
    logger.info(f"Fetching recent data ({start_date} to {end_date})")
    data = download_with_session(symbol, start=start_date, end=end_date, interval="1d")
    
    if data is not None and not data.empty:
        logger.info(f"Recent data retrieved successfully. Shape: {data.shape}")
        logger.debug(f"Data head:\n{data.head()}")
        return True
    else:
        logger.warning(f"No recent data available for {symbol}")
        
        # Try fetching with different intervals
        intervals = ["1d", "1wk", "1mo"]
        for interval in intervals:
            logger.info(f"Attempting with {interval} interval...")
            data = download_with_session(symbol, start=start_date, end=end_date, interval=interval)
            if data is not None and not data.empty:
                logger.info(f"Data found with {interval} interval! Shape: {data.shape}")
                logger.debug(f"Data head:\n{data.head()}")
                return True
        
        # Try a longer time range
        start_date = end_date - timedelta(days=30)
        logger.info(f"Attempting with longer range ({start_date} to {end_date})...")
        data = download_with_session(symbol, start=start_date, end=end_date, interval="1d")
        if data is not None and not data.empty:
            logger.info(f"Data found with longer range! Shape: {data.shape}")
            logger.debug(f"Data head:\n{data.head()}")
            return True
            
        return False

def main():
    problem_symbols = [
        "ADBE",  # Adobe - definitely not delisted
        "ADD",
        "ALAB",
        "ALCE",
        "ALXO",
        "AREB"
    ]
    
    logger.info("Starting problem symbols test")
    results = {}
    
    for symbol in problem_symbols:
        logger.info(f"\nTesting {symbol}...")
        results[symbol] = test_symbol(symbol)
        
    logger.info("\nTest Results Summary:")
    for symbol, success in results.items():
        status = "ACTIVE" if success else "NO DATA"
        logger.info(f"{symbol}: {status}")

if __name__ == "__main__":
    main()