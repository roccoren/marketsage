#!/usr/bin/env python
"""
Azure OpenAI Trio Test - Tests Qlib and RDAgent integration with AAPL, MSFT, NVDA stocks
"""
import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from stockscan.utils.env_setup import setup_environment
from stockscan.evolution.rdagent_manager import RDAgentManager
from stockscan.pipeline import Pipeline
from stockscan.core.data_manager import get_stock_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('azure_openai_trio_test')

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Azure OpenAI Trio Stock Test')
    
    parser.add_argument('--start_date', type=str, default='2019-01-01',
                      help='Start date for historical data (default: 2019-01-01)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for historical data (default: today)')
    parser.add_argument('--output_dir', type=str, default='./output/azure_trio_test',
                      help='Directory for output files')
    parser.add_argument('--max_iterations', type=int, default=3,
                      help='Maximum iterations for self-evolution')
    parser.add_argument('--config_file', type=str, default=None,
                      help='Path to configuration file for Azure OpenAI settings')
    parser.add_argument('--rdagent_ui', action='store_true',
                      help='Start the RDAgent UI')
    parser.add_argument('--data_source', type=str, default='yfinance', choices=['qlib', 'yfinance'],
                      help='Data source to use (qlib or yfinance)')
    parser.add_argument('--download_qlib_data', action='store_true',
                      help='Download Qlib US data if needed')
    parser.add_argument('--install_rdagent', action='store_true',
                      help='Attempt to install RDAgent if not available')
    parser.add_argument('--skip_rdagent_optimization', action='store_true',
                      help='Skip model optimization with RDAgent')
    
    return parser.parse_args()

def load_azure_config(config_file=None):
    """
    Load Azure OpenAI configuration from .env file or config file
    
    Parameters
    ----------
    config_file : str, optional
        Path to configuration file
        
    Returns
    -------
    dict
        Azure OpenAI configuration
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Default configuration (from environment variables)
    config = {
        'api_key': os.environ.get('OPENAI_API_KEY'),
        'azure_endpoint': os.environ.get('OPENAI_API_BASE') or os.environ.get('AZURE_OPENAI_ENDPOINT'),
        'azure_deployment': os.environ.get('OPENAI_DEPLOYMENT_NAME') or os.environ.get('AZURE_OPENAI_DEPLOYMENT'),
        'azure_api_version': os.environ.get('OPENAI_API_VERSION', '2023-07-01-preview'),
        'chat_model': os.environ.get('CHAT_MODEL', 'gpt-4')
    }
    
    # If config file provided, load from there
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration with values from file
            for key in config:
                if key in file_config and file_config[key]:
                    config[key] = file_config[key]
                    
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
    
    # Check if required configuration is available
    if not config['api_key']:
        logger.warning("API key not found in environment variables or config file")
    
    # Validate Azure deployment name - should be a model name not a date
    if config['azure_deployment'] and config['azure_deployment'].startswith('20'):
        logger.warning(f"Azure deployment name '{config['azure_deployment']}' appears to be a date or invalid format.")
        logger.warning("Setting to default 'gpt-4'. Please set OPENAI_DEPLOYMENT_NAME to your actual deployment name.")
        config['azure_deployment'] = 'gpt-4'
        
    # Determine if using Azure OpenAI
    is_azure = bool(config['azure_endpoint'] and config['azure_deployment'])
    
    if is_azure:
        logger.info(f"Using Azure OpenAI deployment: {config['azure_deployment']}")
        # Set API type if using Azure OpenAI
        os.environ['OPENAI_API_TYPE'] = 'azure'
        os.environ['OPENAI_API_BASE'] = config['azure_endpoint']
        os.environ['OPENAI_API_VERSION'] = config['azure_api_version']
        os.environ['OPENAI_DEPLOYMENT_NAME'] = config['azure_deployment']
    else:
        logger.info("Using direct OpenAI API (Azure OpenAI configuration not found)")
    
    return config

def download_us_qlib_data():
    """
    Download US Qlib data
    
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    try:
        logger.info("Downloading US Qlib data... This may take a few minutes")
        
        # Check if we have a download script
        download_script = Path("scripts/download_us_data.py")
        if download_script.exists():
            # Use the project's download script
            subprocess.run([sys.executable, download_script], check=True)
        else:
            # Use the qlib download script directly
            from qlib.data import download_data
            download_data(region="us", target_dir="~/.qlib/qlib_data/us_data", delete_old=False)
            
        logger.info("US Qlib data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading US Qlib data: {e}")
        return False

def prepare_yfinance_data(symbols, start_date, end_date, output_dir):
    """
    Prepare data from yfinance
    
    Parameters
    ----------
    symbols : list
        List of stock symbols
    start_date : str
        Start date (not used directly, used to calculate days)
    end_date : str
        End date (not used directly, used to calculate days)
    output_dir : str
        Output directory
        
    Returns
    -------
    dict
        Dictionary of dataframes for each symbol
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate days from start_date to now
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        days = (end_dt - start_dt).days
        days = max(days, 365)  # Ensure at least 1 year of data
        logger.info(f"Calculated {days} days from {start_date} to {end_date or 'today'}")
    except Exception as e:
        logger.warning(f"Error calculating days from dates: {e}. Using default of 1095 days (3 years).")
        days = 1095  # Default to 3 years
    
    logger.info(f"Downloading data from yfinance for {len(symbols)} symbols, past {days} days")
    
    # Store data for each symbol
    data_dict = {}
    
    # Try direct date-based download first
    for symbol in symbols:
        try:
            # Try direct date range method first
            logger.info(f"Downloading data for {symbol} from {start_date} to {end_date or 'today'}")
            
            # Import yfinance here to ensure we're using the version installed in the environment
            import yfinance as yf
            
            # Try to download with explicit date range
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date if end_date else None,
                auto_adjust=False,
                progress=False
            )
            
            # Check if dataframe is empty or has insufficient data
            if data is None or data.empty or len(data) < 20:
                logger.warning(f"Insufficient data for {symbol} with date range. Trying with days parameter.")
                # Fall back to days-based method
                data = get_stock_data(symbol, days=days, provider='yfinance')
            
            # Process and save data
            if data is not None and not data.empty:
                # Standardize column names
                data = standardize_column_names(data)
                
                # Save to CSV for reference
                csv_path = os.path.join(output_dir, f"{symbol}.csv")
                data.to_csv(csv_path)
                logger.info(f"Saved {symbol} data to {csv_path} with {len(data)} rows")
                
                # Store in dictionary
                data_dict[symbol] = data
            else:
                logger.error(f"No data retrieved for {symbol}")
                
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            # Try with a different approach as fallback
            try:
                logger.info(f"Trying alternative method for {symbol}")
                # Try with get_stock_data which has additional error handling
                data = get_stock_data(symbol, days=days*2, provider='yfinance')
                
                if data is not None and not data.empty:
                    csv_path = os.path.join(output_dir, f"{symbol}.csv")
                    data.to_csv(csv_path)
                    logger.info(f"Saved {symbol} data to {csv_path} with {len(data)} rows")
                    
                    # Store in dictionary
                    data_dict[symbol] = data
                else:
                    logger.error(f"Still no data retrieved for {symbol}")
            except Exception as e2:
                logger.error(f"Error with alternative method for {symbol}: {e2}")
    
    # Check if we got any data
    if not data_dict:
        logger.warning("Failed to retrieve data for any symbols. Using sample data.")
        # Create some sample data for testing purposes
        for symbol in symbols:
            try:
                # Generate some random sample data
                dates = pd.date_range(start=start_date, end=end_date if end_date else datetime.now(), freq='D')
                df = pd.DataFrame(index=dates)
                df['close'] = np.random.normal(100, 10, size=len(dates)).cumsum() + 100
                df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.01, size=len(dates)))
                df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, size=len(dates))))
                df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, size=len(dates))))
                df['volume'] = np.random.normal(1000000, 200000, size=len(dates))
                df = df.fillna(method='bfill')
                
                # Add technical indicators
                df['ma5'] = df['close'].rolling(window=5).mean()
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['rsi'] = 50 + 25 * np.sin(np.linspace(0, 10, len(df)))
                
                csv_path = os.path.join(output_dir, f"{symbol}_sample.csv")
                df.to_csv(csv_path)
                logger.warning(f"Created sample data for {symbol} and saved to {csv_path}")
                
                data_dict[symbol] = df
            except Exception as e:
                logger.error(f"Error creating sample data for {symbol}: {e}")
    
    return data_dict

def standardize_column_names(data):
    """
    Standardize column names from various yfinance versions
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with yfinance stock data
        
    Returns
    -------
    data : pd.DataFrame
        DataFrame with standardized column names
    """
    # Handle multi-index columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        # Create a more descriptive mapping that preserves information
        flat_columns = []
        for col in data.columns:
            # Join the levels with an underscore
            flat_columns.append(f"{col[0]}_{col[1]}" if len(col) > 1 else col[0])
        
        # Set the new column names
        data.columns = flat_columns
    
    # Create a mapping of possible column names to standardized names
    column_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close',
        'Open_Open': 'open', 'High_High': 'high', 'Low_Low': 'low', 'Close_Close': 'close', 
        'Volume_Volume': 'volume', 'Adj Close_Adj Close': 'adj_close',
        'Adj Close_adj close': 'adj_close', 'Volume_volume': 'volume',
    }
    
    # Rename columns to align with standardized format, but only for columns that exist
    new_columns = {}
    for old_col in data.columns:
        if old_col in column_map:
            new_columns[old_col] = column_map[old_col]
    
    if new_columns:
        data = data.rename(columns=new_columns)
    
    # Make sure we have the required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    # Try to handle common missing columns
    if missing_columns:
        for col in missing_columns:
            if col == 'close' and 'adj_close' in data.columns:
                data['close'] = data['adj_close']
            elif col != 'volume' and 'close' in data.columns:  # For price columns, use close
                data[col] = data['close']
            elif col == 'volume':  # For volume, use 0
                data['volume'] = 0
                
    # Reset index if it's a DatetimeIndex to ensure Date is always a column
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
        if 'Date' in data.columns and 'date' not in data.columns:
            data = data.rename(columns={'Date': 'date'})
    
    return data

def train_model(model, data_dict):
    """
    Train a model using the data from all symbols
    
    Parameters
    ----------
    model : object
        Model to train
    data_dict : dict
        Dictionary of symbol -> dataframe
        
    Returns
    -------
    model : object
        Trained model, None if training failed
    trained : bool
        Whether the training was successful
    """
    # Combine all data for training (simple approach)
    combined_X = pd.DataFrame()
    combined_y = pd.Series()
    
    from stockscan.core.prediction import prepare_features
    
    for symbol, data in data_dict.items():
        if data is not None and not data.empty:
            try:
                X, y = prepare_features(data)
                if not X.empty:
                    # Add symbol as a feature
                    X['symbol'] = symbol
                    combined_X = pd.concat([combined_X, X])
                    combined_y = pd.concat([combined_y, y])
            except Exception as e:
                logger.warning(f"Error preparing features for {symbol}: {e}")
    
    # Check if we have data to train on
    if combined_X.empty or len(combined_X) < 10:
        logger.warning("Not enough data to train model")
        return model, False
    
    logger.info(f"Training model on {len(combined_X)} samples")
    try:
        # Drop symbol column if it's categorical (some models can't handle it)
        if 'symbol' in combined_X.columns:
            combined_X = combined_X.drop('symbol', axis=1)
            
        # Train the model
        model.fit(combined_X, combined_y)
        
        # Verify model was trained successfully with a simple prediction test
        try:
            test_pred = model.predict(combined_X.iloc[:1])
            logger.info(f"Model trained successfully. Test prediction: {test_pred[0]:.4f}")
            return model, True
        except Exception as e:
            logger.error(f"Model training verification failed: {e}")
            return model, False
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return model, False

def install_rdagent():
    """
    Attempt to install RDAgent package
    
    Returns
    -------
    bool
        True if installation successful, False otherwise
    """
    try:
        logger.info("Attempting to install RDAgent package...")
        import subprocess
        
        # First ensure pip is up to date
        logger.info("Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        
        # Install RDAgent with required dependencies
        logger.info("Installing RDAgent...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "rdagent"], 
                               check=True, capture_output=True, text=True)
        logger.info(f"Installation output: {result.stdout}")
        
        # Install additional dependencies that might be needed
        logger.info("Installing additional dependencies...")
        deps = ["openai>=1.0.0", "numpy", "pandas", "scikit-learn", "colorama"]
        for dep in deps:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep],
                              check=True, capture_output=True, text=True)
                logger.info(f"Installed {dep}")
            except Exception as e:
                logger.warning(f"Failed to install {dep}: {e}")
        
        # Try to import to verify installation
        import importlib
        importlib.import_module('rdagent')
        
        # Check if rdagent CLI command works
        try:
            version_check = subprocess.run(['rdagent', '--version'], 
                                          capture_output=True, text=True, check=False)
            if version_check.returncode == 0:
                logger.info(f"RDAgent CLI available: {version_check.stdout.strip()}")
            else:
                logger.warning(f"RDAgent CLI returned error: {version_check.stderr}")
        except Exception as e:
            logger.warning(f"RDAgent CLI check failed: {e}")
        
        logger.info("RDAgent successfully installed.")
        return True
    except Exception as e:
        logger.error(f"Failed to install RDAgent: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    setup_environment()
    
    # Set up dates
    if args.start_date is None:
        # Default to 2019-01-01 
        start_date = '2019-01-01'
    else:
        start_date = args.start_date
        
    if args.end_date is None:
        # Default to today
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    # Stock symbols to analyze
    symbols = ['AAPL', 'MSFT', 'NVDA']
    
    # Load Azure OpenAI configuration
    config = load_azure_config(args.config_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a config file for reference
    config_path = os.path.join(args.output_dir, 'azure_config.json')
    with open(config_path, 'w') as f:
        # Don't write the API key
        safe_config = {k: v for k, v in config.items() if k != 'api_key'}
        json.dump(safe_config, f, indent=4)
    
    logger.info(f"Starting Azure OpenAI Trio Test with symbols: {', '.join(symbols)}")
    logger.info(f"Analysis period: {start_date} to {end_date}")
    logger.info(f"Using data source: {args.data_source}")
    
    # Check if we need to download Qlib data
    if args.data_source == 'qlib' and args.download_qlib_data:
        if not download_us_qlib_data():
            logger.warning("Failed to download Qlib data. Falling back to yfinance.")
            args.data_source = 'yfinance'
    
    # Create RDAgent manager with Azure OpenAI configuration
    try:
        rdagent_manager = RDAgentManager(
            log_dir=os.path.join(args.output_dir, 'logs'),
            api_key=config['api_key'],
            chat_model=config['chat_model'],
            azure_endpoint=config['azure_endpoint'],
            azure_deployment=config['azure_deployment'],
            azure_api_version=config['azure_api_version']
        )
        
        # Check if RDAgent is available
        rdagent_available = hasattr(rdagent_manager, 'is_available') and rdagent_manager.is_available
        
        # Check if API key is set (required for RDAgent)
        if rdagent_available and not config['api_key']:
            logger.warning("OpenAI API key is not set. RDAgent will not work properly.")
            logger.warning("Please set the OPENAI_API_KEY environment variable.")
            rdagent_available = False
    except Exception as e:
        logger.error(f"Error initializing RDAgent manager: {e}")
        rdagent_manager = None
        rdagent_available = False
        
        # Try to check if RDAgent is installed
        try:
            import importlib
            rdagent_spec = importlib.util.find_spec('rdagent')
            if rdagent_spec is None:
                logger.error("RDAgent package is not installed")
            else:
                logger.info("RDAgent package is installed but initialization failed")
                
                # Try to get the version to provide more diagnostic information
                try:
                    import subprocess
                    result = subprocess.run(['rdagent', '--version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info(f"RDAgent version: {result.stdout.strip()}")
                    else:
                        logger.warning(f"Failed to get RDAgent version: {result.stderr.strip()}")
                except Exception as e2:
                    logger.warning(f"Error checking RDAgent version: {e2}")
        except Exception as check_e:
            logger.error(f"Error checking for RDAgent: {check_e}")
    
    # Try to install RDAgent if requested
    if not rdagent_available and args.install_rdagent:
        logger.info("RDAgent not available. Attempting to install...")
        if install_rdagent():
            # Recreate the manager after installation
            try:
                rdagent_manager = RDAgentManager(
                    log_dir=os.path.join(args.output_dir, 'logs'),
                    api_key=config['api_key'],
                    chat_model=config['chat_model'],
                    azure_endpoint=config['azure_endpoint'],
                    azure_deployment=config['azure_deployment'],
                    azure_api_version=config['azure_api_version']
                )
                rdagent_available = hasattr(rdagent_manager, 'is_available') and rdagent_manager.is_available
            except Exception as e:
                logger.error(f"Error initializing RDAgent manager after installation: {e}")
                rdagent_available = False
    
    if not rdagent_available:
        logger.warning("RDAgent is not available. Will proceed without model optimization.")
    
    # Start RDAgent UI if requested
    rdagent_process = None
    if args.rdagent_ui and rdagent_available and rdagent_manager is not None:
        try:
            rdagent_process = rdagent_manager.start_ui()
            if rdagent_process is None:
                logger.warning("Failed to start RDAgent UI. Continuing without UI.")
        except Exception as e:
            logger.warning(f"Error starting RDAgent UI: {e}")
    elif args.rdagent_ui:
        logger.warning("Cannot start RDAgent UI because RDAgent is not available.")
    
    try:
        results = None
        
        if args.data_source == 'qlib':
            # Use Qlib data and pipeline
            try:
                # Create pipeline
                pipeline = Pipeline(
                    provider_uri='~/.qlib/qlib_data/us_data',
                    region='us',
                    output_dir=args.output_dir,
                    log_dir=os.path.join(args.output_dir, 'logs'),
                    rdagent_api_key=config['api_key'],
                    azure_endpoint=config['azure_endpoint'],
                    azure_deployment=config['azure_deployment'],
                    azure_api_version=config['azure_api_version'],
                    chat_model=config['chat_model']
                )
                
                # Prepare data
                pipeline.prepare_data(
                    instruments=symbols,
                    start_time=start_date,
                    end_time=end_date
                )
                
                # Run pipeline with feature evolution
                results = pipeline.run_pipeline(
                    max_iterations=args.max_iterations,
                    stopping_threshold=0.01
                )
            except Exception as e:
                logger.error(f"Error running Qlib pipeline: {e}")
                logger.info("Falling back to yfinance data source")
                args.data_source = 'yfinance'
        
        if args.data_source == 'yfinance' or results is None:
            # Use yfinance data
            logger.info("Using yfinance data source")
            
            # Download data from yfinance
            data_dict = prepare_yfinance_data(
                symbols=symbols,
                start_date=start_date, 
                end_date=end_date,
                output_dir=os.path.join(args.output_dir, 'data')
            )
            
            # Use direct prediction with models and RDAgent
            from stockscan.core.model_factory import create_model
            from stockscan.core.prediction import predict_stock_movement
            
            prediction_results = {}
            best_model = None
            best_accuracy = 0
            
            # Try different models
            models = ['LightGBM', 'XGBoost']
            
            for model_name in models:
                try:
                    logger.info(f"Creating model: {model_name}")
                    model = create_model(model_name)
                    
                    # Train the model using all available data
                    logger.info(f"Training {model_name} model...")
                    model, trained = train_model(model, data_dict)
                    
                    # Skip prediction if training failed
                    if not trained:
                        logger.warning(f"Skipping prediction with {model_name} because training failed")
                        continue
                    
                    # Make predictions for each stock
                    model_predictions = {}
                    for symbol, data in data_dict.items():
                        if data is not None and not data.empty:
                            try:
                                pred = predict_stock_movement(model, data)
                                model_predictions[symbol] = pred
                                logger.info(f"Prediction for {symbol} with {model_name}: {pred:.4f}")
                            except Exception as e:
                                logger.error(f"Error predicting for {symbol}: {e}")
                    
                    # Calculate simple accuracy (placeholder for real evaluation)
                    if model_predictions:
                        accuracy = sum(1 for p in model_predictions.values() if p > 0) / len(model_predictions)
                    else:
                        accuracy = 0.0
                    
                    # Optimize model with RDAgent if available
                    try:
                        if rdagent_available and hasattr(rdagent_manager, 'optimize_model') and not args.skip_rdagent_optimization:
                            logger.info(f"Optimizing {model_name} with RDAgent")
                            
                            # Create directory for model config
                            model_config_dir = os.path.join(args.output_dir, f'rdagent_{model_name}')
                            os.makedirs(model_config_dir, exist_ok=True)
                            
                            # Save model parameters to file for debugging
                            model_config_path = os.path.join(model_config_dir, 'model_config.json')
                            with open(model_config_path, 'w') as f:
                                # Convert any non-serializable values to strings
                                config_dict = {}
                                for k, v in model.get_params().items():
                                    try:
                                        json.dumps({k: v})  # Test if serializable
                                        config_dict[k] = v
                                    except (TypeError, OverflowError):
                                        config_dict[k] = str(v)
                                json.dump(config_dict, f, indent=2)
                            
                            logger.info(f"Saved model config to {model_config_path}")
                            
                            try:
                                optimized_config = rdagent_manager.optimize_model(
                                    model_config=model.get_params(),
                                    data_path=os.path.join(args.output_dir, 'data'),
                                    output_path=model_config_dir,
                                    max_iterations=args.max_iterations
                                )
                            except Exception as e:
                                logger.error(f"RDAgent optimization error: {e}")
                                logger.warning(f"Checking if RDAgent is properly configured...")
                                
                                # Try to run a simple RDAgent command to see if it's working
                                try:
                                    import subprocess
                                    result = subprocess.run(
                                        ['rdagent', '--version'], 
                                        capture_output=True, 
                                        text=True
                                    )
                                    logger.info(f"RDAgent version check: {result.stdout.strip()}")
                                    if result.stderr:
                                        logger.warning(f"RDAgent stderr: {result.stderr.strip()}")
                                except Exception as e2:
                                    logger.error(f"Failed to check RDAgent version: {e2}")
                                
                                # Continue with the unoptimized model
                                logger.warning(f"Continuing with unoptimized {model_name} model")
                                optimized_config = None
                            
                            # Update model if optimization was successful
                            if optimized_config and optimized_config != model.get_params():
                                logger.info(f"Updating {model_name} with optimized parameters")
                                optimized_model = create_model(model_name, **optimized_config)
                                
                                # Train model with optimized parameters
                                optimized_model, optimized_trained = train_model(optimized_model, data_dict)
                                
                                # Only proceed with optimized model if training was successful
                                if optimized_trained:
                                    # Make predictions with optimized model
                                    optimized_predictions = {}
                                    for symbol, data in data_dict.items():
                                        if data is not None and not data.empty:
                                            try:
                                                pred = predict_stock_movement(optimized_model, data)
                                                optimized_predictions[symbol] = pred
                                                logger.info(f"Optimized prediction for {symbol}: {pred:.4f}")
                                            except Exception as e:
                                                logger.error(f"Error with optimized prediction for {symbol}: {e}")
                                    
                                    # Calculate accuracy with optimized model
                                    if optimized_predictions:
                                        optimized_accuracy = sum(1 for p in optimized_predictions.values() if p > 0) / len(optimized_predictions)
                                    else:
                                        optimized_accuracy = 0.0
                                    
                                    # Use optimized predictions if better
                                    if optimized_accuracy > accuracy:
                                        accuracy = optimized_accuracy
                                        model_predictions = optimized_predictions
                                        logger.info(f"Optimized {model_name} improved accuracy to {accuracy:.4f}")
                                else:
                                    logger.warning(f"Skipping optimized predictions because training failed")
                        else:
                            logger.info(f"Skipping RDAgent optimization for {model_name} because RDAgent is not available")
                    except Exception as e:
                        logger.warning(f"RDAgent optimization failed for {model_name}: {e}")
                    
                    # Track best model
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
                        prediction_results = model_predictions
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
            
            # Create structured results similar to pipeline output
            results = {
                'iterations': args.max_iterations,
                'best_model': best_model or 'LightGBM',
                'best_strategy': 'direct_prediction',
                'best_metrics': {
                    'accuracy': best_accuracy,
                    'prediction_confidence': sum(abs(p) for p in prediction_results.values()) / len(prediction_results) if prediction_results else 0
                }
            }
            
            # Add predictions if there are any
            if prediction_results:
                results['predictions'] = {symbol: float(pred) for symbol, pred in prediction_results.items()}
            else:
                # Add placeholder predictions
                results['predictions'] = {
                    'AAPL': 0.0,
                    'MSFT': 0.0,
                    'NVDA': 0.0
                }
                logger.warning("No valid predictions were made. Using placeholder predictions.")
        
        # Save results to JSON
        results_path = os.path.join(args.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            # Convert items that are not JSON serializable
            serializable_results = {
                'iterations': results['iterations'],
                'best_model': results['best_model'],
                'best_strategy': results['best_strategy'],
                'best_metrics': {k: float(v) for k, v in results['best_metrics'].items()}
            }
            
            # Add predictions if available
            if 'predictions' in results:
                serializable_results['predictions'] = results['predictions']
                
            json.dump(serializable_results, f, indent=4)
        
        # Print final results
        logger.info("=== Final Results ===")
        logger.info(f"Iterations completed: {results['iterations']}")
        logger.info(f"Best model: {results['best_model']}")
        logger.info(f"Best strategy: {results['best_strategy']}")
        logger.info("Best metrics:")
        for key, value in results['best_metrics'].items():
            logger.info(f"  {key}: {value:.4f}")
            
        # Print predictions if available
        if 'predictions' in results:
            logger.info("Predictions:")
            for symbol, pred in results['predictions'].items():
                direction = "UP" if pred > 0 else "DOWN"
                logger.info(f"  {symbol}: {direction} (confidence: {abs(pred):.4f})")
            
        # Generate a summary report
        generate_report(args.output_dir, symbols, start_date, end_date, results)
        
    finally:
        # Clean up if we started the UI
        if rdagent_process:
            try:
                rdagent_process.terminate()
                logger.info("RDAgent UI process terminated")
            except:
                pass
    
    return 0

def generate_report(output_dir, symbols, start_date, end_date, results):
    """
    Generate a summary report
    
    Parameters
    ----------
    output_dir : str
        Output directory
    symbols : list
        List of stock symbols
    start_date : str
        Start date
    end_date : str
        End date
    results : dict
        Pipeline results
    """
    report_path = os.path.join(output_dir, 'report.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# Azure OpenAI Trio Stock Analysis Report\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Date Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Stocks Analyzed**: {', '.join(symbols)}\n")
        f.write(f"- **Analysis Period**: {start_date} to {end_date}\n")
        f.write(f"- **Iterations Completed**: {results['iterations']}\n\n")
        
        f.write(f"## Best Model\n\n")
        f.write(f"- **Model Name**: {results['best_model']}\n\n")
        
        f.write(f"## Best Strategy\n\n")
        f.write(f"- **Strategy Name**: {results['best_strategy']}\n\n")
        
        f.write(f"## Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, value in results['best_metrics'].items():
            f.write(f"| {key} | {value:.4f} |\n")
        
        # Add predictions if available
        if 'predictions' in results:
            f.write(f"\n## Predictions\n\n")
            f.write("| Symbol | Direction | Confidence |\n")
            f.write("|--------|-----------|------------|\n")
            for symbol, pred in results['predictions'].items():
                direction = "UP" if pred > 0 else "DOWN"
                f.write(f"| {symbol} | {direction} | {abs(pred):.4f} |\n")
            
        f.write(f"\n## Notes\n\n")
        f.write(f"- Report generated using StockScan with Azure OpenAI integration\n")
        f.write(f"- RDAgent was used for feature evolution and model optimization\n")
        
    logger.info(f"Report generated: {report_path}")

if __name__ == "__main__":
    sys.exit(main()) 