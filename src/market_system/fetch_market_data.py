"""Standalone market data fetcher with async processing and rate limiting.
Includes market cap and fundamental data (sector, industry, company name, etc.) in InfluxDB.
"""
import asyncio
import argparse
from datetime import datetime, timedelta
import logging
import signal
import sys
import json
import time
import socket
import os
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from urllib.parse import urlparse

from src.market_data.providers.symbol_list_fetcher import SymbolListFetcher
from influxdb_client import InfluxDBClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for signal handling
shutdown_event = asyncio.Event()
main_task = None

def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date format: {e}")

def validate_symbols(symbols_str: str) -> List[str]:
    """Validate and parse symbols string."""
    if not symbols_str:
        return []
    return [s.strip() for s in symbols_str.split(',') if s.strip()]

def load_symbols_from_file(file_path: str) -> List[str]:
    """Load stock symbols from a file.
    
    The file can be:
    - A JSON file with a 'symbols' key containing a list of symbols
    - A text file with one symbol per line
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Symbol file not found: {file_path}")
        return []
    
    try:
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'symbols' in data:
                    return [s.strip() for s in data['symbols'] if s.strip()]
                elif isinstance(data, list):
                    return [s.strip() for s in data if isinstance(s, str) and s.strip()]
                else:
                    logger.error(f"Invalid JSON format in {file_path}")
                    return []
        else:
            with open(file_path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading symbols from {file_path}: {str(e)}")
        return []

def get_default_dates():
    """Get default start and end dates."""
    end_date = datetime.now()
    start_date = datetime(2020, 1, 1)  # Default start date
    return start_date, end_date

def print_section_header(title):
    """Print a section header to make output more readable."""
    width = 80
    padding = (width - len(title) - 2) // 2
    print("\n" + "=" * width)
    print(" " * padding + f" {title} " + " " * padding)
    print("=" * width + "\n")

def load_influxdb_config() -> Dict:
    """Load InfluxDB configuration from storage_config.json file."""
    default_config = {
        'url': 'http://localhost:8086',
        'token': '',
        'org': '',
        'bucket': 'stocks',
        'timeout': 10000
    }
    
    try:
        config_path = Path("config/storage_config.json")
        if not config_path.exists():
            logger.warning("Storage config file not found, using default configuration")
            return default_config
            
        with open(config_path, 'r') as f:
            storage_config = json.load(f)
            
        if 'influxdb' not in storage_config:
            logger.warning("InfluxDB configuration not found in storage config")
            return default_config
            
        config = storage_config['influxdb']
        
        # 确保所有必需的字段都存在
        for key in default_config:
            if key not in config:
                logger.warning(f"Missing {key} in config file, using default value")
                config[key] = default_config[key]
        
        # 强制使用 stocks bucket
        config['bucket'] = 'stocks'
        
        # 验证 URL 格式
        try:
            parsed = urlparse(config['url'])
            if not all([parsed.scheme, parsed.netloc]):
                logger.warning("Invalid URL format in config, using default URL")
                config['url'] = default_config['url']
        except Exception:
            logger.warning("Invalid URL format in config, using default URL")
            config['url'] = default_config['url']
            
        return config
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in storage config file")
        return default_config
    except Exception as e:
        logger.error(f"Error loading InfluxDB config: {str(e)}")
        return default_config

async def get_last_update_time(symbol: str, influxdb_config: Dict) -> Optional[datetime]:
    """Get the last trading date for a symbol from InfluxDB."""
    try:
        client = InfluxDBClient(
            url=influxdb_config.get('url', 'http://localhost:8086'),
            token=influxdb_config.get('token', ''),
            org=influxdb_config.get('org', ''),
            timeout=10_000  # 增加超时时间到 10 秒
        )
        
        # 使用固定的 bucket 名称 "stocks"
        bucket = "stocks"
        logger.debug(f"Querying last trading date for {symbol} from bucket: {bucket}")
        
        query = f'''
            from(bucket: "{bucket}")
                |> range(start: -10y)
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> filter(fn: (r) => r["_measurement"] == "stock_data")
                |> filter(fn: (r) => r["_field"] == "close")
                |> last()
                |> keep(columns: ["_time"])
        '''
        
        result = client.query_api().query(query)
        if result and len(result) > 0 and len(result[0].records) > 0:
            last_record = result[0].records[0]
            # 获取时间并确保是 offset-naive 的
            last_time = last_record.get_time()
            # 如果时间有时区信息，将其转换为 naive datetime
            if last_time.tzinfo is not None:
                last_time = last_time.replace(tzinfo=None)
                
            logger.info(f"Found last trading date for {symbol}: {last_time.strftime('%Y-%m-%d')}")
            # 返回下一个交易日作为开始日期
            next_date = last_time + timedelta(days=1)
            return next_date
            
    except Exception as e:
        logger.error(f"Error getting last trading date for {symbol}: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
    
    return None

def signal_handler(sig, frame):
    """Handle termination signals for graceful shutdown."""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    shutdown_event.set()
    
    if main_task and not main_task.done():
        logger.info("Cancelling main task...")
        main_task.cancel()

def test_tcp_connection(host: str, port: int, timeout: float = 5.0) -> bool:
    """Test if a TCP connection can be established to the host and port.
    
    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if connection was successful, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error testing TCP connection to {host}:{port}: {str(e)}")
        return False

def parse_influxdb_url(url: str) -> Tuple[str, int]:
    """Parse InfluxDB URL into host and port.
    
    Args:
        url: InfluxDB URL (e.g., http://localhost:8086)
        
    Returns:
        Tuple[str, int]: Host and port
    """
    try:
        if not url or url == 'default URL':
            # Try to get URL from environment variable or use a default
            url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
            
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            raise ValueError(f"Invalid URL format: {url}")
            
        port = parsed.port or 8086
        return host, port
    except Exception as e:
        logger.error(f"Error parsing InfluxDB URL: {str(e)}")
        return 'localhost', 8086

async def fetch_data(args):
    """Fetch market data based on CLI arguments."""
    start_time = time.time()
    
    try:
        print_section_header("Initializing Fetcher")
        
        # Load InfluxDB config
        influxdb_config = load_influxdb_config()
        if influxdb_config:
            db_url = influxdb_config.get('url')
            if db_url:
                logger.info(f"Using InfluxDB at: {db_url}")
            logger.info("Using bucket: stocks")  # 固定使用 stocks bucket
        else:
            logger.info("Using default InfluxDB configuration")
        
        # Initialize the symbol list fetcher
        fetcher = SymbolListFetcher(
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            rate_limit=args.rate_limit,
            cache_ttl_days=args.ttl_days,
            use_influxdb=True,  # Always use InfluxDB
            force_influxdb=args.force_influxdb  # Force InfluxDB regardless of config
        )

        # Set up dates based on different scenarios
        end_date = args.end_date or datetime.now()
        # 确保 end_date 是 offset-naive 的
        if end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
        
        if args.force:
            # If force is true, use provided start_date or default
            start_date = args.start_date or datetime(2020, 1, 1)
            logger.info(f"Force mode enabled, using start date: {start_date.strftime('%Y-%m-%d')}")
        else:
            if args.start_date:
                start_date = args.start_date
                logger.info(f"Using provided start date: {start_date.strftime('%Y-%m-%d')}")
            else:
                # Get the last update time from InfluxDB for each symbol
                try:
                    # 等待 fetcher 初始化完成
                    await asyncio.sleep(1)
                    
                    # 确保 fetcher 已经成功初始化并连接到 InfluxDB
                    if not hasattr(fetcher, 'cache') or not hasattr(fetcher.cache, 'client'):
                        raise ConnectionError("InfluxDB cache not properly initialized")
                    
                    # 从 fetcher.cache 获取配置
                    if not hasattr(fetcher.cache, 'config'):
                        raise ConnectionError("InfluxDB cache configuration not found")
                        
                    # 直接使用已经建立连接的客户端
                    client = fetcher.cache.client
                    influxdb_config = {
                        'url': str(client.url),
                        'token': client.token,
                        'org': client.org,
                        'bucket': 'stocks'  # 固定使用 stocks bucket
                    }
                    
                    logger.info(f"Using fetcher's InfluxDB configuration: {influxdb_config['url']}")
                    logger.info("Using bucket: stocks")
                    
                    host, port = parse_influxdb_url(influxdb_config['url'])
                    logger.info(f"Testing TCP connection to InfluxDB at {host}:{port}")
                    
                    if not test_tcp_connection(host, port):
                        raise ConnectionError(f"Failed to connect to InfluxDB at {host}:{port}")
                    
                    logger.info(f"Successfully connected to InfluxDB at {host}:{port}")
                    
                    # If connected, proceed with fetching last update times
                    last_dates = []
                    symbols = []
                    
                    if args.symbols:
                        symbols = validate_symbols(args.symbols)
                    elif args.symbols_file:
                        symbols = load_symbols_from_file(args.symbols_file)
                    
                    # Only check a few symbols to avoid too many connection attempts
                    sample_symbols = symbols[:5] if len(symbols) > 5 else symbols
                    
                    for symbol in sample_symbols:
                        last_time = await get_last_update_time(symbol, influxdb_config)
                        if last_time:
                            last_dates.append(last_time)
                    
                    if last_dates:
                        # Use the earliest last update time among all symbols
                        start_date = min(last_dates)
                        logger.info(f"Using last update time as start date: {start_date.strftime('%Y-%m-%d')}")
                    else:
                        # If no last update time found, use default start date
                        start_date = datetime(2020, 1, 1)
                        logger.info("No last update time found, using default start date")
                
                except Exception as e:
                    logger.error(f"InfluxDB connection error: {str(e)}")
                    logger.info("Falling back to default start date due to InfluxDB connection issues")
                    start_date = datetime(2020, 1, 1)

        # Validate dates
        if start_date > end_date:
            logger.error("Start date cannot be after end date")
            return 1

        # Check for shutdown signal before continuing
        if shutdown_event.is_set():
            logger.info("Shutdown requested before processing symbols.")
            return 0

        # Get symbols
        print_section_header("Loading Symbols")
        symbols = []
        if args.symbols_file:
            symbols = load_symbols_from_file(args.symbols_file)
            if not symbols:
                logger.error("No valid symbols found in the specified file")
                return 1
            logger.info(f"Loaded {len(symbols)} symbols from file: {args.symbols_file}")
            
            # Show first few symbols as a preview
            preview_count = min(5, len(symbols))
            if preview_count > 0:
                preview = ", ".join(symbols[:preview_count])
                if len(symbols) > preview_count:
                    preview += f", ... ({len(symbols) - preview_count} more)"
                logger.info(f"Symbol preview: {preview}")
                
        elif args.symbols:
            symbols = validate_symbols(args.symbols)
            if not symbols:
                logger.error("No valid symbols provided")
                return 1
            logger.info(f"Using {len(symbols)} symbols from command line: {', '.join(symbols)}")
        
        # Check for shutdown signal before continuing
        if shutdown_event.is_set():
            logger.info("Shutdown requested before fetching data.")
            return 0
            
        print_section_header("Fetching Data")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Storing data in InfluxDB (fallback directory: {args.data_dir})")
        
        if args.timeout > 0:
            logger.info(f"Operation will time out after {args.timeout} seconds")
        
        # If a specific market is provided, filter symbols for that market
        if args.market:
            # Check if market exists
            if args.market not in fetcher.registry.get_markets():
                logger.error(f"Invalid market: {args.market}")
                logger.info(f"Available markets: {', '.join(fetcher.registry.get_markets())}")
                return 1
                
            market_symbols = fetcher.registry.get_symbols(args.market)
            valid_symbols = [s for s in symbols if s in market_symbols]
            
            if not valid_symbols:
                logger.error(f"None of the provided symbols are valid for market: {args.market}")
                return 1
                
            logger.info(f"Processing {len(valid_symbols)} symbols for market: {args.market.upper()}")
            
            # Use market-specific fetch with timeout
            try:
                fetch_task = asyncio.create_task(
                    fetcher.fetch_market_data(
                        market=args.market,
                        start_date=start_date,
                        end_date=end_date,
                        symbols=valid_symbols,
                        chunk_size=args.batch_size
                    )
                )
                
                if args.timeout > 0:
                    # Set a timeout for the operation
                    try:
                        await asyncio.wait_for(fetch_task, timeout=args.timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"Operation timed out after {args.timeout} seconds")
                        fetch_task.cancel()
                else:
                    # No timeout, but check for shutdown events
                    while not fetch_task.done():
                        if shutdown_event.is_set():
                            fetch_task.cancel()
                            break
                        await asyncio.sleep(0.5)
                    
                    if not fetch_task.done():
                        await fetch_task  # Allow task to complete or raise CancelledError
            
            except asyncio.CancelledError:
                logger.info("Market data fetching was cancelled")
            except Exception as e:
                logger.error(f"Error during market fetching: {str(e)}")
                if args.continue_on_error:
                    logger.info("Continuing despite error due to --continue-on-error flag")
                else:
                    raise
        else:
            # No market specified, use direct symbol fetching
            logger.info(f"Fetching data for {len(symbols)} symbols without market constraints")
            
            # Log performance settings
            logger.info(f"Performance settings: max_concurrent={args.max_concurrent}, " +
                       f"batch_size={args.batch_size}, rate_limit={args.rate_limit}")
            
            # Fetch with timeout
            try:
                fetch_task = asyncio.create_task(
                    fetcher.fetch_symbol_data(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date,
                        chunk_size=args.batch_size
                    )
                )
                
                if args.timeout > 0:
                    # Set a timeout for the operation
                    try:
                        await asyncio.wait_for(fetch_task, timeout=args.timeout)
                    except asyncio.TimeoutError:
                        logger.warning(f"Operation timed out after {args.timeout} seconds")
                        fetch_task.cancel()
                else:
                    # No timeout, but check for shutdown events
                    while not fetch_task.done():
                        if shutdown_event.is_set():
                            fetch_task.cancel()
                            break
                        await asyncio.sleep(0.5)
                    
                    if not fetch_task.done():
                        await fetch_task  # Allow task to complete or raise CancelledError
            
            except asyncio.CancelledError:
                logger.info("Symbol data fetching was cancelled")
            except Exception as e:
                logger.error(f"Error during symbol fetching: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        # Calculate total run time
        total_duration = time.time() - start_time
        print_section_header("Summary")
        logger.info(f"Data fetching completed in {total_duration:.2f} seconds")
        
        # Show InfluxDB query information if verbose
        if args.verbose and influxdb_config:
            print_section_header("InfluxDB Access Info")
            logger.info("Data has been stored in InfluxDB and can be accessed with the following parameters:")
            logger.info(f"URL: {influxdb_config.get('url', 'default URL')}")
            logger.info(f"Bucket: stocks")
            logger.info(f"Organization: {influxdb_config.get('org', 'default org')}")
            logger.info("Example Flux query for recent data:")
            query_example = f"""
from(bucket: "stocks")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "stock_data")
  |> filter(fn: (r) => r.symbol == "{symbols[0] if symbols else 'AAPL'}")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
"""
            logger.info(query_example)
            
        return 0

    except asyncio.CancelledError:
        logger.info("Operation was cancelled.")
        total_duration = time.time() - start_time
        logger.info(f"Cancelled after running for {total_duration:.2f} seconds")
        return 0
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

async def async_main(args):
    """Async main function that can be cancelled properly."""
    global main_task
    main_task = asyncio.current_task()
    return await fetch_data(args)

def main():
    """CLI entry point for market data fetcher."""
    parser = argparse.ArgumentParser(description="Fetch stock market data and store it in InfluxDB")
    
    # Symbol selection - either from direct input or file
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument('-s', '--symbols', type=str,
                      help='Comma-separated list of symbols to fetch')
    symbol_group.add_argument('-f', '--symbols-file', type=str,
                      help='Path to file containing symbols (one per line or JSON with "symbols" key)')
    
    # Optional market specification
    parser.add_argument('-m', '--market', type=str,
                      help='Specific market to fetch from (e.g., nyse, nasdaq). If not specified, symbols will be fetched directly.')
    
    # Date range
    parser.add_argument('--start-date', type=parse_date,
                      help='Start date (YYYY-MM-DD format, default: 2020-01-01)')
    parser.add_argument('--end-date', type=parse_date,
                      help='End date (YYYY-MM-DD format, default: today)')
    
    # Configuration
    parser.add_argument('--data-dir', type=str, default="data",
                      help='Directory for storing fallback data if InfluxDB is unavailable')
    parser.add_argument('--config-dir', type=str, default="config/market",
                      help='Directory containing market configuration files')
    parser.add_argument('--ttl-days', type=int, default=30,
                      help='Number of days to keep data in cache')
    
    # InfluxDB specific
    parser.add_argument('--force-influxdb', action='store_true',
                      help='Force using InfluxDB even if config specifies file storage')
    
    # Performance tuning
    parser.add_argument('--max-concurrent', type=int, default=5,
                      help='Maximum concurrent downloads')
    parser.add_argument('--batch-size', type=int, default=20,
                      help='Number of symbols to process in each batch')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                      help='Rate limit in operations per second')
    parser.add_argument('--timeout', type=int, default=0,
                      help='Timeout in seconds for the entire operation (0 for no timeout)')
    
    # Error handling
    parser.add_argument('--continue-on-error', action='store_true',
                      help='Continue processing if a market fails')
    
    # Verbose mode
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output with detailed logging')

    # Force parameter
    parser.add_argument('--force', action='store_true',
                      help='Force fetching data from the start date')

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled")
    
    # Ensure data directory exists (for fallback)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    # Validate that either symbols or symbols-file is provided
    if not args.symbols and not args.symbols_file:
        parser.error("Either --symbols or --symbols-file must be provided")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Run async fetcher
    logger.info("Starting data fetcher - press Ctrl+C to cancel")
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        return 0

if __name__ == "__main__":
    sys.exit(main())
