"""Yahoo Finance market data provider implementation."""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, TypeVar, Callable, Awaitable
from decimal import Decimal
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .base import BaseMarketDataProvider
from ..models.common import (
    MarketData,
    TickerInfo,
    RealTimeQuote,
    ProviderConfig
)
from ..exceptions import (
    DataNotFoundError,
    ConnectionError,
    ValidationError
)

T = TypeVar('T')

class RetryableError(Exception):
    """Error that should trigger a retry."""
    pass

class YahooFinanceProvider(BaseMarketDataProvider):
    """
    Yahoo Finance implementation of market data provider.
    
    Features:
    - Efficient data fetching with connection pooling
    - Automatic request retries with exponential backoff
    - Data validation and cleaning
    - Pandas DataFrame conversion
    """
    
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 30  # seconds
    
    def __init__(self, config: ProviderConfig):
        """Initialize Yahoo Finance provider with config."""
        super().__init__(config)
        
        # Configure yfinance global settings
        yf.set_tz_cache_location(
            config.cache.directory or "./data/yfinance_cache"
        )
        
        # Thread pool for synchronous yfinance operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=min(32, config.rate_limit.burst_size)
        )

    async def _retry_with_backoff(self, operation: Callable[[], Awaitable[T]], symbol: str) -> T:
        """Helper method to implement exponential backoff retry logic."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                print(f"DEBUG: Attempt {attempt + 1}/{self.MAX_RETRIES} for {symbol}")
                # Initial delay based on attempt number to prevent immediate retries
                if attempt > 0:
                    wait_time = self.INITIAL_BACKOFF * (2 ** (attempt - 1))
                    print(f"DEBUG: Waiting {wait_time} seconds before retry")
                    await asyncio.sleep(wait_time)
                
                result = await operation()
                print(f"DEBUG: Operation succeeded for {symbol} on attempt {attempt + 1}")
                return result
            except RetryableError as e:
                last_error = e
                print(f"DEBUG: Retryable error on attempt {attempt + 1} for {symbol}: {str(e)}")
                continue
            except Exception as e:
                print(f"DEBUG: Non-retryable error for {symbol}: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise DataNotFoundError(f"Max retries ({self.MAX_RETRIES}) exceeded for {symbol}: {str(last_error)}")

    async def _validate_symbol_exists(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and is active.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            bool: True if symbol is valid and active
        """
        try:
            async def fetch():
                def _fetch():
                    try:
                        print(f"DEBUG: Fetching info for symbol {symbol}")
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        
                        if not info:
                            print(f"DEBUG: No info returned for {symbol}")
                            return False
                        
                        print(f"DEBUG: Successfully retrieved info for {symbol}")
                        return True
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Invalid JSON response for {symbol}: {str(e)}")
                        raise RetryableError(f"Invalid JSON response: {str(e)}")
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            raise RetryableError(str(e))
                        print(f"DEBUG: Unexpected error for {symbol}: {str(e)}")
                        raise

                return await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool,
                    _fetch
                )

            result = await self._retry_with_backoff(fetch, symbol)
            print(f"DEBUG: Final validation result for {symbol}: {result}")
            return result
        except DataNotFoundError:
            return False
        except Exception as e:
            print(f"DEBUG: Error during symbol validation for {symbol}: {str(e)}")
            return False

    async def _fetch_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[MarketData]:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: Clean symbol string
            start_date: Start date
            end_date: End date
            
        Returns:
            List of MarketData objects
            
        Raises:
            DataNotFoundError: If data fetch fails
            ValidationError: If symbol is invalid
        """
        # Validate symbol first
        print(f"DEBUG: Starting symbol validation for {symbol}")
        is_valid = await self._validate_symbol_exists(symbol)
        print(f"DEBUG: Symbol {symbol} existence validation result: {is_valid}")
        if not is_valid:
            raise ValidationError(f"Invalid or inactive symbol: {symbol}")
            
        try:
            async def fetch():
                def _fetch():
                    try:
                        print(f"DEBUG: Fetching history for {symbol}")
                        ticker = yf.Ticker(symbol)
                        df = ticker.history(
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            interval='1d',
                            auto_adjust=True
                        )
                        if df is None or df.empty:
                            raise RetryableError(f"Empty data returned for {symbol}")
                        return df
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            raise RetryableError(str(e))
                        print(f"DEBUG: Error fetching history for {symbol}: {str(e)}")
                        raise

                return await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool,
                    _fetch
                )
            
            print(f"DEBUG: Starting history fetch for {symbol}")
            df = await self._retry_with_backoff(fetch, symbol)
            
            if df is None or df.empty:
                raise DataNotFoundError(f"No data returned for {symbol}")
                
            print(f"DEBUG: Successfully fetched history for {symbol}")
            return self._transform_df_to_market_data(df)
            
        except Exception as e:
            if "failed to establish a new connection" in str(e).lower():
                raise ConnectionError(f"Connection failed: {str(e)}")
            raise DataNotFoundError(f"Failed to fetch {symbol}: {str(e)}")

    def _transform_df_to_market_data(self, df: pd.DataFrame) -> List[MarketData]:
        """Transform pandas DataFrame to MarketData objects."""
        market_data = []
        
        for index, row in df.iterrows():
            # Skip rows with missing essential data
            if any(pd.isna(row[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                continue
            
            adjusted = None
            if 'Adj Close' in row and not pd.isna(row['Adj Close']):
                adjusted = float(row['Adj Close'])
            else:
                adjusted = float(row['Close'])
            
            # Get dividends and stock splits if available
            dividends = float(row['Dividends']) if 'Dividends' in row and not pd.isna(row['Dividends']) else 0.0
            splits = float(row['Stock Splits']) if 'Stock Splits' in row and not pd.isna(row['Stock Splits']) else 0.0
            
            data = MarketData(
                date=index.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                dividends=dividends,
                stock_splits=splits,
                adjusted_close=adjusted
            )
            market_data.append(data)
            
        return market_data

    async def _fetch_ticker_info(self, symbol: str) -> TickerInfo:
        """
        Fetch ticker information from Yahoo Finance.
        
        Args:
            symbol: Clean symbol string
            
        Returns:
            TickerInfo object
            
        Raises:
            DataNotFoundError: If info fetch fails
        """
        try:
            async def fetch():
                def _fetch():
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        if not info:
                            raise RetryableError(f"No info available for {symbol}")
                        return info
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Invalid JSON response in ticker info for {symbol}: {str(e)}")
                        raise RetryableError(f"Invalid JSON response: {str(e)}")
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            raise RetryableError(str(e))
                        print(f"DEBUG: Error fetching ticker info for {symbol}: {str(e)}")
                        raise

                return await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool,
                    _fetch
                )
            
            info = await self._retry_with_backoff(fetch, symbol)
                
            return TickerInfo(
                symbol=symbol,
                name=info.get('longName'),
                exchange=info.get('exchange'),
                currency=info.get('currency'),
                market_cap=info.get('marketCap'),
                sector=info.get('sector'),
                industry=info.get('industry')
            )
            
        except Exception as e:
            if "failed to establish a new connection" in str(e).lower():
                raise ConnectionError(f"Connection failed: {str(e)}")
            raise DataNotFoundError(f"Failed to fetch info for {symbol}: {str(e)}")

    async def _fetch_real_time_quote(self, symbol: str) -> RealTimeQuote:
        """
        Fetch real-time quote from Yahoo Finance.
        
        Args:
            symbol: Clean symbol string
            
        Returns:
            RealTimeQuote object
            
        Raises:
            DataNotFoundError: If quote cannot be retrieved
            ConnectionError: If provider connection fails
        """
        try:
            async def fetch():
                def _fetch():
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.fast_info
                        
                        # Get quote data
                        last_quote = ticker.history(period='1d', interval='1m').iloc[-1]
                        
                        return {
                            'price': info.last_price,
                            'change': info.last_price - info.previous_close,
                            'change_percent': ((info.last_price - info.previous_close) / info.previous_close) * 100,
                            'volume': info.last_volume,
                            'timestamp': datetime.now(),
                            'bid': last_quote.get('Bid', None),
                            'ask': last_quote.get('Ask', None),
                            'last_trade_time': info.last_trade_time,
                        }
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Invalid JSON response in quote for {symbol}: {str(e)}")
                        raise RetryableError(f"Invalid JSON response: {str(e)}")
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            raise RetryableError(str(e))
                        print(f"DEBUG: Error fetching quote for {symbol}: {str(e)}")
                        raise

                return await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool,
                    _fetch
                )
            
            quote_data = await self._retry_with_backoff(fetch, symbol)
            
            return RealTimeQuote(
                symbol=symbol,
                **quote_data
            )
            
        except Exception as e:
            if "failed to establish a new connection" in str(e).lower():
                raise ConnectionError(f"Connection failed: {str(e)}")
            raise DataNotFoundError(f"Failed to fetch quote for {symbol}: {str(e)}")

    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)