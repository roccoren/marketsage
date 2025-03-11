"""Yahoo Finance market data provider implementation."""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import yfinance as yf
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .base import BaseMarketDataProvider
from ..models.common import (
    MarketData,
    TickerInfo,
    ProviderConfig
)
from ..exceptions import (
    DataNotFoundError,
    ConnectionError,
    ValidationError
)

class YahooFinanceProvider(BaseMarketDataProvider):
    """
    Yahoo Finance implementation of market data provider.
    
    Features:
    - Efficient data fetching with connection pooling
    - Automatic request retries
    - Data validation and cleaning
    - Pandas DataFrame conversion
    """
    
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
        """
        try:
            def fetch():
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    auto_adjust=True
                )
                return df
                
            df = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                fetch
            )
            
            if df is None or df.empty:
                raise DataNotFoundError(f"No data returned for {symbol}")
                
            return self._transform_df_to_market_data(df)
            
        except Exception as e:
            if "failed to establish a new connection" in str(e).lower():
                raise ConnectionError(f"Connection failed: {str(e)}")
            raise DataNotFoundError(f"Failed to fetch {symbol}: {str(e)}")

    def _transform_df_to_market_data(self, df: pd.DataFrame) -> List[MarketData]:
        """Transform pandas DataFrame to MarketData objects."""
        market_data = []
        
        for index, row in df.iterrows():
            # Validate data
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if pd.isna(row[col]):
                    continue
            
            data = MarketData(
                date=index.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                adjusted_close=float(row['Close'])  # Using adjusted prices
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
            def fetch():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return info
                
            info = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                fetch
            )
            
            if not info:
                raise DataNotFoundError(f"No info available for {symbol}")
                
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

    def __del__(self):
        """Cleanup resources on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)