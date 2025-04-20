#!/usr/bin/env python3
"""
Stocks router for the StockScan API.
"""

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api.core.dependencies import (
    AnalystUser,
    CurrentUser,
    DbSession,
    get_current_active_user,
)
from api.schemas.stock import (
    StockData,
    StockDataRequest,
    StockInfo,
    StockList,
    StockRefreshRequest,
    StockRefreshResponse,
    StockSearchRequest,
    StockSearchResponse,
)

# Import existing modules for data acquisition and storage
from modules.data_acquisition import DataAcquisitionManager as DataAcquisition
from modules.storage import StorageManager as Storage
from modules.stock_manager import StockManager

router = APIRouter()


@router.get("/", response_model=StockList)
async def list_stocks(
    search: Optional[str] = None,
    sector: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    List available stocks.
    
    Args:
        search: Search term for stock symbol or name
        sector: Filter by sector
        limit: Maximum number of stocks to return
        offset: Offset for pagination
        current_user: Current user
        
    Returns:
        List of stocks
    """
    # Initialize stock manager
    stock_manager = StockManager()
    
    # Get available stocks
    stocks = stock_manager.get_available_stocks(
        search=search,
        sector=sector,
        limit=limit,
        offset=offset,
    )
    
    # Format response
    stock_list = []
    for stock in stocks:
        stock_info = StockInfo(
            symbol=stock["symbol"],
            name=stock.get("name"),
            sector=stock.get("sector"),
            industry=stock.get("industry"),
            exchange=stock.get("exchange"),
            currency=stock.get("currency", "USD"),
            country=stock.get("country"),
            last_price=stock.get("last_price"),
            price_change=stock.get("price_change"),
            price_change_percent=stock.get("price_change_percent"),
            market_cap=stock.get("market_cap"),
            last_updated=stock.get("last_updated"),
        )
        stock_list.append(stock_info)
    
    return {
        "stocks": stock_list,
        "total": len(stock_list),
    }


@router.get("/{symbol}", response_model=StockInfo)
async def get_stock_details(
    symbol: str,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get stock details.
    
    Args:
        symbol: Stock symbol
        current_user: Current user
        
    Returns:
        Stock details
    """
    # Initialize stock manager
    stock_manager = StockManager()
    
    # Get stock details
    stock = stock_manager.get_stock_details(symbol)
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock with symbol {symbol} not found",
        )
    
    # Format response
    stock_info = StockInfo(
        symbol=stock["symbol"],
        name=stock.get("name"),
        sector=stock.get("sector"),
        industry=stock.get("industry"),
        exchange=stock.get("exchange"),
        currency=stock.get("currency", "USD"),
        country=stock.get("country"),
        last_price=stock.get("last_price"),
        price_change=stock.get("price_change"),
        price_change_percent=stock.get("price_change_percent"),
        market_cap=stock.get("market_cap"),
        last_updated=stock.get("last_updated"),
    )
    
    return stock_info


@router.get("/{symbol}/data", response_model=StockData)
async def get_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Get stock price data.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval (1d, 5m, etc.)
        current_user: Current user
        
    Returns:
        Stock price data
    """
    # Initialize data acquisition
    data_acquisition = DataAcquisition()
    
    # Initialize storage
    storage = Storage()
    
    # Get stock data
    try:
        # First try to get data from storage
        data = storage.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        
        # If no data found, fetch from external source
        if not data or len(data) == 0:
            data = data_acquisition.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
            
            # Store data
            if data and len(data) > 0:
                storage.store_stock_data(
                    symbol=symbol,
                    data=data,
                    interval=interval,
                )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching stock data: {str(e)}",
        )
    
    # Check if data was found
    if not data or len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for stock with symbol {symbol}",
        )
    
    # Format response
    stock_data_points = []
    for point in data:
        stock_data_point = {
            "timestamp": point["timestamp"],
            "open": point["open"],
            "high": point["high"],
            "low": point["low"],
            "close": point["close"],
            "volume": point["volume"],
        }
        
        # Add indicators if available
        if "indicators" in point:
            stock_data_point["indicators"] = point["indicators"]
        
        stock_data_points.append(stock_data_point)
    
    # Get stock details
    stock_manager = StockManager()
    stock = stock_manager.get_stock_details(symbol)
    
    return {
        "symbol": symbol,
        "name": stock.get("name") if stock else None,
        "data": stock_data_points,
        "last_updated": stock.get("last_updated") if stock else None,
    }


@router.post("/refresh", response_model=StockRefreshResponse)
async def refresh_stock_data(
    refresh_request: StockRefreshRequest,
    current_user: dict = Depends(AnalystUser),
) -> Any:
    """
    Manually refresh stock data.
    
    Args:
        refresh_request: Refresh request
        current_user: Current user with analyst role
        
    Returns:
        Refresh response
    """
    # Initialize data acquisition
    data_acquisition = DataAcquisition()
    
    # Initialize storage
    storage = Storage()
    
    # Get stock data for each symbol
    failed_symbols = {}
    completed_symbols = []
    
    for symbol in refresh_request.symbols:
        try:
            # Fetch data from external source
            data = data_acquisition.get_stock_data(
                symbol=symbol,
                start_date=refresh_request.start_date,
                end_date=refresh_request.end_date,
                interval=refresh_request.interval,
            )
            
            # Store data
            if data and len(data) > 0:
                storage.store_stock_data(
                    symbol=symbol,
                    data=data,
                    interval=refresh_request.interval,
                )
                completed_symbols.append(symbol)
            else:
                failed_symbols[symbol] = "No data found"
        except Exception as e:
            failed_symbols[symbol] = str(e)
    
    # Determine status
    if len(failed_symbols) == 0:
        status = "completed"
        message = "All symbols refreshed successfully"
    elif len(completed_symbols) == 0:
        status = "failed"
        message = "Failed to refresh any symbols"
    else:
        status = "partial"
        message = f"Refreshed {len(completed_symbols)} out of {len(refresh_request.symbols)} symbols"
    
    return {
        "symbols": refresh_request.symbols,
        "status": status,
        "message": message,
    }


@router.post("/search", response_model=StockSearchResponse)
async def search_stocks(
    search_request: StockSearchRequest,
    current_user: dict = Depends(get_current_active_user),
) -> Any:
    """
    Search for stocks.
    
    Args:
        search_request: Search request
        current_user: Current user
        
    Returns:
        Search results
    """
    # Initialize stock manager
    stock_manager = StockManager()
    
    # Search for stocks
    stocks = stock_manager.search_stocks(
        query=search_request.query,
        limit=search_request.limit,
    )
    
    # Format response
    stock_list = []
    for stock in stocks:
        stock_info = StockInfo(
            symbol=stock["symbol"],
            name=stock.get("name"),
            sector=stock.get("sector"),
            industry=stock.get("industry"),
            exchange=stock.get("exchange"),
            currency=stock.get("currency", "USD"),
            country=stock.get("country"),
            last_price=stock.get("last_price"),
            price_change=stock.get("price_change"),
            price_change_percent=stock.get("price_change_percent"),
            market_cap=stock.get("market_cap"),
            last_updated=stock.get("last_updated"),
        )
        stock_list.append(stock_info)
    
    return {
        "results": stock_list,
        "total": len(stock_list),
    }