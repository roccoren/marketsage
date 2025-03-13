"""Market symbol loaders implementation."""
import json
import os
from typing import Dict, Any, List
import aiofiles
import aiofiles.os

from .interfaces import MarketSymbolLoader
from .models.common import MarketSource
from .exceptions import ValidationError

class JsonMarketSymbolLoader(MarketSymbolLoader):
    """
    Loads market symbols from JSON files.
    
    Expected JSON format:
    {
        "symbols": ["AAPL", "MSFT", ...],
        "metadata": {
            "market": "NYSE",
            "last_updated": "2024-03-13",
            "total_symbols": 500,
            ...
        }
    }
    """
    
    def __init__(self, config_dir: str = "config/market"):
        """
        Initialize loader with config directory.
        
        Args:
            config_dir: Directory containing market symbol files
        """
        self.config_dir = config_dir
        
    def _get_market_file(self, market: MarketSource) -> str:
        """Get path to market symbols file."""
        filename = f"{market.name.lower()}-listed.json"
        return os.path.join(self.config_dir, filename)
        
    async def _validate_market_data(self, data: Dict[str, Any], market: MarketSource) -> None:
        """
        Validate market data format and content.
        
        Raises:
            ValidationError: If data is invalid
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Invalid JSON format for {market.name}")
            
        if "symbols" not in data:
            raise ValidationError(f"No symbols found for {market.name}")
            
        if not isinstance(data["symbols"], list):
            raise ValidationError(f"Symbols must be a list for {market.name}")
            
        if not all(isinstance(s, str) for s in data["symbols"]):
            raise ValidationError(f"Invalid symbol format in {market.name}")
            
        if "metadata" in data and not isinstance(data["metadata"], dict):
            raise ValidationError(f"Invalid metadata format for {market.name}")
            
    async def load_symbols(self, market: MarketSource) -> List[str]:
        """Load symbols for a specific market."""
        file_path = self._get_market_file(market)
        
        try:
            if not await aiofiles.os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Market symbols file not found: {file_path}"
                )
            
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                data = json.loads(content)
                
            await self._validate_market_data(data, market)
            return data["symbols"]
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")
            
    async def get_market_metadata(self, market: MarketSource) -> Dict[str, Any]:
        """Get metadata for a specific market."""
        file_path = self._get_market_file(market)
        
        try:
            if not await aiofiles.os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Market symbols file not found: {file_path}"
                )
            
            async with aiofiles.open(file_path, mode='r') as f:
                content = await f.read()
                data = json.loads(content)
                
            await self._validate_market_data(data, market)
            return data.get("metadata", {})
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load market metadata: {str(e)}")