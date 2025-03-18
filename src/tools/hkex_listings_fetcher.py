#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import logging
import os
import sys
from datetime import datetime
import pytz
import re
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HKEXListingsFetcher:
    BASE_URL = "https://www1.hkex.com.hk/hkexwidget/data/getequityfilter"
    OUTPUT_FILE = "config/market/hkex-listed.json"
    
    def __init__(self, output_path: Optional[str] = None):
        """Initialize the HKEX listings fetcher.
        
        Args:
            output_path: Optional path to save the output JSON file.
        """
        self.output_path = output_path or self.OUTPUT_FILE
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "referer": "https://www.hkex.com.hk/",
            "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Microsoft Edge";v="134"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "macOS",
            "sec-fetch-dest": "script",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0"
        }
        
    async def fetch_listings(self) -> Dict[str, Any]:
        """Fetch listings data from HKEX."""
        timestamp = int(datetime.now().timestamp() * 1000)
        params = {
            "lang": "eng",
            "token": "evLtsLsBNAUVTPxtGqVeGzU7Byjw9QeA7ndoolVIuMVaILrp0WyYF5oVJ1QBOa7r",
            "sort": "5",
            "order": "0",
            "all": "1",
            "qid": str(timestamp),
            "callback": f"jQuery35106838071961914929_{timestamp - 65000}",
            "_": str(timestamp)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, headers=self.headers, params=params) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP error {response.status}: {await response.text()}")
                    
                    data = await response.text()
                    logger.debug(f"Response length: {len(data)}")
                    # Log a sample of the raw response
                    logger.debug(f"Response sample: {data[:200]}...")
                    return self._parse_jsonp(data)
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching HKEX listings: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error fetching HKEX listings: {str(e)}")
            raise

    def _parse_jsonp(self, jsonp_str: str) -> Dict[str, Any]:
        """Parse JSONP response to extract JSON data."""
        try:
            # Pattern to handle JSONP callback
            pattern = r'jQuery[0-9_]+\((.*)\)'
            match = re.search(pattern, jsonp_str)
            if not match:
                logger.error("Failed to match JSONP pattern")
                logger.debug(f"JSONP string prefix: {jsonp_str[:200]}...")
                raise ValueError("Invalid JSONP response format")
            
            json_str = match.group(1)
            data = json.loads(json_str)
            
            # Log the structure of the parsed data
            logger.debug(f"Parsed data structure keys: {list(data.keys()) if data else 'None'}")
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing JSONP response: {str(e)}")
            logger.debug(f"JSONP response sample: {jsonp_str[:200]}...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing JSONP: {str(e)}")
            raise

    def _format_output(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw data into the required output structure.
        
        Args:
            raw_data: The parsed JSON data from HKEX API
            
        Returns:
            Formatted data in the required schema
        """
        try:
            if not isinstance(raw_data, dict) or 'data' not in raw_data:
                raise ValueError(f"Invalid data structure. Expected dict with 'data' key.")
            
            data = raw_data['data']
            if 'stocklist' not in data:
                raise ValueError(f"Missing 'stocklist' in data. Available keys: {list(data.keys())}")
            
            stock_list = data['stocklist']
            if not isinstance(stock_list, list):
                raise ValueError(f"'stocklist' is not a list: {type(stock_list)}")
            
            # Extract symbols (ric field)
            symbols = []
            for stock in stock_list:
                if 'ric' in stock and stock['ric']:
                    symbols.append(stock['ric'])
            
            # Sort symbols for consistency
            symbols = sorted(symbols)
            
            # Create metadata
            hk_tz = pytz.timezone('Asia/Hong_Kong')
            current_time = datetime.now(hk_tz)
            
            return {
                "symbols": symbols,
                "metadata": {
                    "market": "HKEX",
                    "description": "Hong Kong Stock Exchange Listed Companies",
                    "last_updated": current_time.strftime("%Y-%m-%d"),
                    "total_symbols": len(symbols),
                    "market_hours": {
                        "open": "09:30",
                        "close": "16:00",
                        "timezone": "Asia/Hong_Kong"
                    },
                    "exchange_info": {
                        "mic": "XHKG",
                        "website": "https://www.hkex.com.hk"
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error formatting output data: {str(e)}")
            raise

    def save_to_file(self, data: Dict[str, Any]) -> None:
        """Save formatted data to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully saved listings to {self.output_path}")
            logger.info(f"Total symbols saved: {data['metadata']['total_symbols']}")
        except IOError as e:
            logger.error(f"IO error while saving file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving to file: {str(e)}")
            raise

async def main():
    """Main function to orchestrate the fetching and saving of HKEX listings."""
    try:
        logger.info("Starting HKEX listings fetch...")
        fetcher = HKEXListingsFetcher()
        raw_data = await fetcher.fetch_listings()
        logger.info("Successfully fetched raw data, formatting...")
        formatted_data = fetcher._format_output(raw_data)
        fetcher.save_to_file(formatted_data)
        logger.info("Process completed successfully")
    except Exception as e:
        logger.error(f"Failed to fetch and save HKEX listings: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    # Set log level to INFO for production use
    # For debugging, you can change to logging.DEBUG
    logger.setLevel(logging.INFO)
    sys.exit(asyncio.run(main()))
