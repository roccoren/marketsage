from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .yahoo_finance_client import YahooFinanceClient

app = FastAPI()

class StockRequest(BaseModel):
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.post("/fetch_stock_data")
async def fetch_stock_data(request: StockRequest):
    async with YahooFinanceClient() as client:
        try:
            data = await client.get_history_batch(
                request.symbols,
                start_date=request.start_date,
                end_date=request.end_date
            )
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
