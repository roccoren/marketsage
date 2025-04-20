#!/usr/bin/env python3
"""
Example client for the StockScan API.
This script demonstrates how to interact with the API programmatically.
"""

import argparse
import json
import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default API URL
DEFAULT_API_URL = "http://localhost:8000/api"


class StockScanClient:
    """Client for the StockScan API."""
    
    def __init__(self, api_url: str = DEFAULT_API_URL):
        """Initialize the client."""
        self.api_url = api_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    def register(self, username: str, email: str, password: str, full_name: Optional[str] = None) -> Dict:
        """Register a new user."""
        data = {
            "username": username,
            "email": email,
            "password": password,
        }
        if full_name:
            data["full_name"] = full_name
        
        response = requests.post(
            f"{self.api_url}/auth/register",
            json=data,
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        
        return data
    
    def login(self, username: str, password: str) -> Dict:
        """Login with username and password."""
        data = {
            "username": username,
            "password": password,
        }
        
        response = requests.post(
            f"{self.api_url}/auth/login/alt",
            json=data,
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        
        return data
    
    def refresh(self) -> Dict:
        """Refresh access token."""
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        
        data = {
            "refresh_token": self.refresh_token,
        }
        
        response = requests.post(
            f"{self.api_url}/auth/refresh",
            json=data,
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        
        return data
    
    def logout(self) -> None:
        """Logout user."""
        if not self.refresh_token:
            return
        
        data = {
            "refresh_token": self.refresh_token,
        }
        
        requests.post(
            f"{self.api_url}/auth/logout",
            json=data,
            headers=self.get_headers(),
        )
        
        self.access_token = None
        self.refresh_token = None
    
    def get_current_user(self) -> Dict:
        """Get current user info."""
        response = requests.get(
            f"{self.api_url}/auth/me",
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def list_stocks(self, search: Optional[str] = None, limit: int = 100, offset: int = 0) -> Dict:
        """List available stocks."""
        params = {
            "limit": limit,
            "offset": offset,
        }
        if search:
            params["search"] = search
        
        response = requests.get(
            f"{self.api_url}/stocks",
            params=params,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_stock_data(self, symbol: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None, interval: str = "1d") -> Dict:
        """Get stock price data."""
        params = {"interval": interval}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        response = requests.get(
            f"{self.api_url}/stocks/{symbol}/data",
            params=params,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def refresh_stock_data(self, symbols: list, start_date: Optional[str] = None,
                          end_date: Optional[str] = None, interval: str = "1d") -> Dict:
        """Manually refresh stock data."""
        data = {
            "symbols": symbols,
            "interval": interval,
        }
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date
        
        response = requests.post(
            f"{self.api_url}/stocks/refresh",
            json=data,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def create_prediction(self, symbol: str, model_type: str = "lstm",
                         prediction_period: str = "30d") -> Dict:
        """Create a new prediction."""
        data = {
            "symbol": symbol,
            "model_type": model_type,
            "prediction_period": prediction_period,
        }
        
        response = requests.post(
            f"{self.api_url}/predictions",
            json=data,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def list_predictions(self, symbol: Optional[str] = None, model_type: Optional[str] = None,
                        prediction_period: Optional[str] = None, status: Optional[str] = None,
                        page: int = 1, page_size: int = 10) -> Dict:
        """List predictions."""
        params = {
            "page": page,
            "page_size": page_size,
        }
        if symbol:
            params["symbol"] = symbol
        if model_type:
            params["model_type"] = model_type
        if prediction_period:
            params["prediction_period"] = prediction_period
        if status:
            params["status"] = status
        
        response = requests.get(
            f"{self.api_url}/predictions",
            params=params,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def create_schedule(self, name: str, task_type: str, params: Dict,
                       cron_expression: Optional[str] = None,
                       interval_seconds: Optional[int] = None) -> Dict:
        """Create a new schedule."""
        data = {
            "name": name,
            "task_type": task_type,
            "params": params,
        }
        if cron_expression:
            data["cron_expression"] = cron_expression
        if interval_seconds:
            data["interval_seconds"] = interval_seconds
        
        response = requests.post(
            f"{self.api_url}/schedules",
            json=data,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()
    
    def list_schedules(self, task_type: Optional[str] = None, status: Optional[str] = None,
                      page: int = 1, page_size: int = 10) -> Dict:
        """List schedules."""
        params = {
            "page": page,
            "page_size": page_size,
        }
        if task_type:
            params["task_type"] = task_type
        if status:
            params["status"] = status
        
        response = requests.get(
            f"{self.api_url}/schedules",
            params=params,
            headers=self.get_headers(),
        )
        response.raise_for_status()
        
        return response.json()


def main():
    """Run the example client."""
    parser = argparse.ArgumentParser(description="StockScan API Client Example")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("--username", default="testuser", help="Username")
    parser.add_argument("--password", default="Password123", help="Password")
    parser.add_argument("--email", default="testuser@example.com", help="Email")
    args = parser.parse_args()
    
    # Create client
    client = StockScanClient(api_url=args.api_url)
    
    # Try to login, if it fails, register
    try:
        print("Logging in...")
        client.login(args.username, args.password)
        print("Logged in successfully.")
    except requests.HTTPError:
        print("Login failed, trying to register...")
        try:
            client.register(args.username, args.email, args.password)
            print("Registered and logged in successfully.")
        except requests.HTTPError as e:
            print(f"Registration failed: {e}")
            return
    
    # Get current user
    try:
        user = client.get_current_user()
        print(f"Current user: {user['username']} ({user['email']})")
    except requests.HTTPError as e:
        print(f"Failed to get current user: {e}")
    
    # List stocks
    try:
        stocks = client.list_stocks(limit=5)
        print(f"Found {stocks['total']} stocks. First 5:")
        for stock in stocks['stocks'][:5]:
            print(f"  {stock['symbol']}: {stock.get('name', 'N/A')}")
    except requests.HTTPError as e:
        print(f"Failed to list stocks: {e}")
    
    # Create a prediction
    try:
        symbol = "AAPL"  # Example symbol
        print(f"Creating prediction for {symbol}...")
        prediction = client.create_prediction(symbol)
        print(f"Prediction created: {prediction['id']}")
        print(f"Status: {prediction['status']}")
        if prediction.get('results'):
            print(f"Forecast change: {prediction['results']['forecast_change_percent']:.2f}%")
    except requests.HTTPError as e:
        print(f"Failed to create prediction: {e}")
    
    # List predictions
    try:
        predictions = client.list_predictions(page_size=5)
        print(f"Found {predictions['total']} predictions. First 5:")
        for pred in predictions['predictions'][:5]:
            print(f"  {pred['symbol']} ({pred['model_type']}): {pred['status']}")
    except requests.HTTPError as e:
        print(f"Failed to list predictions: {e}")
    
    # Logout
    client.logout()
    print("Logged out successfully.")


if __name__ == "__main__":
    main()