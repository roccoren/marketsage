#!/usr/bin/env python3
"""
Development script for the StockScan API.
This script starts the API server in development mode with hot reloading.
"""

import os
import sys
from dotenv import load_dotenv

import uvicorn

# Load environment variables from .env file
load_dotenv()

# Set development environment variables
os.environ["DEBUG"] = "true"


def main():
    """Run the API server in development mode."""
    print("Starting StockScan API server in development mode...")
    print("API documentation available at http://localhost:8000/api/docs")
    print("Hot reloading is enabled - changes to code will restart the server automatically")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        sys.exit(0)