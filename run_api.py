#!/usr/bin/env python3
"""
Run script for the StockScan API.
This script starts the API server using uvicorn.
"""

import argparse
import os
import sys
from dotenv import load_dotenv

import uvicorn

# Load environment variables from .env file
load_dotenv()

# Default values
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))
DEFAULT_RELOAD = os.getenv("DEBUG", "false").lower() == "true"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the StockScan API server")
    
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host to bind the server to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind the server to (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=DEFAULT_RELOAD,
        help="Enable auto-reload on code changes (default: {})".format(
            "enabled" if DEFAULT_RELOAD else "disabled"
        ),
    )
    
    return parser.parse_args()


def main():
    """Run the API server."""
    args = parse_args()
    
    print(f"Starting StockScan API server at http://{args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/api/docs")
    print(f"Auto-reload: {'enabled' if args.reload else 'disabled'}")
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        sys.exit(0)