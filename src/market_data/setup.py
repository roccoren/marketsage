"""Setup script for market_data package."""
from setuptools import setup, find_packages

setup(
    name="market_data",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "yfinance>=0.2.36",
        "pandas>=2.0.0",
        "aiofiles>=23.1.0",
    ],
    python_requires=">=3.7",
    author="Roo",
    description="A flexible market data fetching system",
    keywords="finance,market-data,stocks",
)