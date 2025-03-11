"""Setup configuration for market data package."""
from setuptools import setup, find_namespace_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="market_data",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A robust package for fetching and managing market data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/market_data",
    packages=find_namespace_packages(include=["market_data*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "aiohttp>=3.8.0",
        "aiofiles>=0.8.0",
        "typing_extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.930",
            "pylint>=2.12.0",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.17.0",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/market_data/issues",
        "Documentation": "https://market-data.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/market_data",
    },
    entry_points={
        "console_scripts": [
            "market-data=market_data.cli:main",
        ],
    },
)