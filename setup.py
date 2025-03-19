#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name="stockscan",
    version="0.1.0",
    author="StockScan Team",
    author_email="yourname@example.com",
    description="A comprehensive framework for stock analysis and backtesting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stockscan",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "stockscan": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "tabulate>=0.8.9",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
            "pytest-cov>=2.12.1",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.2",
            "sphinx-rtd-theme>=0.5.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "stockscan=stockscan.main:main",
        ],
    },
)