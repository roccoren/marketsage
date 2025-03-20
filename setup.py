#!/usr/bin/env python
from setuptools import setup, find_packages

# Explicitly exclude the data directory from package discovery
setup(
    name="stockscan",
    version="0.1.0",
    packages=find_packages(exclude=["data", "data.*"]),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "yfinance>=0.1.70",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "tabulate>=0.8.9",
        "pyqlib>=0.8.7",
        "rdagent>=0.0.1",
        "python-dotenv>=1.0.0",
        "lightgbm>=3.3.5",
        "xgboost>=1.7.0"
    ],
) 