#!/bin/bash
set -e

echo "Setting up a Python virtual environment with venv instead of conda"
echo "This is a fallback in case conda has permission issues"

# Define the environment directory
VENV_DIR="$(pwd)/.venv"

# Create and activate a virtual environment
echo "Creating virtual environment in $VENV_DIR"
python -m venv "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages from your dependencies
echo "Installing packages..."

# Core dependencies
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tqdm tabulate
pip install lightgbm xgboost python-dotenv

# Development dependencies
pip install pytest black isort flake8

# Project-specific dependencies
pip install pyqlib yfinance

# Try to install rdagent 
pip install rdagent || echo "Warning: Could not install rdagent, you may need to install it manually"

echo ""
echo "Virtual environment setup complete at $VENV_DIR"
echo ""
echo "To activate the environment, run:"
echo "source $VENV_DIR/bin/activate"
echo ""
echo "For Qlib development, you may need to download US stock data:"
echo "1. Activate the environment: source $VENV_DIR/bin/activate"
echo "2. Run: python scripts/download_us_data.py"
echo ""
echo "Or use the built-in Qlib data collector:"
echo "python -m qlib.data.data_collector_us" 