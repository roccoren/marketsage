#!/bin/bash
set -e

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Define environment location in the project directory to avoid permission issues
ENV_DIR="$(pwd)/.conda_env"

# Create directories if they don't exist
mkdir -p "$ENV_DIR"

# Create conda environment from environment.yml in the local directory
echo "Creating conda environment for Qlib development in: $ENV_DIR"

# First, create a minimal environment with just Python
echo "Creating minimal environment with Python..."
conda create -y -p "$ENV_DIR" python=3.10

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

# Install packages using pip (more reliable than conda for these packages)
echo "Installing packages using pip..."
pip install --upgrade pip

# Core dependencies
echo "Installing core dependencies..."
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tqdm tabulate
pip install lightgbm xgboost python-dotenv

# Development dependencies
echo "Installing development dependencies..."
pip install pytest black isort flake8

# Project-specific dependencies
echo "Installing project-specific dependencies..."
pip install pyqlib yfinance

# Try to install rdagent
echo "Installing rdagent..."
pip install rdagent || echo "Warning: Could not install rdagent, you may need to install it manually"

# Activation instructions
echo ""
echo "To activate the environment, run:"
echo "conda activate $ENV_DIR"
echo "or"
echo "source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_DIR"

# Instructions for Qlib data setup
echo ""
echo "For Qlib development, you may need to download US stock data:"
echo "1. Activate the environment: conda activate $ENV_DIR"
echo "2. Run: python scripts/download_us_data.py"
echo ""
echo "Or use the built-in Qlib data collector:"
echo "python -m qlib.data.data_collector_us"
echo ""
echo "Environment setup complete!"
echo ""
echo "NOTE: If you still encounter permission issues, try running:"
echo "conda clean --all"
echo "conda config --set pkgs_dirs $(pwd)/.conda_pkgs"
echo "And then run this script again." 