#!/bin/bash
# Script to deploy Dash application as a standalone Python application

set -e

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if config file exists
if [ ! -f "config/default_config.yaml" ]; then
    echo "Error: Configuration file not found. Please ensure config/default_config.yaml exists."
    exit 1
fi

# Set environment variables for local deployment
export CONFIG_FILE="$(pwd)/config/default_config.yaml"

echo "Starting Dash application..."
echo "You can access the dashboard at http://localhost:8050"
echo "Press Ctrl+C to stop the application"

# Run the Dash application
python dash_app.py

# Deactivate virtual environment (this will only run if the application is stopped)
deactivate

exit 0