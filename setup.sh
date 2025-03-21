#!/bin/bash

# Exit on error
set -e

echo "Setting up StockScan environment..."

# Parse command-line options
RECREATE_ENV=false
NON_INTERACTIVE=false

# Print help message if needed
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -r  Recreate the conda environment even if it exists"
    echo "  -n  Non-interactive mode (uses defaults, doesn't prompt)"
    echo "  -h, --help  Show this help message"
    exit 0
fi

while getopts "rn" opt; do
  case $opt in
    r) RECREATE_ENV=true ;;
    n) NON_INTERACTIVE=true ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Determine the script's directory (not the current working directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Get environment name from environment.yml
ENV_NAME=$(grep "^name:" environment.yml | head -n 1 | cut -d ":" -f 2 | tr -d " ")
if [ -z "$ENV_NAME" ]; then
    ENV_NAME="stockscan"  # Default fallback name
    echo "Warning: Could not extract environment name from environment.yml, using default: $ENV_NAME"
else
    echo "Using environment name from environment.yml: $ENV_NAME"
fi

# Warn about conda environments.txt
echo "Note: You may see a warning about conda environments during setup - these can be safely ignored."

# Check if the environment already exists
if conda env list | grep -q "\b$ENV_NAME\b"; then
    echo "Conda environment '$ENV_NAME' already exists"
    
    use_existing="y"
    if [ "$RECREATE_ENV" = true ]; then
        use_existing="n"
    elif [ "$NON_INTERACTIVE" = false ]; then
        read -p "Do you want to use the existing environment? (y/n, default: y): " user_input
        if [ -n "$user_input" ]; then
            use_existing=$user_input
        fi
    fi
    
    if [ "$use_existing" != "y" ]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
        echo "Creating new conda environment..."
        conda env create -f environment.yml
    else
        echo "Using existing conda environment..."
    fi
else
    echo "Creating conda environment..."
    conda env create -f environment.yml
fi

# Activate environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install system dependencies for macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing system dependencies for macOS..."
    
    # Install Homebrew if not installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install required system libraries
    brew install hdf5 c-blosc
    
    # Set environment variables for compilation
    export HDF5_DIR="$(brew --prefix hdf5)"
    export BLOSC_DIR="$(brew --prefix c-blosc)"
    export CFLAGS="-I$HDF5_DIR/include -I$BLOSC_DIR/include"
    export LDFLAGS="-L$HDF5_DIR/lib -L$BLOSC_DIR/lib"
    
    # Set additional compilation flags for Apple Silicon if needed
    if [[ "$(uname -m)" == "arm64" ]]; then
        export ARCHFLAGS="-arch arm64"
    fi
fi

# Variable to track if we're using a fresh environment
is_fresh_env=false
if [ "$use_existing" != "y" ] || ! conda env list | grep -q "\b$ENV_NAME\b"; then
    is_fresh_env=true
fi

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install remaining dependencies from requirements.txt
if [ "$is_fresh_env" = true ] || [ "$use_existing" != "y" ]; then
    echo "Installing additional Python dependencies..."
    pip install -r requirements.txt

    # Install pyqlib from PyPI
    echo "Installing pyqlib package..."
    pip uninstall -y qlib  # Remove any conflicting packages
    pip install pyqlib==0.9.6
else
    echo "Skipping dependencies installation as using existing environment..."
    echo "If you need to update dependencies, run: pip install -r requirements.txt"
    
    # Make sure pyqlib is installed
    echo "Ensuring pyqlib package is installed..."
    pip uninstall -y qlib  # Remove any conflicting packages
    pip install pyqlib==0.9.6
fi

# Set up Qlib data directory
echo "Setting up Qlib data directory..."
QLIB_DATA_PATH=~/.qlib/qlib_data
mkdir -p $QLIB_DATA_PATH

# Download Qlib data if not exists
if [ ! -d "$QLIB_DATA_PATH/cn_data" ]; then
    echo "Downloading Qlib data..."
    max_attempts=3
    attempt=1
    success=false
    
    # Create basic directory structure if it doesn't exist
    mkdir -p "$QLIB_DATA_PATH/cn_data/1d"
    
    # Download basic data manually if needed
    while [ $attempt -le $max_attempts ] && [ "$success" = false ]; do
        echo "Attempt $attempt of $max_attempts to download Qlib data..."
        python -c "
try:
    import qlib
    print('Successfully imported qlib from pyqlib package')
    
    # Create basic structure and metadata for Qlib
    import os
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Define the data directory
    provider_uri = '$QLIB_DATA_PATH'
    region = 'cn'
    
    # Create basic calendar file if it doesn't exist
    calendar_path = os.path.join(provider_uri, region + '_data', 'calendars', 'calendar.txt')
    if not os.path.exists(os.path.dirname(calendar_path)):
        os.makedirs(os.path.dirname(calendar_path), exist_ok=True)
    
    if not os.path.exists(calendar_path):
        print('Creating basic calendar file...')
        # Generate a simple calendar (trading days)
        start_date = datetime(2010, 1, 1)
        end_date = datetime.now() + timedelta(days=30)
        all_days = pd.date_range(start=start_date, end=end_date)
        # Filter out weekends (6=Saturday, 0=Sunday)
        trading_days = all_days[~all_days.day_of_week.isin([5, 6])]
        # Format and save
        formatted_days = trading_days.strftime('%Y-%m-%d')
        with open(calendar_path, 'w') as f:
            f.write('\\n'.join(formatted_days))
        print(f'Created calendar file at {calendar_path}')
    
    # Create instruments directory if it doesn't exist
    instruments_dir = os.path.join(provider_uri, region + '_data', 'instruments')
    if not os.path.exists(instruments_dir):
        os.makedirs(instruments_dir, exist_ok=True)
        print(f'Created instruments directory at {instruments_dir}')
    
    # Create a basic instruments file if it doesn't exist
    instruments_path = os.path.join(instruments_dir, 'all.txt')
    if not os.path.exists(instruments_path):
        print('Creating basic instruments file...')
        # Just add a few sample stocks
        sample_stocks = ['000001.SZ', '600000.SH', '601398.SH', '000002.SZ', '600519.SH']
        with open(instruments_path, 'w') as f:
            f.write('\\n'.join(sample_stocks))
        print(f'Created instruments file at {instruments_path}')
    
    print('Basic Qlib data structure has been set up successfully')
    exit(0)  # Success
except ImportError:
    print('Error: Please make sure pyqlib is installed correctly')
    print('Note: pyqlib installs the qlib package which is imported as \"import qlib\"')
except Exception as e:
    print(f'Error setting up Qlib data: {str(e)}')
    exit(1)  # Failure
" && success=true || echo "Qlib data setup failed on attempt $attempt"
        
        if [ "$success" = false ]; then
            if [ $attempt -lt $max_attempts ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
            else
                echo "Maximum attempts reached. Qlib data setup failed."
                if [ "$NON_INTERACTIVE" = false ]; then
                    read -p "Continue anyway? (y/n, default: n): " continue_anyway
                    if [ "$continue_anyway" != "y" ]; then
                        echo "Setup aborted."
                        exit 1
                    fi
                else
                    echo "Setup will continue, but Qlib functionality might be limited."
                fi
            fi
        fi
        
        attempt=$((attempt+1))
    done
else
    echo "Qlib data directory already exists at $QLIB_DATA_PATH/cn_data"
fi

# Export environment variables
echo "Setting up environment variables..."

# Function to update shell config files more safely
update_shell_config() {
    local config_file="$1"
    local start_marker="# BEGIN StockScan environment variables"
    local end_marker="# END StockScan environment variables"
    
    # Only proceed if the file exists
    if [ -f "$config_file" ]; then
        # Remove existing StockScan config block if present
        if grep -q "$start_marker" "$config_file"; then
            echo "Updating existing environment variables in $config_file"
            sed -i.bak "/$start_marker/,/$end_marker/d" "$config_file" && rm -f "${config_file}.bak"
        else
            echo "Adding new environment variables to $config_file"
        fi
        
        # Append new config
        cat >> "$config_file" << EOL
$start_marker
export QLIB_DATA_PATH=$QLIB_DATA_PATH
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR
EOL

        # Add macOS specific variables if needed
        if [[ "$OSTYPE" == "darwin"* ]]; then
            cat >> "$config_file" << EOL
export HDF5_DIR=$HDF5_DIR
export BLOSC_DIR=$BLOSC_DIR
export CFLAGS="$CFLAGS"
export LDFLAGS="$LDFLAGS"
EOL
        fi
        
        # Add end marker
        echo "$end_marker" >> "$config_file"
    fi
}

# Update shell configuration files
update_shell_config ~/.bashrc
update_shell_config ~/.zshrc

# Source the updated environment variables
source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p results/logs
mkdir -p results/optimize
mkdir -p results/models
mkdir -p results/predictions
mkdir -p mlruns

# Initialize git if not already initialized
if [ ! -d .git ]; then
    git init
    echo "mlruns/" >> .gitignore
    echo "data/" >> .gitignore
    echo "results/" >> .gitignore
    echo ".DS_Store" >> .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo ".env" >> .gitignore
    echo ".conda_env/" >> .gitignore
fi

# Make scripts executable
chmod +x run_stockscan.py optimize_qlib.py

# Create empty .env file for environment variables
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# StockScan Environment Variables
QLIB_DATA_PATH=$QLIB_DATA_PATH
MLFLOW_TRACKING_URI=./mlruns
PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR
EOL
fi

# Create empty __init__.py files in src directories
echo "Creating __init__.py files..."
mkdir -p src/data src/models src/strategies src/utils
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/strategies/__init__.py
touch src/utils/__init__.py

echo "Setup completed successfully!"
echo "To get started, run:"
echo "  conda activate \"$ENV_NAME\""
echo "  ./run_stockscan.py --help"
echo "  ./optimize_qlib.py --help"

echo -e "\nTo re-run this setup script with different options:"
echo "  ./setup.sh -r    # Recreate the conda environment"
echo "  ./setup.sh -n    # Non-interactive mode (uses defaults)"
echo "  ./setup.sh -h    # Show help message"

# Print environment check
echo -e "\nChecking environment..."
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "QLIB_DATA_PATH: $QLIB_DATA_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# Test Qlib installation
echo -e "\nTesting Qlib installation..."
python -c "
try:
    import qlib
    print('Successfully imported qlib package (via pyqlib)')
    
    # Check if basic data structure exists
    import os
    
    qlib_data_path = '$QLIB_DATA_PATH'
    required_paths = [
        os.path.join(qlib_data_path, 'cn_data'),
        os.path.join(qlib_data_path, 'cn_data', 'calendars'),
        os.path.join(qlib_data_path, 'cn_data', 'instruments')
    ]
    
    all_exist = True
    for path in required_paths:
        if not os.path.exists(path):
            print(f'Warning: Required path {path} does not exist')
            all_exist = False
    
    if all_exist:
        print('Basic Qlib data structure verification successful')
    else:
        print('Warning: Some Qlib data directories are missing')
    
    # Print version if available
    try:
        print('Qlib version:', qlib.__version__)
    except AttributeError:
        print('Note: Qlib version information not available')
        
except ImportError:
    print('Error: Please make sure pyqlib is installed correctly')
    print('Note: pyqlib installs the qlib package which is imported as \"import qlib\"')
except Exception as e:
    print(f'Error testing Qlib: {str(e)}')
"

echo -e "\nInstalled packages:"
pip list 