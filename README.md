# StockScan - Quantitative Trading with Qlib and MLflow

A quantitative trading analytics system that integrates Qlib with MLflow for tracking experiments and results.

## Quick Setup

1. Install Miniconda or Anaconda if you haven't already:
   ```bash
   # For macOS
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda
   
   # For Linux
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
   bash ~/miniconda.sh -b -p $HOME/miniconda
   ```

2. Clone and set up the project:
   ```bash
   git clone https://github.com/yourusername/stockscan.git
   cd stockscan
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the environment:
   ```bash
   conda activate stockscan
   ```

4. Run a quick test:
   ```bash
   ./run_stockscan.py --quick
   ```

## Features

- Fetch historical stock data up to today
- Analyze data with Qlib's built-in strategies
- Make predictions using Qlib's built-in models
- Fine-tune models for improved performance
- Track all experiments with MLflow
- **New:** Advanced model and strategy optimization
  - Model hyperparameter tuning
  - Feature importance analysis and selection
  - Model ensemble methods
  - Strategy parameter optimization

## Project Structure

The codebase has been organized as follows:

- `src/` - Core implementation
  - `data/` - Data handling modules
  - `models/` - Prediction models
  - `strategies/` - Trading strategies
  - `utils/` - Utility functions
    - `optimize.py` - New optimization module
  - `run_workflow.py` - Workflow implementation
  - `config.py` - Configuration parameters

- `run_stockscan.py` - Main CLI entry point
- `test_workflow.py` - Testing framework 
- `fix_qlib_imports.py` - Utility to fix Qlib import issues
- `check_results.py` - Results validation tool
- `optimize_qlib.py` - New command-line tool for optimization

## Usage Examples

### Basic Usage

```bash
# Run the complete workflow for Chinese market
./run_stockscan.py --region cn --market csi300

# Run a quick test with minimal data
./run_stockscan.py --quick

# Run the full workflow with all features
./run_stockscan.py --full
```

### Model Optimization

```bash
# Using run_stockscan.py
./run_stockscan.py --optimize --optimize-mode model \
    --model LGBM \
    --n-trials 50 \
    --opt-metric IC

# Using standalone optimize_qlib.py
./optimize_qlib.py optimize-model \
    --model LGBM \
    --n-trials 50 \
    --metric IC \
    --train-start 2015-01-01 \
    --train-end 2020-12-31 \
    --valid-start 2021-01-01 \
    --valid-end 2021-06-30 \
    --test-start 2021-07-01 \
    --test-end 2022-12-31
```

### Strategy Optimization

```bash
# Optimize TopkDropoutStrategy
./optimize_qlib.py optimize-strategy \
    --prediction-path results/predictions/lgbm_pred.pkl \
    --strategy TopkDropoutStrategy \
    --n-trials 30 \
    --metric sharpe
```

### Feature Importance Analysis

```bash
# Analyze feature importance
./optimize_qlib.py feature-importance \
    --model-path results/models/lgbm_model \
    --top-k 50
```

### Model Ensemble

```bash
# Create model ensemble
./optimize_qlib.py model-ensemble \
    --model-paths results/models/model1 results/models/model2 \
    --weights 0.6 0.4
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: If you see errors about missing modules, try:
   ```bash
   conda activate stockscan
   pip install -r requirements.txt
   ```

2. **Qlib Data Error**: If you have issues with Qlib data:
   ```bash
   export QLIB_DATA_PATH=~/.qlib/qlib_data
   mkdir -p $QLIB_DATA_PATH
   ```

3. **Permission Issues**: If you can't execute the scripts:
   ```bash
   chmod +x run_stockscan.py optimize_qlib.py
   ```

### Getting Help

If you encounter any issues:
1. Check the logs in `results/logs/`
2. Look for error messages in MLflow UI (`mlflow ui`)
3. Open an issue on GitHub

## License

[MIT License](LICENSE)

## MLflow Integration

StockScan provides enhanced MLflow integration for experiment tracking:

```bash
# Run the MLflow demo
python mlflow_demo.py

# View experiments and results
mlflow ui --backend-store-uri ./mlruns
```

For detailed information about our MLflow integration, see [MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md). 