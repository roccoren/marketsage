# Enhanced MLflow Integration for StockScan

This document explains how to use the enhanced MLflow integration capabilities for the StockScan project.

## Overview

The enhanced MLflow integration provides the following capabilities:

1. **Hierarchical Experiment Tracking**: Parent and nested runs for organizing complex workflows
2. **Rich Parameter Logging**: Support for complex parameter types and hierarchical configurations
3. **Dataset Management**: Log datasets with metadata, visualizations, and samples
4. **Advanced Model Tracking**: Log models with input/output signatures and conda environments
5. **Visualization Tools**: Automated visualization creation for predictions and backtest results
6. **Run Comparison**: Find best runs and compare multiple runs across metrics
7. **Performance Analytics**: Track and visualize backtest metrics with interactive charts

## Quick Start

To run a demonstration of all MLflow capabilities:

```bash
# Run the demonstration with default settings
python mlflow_demo.py

# Customize the demonstration
python mlflow_demo.py --experiment-name "my_experiment" --runs 5 --stocks 50 --days 200
```

## Usage in Your Workflows

### Basic Setup

```python
from src.utils import setup_mlflow

# Initialize with enhanced functionality
experiment_id, run_id = setup_mlflow()
```

### Logging Complex Parameters

```python
from src.utils import log_extended_params

# Log complex configurations with nested structures
config = {
    "qlib_config": {...},
    "model_config": {...},
    "strategy_params": {
        "topk": 10,
        "complex_setting": {
            "risk_limits": {...}
        }
    }
}

log_extended_params(config, prefix="workflow")
```

### Logging Datasets with Metadata

```python
from src.utils import log_dataset

# Log dataset with rich metadata
log_dataset(
    data_df, 
    "training_data", 
    description="Training data for stock prediction model",
    metadata={
        "source": "yahoo_finance",
        "tickers": ["AAPL", "MSFT", ...],
        "date_range": "2020-01-01 to 2022-12-31"
    }
)
```

### Working with Nested Runs

```python
from src.utils import create_nested_run
import mlflow

# Create a parent run
with mlflow.start_run() as parent_run:
    parent_id = parent_run.info.run_id
    
    # Create nested runs for different models
    for model_type in ["Linear", "LGBM", "DEnsemble"]:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_type", model_type)
            # Train model, log results, etc.
```

### Handling NumPy Data Types

MLflow doesn't natively support NumPy data types for parameter and metric logging. Our integration includes a custom JSON encoder that automatically handles NumPy types:

```python
# Import the NumPy-compatible JSON encoder
from src.utils.mlflow_utils import NumpyEncoder
import json
import numpy as np

# Example of using the encoder
data = {
    "int_value": np.int64(42),
    "float_value": np.float32(3.14),
    "array": np.array([1, 2, 3]),
    "bool_value": np.bool_(True)
}

# Serialize with the custom encoder
json_string = json.dumps(data, cls=NumpyEncoder)

# When logging metrics with NumPy types, convert to Python native types
metrics = {
    "accuracy": np.float32(0.95),
    "error_rate": np.float64(0.05)
}

# Convert to native Python types
clean_metrics = {k: float(v) if isinstance(v, np.number) else v for k, v in metrics.items()}
mlflow.log_metrics(clean_metrics)
```

### Logging Models with Signatures

```python
from src.utils import log_model_with_signature

# Log model with input/output signatures
log_model_with_signature(
    model,
    "my_model",
    sample_input=X_test,
    sample_output=y_pred,
    conda_env={
        "channels": ["defaults", "conda-forge"],
        "dependencies": ["python=3.8", "scikit-learn=1.0"]
    }
)
```

### Logging Predictions with Visualizations

```python
from src.utils import log_prediction_results

# Log predictions with auto-generated visualizations
log_prediction_results(predictions_df, "model_predictions")
```

### Logging Backtest Results

```python
from src.utils import log_backtest_metrics

# Log backtest results with rich visualizations
log_backtest_metrics(
    metrics={"sharpe": 1.5, "total_return": 0.25, "max_drawdown": -0.1},
    portfolio_values=portfolio_value_series,
    positions=positions_df,
    trades=trades_df,
    name="my_strategy"
)
```

### Finding Best Runs

```python
from src.utils import find_best_runs

# Find top runs by a specific metric
best_runs = find_best_runs(
    experiment_name="my_experiment",
    metric_name="sharpe",
    max_results=5,
    higher_is_better=True
)

# Process results
for run in best_runs:
    print(f"Run {run['run_name']} achieved {run['sharpe']} Sharpe ratio")
```

### Comparing Multiple Runs

```python
from src.utils import compare_runs, create_comparison_chart

# Compare runs across multiple metrics
comparison_df = compare_runs(
    run_ids=["run-id-1", "run-id-2", "run-id-3"],
    metric_names=["sharpe", "total_return", "max_drawdown"]
)

# Create visual comparison
chart_path = create_comparison_chart(
    comparison_df,
    metric_name="sharpe",
    title="Sharpe Ratio Comparison Across Models"
)
```

## MLflow UI

To view all tracked experiments and runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then navigate to http://127.0.0.1:5000 in your browser.

## Key Files

- `src/utils/mlflow_utils.py`: Enhanced MLflow utility functions
- `mlflow_demo.py`: Demonstration of all MLflow capabilities
- `src/models/predict.py`: Integration of MLflow with prediction workflows

## Best Practices

1. **Use Parent-Child Hierarchy**: Structure complex workflows with parent and nested runs
2. **Log Everything**: Parameters, datasets, models, metrics, and artifacts
3. **Create Visualizations**: Use the visualization utilities for better insights
4. **Compare Runs**: Regularly compare different runs to identify best configurations
5. **Tag Runs**: Add meaningful tags to make filtering and searching easier
6. **Convert NumPy Types**: Always convert NumPy types to native Python types before logging metrics and parameters
7. **Proper Run Management**: End parent runs before creating children to avoid parameter conflicts
8. **Use Context Managers**: Prefer using `with mlflow.start_run():` to ensure runs are properly ended
9. **Exception Handling**: Wrap MLflow operations in try/finally blocks to ensure runs are properly closed

## Troubleshooting

If you encounter issues with MLflow:

1. **JSON Serialization Errors**: If you see errors like `TypeError: Object of type int64 is not JSON serializable`, use the provided `NumpyEncoder` class:
   ```python
   from src.utils.mlflow_utils import NumpyEncoder
   import json
   
   # Example usage
   with open('data.json', 'w') as f:
       json.dump(numpy_data, f, cls=NumpyEncoder)
   ```

2. **Parameter/Metric Type Errors**: Convert NumPy types to native Python types before logging:
   ```python
   # Convert metrics with NumPy types
   clean_metrics = {k: float(v) if isinstance(v, np.number) else v 
                    for k, v in metrics.items()}
   mlflow.log_metrics(clean_metrics)
   ```

3. **Nested Run Issues**: If you encounter errors like `Changing param values is not allowed`, use separate runs with linking tags instead of true nested runs:
   ```python
   # Initialize experiment
   experiment_id, parent_run_id = setup_mlflow()
   mlflow.end_run()  # End parent run
   
   # Create independent runs
   with mlflow.start_run(experiment_id=experiment_id, run_name="child_run") as run:
       mlflow.set_tag("parent_run_id", parent_run_id)  # Link to parent
       # Log parameters, metrics, etc.
   ```

4. **Check if the MLflow server is running**
5. **Verify that the tracking URI is correctly configured**
6. **Ensure artifacts are being saved to the correct location**
7. **Check file permissions for the MLflow storage directories** 