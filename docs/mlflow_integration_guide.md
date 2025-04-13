# MLflow Integration Guide for Qlib Workflow

This guide explains how to use MLflow for experiment tracking, workflow management, and result visualization in your Qlib-based quantitative analysis project.

## Table of Contents

1. [Setting Up MLflow](#setting-up-mlflow)
2. [Using the MLflow UI](#using-the-mlflow-ui)
3. [Tracking Experiments](#tracking-experiments)
4. [Workflow Management](#workflow-management)
5. [Result Visualization](#result-visualization)
6. [Integration with Other Tools](#integration-with-other-tools)
7. [Best Practices](#best-practices)

## Setting Up MLflow

### Installation

First, ensure MLflow is installed:

```bash
pip install mlflow>=2.8.0
```

### Configuration

MLflow can be configured to store tracking data locally or on a remote server:

#### Local Storage (Default)

By default, the project is set up to use local storage in the `mlruns` directory.

#### Remote Tracking Server

For team collaboration, you can set up a remote tracking server:

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

Then configure your application to use this server:

```bash
# In your terminal before running the application
export MLFLOW_TRACKING_URI=http://localhost:5000

# Or specify via command line
python main.py --mlflow-tracking-uri=http://localhost:5000
```

## Using the MLflow UI

MLflow provides a web-based UI for visualizing experiments and results.

### Starting the UI

```bash
# If using local storage
mlflow ui

# If using a remote tracking server, the UI is already available at:
# http://localhost:5000
```

### UI Features

- **Experiments**: View all experiments and runs
- **Run Comparison**: Compare metrics across different runs
- **Artifact Viewing**: View logged artifacts like charts and data descriptions
- **Parameter Analysis**: Analyze how different parameters affect results

## Tracking Experiments

The project is set up to track the following:

### Parameters

- Stock symbols
- Date ranges
- Model configurations
- Forecasting and backtesting settings

### Metrics

- Backtesting returns
- Forecast accuracy metrics (MAE, RMSE, MAPE, R²)
- Risk metrics (VaR, drawdown)

### Artifacts

- Data descriptions
- Price charts
- Forecast visualizations
- Model files

### Custom Tracking

You can add custom tracking in your modules:

```python
from modules.mlflow_integration import MLflowManager

# Initialize MLflow manager
mlflow_manager = MLflowManager()
mlflow_manager.start_run("custom_analysis")

# Log custom parameters
mlflow_manager.log_parameters({"param_name": param_value})

# Log custom metrics
mlflow_manager.log_metrics({"metric_name": metric_value})

# Log custom artifacts
mlflow_manager.log_artifact("path/to/artifact.png")

# End the run
mlflow_manager.end_run()
```

## Workflow Management

### Basic Workflow with MLflow

MLflow can be used to track the entire quantitative analysis workflow:

1. **Data Acquisition**: Track data sources and preprocessing steps
2. **Forecasting**: Track model parameters and performance
3. **Backtesting**: Track strategy performance and risk metrics
4. **Analysis**: Track insights and decisions

### Run Management

```bash
# Run the full workflow
python main.py --symbols=AAPL,MSFT,GOOG --start-date=2020-01-01

# Run specific parts of the workflow
python main.py --symbols=AAPL --skip-forecasting
python main.py --symbols=AAPL --skip-backtesting
```

## Result Visualization

MLflow automatically visualizes:

1. **Metric Comparisons**: Compare metrics across different runs
2. **Parameter Effects**: Analyze how parameters affect results
3. **Run History**: Track performance over time

### Custom Visualizations

You can create and log custom visualizations:

```python
import matplotlib.pyplot as plt
from modules.mlflow_integration import MLflowManager

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(data['Close'])
plt.title('Stock Close Prices')

# Log to MLflow
mlflow_manager = MLflowManager()
mlflow_manager.start_run()
mlflow_manager.log_figure(plt.gcf(), "custom_charts")
mlflow_manager.end_run()
```

## Integration with Other Tools

### Dagster for Advanced Workflow Management

[Dagster](https://dagster.io/) is a data orchestrator that can be used for more complex workflows:

```bash
pip install dagster dagster-mlflow
```

Example Dagster integration:

```python
from dagster import job, op, Out, In
from dagster_mlflow import mlflow_tracking

@op(out=Out(dict))
def fetch_data(context):
    # Your data acquisition code
    return {"data": data}

@op(ins={"data_dict": In(dict)}, out=Out(dict))
def forecast(context, data_dict):
    # Your forecasting code
    return {"data": data_dict["data"], "forecasts": forecasts}

@op(ins={"forecast_dict": In(dict)})
@mlflow_tracking
def backtest(context, forecast_dict):
    # Your backtesting code with MLflow tracking
    return results

@job
def quant_workflow():
    forecast_dict = forecast(fetch_data())
    backtest(forecast_dict)

# Execute the workflow
quant_workflow.execute_in_process()
```

### Airflow for Scheduling

[Apache Airflow](https://airflow.apache.org/) can be used for scheduling and monitoring workflows:

```bash
pip install apache-airflow apache-airflow-providers-mlflow
```

Example Airflow DAG:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'quant',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'quant_workflow',
    default_args=default_args,
    description='Quantitative analysis workflow',
    schedule_interval=timedelta(days=1),
)

def run_data_acquisition(**kwargs):
    # Your data acquisition code
    pass

def run_forecasting(**kwargs):
    # Your forecasting code
    pass

def run_backtesting(**kwargs):
    # Your backtesting code
    pass

data_acquisition_task = PythonOperator(
    task_id='data_acquisition',
    python_callable=run_data_acquisition,
    dag=dag,
)

forecasting_task = PythonOperator(
    task_id='forecasting',
    python_callable=run_forecasting,
    dag=dag,
)

backtesting_task = PythonOperator(
    task_id='backtesting',
    python_callable=run_backtesting,
    dag=dag,
)

data_acquisition_task >> forecasting_task >> backtesting_task
```

## Best Practices

1. **Consistent Experiment Naming**: Use meaningful names for experiments and runs
2. **Parameter Tracking**: Track all relevant parameters for reproducibility
3. **Metric Standardization**: Use consistent metrics across experiments
4. **Regular UI Review**: Regularly review the MLflow UI to analyze results
5. **Artifact Organization**: Organize artifacts logically for easy retrieval
6. **Version Control**: Track code versions alongside MLflow experiments
7. **Documentation**: Document experiment purpose and findings

## Example Usage

### Basic Usage

```bash
# Run with default settings (local MLflow storage)
python main.py --symbols=AAPL

# Run with custom MLflow settings
python main.py --symbols=AAPL,MSFT --mlflow-tracking-uri=http://localhost:5000 --experiment-name=production_run
```

### Advanced Usage

```bash
# Run with specific date range and skip certain steps
python main.py --symbols=stock_symbols.txt --start-date=2021-01-01 --skip-forecasting

# Run with custom parameters and view results in MLflow UI
python main.py --symbols=AAPL --start-date=2020-01-01
mlflow ui  # Open http://localhost:5000 in your browser
```

## Conclusion

MLflow provides a comprehensive solution for experiment tracking, workflow management, and result visualization in your Qlib-based quantitative analysis project. By following this guide, you can effectively manage your experiments, compare results, and make data-driven decisions.

For more advanced workflow management, consider integrating with tools like Dagster or Airflow as described in this guide.