# Stock Analysis Visualization Guide

This guide explains how to use the visualization tools added to the stock analysis system.

## Overview

The visualization system consists of two main components:

1. **MLflow Tracking**: Records experiment results, metrics, and artifacts
2. **Streamlit Dashboard**: Provides an interactive web interface to view results

## MLflow Tracking

### Running Analysis with MLflow

The main script now integrates with MLflow to track all analysis results. To run an analysis with MLflow tracking:

```bash
python main.py --symbols-file=sample_stocks.txt --experiment-name=my_analysis
```

Additional MLflow options:

- `--mlflow-tracking-uri`: Specify a custom MLflow tracking server (default: local mlruns directory)
- `--experiment-name`: Name for the MLflow experiment (default: stock_analysis)
- `--disable-mlflow`: Disable MLflow tracking if needed

### Viewing MLflow Results

To view the MLflow dashboard:

```bash
mlflow ui
```

Then open your browser to http://localhost:5000

In the MLflow UI, you can:
- Browse experiments and runs
- Compare metrics across different runs
- View signal distribution charts
- Analyze performance metrics
- Download artifacts

## Streamlit Dashboard

### Running the Dashboard

To launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

### Dashboard Features

The Streamlit dashboard provides:

1. **Trading Signals View**:
   - Signal distribution chart
   - Buy signals table with confidence levels
   - Sell signals table with confidence levels

2. **Performance Metrics View**:
   - Key performance indicators chart
   - Detailed metrics table

3. **Data Source Options**:
   - Load from MLflow experiments (recommended)
   - Load from local CSV files (fallback)

### Connecting to MLflow

The dashboard can connect to MLflow to load experiment data:

1. In the sidebar, set the MLflow Tracking URI (default: file:mlruns)
2. Select the experiment name (default: stock_analysis)
3. Choose a specific run to analyze

## Workflow Example

A typical workflow would be:

1. Run analysis with MLflow tracking:
   ```bash
   python main.py --symbols-file=sample_stocks.txt --experiment-name=daily_analysis
   ```

2. View results in the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

3. In the dashboard:
   - Select "MLflow" as the data source
   - Set experiment name to "daily_analysis"
   - Choose the latest run
   - Explore signals and metrics

## Customization

### Adding New Visualizations

To add new visualizations to the dashboard, modify the `app.py` file:

1. Create a new section using `st.header()` and `st.subheader()`
2. Load the required data from MLflow or CSV
3. Use Streamlit's plotting capabilities to create visualizations

### Tracking Additional Metrics

To track additional metrics in MLflow:

1. In `main.py`, add your metrics to the parameters or metrics dictionaries
2. Use the `mlflow_manager.log_metrics()` method to log numeric metrics
3. Use the `mlflow_manager.log_parameters()` method to log parameters
4. For artifacts, use the `mlflow_manager.log_artifact()` method

## Troubleshooting

### MLflow Connection Issues

If you encounter issues connecting to MLflow:

1. Verify the tracking URI is correct
2. Ensure the mlruns directory exists
3. Check permissions on the mlruns directory

### Missing Data in Dashboard

If data is not appearing in the dashboard:

1. Verify that you've run the analysis with MLflow tracking enabled
2. Check that the experiment name matches between the analysis run and the dashboard
3. Try using the CSV fallback option to see if data is available locally