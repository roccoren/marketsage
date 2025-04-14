#!/usr/bin/env python3
"""
Streamlit dashboard for stock analysis and signal visualization.
This app connects to MLflow to display trading signals, performance metrics,
and historical analysis.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from modules.mlflow_integration import MLflowManager

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for loading data
def get_mlflow_runs(experiment_name="stock_analysis", max_results=100):
    """Get MLflow runs for the specified experiment."""
    try:
        # Get experiment by name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            st.error(f"Experiment '{experiment_name}' not found.")
            return []
        
        # Get runs for the experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results
        )
        return runs
    except Exception as e:
        st.error(f"Error retrieving MLflow runs: {e}")
        return []

def load_signals_from_mlflow(run_id):
    """Load trading signals from MLflow artifacts."""
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        signal_files = [a.path for a in artifacts if a.path.startswith('signals_')]
        if not signal_files:
            return None
        
        # Get the latest signals file
        latest_signal_file = sorted(signal_files)[-1]
        
        # Download the artifact to a temporary location
        local_path = client.download_artifacts(run_id, latest_signal_file, "temp")
        signals_df = pd.read_csv(local_path)
        
        # Clean up
        os.remove(local_path)
        
        return signals_df
    except Exception as e:
        st.error(f"Error loading signals from MLflow: {e}")
        return None

def load_analysis_from_mlflow(run_id):
    """Load stock analysis results from MLflow artifacts."""
    try:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        analysis_files = [a.path for a in artifacts if a.path.startswith('analysis_')]
        if not analysis_files:
            return None
        
        # Get the latest analysis file
        latest_analysis_file = sorted(analysis_files)[-1]
        
        # Download the artifact to a temporary location
        local_path = client.download_artifacts(run_id, latest_analysis_file, "temp")
        analysis_df = pd.read_csv(local_path)
        
        # Clean up
        os.remove(local_path)
        
        return analysis_df
    except Exception as e:
        st.error(f"Error loading analysis from MLflow: {e}")
        return None

def load_latest_csv_files():
    """Load the latest CSV files from the results directory as fallback."""
    try:
        results_dir = Path("data/results")
        if not results_dir.exists():
            return None, None
        
        # Find the latest signal predictions file
        signal_files = list(results_dir.glob("signal_predictions_*.csv"))
        if signal_files:
            latest_signal_file = max(signal_files, key=lambda x: x.stat().st_mtime)
            signals_df = pd.read_csv(latest_signal_file)
        else:
            signals_df = None
            
        # Find the latest stock analysis file
        analysis_files = list(results_dir.glob("stock_analysis_*.csv"))
        if analysis_files:
            latest_analysis_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
            analysis_df = pd.read_csv(latest_analysis_file)
        else:
            analysis_df = None
            
        return signals_df, analysis_df
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None

# Main app
def main():
    """Main Streamlit app function."""
    # Sidebar
    st.sidebar.title("Stock Analysis Dashboard")
    
    # MLflow connection settings
    st.sidebar.header("MLflow Connection")
    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        value="file:mlruns"
    )
    
    # Set MLflow tracking URI
    try:
        mlflow.set_tracking_uri(tracking_uri)
        st.sidebar.success(f"Connected to MLflow at: {tracking_uri}")
    except Exception as e:
        st.sidebar.error(f"Failed to connect to MLflow: {e}")
    
    # Experiment selection
    experiment_name = st.sidebar.text_input(
        "Experiment Name",
        value="stock_analysis"
    )
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["MLflow", "CSV Files"]
    )
    
    # Main content
    st.title("Stock Analysis Dashboard")
    
    # Load data based on selected source
    if data_source == "MLflow":
        runs = get_mlflow_runs(experiment_name)
        
        if runs.empty:
            st.warning(f"No runs found for experiment '{experiment_name}'.")
            signals_df, analysis_df = load_latest_csv_files()
            if signals_df is None and analysis_df is None:
                st.error("No data available. Please run the analysis first.")
                return
        else:
            # Display run selection
            st.subheader("Available Runs")
            run_options = {}
            for _, run in runs.iterrows():
                # Handle different timestamp formats
                try:
                    if isinstance(run.start_time, (int, float)):
                        # Convert milliseconds to datetime
                        timestamp = pd.to_datetime(run.start_time, unit='ms')
                    else:
                        # Already a timestamp
                        timestamp = run.start_time
                    
                    run_options[f"{run.run_id[:8]} - {timestamp}"] = run.run_id
                except Exception as e:
                    # Fallback if timestamp conversion fails
                    run_options[f"{run.run_id[:8]}"] = run.run_id
            selected_run = st.selectbox(
                "Select a run",
                options=list(run_options.keys())
            )
            run_id = run_options[selected_run]
            
            # Load data from selected run
            signals_df = load_signals_from_mlflow(run_id)
            analysis_df = load_analysis_from_mlflow(run_id)
    else:
        # Load from CSV files
        signals_df, analysis_df = load_latest_csv_files()
        if signals_df is None and analysis_df is None:
            st.error("No data available. Please run the analysis first.")
            return
    
    # Display trading signals
    if signals_df is not None:
        st.header("Trading Signals")
        
        # Signal distribution chart
        st.subheader("Signal Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        signal_counts = signals_df['Signal'].value_counts()
        signal_counts.plot(kind='bar', ax=ax)
        ax.set_title('Trading Signal Distribution')
        ax.set_xlabel('Signal Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
        # Signal tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Buy Signals")
            buy_signals = signals_df[signals_df['Signal'] == 'BUY'].sort_values('Confidence', ascending=False)
            if not buy_signals.empty:
                st.dataframe(buy_signals[['Symbol', 'Last Close', 'Predicted Price', 'Expected Return', 'Confidence']])
            else:
                st.info("No buy signals available.")
        
        with col2:
            st.subheader("Sell Signals")
            sell_signals = signals_df[signals_df['Signal'] == 'SELL'].sort_values('Confidence', ascending=False)
            if not sell_signals.empty:
                st.dataframe(sell_signals[['Symbol', 'Last Close', 'Predicted Price', 'Expected Return', 'Confidence']])
            else:
                st.info("No sell signals available.")
    
    # Display performance metrics
    if analysis_df is not None:
        st.header("Performance Metrics")
        
        # Performance chart
        st.subheader("Key Performance Indicators")
        fig, ax = plt.subplots(figsize=(12, 6))
        performance_metrics = ['Base Sharpe', 'Opt Sharpe', 'Win Rate']
        analysis_df[performance_metrics].plot(kind='bar', ax=ax)
        ax.set_title('Performance Metrics by Stock')
        ax.set_xlabel('Stock')
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(analysis_df)))
        ax.set_xticklabels(analysis_df['Symbol'], rotation=45)
        st.pyplot(fig)
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        st.dataframe(analysis_df)

if __name__ == "__main__":
    main()