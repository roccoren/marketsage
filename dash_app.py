#!/usr/bin/env python3
"""
Interactive Dash application for stock data visualization and model management.
Integrates with InfluxDB for data storage and MLflow for experiment tracking.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import traceback

# Import storage backend and configuration
from modules.storage import StorageManager
from modules.connection_config import get_influxdb_config, get_mlflow_config
from modules.config_manager import load_config_file
from modules.logging_manager import get_logger
import mlflow
import os

# Initialize logger
logger = get_logger("dash_app")

# Load configuration file
config_file = os.path.join(os.path.dirname(__file__), "config/default_config.yaml")
load_config_file(config_file)
logger.info(f"Loaded configuration from {config_file}")

# Get connection configurations
influxdb_config = get_influxdb_config()
mlflow_config = get_mlflow_config()
logger.info(f"Using InfluxDB URL: {influxdb_config['url']}")
logger.info(f"Using MLflow tracking URI: {mlflow_config['tracking_uri']}")

# Initialize storage manager with InfluxDB backend
try:
    logger.info("Initializing storage manager with InfluxDB backend")
    storage_manager = StorageManager(
        backend_type="influxdb",
        url=influxdb_config["url"],
        token=influxdb_config["token"],
        org=influxdb_config["org"],
        bucket=influxdb_config["bucket"]
    )
    logger.info("Storage manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize storage manager: {e}")
    logger.debug(f"Detailed error: {traceback.format_exc()}")
    raise

# Set MLflow tracking URI
try:
    logger.info("Setting MLflow tracking URI")
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    logger.info("MLflow tracking URI set successfully")
except Exception as e:
    logger.error(f"Failed to set MLflow tracking URI: {e}")
    logger.debug(f"Detailed error: {traceback.format_exc()}")

# Initialize Dash app with Bootstrap styling
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

server = app.server
app.title = "StockScan Dashboard"

# Helper functions for data processing and visualization
def get_stock_symbols():
    """Get list of available stock symbols."""
    try:
        logger.debug("Getting list of available stock symbols")
        symbols = storage_manager.list_symbols()
        logger.debug(f"Found {len(symbols)} stock symbols: {symbols}")
        return symbols
    except Exception as e:
        logger.error(f"Failed to get stock symbols: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return []

def get_stock_data(symbol, start_date=None, end_date=None):
    """Get stock data for a symbol with optional date filtering."""
    try:
        if not start_date:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.debug(f"Getting stock data for {symbol} from {start_date} to {end_date}")
        data = storage_manager.load_data(symbol, start_date, end_date)
        logger.debug(f"Retrieved {len(data)} data points for {symbol}")
        return data
    except Exception as e:
        logger.error(f"Failed to get stock data for {symbol}: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return pd.DataFrame()

def get_model_list(symbol):
    """Get list of available models for a symbol."""
    try:
        logger.debug(f"Getting list of models for {symbol}")
        models = storage_manager.list_models(symbol)
        logger.debug(f"Found {len(models)} models for {symbol}: {models}")
        return models
    except Exception as e:
        logger.error(f"Failed to get model list for {symbol}: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return []

def get_model_versions(symbol, model_name):
    """Get list of versions for a model."""
    try:
        from modules.storage_backends.influxdb_storage import InfluxDBStorage
        logger.debug(f"Getting versions for model {model_name} of {symbol}")
        if isinstance(storage_manager.backend, InfluxDBStorage):
            versions = storage_manager.backend.list_model_versions(symbol, model_name)
            logger.debug(f"Found {len(versions)} versions for {model_name} of {symbol}")
            return versions
        logger.warning(f"Storage backend is not InfluxDBStorage, cannot get model versions")
        return []
    except Exception as e:
        logger.error(f"Failed to get model versions for {symbol}/{model_name}: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return []

def create_candlestick_chart(df, symbol):
    """Create a candlestick chart for stock data."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.7, 0.3],
                        subplot_titles=(f"{symbol} Stock Price", "Volume"))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name="Volume",
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Add moving averages if we have enough data
    if len(df) > 20:
        df['MA20'] = df['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA20'],
                name="20-day MA",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if len(df) > 50:
        df['MA50'] = df['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA50'],
                name="50-day MA",
                line=dict(color='green', width=1)
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_model_comparison_chart(symbol, model_name, versions):
    """Create a chart comparing model performance across versions."""
    if not versions:
        return go.Figure().update_layout(
            title="No model versions available",
            template="plotly_dark",
            height=400
        )
    
    # Extract metrics from versions
    metrics = []
    for version in versions:
        version_id = version.get("version")
        timestamp = version.get("timestamp")
        
        # Extract performance metrics
        metrics_data = {
            "version": version_id,
            "timestamp": timestamp,
            "mae": version.get("mae", None),
            "mse": version.get("mse", None),
            "rmse": version.get("rmse", None),
            "mape": version.get("mape", None),
            "r2": version.get("r2", None)
        }
        metrics.append(metrics_data)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    if metrics_df.empty:
        return go.Figure().update_layout(
            title="No metrics available for comparison",
            template="plotly_dark",
            height=400
        )
    
    # Create comparison chart
    fig = go.Figure()
    
    # Add metrics that exist
    for metric in ["mae", "mse", "rmse", "mape"]:
        if metric in metrics_df.columns and not metrics_df[metric].isna().all():
            fig.add_trace(
                go.Bar(
                    x=metrics_df["version"],
                    y=metrics_df[metric],
                    name=metric.upper()
                )
            )
    
    # Add R2 on secondary axis if available
    if "r2" in metrics_df.columns and not metrics_df["r2"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=metrics_df["version"],
                y=metrics_df["r2"],
                name="R²",
                mode="lines+markers",
                yaxis="y2"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Model Performance Comparison: {model_name}",
        xaxis_title="Version",
        yaxis_title="Error Metrics",
        template="plotly_dark",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis2=dict(
            title="R²",
            overlaying="y",
            side="right"
        )
    )
    
    return fig

def create_training_history_chart(symbol, model_name, version):
    """Create a chart showing model training history."""
    try:
        from modules.storage_backends.influxdb_storage import InfluxDBStorage
        if isinstance(storage_manager.backend, InfluxDBStorage):
            history = storage_manager.backend.get_training_history(symbol, model_name, version)
            
            if not history:
                return go.Figure().update_layout(
                    title="No training history available",
                    template="plotly_dark",
                    height=400
                )
            
            # Create figure
            fig = go.Figure()
            
            # Add loss curve
            if "loss" in history and len(history["loss"]) > 0:
                epochs = list(range(1, len(history["loss"]) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history["loss"],
                        name="Training Loss",
                        mode="lines"
                    )
                )
            
            # Add validation loss curve
            if "val_loss" in history and len(history["val_loss"]) > 0:
                epochs = list(range(1, len(history["val_loss"]) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=history["val_loss"],
                        name="Validation Loss",
                        mode="lines"
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f"Training History: {model_name} (Version: {version})",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_dark",
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
    except Exception as e:
        logger.error(f"Error creating training history chart: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
    
    return go.Figure().update_layout(
        title="No training history available",
        template="plotly_dark",
        height=400
    )

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("StockScan Dashboard", className="dashboard-title"),
            html.P("Interactive visualization for stock analysis and model management", className="dashboard-subtitle"),
        ], width=12)
    ], className="header-section"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Stock Symbol:"),
                            dcc.Dropdown(
                                id="stock-dropdown",
                                options=[
                                    {"label": symbol, "value": symbol}
                                    for symbol in get_stock_symbols()
                                ],
                                value=get_stock_symbols()[0] if get_stock_symbols() else None,
                                clearable=False,
                                className="dropdown"
                            )
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Date Range:"),
                            dcc.DatePickerRange(
                                id="date-picker",
                                start_date=(datetime.now() - timedelta(days=90)).date(),
                                end_date=datetime.now().date(),
                                display_format="YYYY-MM-DD",
                                className="date-picker"
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("Technical Indicators:"),
                            dbc.Checklist(
                                id="indicators-checklist",
                                options=[
                                    {"label": "Moving Averages", "value": "ma"},
                                    {"label": "Bollinger Bands", "value": "bb"},
                                    {"label": "RSI", "value": "rsi"},
                                    {"label": "MACD", "value": "macd"}
                                ],
                                value=["ma"],
                                inline=True,
                                className="checklist"
                            )
                        ], width=12)
                    ], className="mt-3")
                ])
            ], className="mb-4 shadow")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Price Chart"),
                dbc.CardBody([
                    dcc.Graph(id="stock-chart")
                ])
            ], className="shadow")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Management"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Model:"),
                            dcc.Dropdown(
                                id="model-dropdown",
                                placeholder="Select a model",
                                className="dropdown"
                            )
                        ], width=6),
                        
                        dbc.Col([
                            html.Label("Select Version:"),
                            dcc.Dropdown(
                                id="version-dropdown",
                                placeholder="Select a version",
                                className="dropdown"
                            )
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="model-info", className="mt-3")
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Rollback to this Version",
                                      id="rollback-button",
                                      color="warning",
                                      className="mt-3 me-2"),
                            dbc.Button("Compare Versions",
                                      id="compare-button",
                                      color="info",
                                      className="mt-3")
                        ], width=12)
                    ])
                ])
            ], className="shadow")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Performance"),
                dbc.CardBody([
                    dcc.Graph(id="model-performance-chart")
                ])
            ], className="shadow")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Training History"),
                dbc.CardBody([
                    dcc.Graph(id="training-history-chart")
                ])
            ], className="shadow")
        ], width=12)
    ], className="mb-4"),
    
    # MLflow integration section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("MLflow Integration"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(label="Experiments & Runs", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("MLflow Experiments:"),
                                    dcc.Dropdown(
                                        id="mlflow-experiment-dropdown",
                                        placeholder="Select an experiment",
                                        className="dropdown"
                                    )
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label("MLflow Runs:"),
                                    dcc.Dropdown(
                                        id="mlflow-run-dropdown",
                                        placeholder="Select a run",
                                        className="dropdown"
                                    )
                                ], width=6)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="mlflow-run-info", className="mt-3")
                                ], width=12)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Link to InfluxDB Version",
                                              id="link-button",
                                              color="primary",
                                              className="mt-3 me-2"),
                                    dbc.Button("View in MLflow UI",
                                              id="mlflow-ui-button",
                                              color="secondary",
                                              className="mt-3",
                                              href=mlflow_config["tracking_uri"],
                                              target="_blank")
                                ], width=12)
                            ])
                        ]),
                        dbc.Tab(label="Prediction Results", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Generate Prediction:"),
                                    dbc.Button("Generate New Prediction",
                                              id="generate-prediction-button",
                                              color="primary",
                                              className="mt-2")
                                ], width=6),
                                dbc.Col([
                                    html.Label("Save Prediction:"),
                                    dbc.Input(id="prediction-tag-input",
                                             placeholder="Enter tag name",
                                             type="text",
                                             className="mt-2"),
                                    dbc.Button("Save to MLflow",
                                              id="save-prediction-button",
                                              color="success",
                                              className="mt-2 ms-2")
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="current-prediction-info", className="mt-3")
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="prediction-chart")
                                ], width=12)
                            ])
                        ]),
                        dbc.Tab(label="Saved Predictions", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Saved Prediction:"),
                                    dcc.Dropdown(
                                        id="saved-prediction-dropdown",
                                        placeholder="Select a saved prediction",
                                        className="dropdown"
                                    )
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="saved-prediction-info", className="mt-3")
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="saved-prediction-chart")
                                ], width=12)
                            ])
                        ]),
                        dbc.Tab(label="Comparison", children=[
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Predictions to Compare:"),
                                    dcc.Dropdown(
                                        id="comparison-dropdown",
                                        placeholder="Select predictions to compare",
                                        multi=True,
                                        className="dropdown"
                                    )
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id="comparison-prediction-chart")
                                ], width=12)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="comparison-metrics-table", className="mt-3")
                                ], width=12)
                            ])
                        ])
                    ])
                ])
            ], className="shadow")
        ], width=12)
    ], className="mb-4"),
    
    # Modal for version comparison
    dbc.Modal([
        dbc.ModalHeader("Model Version Comparison"),
        dbc.ModalBody([
            dcc.Graph(id="comparison-chart")
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-comparison-modal", className="ms-auto", n_clicks=0)
        ),
    ], id="comparison-modal", size="lg"),
    
    # Modal for rollback confirmation
    dbc.Modal([
        dbc.ModalHeader("Confirm Rollback"),
        dbc.ModalBody("Are you sure you want to rollback to this version?"),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="cancel-rollback", className="me-1", n_clicks=0),
            dbc.Button("Confirm", id="confirm-rollback", color="danger", n_clicks=0)
        ]),
    ], id="rollback-modal"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("StockScan Dashboard - Powered by InfluxDB and Plotly Dash", className="text-center text-muted")
        ], width=12)
    ])
], fluid=True, className="dashboard-container")

# Callbacks for interactive functionality
@app.callback(
    Output("stock-chart", "figure"),
    [
        Input("stock-dropdown", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("indicators-checklist", "value")
    ]
)
def update_stock_chart(symbol, start_date, end_date, indicators):
    """Update stock chart based on selected symbol and date range."""
    if not symbol:
        return go.Figure().update_layout(title="No stock selected", template="plotly_dark")
    
    # Get stock data
    logger.info(f"Updating stock chart for {symbol} from {start_date} to {end_date}")
    df = get_stock_data(symbol, start_date, end_date)
    if df.empty:
        logger.warning(f"No data available for {symbol}")
        return go.Figure().update_layout(title=f"No data available for {symbol}", template="plotly_dark")
    
    logger.debug(f"Stock data shape: {df.shape}")
    
    # Create candlestick chart
    fig = create_candlestick_chart(df, symbol)
    
    # Add technical indicators
    if indicators:
        if "bb" in indicators and len(df) > 20:
            # Add Bollinger Bands (20-day, 2 standard deviations)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['BOLU'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
            df['BOLD'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BOLU'],
                    name="Upper Bollinger Band",
                    line=dict(color='rgba(250, 120, 120, 0.7)', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['BOLD'],
                    name="Lower Bollinger Band",
                    line=dict(color='rgba(120, 250, 120, 0.7)', width=1)
                ),
                row=1, col=1
            )
        
        if "rsi" in indicators and len(df) > 14:
            # Calculate RSI (14-day)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Add RSI to a new subplot
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name="RSI (14-day)",
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            
            # Add reference lines at 70 and 30
            fig.add_trace(
                go.Scatter(
                    x=[df.index[0], df.index[-1]],
                    y=[70, 70],
                    name="Overbought (70)",
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[df.index[0], df.index[-1]],
                    y=[30, 30],
                    name="Oversold (30)",
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=3, col=1
            )
            
            # Update layout for RSI
            fig.update_layout(
                height=800,
                yaxis3=dict(
                    title="RSI",
                    range=[0, 100]
                )
            )
        
        if "macd" in indicators and len(df) > 26:
            # Calculate MACD
            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA12'] - df['EMA26']
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['Histogram'] = df['MACD'] - df['Signal']
            
            # Add MACD to a new subplot
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['MACD'],
                    name="MACD",
                    line=dict(color='blue', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Signal'],
                    name="Signal",
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Histogram'],
                    name="Histogram",
                    marker_color='rgba(100, 100, 255, 0.5)'
                ),
                row=4, col=1
            )
            
            # Update layout for MACD
            fig.update_layout(
                height=900,
                yaxis4=dict(title="MACD")
            )
    
    return fig

@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    [Input("stock-dropdown", "value")]
)
def update_model_dropdown(symbol):
    """Update model dropdown based on selected stock."""
    logger.debug(f"Updating model dropdown for symbol: {symbol}")
    if not symbol:
        logger.debug("No symbol selected, returning empty model dropdown")
        return [], None
    
    models = get_model_list(symbol)
    options = [{"label": model, "value": model} for model in models]
    value = models[0] if models else None
    
    logger.debug(f"Model dropdown options: {options}")
    logger.debug(f"Model dropdown default value: {value}")
    return options, value

@app.callback(
    Output("version-dropdown", "options"),
    Output("version-dropdown", "value"),
    [Input("stock-dropdown", "value"),
     Input("model-dropdown", "value")]
)
def update_version_dropdown(symbol, model_name):
    """Update version dropdown based on selected stock and model."""
    logger.debug(f"Updating version dropdown for {symbol}/{model_name}")
    if not symbol or not model_name:
        logger.debug("No symbol or model selected, returning empty version dropdown")
        return [], None
    
    versions = get_model_versions(symbol, model_name)
    options = [{"label": version.get("version"), "value": version.get("version")}
               for version in versions]
    value = options[0]["value"] if options else None
    
    logger.debug(f"Version dropdown options count: {len(options)}")
    logger.debug(f"Version dropdown default value: {value}")
    return options, value

@app.callback(
    Output("model-info", "children"),
    [Input("stock-dropdown", "value"),
     Input("model-dropdown", "value"),
     Input("version-dropdown", "value")]
)
def update_model_info(symbol, model_name, version):
    """Update model information based on selected stock, model, and version."""
    logger.debug(f"Updating model info for {symbol}/{model_name}/{version}")
    if not symbol or not model_name or not version:
        logger.debug("No symbol, model, or version selected")
        return html.P("No model selected")
    
    try:
        # Get model parameters
        logger.debug(f"Loading model parameters for {symbol}/{model_name}")
        params = storage_manager.load_model_params(symbol, model_name)
        if not params:
            logger.warning(f"No parameters found for model {symbol}/{model_name}")
            return html.P("No parameters found for this model")
        
        # Get version details
        logger.debug(f"Getting version details for {symbol}/{model_name}/{version}")
        versions = get_model_versions(symbol, model_name)
        version_info = next((v for v in versions if v.get("version") == version), None)
        
        if not version_info:
            logger.warning(f"Version information not found for {symbol}/{model_name}/{version}")
            return html.P("Version information not found")
        
        # Create info cards
        cards = []
        
        # Version info card
        version_card = dbc.Card([
            dbc.CardHeader("Version Information"),
            dbc.CardBody([
                html.P(f"Version: {version}"),
                html.P(f"Status: {version_info.get('status', 'Unknown')}"),
                html.P(f"Created: {version_info.get('timestamp', 'Unknown')}"),
                html.P(f"Description: {version_info.get('description', 'No description')}")
            ])
        ], className="mb-3")
        cards.append(version_card)
        
        # Model parameters card
        param_items = []
        for key, value in params.items():
            if isinstance(value, dict):
                param_items.append(html.P(f"{key}: {json.dumps(value, indent=2)}"))
            else:
                param_items.append(html.P(f"{key}: {value}"))
        
        params_card = dbc.Card([
            dbc.CardHeader("Model Parameters"),
            dbc.CardBody(param_items)
        ], className="mb-3")
        cards.append(params_card)
        
        # Performance metrics card
        metrics = {k: v for k, v in version_info.items()
                  if k in ["mae", "mse", "rmse", "mape", "r2"]}
        
        if metrics:
            metric_items = [html.P(f"{k.upper()}: {v}") for k, v in metrics.items()]
            metrics_card = dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody(metric_items)
            ])
            cards.append(metrics_card)
        
        return html.Div(cards)
    except Exception as e:
        logger.error(f"Error loading model information: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return html.P(f"Error loading model information: {str(e)}")

@app.callback(
    Output("model-performance-chart", "figure"),
    [Input("stock-dropdown", "value"),
     Input("model-dropdown", "value")]
)
def update_model_performance(symbol, model_name):
    """Update model performance chart based on selected stock and model."""
    if not symbol or not model_name:
        return go.Figure().update_layout(
            title="Select a stock and model to view performance",
            template="plotly_dark"
        )
    
    versions = get_model_versions(symbol, model_name)
    return create_model_comparison_chart(symbol, model_name, versions)

@app.callback(
    Output("training-history-chart", "figure"),
    [Input("stock-dropdown", "value"),
     Input("model-dropdown", "value"),
     Input("version-dropdown", "value")]
)
def update_training_history(symbol, model_name, version):
    """Update training history chart based on selected model version."""
    if not symbol or not model_name or not version:
        return go.Figure().update_layout(
            title="Select a model version to view training history",
            template="plotly_dark"
        )
    
    return create_training_history_chart(symbol, model_name, version)

@app.callback(
    Output("comparison-modal", "is_open"),
    Output("comparison-chart", "figure"),
    [Input("compare-button", "n_clicks"),
     Input("close-comparison-modal", "n_clicks")],
    [State("stock-dropdown", "value"),
     State("model-dropdown", "value"),
     State("comparison-modal", "is_open")]
)
def toggle_comparison_modal(compare_clicks, close_clicks, symbol, model_name, is_open):
    """Toggle comparison modal and update chart."""
    ctx = callback_context
    if not ctx.triggered:
        return is_open, go.Figure()
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "compare-button" and symbol and model_name:
        versions = get_model_versions(symbol, model_name)
        fig = create_model_comparison_chart(symbol, model_name, versions)
        return True, fig
    
    if button_id == "close-comparison-modal":
        return False, go.Figure()
    
    return is_open, go.Figure()

@app.callback(
    Output("rollback-modal", "is_open"),
    [Input("rollback-button", "n_clicks"),
     Input("cancel-rollback", "n_clicks"),
     Input("confirm-rollback", "n_clicks")],
    [State("rollback-modal", "is_open")]
)
def toggle_rollback_modal(rollback_clicks, cancel_clicks, confirm_clicks, is_open):
    """Toggle rollback confirmation modal."""
    ctx = callback_context
    if not ctx.triggered:
        return is_open
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "rollback-button":
        return True
    elif button_id in ["cancel-rollback", "confirm-rollback"]:
        return False
    
    return is_open

@app.callback(
    Output("version-dropdown", "options", allow_duplicate=True),
    Output("version-dropdown", "value", allow_duplicate=True),
    [Input("confirm-rollback", "n_clicks")],
    [State("stock-dropdown", "value"),
     State("model-dropdown", "value"),
     State("version-dropdown", "value")],
    prevent_initial_call=True
)
def rollback_to_version(n_clicks, symbol, model_name, version):
    """Rollback to selected version when confirmed."""
    if not n_clicks or not symbol or not model_name or not version:
        raise PreventUpdate
    
    from modules.storage_backends.influxdb_storage import InfluxDBStorage
    if isinstance(storage_manager.backend, InfluxDBStorage):
        success = storage_manager.backend.rollback_to_version(symbol, model_name, version)
        if success:
            # Refresh versions list
            versions = get_model_versions(symbol, model_name)
            options = [{"label": v.get("version"), "value": v.get("version")} for v in versions]
            # Return the active version (should be the restored one)
            active_version = next((v.get("version") for v in versions if v.get("status") == "active"), None)
            return options, active_version
    
    # If rollback failed or not using InfluxDB, just return current options and value
    versions = get_model_versions(symbol, model_name)
    options = [{"label": v.get("version"), "value": v.get("version")} for v in versions]
    return options, version

# Prediction-related callbacks
@app.callback(
    Output("current-prediction-info", "children"),
    Output("prediction-chart", "figure"),
    [Input("generate-prediction-button", "n_clicks")],
    [State("stock-dropdown", "value"),
     State("date-picker", "start_date"),
     State("date-picker", "end_date")]
)
def generate_and_display_prediction(n_clicks, symbol, start_date, end_date):
    """Generate and display a new prediction."""
    if not n_clicks or not symbol:
        return html.P("Click 'Generate New Prediction' to create a prediction"), go.Figure()
    
    # Get stock data
    df = get_stock_data(symbol, start_date, end_date)
    if df.empty:
        return html.P(f"No data available for {symbol}"), go.Figure()
    
    # Generate prediction
    from modules.trading_prediction.integration import get_trading_prediction
    df.name = symbol  # Set name attribute for the dataframe
    prediction = get_trading_prediction(symbol, df)
    
    # Create prediction info
    info_card = dbc.Card([
        dbc.CardHeader("Prediction Information"),
        dbc.CardBody([
            html.P(f"Symbol: {symbol}"),
            html.P(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            html.P(f"Action: {prediction['overall_action']}"),
            html.P(f"Confidence: {prediction['confidence']:.2f}"),
            html.P(f"Forecast Change: {prediction['forecast_change_percent']:.2f}%"),
            html.P(f"Trend: {prediction['trend']:.2f}"),
            html.P(f"Volatility: {prediction['volatility']:.2f}")
        ])
    ])
    
    # Create prediction chart
    fig = create_prediction_chart(df, prediction, symbol)
    
    return info_card, fig

@app.callback(
    Output("saved-prediction-dropdown", "options"),
    [Input("save-prediction-button", "n_clicks")],
    [State("stock-dropdown", "value"),
     State("prediction-tag-input", "value")]
)
def save_prediction_to_mlflow(n_clicks, symbol, tag):
    """Save the current prediction to MLflow with a tag."""
    if not n_clicks or not symbol or not tag:
        # Just return the current saved predictions
        return get_saved_predictions()
    
    # Get stock data
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = get_stock_data(symbol, start_date, end_date)
    if df.empty:
        return get_saved_predictions()
    
    # Generate prediction and log to MLflow
    from modules.trading_prediction.integration import get_trading_prediction
    df.name = symbol
    prediction = get_trading_prediction(symbol, df, log_to_mlflow=True,
                                       mlflow_tags={"tag": tag, "saved_by_user": "true"})
    
    # Return updated saved predictions
    return get_saved_predictions()

def get_saved_predictions():
    """Get list of saved predictions from MLflow."""
    import mlflow
    
    try:
        # Set MLflow tracking URI
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # Get experiment ID
        experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
        if not experiment:
            logger.warning(f"MLflow experiment '{mlflow_config['experiment_name']}' not found")
            return []
        
        # Get runs with "saved_by_user" tag
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.saved_by_user = 'true'"
        )
        
        # Format options for dropdown
        options = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            symbol = run["params.symbol"] if "params.symbol" in run else "Unknown"
            tag = run["tags.tag"] if "tags.tag" in run else "Untitled"
            date = run["start_time"].split(" ")[0] if "start_time" in run else "Unknown"
            
            options.append({
                "label": f"{symbol} - {tag} ({date})",
                "value": run_id
            })
        
        return options
    except Exception as e:
        logger.error(f"Error getting saved predictions: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return []

@app.callback(
    Output("saved-prediction-info", "children"),
    Output("saved-prediction-chart", "figure"),
    [Input("saved-prediction-dropdown", "value")]
)
def display_saved_prediction(run_id):
    """Display a saved prediction from MLflow."""
    logger.debug(f"Displaying saved prediction for run_id: {run_id}")
    if not run_id:
        logger.debug("No run_id selected")
        return html.P("Select a saved prediction to view"), go.Figure()
    
    import mlflow
    
    try:
        # Set MLflow tracking URI
        logger.debug("Setting MLflow tracking URI")
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # Get run data
        logger.debug(f"Getting MLflow run data for run_id: {run_id}")
        run = mlflow.get_run(run_id)
        logger.debug(f"Retrieved run data: {run.info.run_id}")
        
        # Extract prediction info
        params = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags
        
        # Create prediction info card
        info_card = dbc.Card([
            dbc.CardHeader("Saved Prediction"),
            dbc.CardBody([
                html.P(f"Symbol: {params.get('symbol', 'Unknown')}"),
                html.P(f"Date: {params.get('prediction_date', 'Unknown')}"),
                html.P(f"Action: {params.get('prediction_type', 'Unknown')}"),
                html.P(f"Confidence: {metrics.get('confidence', 0):.2f}"),
                html.P(f"Forecast Change: {metrics.get('forecast_change_percent', 0):.2f}%"),
                html.P(f"Trend: {metrics.get('trend', 0):.2f}"),
                html.P(f"Volatility: {metrics.get('volatility', 0):.2f}"),
                html.P(f"Tag: {tags.get('tag', 'Untitled')}")
            ])
        ])
        
        # Try to get the artifact URI for the visualization
        artifact_uri = run.info.artifact_uri
        
        # Create a figure from the saved data or a placeholder
        fig = go.Figure()
        
        # Try to reconstruct the prediction from the saved parameters and metrics
        symbol = params.get('symbol', 'Unknown')
        action = params.get('prediction_type', 'Unknown')
        confidence = metrics.get('confidence', 0.5)
        forecast_change = metrics.get('forecast_change_percent', 0)
        trend = metrics.get('trend', 0)
        volatility = metrics.get('volatility', 0.5)
        
        # Create a placeholder chart
        fig.add_trace(go.Scatter(
            x=[0, 1, 2, 3, 4, 5],
            y=[100, 100 * (1 + forecast_change/100)],
            mode='lines',
            name='Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{symbol} Saved Prediction",
            xaxis_title="Days",
            yaxis_title="Price (Normalized)",
            template="plotly_dark",
            height=500,
            annotations=[
                dict(
                    x=2.5,
                    y=100 * (1 + forecast_change/200),
                    xref="x",
                    yref="y",
                    text=f"{action} ({confidence:.2f} confidence)<br>{forecast_change:.2f}% change",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=0,
                    ay=-40
                )
            ]
        )
        
        return info_card, fig
    except Exception as e:
        logger.error(f"Error displaying saved prediction: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return html.P(f"Error: {str(e)}"), go.Figure()

@app.callback(
    Output("mlflow-experiment-dropdown", "options"),
    Output("mlflow-experiment-dropdown", "value"),
    [Input("stock-dropdown", "value")]  # Using stock dropdown as a trigger
)
def update_mlflow_experiment_dropdown(stock_symbol):
    """Update MLflow experiment dropdown."""
    import mlflow
    
    try:
        # Set MLflow tracking URI
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # List all experiments
        experiments = mlflow.search_experiments()
        
        # Format options for dropdown
        options = []
        for exp in experiments:
            options.append({
                "label": exp.name,
                "value": exp.experiment_id
            })
        
        # Set default value to the configured experiment
        default_experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
        default_value = default_experiment.experiment_id if default_experiment else None
        
        logger.debug(f"Found {len(options)} MLflow experiments")
        return options, default_value
    except Exception as e:
        logger.error(f"Error getting MLflow experiments: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return [], None

@app.callback(
    Output("mlflow-run-dropdown", "options"),
    Output("mlflow-run-dropdown", "value"),
    [Input("mlflow-experiment-dropdown", "value")]
)
def update_mlflow_run_dropdown(experiment_id):
    """Update MLflow run dropdown based on selected experiment."""
    if not experiment_id:
        return [], None
    
    import mlflow
    
    try:
        # Set MLflow tracking URI
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # Get runs for the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        
        # Format options for dropdown
        options = []
        for _, run in runs.iterrows():
            run_id = run["run_id"]
            
            # Handle timestamp formatting safely
            if "start_time" in run:
                if isinstance(run["start_time"], str):
                    start_time = run["start_time"].split(" ")[0]
                else:
                    # Handle pandas Timestamp object
                    start_time = str(run["start_time"]).split(" ")[0]
            else:
                start_time = "Unknown"
                
            status = run["status"]
            
            # Try to get a meaningful name for the run
            run_name = run["tags.mlflow.runName"] if "tags.mlflow.runName" in run else run_id[:8]
            
            # Try to get symbol if available
            symbol = run["params.symbol"] if "params.symbol" in run else ""
            if symbol:
                run_name = f"{symbol} - {run_name}"
            
            options.append({
                "label": f"{run_name} ({start_time}, {status})",
                "value": run_id
            })
        
        logger.debug(f"Found {len(options)} MLflow runs for experiment {experiment_id}")
        value = options[0]["value"] if options else None
        return options, value
    except Exception as e:
        logger.error(f"Error getting MLflow runs: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return [], None

@app.callback(
    Output("mlflow-run-info", "children"),
    [Input("mlflow-run-dropdown", "value")]
)
def update_mlflow_run_info(run_id):
    """Update MLflow run information based on selected run."""
    if not run_id:
        return html.P("Select a run to view details")
    
    import mlflow
    
    try:
        # Set MLflow tracking URI
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # Get run data
        run = mlflow.get_run(run_id)
        
        # Extract run info
        info = run.info
        params = run.data.params
        metrics = run.data.metrics
        tags = run.data.tags
        
        # Create info cards
        cards = []
        
        # Run info card
        run_info_items = [
            html.P(f"Run ID: {info.run_id}"),
            html.P(f"Status: {info.status}"),
            html.P(f"Start Time: {info.start_time}"),
            html.P(f"End Time: {info.end_time or 'N/A'}"),
            html.P(f"Artifact URI: {info.artifact_uri}")
        ]
        
        run_info_card = dbc.Card([
            dbc.CardHeader("Run Information"),
            dbc.CardBody(run_info_items)
        ], className="mb-3")
        cards.append(run_info_card)
        
        # Parameters card
        if params:
            param_items = [html.P(f"{k}: {v}") for k, v in params.items()]
            params_card = dbc.Card([
                dbc.CardHeader("Parameters"),
                dbc.CardBody(param_items)
            ], className="mb-3")
            cards.append(params_card)
        
        # Metrics card
        if metrics:
            metric_items = [html.P(f"{k}: {v}") for k, v in metrics.items()]
            metrics_card = dbc.Card([
                dbc.CardHeader("Metrics"),
                dbc.CardBody(metric_items)
            ], className="mb-3")
            cards.append(metrics_card)
        
        # Tags card
        if tags:
            tag_items = [html.P(f"{k}: {v}") for k, v in tags.items()
                        if not k.startswith("mlflow.")]
            if tag_items:
                tags_card = dbc.Card([
                    dbc.CardHeader("Tags"),
                    dbc.CardBody(tag_items)
                ], className="mb-3")
                cards.append(tags_card)
        
        return html.Div(cards)
    except Exception as e:
        logger.error(f"Error getting MLflow run info: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return html.P(f"Error: {str(e)}")

@app.callback(
    Output("comparison-dropdown", "options"),
    [Input("saved-prediction-dropdown", "options")]
)
def update_comparison_dropdown(saved_predictions):
    """Update comparison dropdown with saved predictions."""
    return saved_predictions

@app.callback(
    Output("comparison-prediction-chart", "figure"),
    Output("comparison-metrics-table", "children"),
    [Input("comparison-dropdown", "value")]
)
def update_comparison_view(selected_runs):
    """Update comparison view based on selected runs."""
    if not selected_runs or len(selected_runs) == 0:
        return go.Figure(), html.P("Select predictions to compare")
    
    import mlflow
    
    try:
        # Set MLflow tracking URI
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        
        # Get data for all selected runs
        runs_data = []
        for run_id in selected_runs:
            run = mlflow.get_run(run_id)
            
            # Extract data
            params = run.data.params
            metrics = run.data.metrics
            tags = run.data.tags
            
            run_data = {
                "run_id": run_id,
                "symbol": params.get("symbol", "Unknown"),
                "date": params.get("prediction_date", "Unknown"),
                "action": params.get("prediction_type", "Unknown"),
                "confidence": metrics.get("confidence", 0),
                "forecast_change": metrics.get("forecast_change_percent", 0),
                "trend": metrics.get("trend", 0),
                "volatility": metrics.get("volatility", 0),
                "tag": tags.get("tag", "Untitled")
            }
            
            runs_data.append(run_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add bars for forecast change
        fig.add_trace(go.Bar(
            x=[f"{d['symbol']} - {d['tag']}" for d in runs_data],
            y=[d["forecast_change"] for d in runs_data],
            name="Forecast Change %",
            marker_color='blue'
        ))
        
        # Add confidence as a line
        fig.add_trace(go.Scatter(
            x=[f"{d['symbol']} - {d['tag']}" for d in runs_data],
            y=[d["confidence"] for d in runs_data],
            name="Confidence",
            mode="lines+markers",
            yaxis="y2",
            marker_color='red'
        ))
        
        # Update layout
        fig.update_layout(
            title="Prediction Comparison",
            xaxis_title="Predictions",
            yaxis_title="Forecast Change %",
            template="plotly_dark",
            height=500,
            yaxis2=dict(
                title="Confidence",
                overlaying="y",
                side="right",
                range=[0, 1]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create metrics table
        table_header = [
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Tag"),
                html.Th("Date"),
                html.Th("Action"),
                html.Th("Forecast Change %"),
                html.Th("Confidence"),
                html.Th("Trend"),
                html.Th("Volatility")
            ]))
        ]
        
        table_body = [html.Tbody([
            html.Tr([
                html.Td(d["symbol"]),
                html.Td(d["tag"]),
                html.Td(d["date"]),
                html.Td(d["action"]),
                html.Td(f"{d['forecast_change']:.2f}%"),
                html.Td(f"{d['confidence']:.2f}"),
                html.Td(f"{d['trend']:.2f}"),
                html.Td(f"{d['volatility']:.2f}")
            ]) for d in runs_data
        ])]
        
        table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)
        
        return fig, table
    except Exception as e:
        logger.error(f"Error updating comparison view: {e}")
        logger.debug(f"Detailed error: {traceback.format_exc()}")
        return go.Figure(), html.P(f"Error: {str(e)}")

def create_prediction_chart(df, prediction, symbol):
    """Create a chart visualizing the prediction."""
    # Create figure
    fig = make_subplots(rows=1, cols=1)
    
    # Add historical price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            name="Historical Price",
            line=dict(color='blue')
        )
    )
    
    # Add prediction
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    
    # Calculate predicted prices
    last_price = df['Close'].iloc[-1]
    forecast_change = prediction['forecast_change_percent'] / 100
    predicted_price = last_price * (1 + forecast_change)
    
    # Create a simple linear path from last price to predicted price
    predicted_prices = [last_price]
    for i in range(1, 6):
        step_change = last_price * (forecast_change * i / 5)
        predicted_prices.append(last_price + step_change)
    
    # Add prediction line
    fig.add_trace(
        go.Scatter(
            x=[last_date] + forecast_dates,
            y=predicted_prices,
            name="Prediction",
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add confidence interval
    volatility = prediction.get('volatility', 0.5)
    upper_bound = [price * (1 + volatility) for price in predicted_prices]
    lower_bound = [price * (1 - volatility) for price in predicted_prices]
    
    fig.add_trace(
        go.Scatter(
            x=[last_date] + forecast_dates,
            y=upper_bound,
            name="Upper Bound",
            line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[last_date] + forecast_dates,
            y=lower_bound,
            name="Lower Bound",
            line=dict(color='rgba(255,0,0,0.2)', dash='dot'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=False
        )
    )
    
    # Add action markers
    daily_actions = prediction.get('daily_actions', [])
    for i, action in enumerate(daily_actions):
        if action.lower() == 'buy':
            fig.add_trace(
                go.Scatter(
                    x=[forecast_dates[i]],
                    y=[predicted_prices[i+1]],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name="Buy Signal"
                )
            )
        elif action.lower() == 'sell':
            fig.add_trace(
                go.Scatter(
                    x=[forecast_dates[i]],
                    y=[predicted_prices[i+1]],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name="Sell Signal"
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main entry point
if __name__ == "__main__":
    # Run the Dash app
    logger.info("Starting Dash app server on host=0.0.0.0, port=8050")
    app.run_server(debug=True, host="0.0.0.0", port=8050)