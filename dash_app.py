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

# Import storage backend and configuration
from modules.storage import StorageManager
from modules.connection_config import get_influxdb_config, get_mlflow_config
from modules.config_manager import load_config_file
import mlflow
import os

# Load configuration file
config_file = os.path.join(os.path.dirname(__file__), "config/default_config.yaml")
load_config_file(config_file)
print(f"Loaded configuration from {config_file}")

# Get connection configurations
influxdb_config = get_influxdb_config()
mlflow_config = get_mlflow_config()
print(f"Using InfluxDB URL: {influxdb_config['url']}")

# Initialize storage manager with InfluxDB backend
storage_manager = StorageManager(
    backend_type="influxdb",
    url=influxdb_config["url"],
    token=influxdb_config["token"],
    org=influxdb_config["org"],
    bucket=influxdb_config["bucket"]
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

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
    return storage_manager.list_symbols()

def get_stock_data(symbol, start_date=None, end_date=None):
    """Get stock data for a symbol with optional date filtering."""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    return storage_manager.load_data(symbol, start_date, end_date)

def get_model_list(symbol):
    """Get list of available models for a symbol."""
    return storage_manager.list_models(symbol)

def get_model_versions(symbol, model_name):
    """Get list of versions for a model."""
    from modules.storage_backends.influxdb_storage import InfluxDBStorage
    if isinstance(storage_manager.backend, InfluxDBStorage):
        return storage_manager.backend.list_model_versions(symbol, model_name)
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
        print(f"Error creating training history chart: {e}")
    
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
    df = get_stock_data(symbol, start_date, end_date)
    if df.empty:
        return go.Figure().update_layout(title=f"No data available for {symbol}", template="plotly_dark")
    
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
    if not symbol:
        return [], None
    
    models = get_model_list(symbol)
    options = [{"label": model, "value": model} for model in models]
    value = models[0] if models else None
    
    return options, value

@app.callback(
    Output("version-dropdown", "options"),
    Output("version-dropdown", "value"),
    [Input("stock-dropdown", "value"),
     Input("model-dropdown", "value")]
)
def update_version_dropdown(symbol, model_name):
    """Update version dropdown based on selected stock and model."""
    if not symbol or not model_name:
        return [], None
    
    versions = get_model_versions(symbol, model_name)
    options = [{"label": version.get("version"), "value": version.get("version")}
               for version in versions]
    value = options[0]["value"] if options else None
    
    return options, value

@app.callback(
    Output("model-info", "children"),
    [Input("stock-dropdown", "value"),
     Input("model-dropdown", "value"),
     Input("version-dropdown", "value")]
)
def update_model_info(symbol, model_name, version):
    """Update model information based on selected stock, model, and version."""
    if not symbol or not model_name or not version:
        return html.P("No model selected")
    
    try:
        # Get model parameters
        params = storage_manager.load_model_params(symbol, model_name)
        if not params:
            return html.P("No parameters found for this model")
        
        # Get version details
        versions = get_model_versions(symbol, model_name)
        version_info = next((v for v in versions if v.get("version") == version), None)
        
        if not version_info:
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

# Main entry point
if __name__ == "__main__":
    # Run the Dash app
    app.run_server(debug=True, host="0.0.0.0", port=8050)