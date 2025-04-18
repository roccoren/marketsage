#!/usr/bin/env python3
"""
Connection configuration utility for the StockScan project.
This module provides easy access to connection strings for databases and services.
"""

import os
from modules.config_manager import get_config

def get_influxdb_config():
    """
    Get InfluxDB connection configuration.
    
    Returns:
        dict: InfluxDB connection parameters
    """
    # Print the current config for debugging
    from modules.config_manager import get_config_manager
    config = get_config_manager().get_all()
    print(f"Current config: {config}")
    
    # Get InfluxDB config with explicit debugging
    url = get_config("connections.influxdb.url", os.environ.get("INFLUXDB_URL", "http://localhost:8086"))
    token = get_config("connections.influxdb.token", os.environ.get("INFLUXDB_TOKEN", "stockscantoken"))
    org = get_config("connections.influxdb.org", os.environ.get("INFLUXDB_ORG", "stockscan"))
    bucket = get_config("connections.influxdb.bucket", os.environ.get("INFLUXDB_BUCKET", "stock_data"))
    
    print(f"InfluxDB config from get_config: url={url}, org={org}, bucket={bucket}")
    
    return {
        "url": url,
        "token": token,
        "org": org,
        "bucket": bucket
    }

def get_mlflow_config():
    """
    Get MLflow connection configuration.
    
    Returns:
        dict: MLflow connection parameters
    """
    return {
        "tracking_uri": get_config("connections.mlflow.tracking_uri", 
                                  os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")),
        "experiment_name": get_config("connections.mlflow.experiment_name", "marketsage"),
        "artifact_dir": get_config("connections.mlflow.artifact_dir", "mlruns")
    }

def get_postgres_config():
    """
    Get PostgreSQL connection configuration.
    
    Returns:
        dict: PostgreSQL connection parameters
    """
    return {
        "host": get_config("connections.postgres.host", os.environ.get("POSTGRES_HOST", "localhost")),
        "port": get_config("connections.postgres.port", os.environ.get("POSTGRES_PORT", "5432")),
        "user": get_config("connections.postgres.user", os.environ.get("POSTGRES_USER", "mlflow")),
        "password": get_config("connections.postgres.password", os.environ.get("POSTGRES_PASSWORD", "mlflow")),
        "database": get_config("connections.postgres.database", os.environ.get("POSTGRES_DB", "mlflow"))
    }

def get_postgres_uri():
    """
    Get PostgreSQL connection URI.
    
    Returns:
        str: PostgreSQL connection URI
    """
    pg_config = get_postgres_config()
    return f"postgresql://{pg_config['user']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"