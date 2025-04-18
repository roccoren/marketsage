# Configuration and Deployment Plan

This document outlines the plan for:
1. Moving database and service connection strings to a centralized configuration file
2. Separating Docker deployment for middleware and applications
3. Providing multiple deployment options for the Dash application

## 1. Extend Configuration Management

### 1.1. Update default_config.yaml

We'll extend the existing default_config.yaml file to include a new section for database and service connections:

```yaml
# Database and service connections
connections:
  # InfluxDB configuration
  influxdb:
    url: http://localhost:8086
    token: stockscantoken
    org: stockscan
    bucket: stock_data
  
  # MLflow configuration
  mlflow:
    tracking_uri: http://localhost:5000
    experiment_name: marketsage
    artifact_dir: mlruns
```

### 1.2. Create Connection Config Utility

Create a utility module `modules/connection_config.py` to provide easy access to connection strings across different applications:

```python
from modules.config_manager import get_config

def get_influxdb_config():
    """Get InfluxDB connection configuration."""
    return {
        "url": get_config("connections.influxdb.url", "http://localhost:8086"),
        "token": get_config("connections.influxdb.token", "stockscantoken"),
        "org": get_config("connections.influxdb.org", "stockscan"),
        "bucket": get_config("connections.influxdb.bucket", "stock_data")
    }

def get_mlflow_config():
    """Get MLflow connection configuration."""
    return {
        "tracking_uri": get_config("connections.mlflow.tracking_uri", "http://localhost:5000"),
        "experiment_name": get_config("connections.mlflow.experiment_name", "marketsage"),
        "artifact_dir": get_config("connections.mlflow.artifact_dir", "mlruns")
    }
```

### 1.3. Update Applications to Use Centralized Config

Modify the following files to use the centralized configuration:

- dash_app.py
- migrate_to_influxdb.py
- Any other files using hardcoded connection strings

## 2. Docker Deployment Separation

### 2.1. Create Middleware Docker Compose File

Create a new docker-compose file specifically for middleware components (InfluxDB and MLflow):

`docker/docker-compose.middleware.yml`:
```yaml
version: '3'

services:
  postgres:
    image: postgres:14
    container_name: mlflow-postgres
    restart: always
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=mlflow
      - POSTGRES_DB=mlflow
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - middleware-network

  mlflow:
    image: python:3.9
    container_name: mlflow-server
    restart: always
    depends_on:
      - postgres
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
    volumes:
      - ./mlflow:/mlflow
      - mlflow_artifacts:/mlflow/artifacts
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000
               --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
               --default-artifact-root /mlflow/artifacts"
    networks:
      - middleware-network

  influxdb:
    image: influxdb:2.0
    container_name: stockscan-influxdb
    ports:
      - "8086:8086"
    volumes:
      - influxdb-data:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=stockscanadmin
      - DOCKER_INFLUXDB_INIT_ORG=stockscan
      - DOCKER_INFLUXDB_INIT_BUCKET=stock_data
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=stockscantoken
    restart: unless-stopped
    networks:
      - middleware-network

networks:
  middleware-network:
    driver: bridge

volumes:
  postgres_data:
  mlflow_artifacts:
  influxdb-data:
```

### 2.2. Create Dash Docker Compose File

Create a docker-compose file specifically for the Dash application:

`docker/docker-compose.dash.yml`:
```yaml
version: '3'

services:
  dash:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dash
    container_name: stockscan-dash
    ports:
      - "8050:8050"
    volumes:
      - ../config:/app/config
    environment:
      - CONFIG_FILE=/app/config/default_config.yaml
    restart: unless-stopped
    networks:
      - dash-network
      - middleware-network

networks:
  dash-network:
    driver: bridge
  middleware-network:
    external: true
```

## 3. Dash Deployment Scripts

### 3.1. Docker Deployment Script

Create a script for Docker deployment of the Dash application:

`scripts/deploy_dash_docker.sh`:
```bash
#!/bin/bash

# Deploy Dash application using Docker
cd docker
docker-compose -f docker-compose.dash.yml up -d
```

### 3.2. Standalone Python Deployment Script

Create a script for standalone Python deployment:

`scripts/deploy_dash_standalone.sh`:
```bash
#!/bin/bash

# Install required dependencies
pip install -r requirements.txt

# Run Dash application
python dash_app.py
```

### 3.3. Systemd Service File for Linux Deployment

Create a systemd service file for Linux deployment:

`scripts/stockscan-dash.service`:
```ini
[Unit]
Description=StockScan Dash Application
After=network.target

[Service]
User=stockscan
WorkingDirectory=/path/to/stockscan
ExecStart=/usr/bin/python3 /path/to/stockscan/dash_app.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=stockscan-dash
Environment="CONFIG_FILE=/path/to/stockscan/config/default_config.yaml"

[Install]
WantedBy=multi-user.target
```

### 3.4. Cloud Deployment Scripts

#### 3.4.1. AWS Elastic Beanstalk Deployment

Create a configuration file for AWS Elastic Beanstalk:

`scripts/cloud/aws/.ebextensions/01_flask.config`:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: dash_app:server
  aws:elasticbeanstalk:application:environment:
    CONFIG_FILE: /var/app/current/config/default_config.yaml
```

`scripts/cloud/aws/deploy_aws.sh`:
```bash
#!/bin/bash

# Initialize Elastic Beanstalk application
eb init -p python-3.8 stockscan-dash

# Create environment and deploy
eb create stockscan-dash-env
```

#### 3.4.2. Google Cloud Run Deployment

Create a Dockerfile for Google Cloud Run:

`scripts/cloud/gcp/Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CONFIG_FILE=/app/config/default_config.yaml
ENV PORT=8080

CMD exec gunicorn --bind :$PORT dash_app:server
```

`scripts/cloud/gcp/deploy_gcp.sh`:
```bash
#!/bin/bash

# Build and deploy to Google Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/stockscan-dash
gcloud run deploy stockscan-dash --image gcr.io/PROJECT_ID/stockscan-dash --platform managed
```

## Implementation Flow Diagram

```mermaid
graph TD
    A[Extend default_config.yaml] --> B[Create connection_config.py utility]
    B --> C[Update applications to use centralized config]
    
    D[Create docker-compose.middleware.yml] --> E[Create docker-compose.dash.yml]
    
    F[Create deployment scripts] --> G[Docker deployment]
    F --> H[Standalone Python deployment]
    F --> I[Systemd service]
    F --> J[Cloud deployment options]
    
    C --> F
    E --> G
```

## Implementation Steps

1. First, extend the default_config.yaml file with the new connections section
2. Create the connection_config.py utility module
3. Update the applications (dash_app.py, migrate_to_influxdb.py) to use the centralized configuration
4. Split the Docker Compose files into middleware and application components
5. Create the deployment scripts for different environments
6. Test each deployment option to ensure it works correctly