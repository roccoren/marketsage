# StockScan Deployment Scripts

This directory contains scripts for deploying the StockScan application in various environments. The scripts are organized by deployment type and target environment.

## Local Deployment

### Docker Deployment

- **deploy_middleware_docker.sh**: Deploys middleware components (InfluxDB, PostgreSQL, MLflow) using Docker.
  ```bash
  ./scripts/deploy_middleware_docker.sh
  ```

- **deploy_dash_docker.sh**: Deploys the Dash application using Docker. Requires middleware to be deployed first.
  ```bash
  ./scripts/deploy_dash_docker.sh
  ```

### Standalone Python Deployment

- **deploy_dash_standalone.sh**: Deploys the Dash application as a standalone Python application. Useful for development or small-scale deployments.
  ```bash
  ./scripts/deploy_dash_standalone.sh
  ```

### Systemd Service Deployment

- **stockscan-dash.service**: Systemd service file for running the Dash application as a service on Linux.
- **install_systemd_service.sh**: Installs the Dash application as a systemd service on Linux.
  ```bash
  sudo ./scripts/install_systemd_service.sh
  ```

## Cloud Deployment

### Azure Deployment

- **cloud/azure/prepare_deployment.sh**: Prepares a deployment package for Azure App Service.
  ```bash
  ./scripts/cloud/azure/prepare_deployment.sh
  ```

- **cloud/azure/deploy_app_service.sh**: Deploys the Dash application to Azure App Service.
  ```bash
  ./scripts/cloud/azure/deploy_app_service.sh
  ```

- **cloud/azure/deploy_container_app.sh**: Deploys the Dash application to Azure Container Apps.
  ```bash
  ./scripts/cloud/azure/deploy_container_app.sh
  ```

- **cloud/azure/Dockerfile**: Dockerfile for building the Dash application for Azure deployment.

## Deployment Configuration

All deployment scripts use the centralized configuration from `config/default_config.yaml`. The connection strings for databases and services are stored in this file and can be overridden using environment variables.

For cloud deployments, you need to update the connection strings in the deployment scripts to point to your cloud-hosted middleware services (InfluxDB, MLflow).

## Prerequisites

- Docker and docker-compose (for Docker deployment)
- Python 3.9+ (for standalone deployment)
- Azure CLI (for Azure deployment)
- Systemd (for systemd service deployment)