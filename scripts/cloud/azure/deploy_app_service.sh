#!/bin/bash
# Script to deploy the StockScan Dash application to Azure App Service
# This script assumes middleware services (InfluxDB, MLflow) are already available

set -e

# Navigate to the project root directory
cd "$(dirname "$0")/../../.."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed. Please install it first."
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if user is logged in to Azure
if ! az account show &> /dev/null; then
    echo "You are not logged in to Azure. Please login first."
    az login
fi

# Configuration variables - modify these as needed
RESOURCE_GROUP="stockscan-rg"
LOCATION="eastus"
APP_SERVICE_PLAN="stockscan-plan"
APP_NAME="stockscan-dash"
SKU="B1"

# External middleware connection settings - update these with your actual endpoints
INFLUXDB_URL="https://your-influxdb-endpoint:8086"
INFLUXDB_TOKEN="your-influxdb-token"
INFLUXDB_ORG="stockscan"
INFLUXDB_BUCKET="stock_data"
MLFLOW_TRACKING_URI="https://your-mlflow-endpoint:5000"

# Ask for confirmation
echo "This script will deploy the StockScan Dash application to Azure App Service with the following configuration:"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "App Service Plan: $APP_SERVICE_PLAN"
echo "App Name: $APP_NAME"
echo "SKU: $SKU"
echo
echo "The application will connect to these external services:"
echo "InfluxDB URL: $INFLUXDB_URL"
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 1
fi

# Create resource group if it doesn't exist
echo "Creating resource group if it doesn't exist..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create App Service Plan if it doesn't exist
echo "Creating App Service Plan if it doesn't exist..."
az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --is-linux --sku $SKU

# Create Web App for Containers if it doesn't exist
echo "Creating Web App for Containers if it doesn't exist..."
az webapp create --resource-group $RESOURCE_GROUP --plan $APP_SERVICE_PLAN --name $APP_NAME --runtime "PYTHON|3.9"

# Configure Web App settings
echo "Configuring Web App settings..."
az webapp config set --resource-group $RESOURCE_GROUP --name $APP_NAME --always-on true --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 dash_app:server"

# Set environment variables
echo "Setting environment variables..."
az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $APP_NAME --settings \
    CONFIG_FILE=/home/site/wwwroot/config/default_config.yaml \
    INFLUXDB_URL="$INFLUXDB_URL" \
    INFLUXDB_TOKEN="$INFLUXDB_TOKEN" \
    INFLUXDB_ORG="$INFLUXDB_ORG" \
    INFLUXDB_BUCKET="$INFLUXDB_BUCKET" \
    MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI"

# Enable system-assigned managed identity
echo "Enabling system-assigned managed identity..."
az webapp identity assign --resource-group $RESOURCE_GROUP --name $APP_NAME

# Deploy the application code
echo "Deploying application code..."
az webapp deployment source config-zip --resource-group $RESOURCE_GROUP --name $APP_NAME --src ./deploy.zip

# Restart the web app to apply changes
echo "Restarting the web app..."
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

echo "Deployment completed successfully!"
echo "Your application is available at: https://$APP_NAME.azurewebsites.net"

exit 0