#!/bin/bash
# Script to deploy the MarketSage Dash application to Azure App Service

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
RESOURCE_GROUP="marketsage-rg"
LOCATION="eastus"
APP_SERVICE_PLAN="marketsage-plan"
APP_NAME="marketsage-dash"
SKU="B1"
REGISTRY_NAME="marketsageregistry"
IMAGE_NAME="marketsage-dash"
IMAGE_TAG="latest"

# Ask for confirmation
echo "This script will deploy the MarketSage Dash application to Azure App Service with the following configuration:"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "App Service Plan: $APP_SERVICE_PLAN"
echo "App Name: $APP_NAME"
echo "SKU: $SKU"
echo "Container Registry: $REGISTRY_NAME"
echo "Image Name: $IMAGE_NAME:$IMAGE_TAG"
read -p "Do you want to continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 1
fi

# Create resource group if it doesn't exist
echo "Creating resource group if it doesn't exist..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry if it doesn't exist..."
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic

# Log in to ACR
echo "Logging in to Azure Container Registry..."
az acr login --name $REGISTRY_NAME

# Build and push Docker image to ACR
echo "Building and pushing Docker image to ACR..."
az acr build --registry $REGISTRY_NAME --image $IMAGE_NAME:$IMAGE_TAG --file scripts/cloud/azure/Dockerfile .

# Create App Service Plan if it doesn't exist
echo "Creating App Service Plan if it doesn't exist..."
az appservice plan create --name $APP_SERVICE_PLAN --resource-group $RESOURCE_GROUP --is-linux --sku $SKU

# Create Web App for Containers if it doesn't exist
echo "Creating Web App for Containers if it doesn't exist..."
az webapp create --resource-group $RESOURCE_GROUP --plan $APP_SERVICE_PLAN --name $APP_NAME --deployment-container-image-name "$REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"

# Configure Web App settings
echo "Configuring Web App settings..."
az webapp config set --resource-group $RESOURCE_GROUP --name $APP_NAME --always-on true

# Set environment variables
echo "Setting environment variables..."
az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $APP_NAME --settings \
    CONFIG_FILE=/app/config/default_config.yaml \
    INFLUXDB_URL="https://your-influxdb-instance.azurewebsites.net:8086" \
    INFLUXDB_TOKEN="your-influxdb-token" \
    INFLUXDB_ORG="marketsage" \
    INFLUXDB_BUCKET="stock_data" \
    MLFLOW_TRACKING_URI="https://your-mlflow-instance.azurewebsites.net"

# Enable system-assigned managed identity
echo "Enabling system-assigned managed identity..."
az webapp identity assign --resource-group $RESOURCE_GROUP --name $APP_NAME

# Configure continuous deployment
echo "Configuring continuous deployment..."
az webapp deployment container config --enable-cd true --name $APP_NAME --resource-group $RESOURCE_GROUP

# Restart the web app to apply changes
echo "Restarting the web app..."
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP

echo "Deployment completed successfully!"
echo "Your application is available at: https://$APP_NAME.azurewebsites.net"

exit 0