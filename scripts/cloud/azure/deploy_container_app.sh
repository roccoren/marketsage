#!/bin/bash
# Script to deploy the MarketSage Dash application to Azure Container Apps

set -e

# Navigate to the project root directory
cd "$(dirname "$0")/../../.."

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "Error: Azure CLI is not installed. Please install it first."
    echo "Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if Azure Container Apps extension is installed
if ! az extension show --name containerapp &> /dev/null; then
    echo "Installing Azure Container Apps extension..."
    az extension add --name containerapp --upgrade
fi

# Check if user is logged in to Azure
if ! az account show &> /dev/null; then
    echo "You are not logged in to Azure. Please login first."
    az login
fi

# Configuration variables - modify these as needed
RESOURCE_GROUP="marketsage-rg"
LOCATION="eastus"
CONTAINER_APP_ENV="marketsage-env"
CONTAINER_APP_NAME="marketsage-dash"
REGISTRY_NAME="marketsageregistry"
IMAGE_NAME="marketsage-dash"
IMAGE_TAG="latest"

# External middleware connection settings - update these with your actual endpoints
INFLUXDB_URL="https://your-influxdb-endpoint:8086"
INFLUXDB_TOKEN="your-influxdb-token"
INFLUXDB_ORG="marketsage"
INFLUXDB_BUCKET="stock_data"
MLFLOW_TRACKING_URI="https://your-mlflow-endpoint:5000"

# Ask for confirmation
echo "This script will deploy the MarketSage Dash application to Azure Container Apps with the following configuration:"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Container App Environment: $CONTAINER_APP_ENV"
echo "Container App Name: $CONTAINER_APP_NAME"
echo "Container Registry: $REGISTRY_NAME"
echo "Image Name: $IMAGE_NAME:$IMAGE_TAG"
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

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry if it doesn't exist..."
az acr create --resource-group $RESOURCE_GROUP --name $REGISTRY_NAME --sku Basic

# Log in to ACR
echo "Logging in to Azure Container Registry..."
az acr login --name $REGISTRY_NAME

# Build and push Docker image to ACR
echo "Building and pushing Docker image to ACR..."
az acr build --registry $REGISTRY_NAME --image $IMAGE_NAME:$IMAGE_TAG --file scripts/cloud/azure/Dockerfile .

# Create Container App Environment if it doesn't exist
echo "Creating Container App Environment if it doesn't exist..."
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Create Container App
echo "Creating Container App..."
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image "$REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" \
  --target-port 8050 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --env-vars \
    "CONFIG_FILE=/app/config/default_config.yaml" \
    "INFLUXDB_URL=$INFLUXDB_URL" \
    "INFLUXDB_TOKEN=$INFLUXDB_TOKEN" \
    "INFLUXDB_ORG=$INFLUXDB_ORG" \
    "INFLUXDB_BUCKET=$INFLUXDB_BUCKET" \
    "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"

# Enable managed identity
echo "Enabling managed identity..."
az containerapp identity assign \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --system-assigned

# Get the application URL
APP_URL=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "Deployment completed successfully!"
echo "Your application is available at: https://$APP_URL"

exit 0