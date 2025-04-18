#!/bin/bash
# Script to deploy middleware components (InfluxDB, PostgreSQL, MLflow) using Docker

set -e

# Navigate to the docker directory
cd "$(dirname "$0")/../docker"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install it first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "Deploying middleware components (InfluxDB, PostgreSQL, MLflow)..."

# Create directories for volumes if they don't exist
mkdir -p ./mlflow

# Start the containers
docker-compose -f docker-compose.middleware.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose -f docker-compose.middleware.yml ps | grep -q "Up"; then
    echo "Middleware services are up and running."
    echo "InfluxDB UI: http://localhost:8086"
    echo "MLflow UI: http://localhost:5000"
else
    echo "Error: Some services failed to start. Check the logs with: docker-compose -f docker-compose.middleware.yml logs"
    exit 1
fi

exit 0