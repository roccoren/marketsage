#!/bin/bash
# Script to deploy Dash application using Docker

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

# Check if middleware network exists
if ! docker network ls | grep -q "middleware-network"; then
    echo "Error: middleware-network not found. Please deploy middleware components first with scripts/deploy_middleware_docker.sh"
    exit 1
fi

echo "Deploying Dash application..."

# Build and start the container
docker-compose -f docker-compose.dash.yml up -d --build

# Wait for service to be ready
echo "Waiting for service to be ready..."
sleep 5

# Check if service is running
if docker-compose -f docker-compose.dash.yml ps | grep -q "Up"; then
    echo "Dash application is up and running."
    echo "Dash UI: http://localhost:8050"
else
    echo "Error: Dash application failed to start. Check the logs with: docker-compose -f docker-compose.dash.yml logs"
    exit 1
fi

exit 0