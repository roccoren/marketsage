# MLflow Docker Setup

This directory contains Docker Compose configuration for setting up MLflow with PostgreSQL for tracking storage.

## Prerequisites

- Docker and Docker Compose installed on your system
- Basic understanding of Docker and MLflow

## Components

1. **PostgreSQL Database**: Stores MLflow tracking data
2. **MLflow Server**: Provides the MLflow tracking server and UI

## Getting Started

### Starting the Services

1. Navigate to this directory:
   ```bash
   cd docker
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Verify the services are running:
   ```bash
   docker-compose ps
   ```

### Accessing MLflow UI

Once the services are running, you can access the MLflow UI at:
```
http://localhost:5000
```

### Using MLflow with Your Project

To use this MLflow server with your project, set the tracking URI:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

Or when running your scripts:

```bash
python examples/workflow_example.py --mlflow-tracking-uri=http://localhost:5000
```

## Configuration

### Environment Variables

You can modify the following environment variables in the `docker-compose.yml` file:

- `POSTGRES_USER`: PostgreSQL username (default: mlflow)
- `POSTGRES_PASSWORD`: PostgreSQL password (default: mlflow)
- `POSTGRES_DB`: PostgreSQL database name (default: mlflow)

### Volumes

The setup uses the following Docker volumes:

- `postgres_data`: Persists PostgreSQL data
- `mlflow_artifacts`: Stores MLflow artifacts

## Stopping the Services

To stop the services:

```bash
docker-compose down
```

To stop the services and remove the volumes:

```bash
docker-compose down -v
```

## Troubleshooting

### Database Connection Issues

If MLflow cannot connect to the database, check:

1. The PostgreSQL service is running:
   ```bash
   docker-compose ps postgres
   ```

2. The database credentials in `docker-compose.yml` match what MLflow is using.

### MLflow Server Issues

If the MLflow server is not starting properly:

1. Check the logs:
   ```bash
   docker-compose logs mlflow
   ```

2. Ensure the PostgreSQL service is fully initialized before MLflow tries to connect.

## Security Note

This setup is intended for development and testing. For production use:

1. Change the default PostgreSQL credentials
2. Set up proper authentication for MLflow
3. Consider using HTTPS for the MLflow server
4. Implement proper backup strategies for the volumes