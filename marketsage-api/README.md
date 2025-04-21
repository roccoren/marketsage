# MarketSage API

MarketSage API is a RESTful API service that provides a unified interface for the MarketSage ecosystem. It serves as the central communication hub between the different components and the user interface.

## Overview

The MarketSage API provides endpoints for:

- Authentication and authorization
- Stock data access and management
- Prediction generation and retrieval
- Task scheduling and management
- Integration with both DataHub and MarketSage Core

## Architecture

The API is built using FastAPI, a modern, fast web framework for building APIs with Python. It follows a modular architecture with the following components:

- **Routers**: Handle HTTP requests and responses
- **Schemas**: Define data models and validation
- **Services**: Implement business logic and integration with other components
- **Core**: Provide configuration, dependencies, and security

## Directory Structure

```
marketsage-api/
├── src/
│   ├── core/         # Core functionality (config, dependencies, security)
│   ├── routers/      # API route handlers
│   ├── schemas/      # Pydantic models for request/response validation
│   ├── services/     # Business logic and integration with other components
│   └── utils/        # Utility functions
├── tests/            # Unit and integration tests
├── alembic/          # Database migrations
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL
- Access to other MarketSage components (DataHub, MarketSage Core)

### Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root directory with the following variables:

```
# API Settings
PROJECT_NAME=MarketSage API
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=marketsage
POSTGRES_PORT=5432

# CORS
CORS_ORIGINS=http://localhost,http://localhost:3000

# Component Integration
DATAHUB_URL=http://localhost:8001
MARKETSAGE_CORE_URL=http://localhost:8002
TSDS_HOST=localhost
TSDS_PORT=50051

# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
```

### Database Setup

Initialize the database and run migrations:

```bash
# Initialize Alembic
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Run migrations
alembic upgrade head
```

### Running the API

```bash
# Development mode
uvicorn src.main:app --reload

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Docker

Build and run with Docker:

```bash
# Build the image
docker build -t marketsage-api .

# Run the container
docker run -p 8000:8000 --env-file .env marketsage-api
```

## API Documentation

When the API is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Development

### Running Tests

```bash
pytest
```

### Code Style

This project uses Black for code formatting and isort for import sorting:

```bash
# Format code
black src tests

# Sort imports
isort src tests
```

## Integration with Other Components

The MarketSage API integrates with other components in the following ways:

1. **DataHub**: Retrieves stock data and manages symbols through REST API calls and gRPC
2. **MarketSage Core**: Submits analysis tasks and retrieves results through REST API calls and message queue
3. **Time Series Data Service**: Accesses time-series data through gRPC
4. **MarketSage Portal**: Provides REST API endpoints for the web interface

## Message Queue Integration

The API uses RabbitMQ for asynchronous communication with other components:

- **Producers**: Send messages to request data acquisition, analysis tasks, etc.
- **Consumers**: Receive notifications about completed tasks, data updates, etc.

## License

[MIT License](LICENSE)