# RD Agent

A Docker-based RD Agent service for stock market analysis.

## Setup

1. Make sure Docker and Docker Compose are installed on your system.
2. Configure the environment variables in the `.env` file.

## Running with Docker

### Running the API Server

To build and run the RD Agent API server:

```bash
# Build the Docker image
docker-compose build

# Run the container
docker-compose up
```

To run in detached mode:

```bash
docker-compose up -d
```

### Running Different Commands

You can run different RD Agent commands using the following syntax:

```bash
# Run fin_agent
docker-compose run rdagent python -m rdagent.fin_agent

# Run any other command
docker-compose run rdagent <your-command>
```

For example:
```bash
# Run a specific script
docker-compose run rdagent python -m rdagent.scripts.your_script

# Run with specific arguments
docker-compose run rdagent python -m rdagent.fin_agent --arg1 value1 --arg2 value2
```

## API Endpoints

- `GET /`: Root endpoint that confirms the API is running
- `GET /health`: Health check endpoint

## Environment Variables

Configuration is done via environment variables in the `.env` file:

- `RDAGENT_PORT`: Port to run the service on (default: 8000)
- `RDAGENT_HOST`: Host to bind to (default: 0.0.0.0)
- `RDAGENT_LOG_LEVEL`: Logging level (default: INFO)
- `RDAGENT_DATA_DIR`: Directory for storing data (default: /app/data)
- `RDAGENT_LOG_DIR`: Directory for storing logs (default: /app/logs)

Additional Azure OpenAI configuration variables are also required. 