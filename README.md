# StockScan API

A modern RESTful API for stock data analysis, predictions, and portfolio management, built on top of the existing StockScan functionality.

## Features

- **Stock Data Management**: Fetch, store, and retrieve stock data
- **Predictions**: Generate predictions using various models and timeframes
- **Scheduling**: Schedule data refresh and prediction tasks
- **Authentication**: Secure user authentication and authorization
- **Portfolio Management**: Manage stock portfolios and positions
- **Alerts**: Set up and receive alerts based on various conditions

## Project Structure

```
stockscan/
├── api/                   # API code
│   ├── core/              # Core modules
│   ├── db/                # Database models and utilities
│   ├── models/            # Pydantic models
│   ├── routers/           # API endpoints
│   ├── schemas/           # Request/response schemas
│   ├── services/          # Business logic
│   └── utils/             # Utility functions
├── alembic/               # Database migrations
├── docker/                # Docker configuration
├── examples/              # Example code
├── modules/               # Existing StockScan modules
├── tests/                 # Test code
├── .env                   # Environment variables
├── alembic.ini            # Alembic configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── run_api.py             # API runner script
├── run_dev.py             # Development runner script
└── setup.py               # Setup script
```

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL
- Redis (for caching)
- InfluxDB (for time series data)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd stockscan
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Copy the `.env.example` file to `.env` and update the values:

```bash
cp .env.example .env
```

5. Run the setup script:

```bash
python setup.py
```

### Running the API

#### Development Mode

```bash
python run_dev.py
```

#### Production Mode

```bash
python run_api.py
```

### Using Docker

```bash
docker-compose up -d
```

## API Documentation

Once the API is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Development

### Running Tests

```bash
pytest
```

### Database Migrations

To create a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
alembic upgrade head
```

## Example Client

An example client is provided in `examples/api_client_example.py`. You can run it with:

```bash
python examples/api_client_example.py
```

## License

[MIT License](LICENSE)