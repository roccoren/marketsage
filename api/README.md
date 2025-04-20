# StockScan API

A RESTful API for stock data analysis, predictions, and portfolio management.

## Overview

StockScan API provides a modern interface to the existing StockScan functionality, enabling:

- Stock data acquisition and management
- Predictions with various models and timeframes
- Scheduled tasks for data refresh and predictions
- User authentication and authorization
- Portfolio management
- Alerts and notifications

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL
- Redis (for caching)
- InfluxDB (for time series data)

### Setup

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

5. Initialize the database:

```bash
alembic upgrade head
```

## Running the API

### Development

```bash
uvicorn api.main:app --reload
```

### Production

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the API is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## API Endpoints

### Authentication

- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login user
- `POST /api/auth/refresh` - Refresh JWT token
- `POST /api/auth/logout` - Logout user
- `GET /api/auth/me` - Get current user info

### Stock Data

- `GET /api/stocks` - List available stocks
- `GET /api/stocks/{symbol}` - Get stock details
- `GET /api/stocks/{symbol}/data` - Get stock price data
- `POST /api/stocks/refresh` - Manually refresh stock data

### Predictions

- `GET /api/predictions` - List predictions
- `GET /api/predictions/{id}` - Get prediction details
- `POST /api/predictions` - Create new prediction
- `GET /api/predictions/history` - Get prediction history
- `POST /api/predictions/batch` - Create batch predictions

### Scheduling

- `GET /api/schedules` - List scheduled tasks
- `POST /api/schedules` - Create schedule
- `PUT /api/schedules/{id}` - Update schedule
- `DELETE /api/schedules/{id}` - Delete schedule
- `POST /api/schedules/{id}/execute` - Execute schedule immediately

## Development

### Database Migrations

To create a new migration:

```bash
alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
alembic upgrade head
```

### Testing

Run tests with pytest:

```bash
pytest
```

## License

[MIT License](LICENSE)