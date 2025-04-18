FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV CONFIG_FILE=/app/config/default_config.yaml
ENV PORT=8050

# Expose port
EXPOSE 8050

# Run the application with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT dash_app:server