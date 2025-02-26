# Stage 1: Builder
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY config/requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -r stockscanner && \
    mkdir -p /app/data /app/output /app/logs && \
    chown -R stockscanner:stockscanner /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code with proper structure
COPY --chown=stockscanner:stockscanner src/ /app/src/
COPY --chown=stockscanner:stockscanner config/ /app/config/
COPY --chown=stockscanner:stockscanner setup.py /app/

# Switch to non-root user
USER stockscanner

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output \
    LOG_DIR=/app/logs \
    PYTHONPATH=/app

# Volume configuration for persistent data
VOLUME ["/app/data", "/app/output", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose port if needed (for future API endpoints)
EXPOSE 8080

# Install package in development mode
RUN pip install -e .

# Entrypoint
ENTRYPOINT ["python", "src/main.py"]

# Default command (can be overridden)
CMD ["--strategies", "rsi_stoch,atr_ma,adx_macd_cci,obv"]