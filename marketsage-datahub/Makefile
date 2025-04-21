# MarketSage DataHub Makefile

.PHONY: help install test lint format clean build docker-build docker-run

# Default target
help:
	@echo "MarketSage DataHub Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install    Install dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linting"
	@echo "  make format     Format code"
	@echo "  make clean      Clean up build artifacts"
	@echo "  make build      Build package"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	pytest tests/

# Run linting
lint:
	flake8 src/ tests/
	mypy src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python setup.py sdist bdist_wheel

# Build Docker image
docker-build:
	docker build -t marketsage-datahub .

# Run Docker container
docker-run:
	docker-compose up