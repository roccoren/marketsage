#!/bin/bash

# Activate virtual environment if it exists
if [ -d "/app/venv" ]; then
    source /app/venv/bin/activate
fi

# Default command if none provided
if [ $# -eq 0 ]; then
    exec python -m rdagent.main
else
    # Execute the provided command
    exec "$@"
fi 