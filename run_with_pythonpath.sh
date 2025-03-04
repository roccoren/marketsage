#!/bin/bash

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PYTHONPATH:/home/roccoren/stockscan"

# Execute the passed command
"$@"
