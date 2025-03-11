#!/bin/bash

# Set PYTHONPATH to include the project root
export PYTHONPATH="/Volumes/External Data/workspaces/stockscan"

# Pass all arguments to the main script
python -m src.main "$@"