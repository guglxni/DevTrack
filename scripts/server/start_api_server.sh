#!/bin/bash

# Start the API server
echo "Starting ASD Assessment API server..."
echo "API will be available at http://localhost:8003"

# Get the absolute path to the Python interpreter
PYTHON_PATH=$(which python3)

# Start the API server
$PYTHON_PATH -m uvicorn src.api.app:app --host 0.0.0.0 --port 8003

# Exit with the status of the last command
exit $?
 