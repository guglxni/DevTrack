#!/bin/bash

# Start the API server with R2R enabled
# This script starts the API server with the R2R (Reason to Retrieve) system enabled

# Set environment variables
export ENABLE_R2R=true
export MISTRAL_API_KEY=""  # Replace with your Mistral API key if available

# Optional: also enable active learning if desired
export ENABLE_ACTIVE_LEARNING=true

# Start the API server
echo "Starting API server with R2R enabled..."

# Get the absolute path to the Python interpreter
PYTHON_PATH=$(which python3)

# Kill any existing uvicorn processes
pkill -f "uvicorn src.app:app" || true

# Wait for processes to terminate
sleep 2

# Start the API server with R2R enabled
ENABLE_R2R=true ENABLE_ACTIVE_LEARNING=true $PYTHON_PATH -m uvicorn src.app:app --port 8003

# Exit with the status of the last command
exit $?

# The R2R dashboard will be available at http://localhost:8003/r2r-dashboard/ 