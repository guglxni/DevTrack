#!/bin/bash

# Start the API server with Active Learning enabled
# This script starts the API server with the Active Learning system enabled

# Set the environment variable to enable Active Learning
export ENABLE_ACTIVE_LEARNING=true

# Start the API server
echo "Starting API server with Active Learning enabled..."
python3 -m uvicorn src.app:app --port 8003

# Note: The feedback interface will be available at http://localhost:8003/feedback/ 