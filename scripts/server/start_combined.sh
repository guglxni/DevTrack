#!/bin/bash

echo "Starting DevTrack Assessment System..."

# Kill any existing processes
echo "Stopping any existing servers..."
pkill -f "uvicorn src.app:app" || true
pkill -f "node server.js" || true

# Start the API server with R2R and Active Learning enabled
echo "Starting API server with R2R and Active Learning enabled..."
ENABLE_R2R=true ENABLE_ACTIVE_LEARNING=true python3 -m uvicorn src.app:app --port 8003 > api_server.log 2>&1 &
API_PID=$!

# Wait for the API server to start
echo "Waiting for API server to start..."
sleep 5

# Check if API server is running
if ! curl -s http://localhost:8003/health > /dev/null; then
  echo "Error: API server failed to start. Check api_server.log for details."
  exit 1
fi

echo "API server is running at http://localhost:8003"

# Start the webapp server
echo "Starting web application..."
cd webapp && node server.js > ../webapp_server.log 2>&1 &
WEBAPP_PID=$!

# Wait for the webapp server to start
echo "Waiting for webapp server to start..."
sleep 3

echo "Web application is running at http://localhost:3000"
echo "DevTrack Assessment System is now running!"
echo "API server PID: $API_PID"
echo "Webapp server PID: $WEBAPP_PID"
echo "To stop the servers, run: pkill -f 'uvicorn src.app:app' && pkill -f 'node server.js'" 