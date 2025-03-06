#!/bin/bash

# Kill any existing processes using port 8003
echo "Checking for existing processes using port 8003..."
PID=$(lsof -i :8003 | grep LISTEN | awk '{print $2}')
if [ ! -z "$PID" ]; then
    echo "Killing process $PID using port 8003..."
    kill -9 $PID
    sleep 1
fi

# Start the API server
echo "Starting API server..."
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m uvicorn src.api.app:app --port 8003 &
SERVER_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8003/health > /dev/null; then
        echo "Server is up and running!"
        break
    fi
    echo "Waiting for server to start... (attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Server failed to start after $MAX_RETRIES attempts."
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# Run the tests
echo "Running tests..."
./test_end_to_end.py

# Cleanup
echo "Cleaning up..."
kill -9 $SERVER_PID 2>/dev/null
echo "Done!" 