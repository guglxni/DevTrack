#!/bin/bash

# Script to start the API server with LLM scoring properly configured

echo "Starting DevTrack Assessment System with LLM Scoring..."

# Stop any existing servers
echo "Stopping any existing servers..."
pkill -f 'uvicorn src.app:app' 2>/dev/null
pkill -f 'node server.js' 2>/dev/null

# Sleep briefly to ensure ports are released
sleep 2

# Set the absolute path to the model file
MODEL_DIR="$(pwd)/models"
MODEL_FILE="${MODEL_DIR}/mistral-7b-instruct-v0.2.Q3_K_S.gguf"

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    echo "Please ensure the Mistral model file is in the models directory"
    exit 1
fi

echo "Using LLM model at: $MODEL_FILE"

# Set environment variables
export LLM_MODEL_PATH="$MODEL_FILE"
export ENABLE_LLM_SCORING=true
export ENABLE_SMART_SCORING=true

# Set scorer weights to prioritize LLM
export SCORE_WEIGHT_LLM=0.7
export SCORE_WEIGHT_KEYWORD=0.1
export SCORE_WEIGHT_EMBEDDING=0.1
export SCORE_WEIGHT_TRANSFORMER=0.1

# Start API server
echo "Starting API server with LLM scoring enabled..."
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8003 > api_server.log 2>&1 &
API_PID=$!

echo "API server process ID: $API_PID"

# Wait for API server to start
echo "Waiting for API server to start..."
MAX_RETRIES=15
RETRY_COUNT=0
API_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8003/health > /dev/null; then
        API_READY=true
        echo "API server is running at http://localhost:8003"
        break
    fi
    echo "API server not responding yet, retrying in 2 seconds..."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ "$API_READY" = false ]; then
    echo "Error: API server failed to respond to health check"
    echo "Showing last 50 lines of the log:"
    tail -n 50 api_server.log
    exit 1
fi

# Check LLM scoring health
echo "Checking LLM scoring health..."
LLM_HEALTH=$(curl -s http://localhost:8003/llm-scoring/health)
echo "LLM scoring status: $LLM_HEALTH"

# Start web application
echo "Starting web application..."
cd webapp && node server.js > ../webapp_server.log 2>&1 &
WEBAPP_PID=$!
cd ..

echo "Web application process ID: $WEBAPP_PID"

# Wait for webapp to start
echo "Waiting for webapp server to start..."
MAX_RETRIES=10
RETRY_COUNT=0
WEBAPP_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3000 > /dev/null; then
        WEBAPP_READY=true
        echo "Web application is running at http://localhost:3000"
        break
    fi
    echo "Web application not responding yet, retrying in 2 seconds..."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ "$WEBAPP_READY" = false ]; then
    echo "Error: Web application failed to respond"
    echo "Showing last 50 lines of the log:"
    tail -n 50 webapp_server.log
    exit 1
fi

echo "DevTrack Assessment System with LLM Scoring is now running!"
echo "API server PID: $API_PID"
echo "Webapp server PID: $WEBAPP_PID"
echo "To stop the servers, run: pkill -f 'uvicorn src.app:app' && pkill -f 'node server.js'"
echo "You can access the web application at: http://localhost:3000"
echo "API documentation is available at: http://localhost:8003/docs" 