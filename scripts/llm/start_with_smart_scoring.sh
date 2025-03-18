#!/bin/bash

# Start the API server with smart scoring enabled and the web application
echo "Starting DevTrack Assessment System with Smart Scoring..."

# Stop any existing servers
echo "Stopping any existing servers..."
pkill -f "uvicorn src.app:app" || true
pkill -f "node server.js" || true

# Make sure the port is free
echo "Checking if port 8003 is in use..."
if lsof -i :8003 > /dev/null; then
    echo "Port 8003 is still in use. Trying to force kill the process..."
    lsof -i :8003 -t | xargs kill -9
    sleep 2
fi

# Check if port 3000 is in use
echo "Checking if port 3000 is in use..."
if lsof -i :3000 > /dev/null; then
    echo "Port 3000 is still in use. Trying to force kill the process..."
    lsof -i :3000 -t | xargs kill -9
    sleep 2
fi

# Set environment variables to enable advanced features
echo "Setting environment variables..."
export ENABLE_R2R=true
export ENABLE_ACTIVE_LEARNING=true
export ENABLE_LLM_SCORING=true
export ENABLE_SMART_SCORING=true
export ENABLE_CONTINUOUS_LEARNING=true

# Set the path to the Mistral model
export LLM_MODEL_PATH="$(pwd)/models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"
echo "Using Mistral model at: $LLM_MODEL_PATH"

# Configure scoring weights to prioritize LLM and embedding scoring over keyword scoring
export SCORE_WEIGHT_KEYWORD=0.2
export SCORE_WEIGHT_EMBEDDING=0.3
export SCORE_WEIGHT_TRANSFORMER=0.2
export SCORE_WEIGHT_LLM=0.5

# Set environment variables for remote LLM API (uncomment and set one of these if you have an API key)
# export OPENAI_API_KEY="your-openai-api-key"
# export MISTRAL_API_KEY="your-mistral-api-key"

# Configure the API endpoint (default is Mistral AI)
export LLM_API_BASE="https://api.mistral.ai/v1"
export LLM_API_MODEL="mistral-small"

# Start the API server
echo "Starting API server with smart scoring enabled..."
python3 -m uvicorn src.app:app --port 8003 > api_server.log 2>&1 &
API_PID=$!
echo "API server process ID: $API_PID"

# Wait for the API server to start
echo "Waiting for API server to start..."
sleep 15  # Increased wait time

# Check if the API server process is still running
if ! ps -p $API_PID > /dev/null; then
    echo "Error: API server process is not running"
    echo "Showing last 50 lines of the log:"
    tail -n 50 api_server.log
    exit 1
fi

# Try multiple times to check if the API server is running
MAX_RETRIES=5
RETRY_COUNT=0
API_RUNNING=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8003/health > /dev/null; then
        API_RUNNING=true
        break
    else
        echo "API server not responding yet, retrying in 2 seconds..."
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ "$API_RUNNING" = true ]; then
    echo "API server is running at http://localhost:8003"
else
    echo "Error: API server failed to respond to health check"
    echo "Showing last 50 lines of the log:"
    tail -n 50 api_server.log
    echo "The API server process is still running, continuing with startup..."
fi

# Check if LLM scoring is available
if curl -s http://localhost:8003/llm-scoring/health > /dev/null; then
    LLM_STATUS=$(curl -s http://localhost:8003/llm-scoring/health)
    echo "LLM scoring is available: $LLM_STATUS"
else
    echo "Warning: LLM scoring endpoint is not available. Using fallback scoring methods."
fi

# Start the web application
echo "Starting web application..."
cd webapp && node server.js > ../webapp_server.log 2>&1 &
WEBAPP_PID=$!
echo "Web application process ID: $WEBAPP_PID"

# Wait for the web application to start
echo "Waiting for webapp server to start..."
sleep 10  # Increased wait time

# Check if the web application process is still running
if ! ps -p $WEBAPP_PID > /dev/null; then
    echo "Error: Web application process is not running"
    echo "Showing last 50 lines of the log:"
    tail -n 50 webapp_server.log
    exit 1
fi

# Try multiple times to check if the web application is running
MAX_RETRIES=5
RETRY_COUNT=0
WEBAPP_RUNNING=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3000 > /dev/null; then
        WEBAPP_RUNNING=true
        break
    else
        echo "Web application not responding yet, retrying in 2 seconds..."
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ "$WEBAPP_RUNNING" = true ]; then
    echo "Web application is running at http://localhost:3000"
else
    echo "Error: Web application failed to respond to health check"
    echo "Showing last 50 lines of the log:"
    tail -n 50 webapp_server.log
    echo "The web application process is still running, continuing with startup..."
fi

echo "DevTrack Assessment System with Smart Scoring is now running!"
echo "API server PID: $API_PID"
echo "Webapp server PID: $WEBAPP_PID"
echo "To stop the servers, run: pkill -f 'uvicorn src.app:app' && pkill -f 'node server.js'"

# Create scripts to open the web application and API documentation
echo "open http://localhost:3000" > open_webapp.sh
echo "open http://localhost:8003/docs" > open_api_docs.sh
chmod +x open_webapp.sh open_api_docs.sh

echo "You can run ./open_webapp.sh to open the web application in your browser"
echo "You can run ./open_api_docs.sh to open the API documentation in your browser"

echo ""
echo "IMPORTANT: For best results with LLM scoring:"
echo "1. Install dependencies: ./scripts/install_llm_dependencies.sh"
echo "2. Download a model: python3 scripts/download_model.py"
echo "3. Set the LLM_MODEL_PATH in this script to point to your downloaded model"
echo "   or set an API key for OpenAI or Mistral AI" 