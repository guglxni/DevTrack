#!/bin/bash

# Start the API server with advanced scoring enabled and the web application
echo "Starting DevTrack Assessment System with Advanced Scoring..."

# Stop any existing servers
echo "Stopping any existing servers..."
pkill -f "uvicorn src.app:app" || true
pkill -f "node server.js" || true

# Set environment variables to enable advanced features
export ENABLE_R2R=true
export ENABLE_ACTIVE_LEARNING=true
export ENABLE_LLM_SCORING=true

# Set environment variables for remote LLM API (uncomment and set one of these if you have an API key)
# export OPENAI_API_KEY="your-openai-api-key"
# export MISTRAL_API_KEY="your-mistral-api-key"

# Configure the API endpoint (default is Mistral AI)
export LLM_API_BASE="https://api.mistral.ai/v1"
export LLM_API_MODEL="mistral-small"

# Start the API server
echo "Starting API server with advanced scoring enabled..."
python3 -m uvicorn src.app:app --port 8003 > api_server.log 2>&1 &
API_PID=$!

# Wait for the API server to start
echo "Waiting for API server to start..."
sleep 5

# Check if the API server is running
if curl -s http://localhost:8003/health > /dev/null; then
    echo "API server is running at http://localhost:8003"
else
    echo "Error: API server failed to start"
    cat api_server.log
    exit 1
fi

# Start the web application
echo "Starting web application..."
cd webapp && node server.js > ../webapp_server.log 2>&1 &
WEBAPP_PID=$!

# Wait for the web application to start
echo "Waiting for webapp server to start..."
sleep 5

# Check if the web application is running
if curl -s http://localhost:3000 > /dev/null; then
    echo "Web application is running at http://localhost:3000"
else
    echo "Error: Web application failed to start"
    cat webapp_server.log
    exit 1
fi

echo "DevTrack Assessment System with Advanced Scoring is now running!"
echo "API server PID: $API_PID"
echo "Webapp server PID: $WEBAPP_PID"
echo "To stop the servers, run: pkill -f 'uvicorn src.app:app' && pkill -f 'node server.js'"

# Create scripts to open the web application and API documentation
echo "open http://localhost:3000" > open_webapp.sh
echo "open http://localhost:8003/docs" > open_api_docs.sh
chmod +x open_webapp.sh open_api_docs.sh

echo "You can run ./open_webapp.sh to open the web application in your browser"
echo "You can run ./open_api_docs.sh to open the API documentation in your browser" 
 