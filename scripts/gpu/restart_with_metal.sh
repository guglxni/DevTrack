#!/bin/bash

# Script to restart the API server with Metal GPU acceleration for Apple Silicon

echo "Restarting the integrated DevTrack Assessment API server with Metal GPU acceleration..."

# Stop any existing servers
echo "Stopping any existing servers..."
pkill -f "uvicorn src.app:app" || true
pkill -f "node server.js" || true

# Wait to ensure processes are fully terminated
sleep 2

# Set up environment variables
export ENABLE_R2R=true
export ENABLE_ACTIVE_LEARNING=true
export ENABLE_LLM_SCORING=true
export ENABLE_SMART_SCORING=true
export ENABLE_CONTINUOUS_LEARNING=true

# Set the path to the Mistral model
export LLM_MODEL_PATH="$(pwd)/models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"
echo "Using Mistral model at: $LLM_MODEL_PATH"

# Enable Metal GPU acceleration for Apple Silicon
export USE_METAL=true
echo "Metal GPU acceleration enabled for Apple Silicon"

# Configure LLM to use Metal
export N_GPU_LAYERS=-1  # Use all layers on GPU
export METAL_DEVICE=true
export F16_KV=true      # Use half-precision for key/value cache

# Configure scoring weights
export SCORE_WEIGHT_KEYWORD=0.2
export SCORE_WEIGHT_EMBEDDING=0.3
export SCORE_WEIGHT_TRANSFORMER=0.2
export SCORE_WEIGHT_LLM=0.5

# Start the server
echo "Starting integrated API server with Metal GPU acceleration..."
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8003 > api_server.log 2>&1 &
SERVER_PID=$!

echo "API server started with PID: $SERVER_PID"

# Wait for the server to start
echo "Waiting for the server to start..."
MAX_RETRIES=15
RETRY_COUNT=0
SERVER_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8003/health > /dev/null; then
        SERVER_READY=true
        echo "API server is running!"
        break
    fi
    echo "API server not responding yet, retrying in 2 seconds... (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
    
    # Check if the process is still running
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Error: API server process has terminated. Check api_server.log for details."
        tail -n 50 api_server.log
        exit 1
    fi
done

if [ "$SERVER_READY" = false ]; then
    echo "Error: API server failed to respond to health check after $MAX_RETRIES attempts."
    echo "Showing last 50 lines of the log:"
    tail -n 50 api_server.log
    exit 1
fi

# Check if LLM scoring is available
echo "Checking LLM scoring endpoint..."
if curl -s http://localhost:8003/llm-scoring/health > /dev/null; then
    LLM_STATUS=$(curl -s http://localhost:8003/llm-scoring/health)
    echo "LLM scoring status: $LLM_STATUS"
else
    echo "Warning: LLM scoring endpoint is not available."
    echo "Showing last 50 lines of the log:"
    tail -n 50 api_server.log
fi

echo ""
echo "DevTrack Assessment API with integrated webapp and Metal GPU acceleration is now running!"
echo "API server PID: $SERVER_PID"
echo "You can access the web interface at: http://localhost:8003"
echo "API documentation is available at: http://localhost:8003/docs"
echo ""
echo "To stop the server, run: pkill -f 'uvicorn src.app:app'" 