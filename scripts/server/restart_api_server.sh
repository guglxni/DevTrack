#!/bin/bash
# Restart script for the API server with improved LLM scoring

echo "Restarting the API server with improved LLM scoring..."

# Define the model path with absolute path
MISTRAL_MODEL="/Volumes/MacExt/asd-assessment-api/models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"

# Make sure we're in the right directory
cd /Volumes/MacExt/asd-assessment-api

# Stop any existing API server
echo "Stopping any existing API server processes..."
pkill -f "uvicorn src.app:app" || true

# Wait to ensure processes are fully terminated
sleep 2

# Set up environment variables
export LLM_MODEL_PATH="${MISTRAL_MODEL}"
export ENABLE_LLM_SCORING=true
export ENABLE_SMART_SCORING=true

# Scoring weights for ensemble
export SCORE_WEIGHT_KEYWORD=0.2
export SCORE_WEIGHT_TRANSFORMER=0.2
export SCORE_WEIGHT_LLM=0.6

# Start the server
echo "Starting API server with improved LLM scoring..."
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
    echo
    echo "To test the direct LLM scoring endpoint, use:"
    echo "curl -X POST -H \"Content-Type: application/json\" -d '{\"response\":\"My child does this all the time\",\"milestone_behavior\":\"Points to ask for something\",\"domain\":\"SOC\",\"age_range\":\"12-24 months\"}' http://localhost:8003/llm-scoring/score"
    echo
    echo "To test the /direct-test endpoint, use:"
    echo "curl -X POST -H \"Content-Type: application/json\" -d '{\"question\":\"Does your child point to things to ask for them?\",\"milestone\":\"Points to ask for something\",\"response\":\"Yes, my child does this frequently\"}' http://localhost:8003/direct-test"
else
    echo "Warning: LLM scoring endpoint is not available."
    echo "Showing last 50 lines of the log:"
    tail -n 50 api_server.log
fi 