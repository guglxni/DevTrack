#!/bin/bash
# Script to start the ASD Assessment API and the demo web application

# Display a banner
echo "===================================================="
echo "    ASD Assessment API Demo - Startup Script        "
echo "===================================================="
echo ""

# Check if API is already running
if nc -z localhost 8003 2>/dev/null; then
    echo "✅ API server already running at http://localhost:8003"
else
    echo "🚀 Starting API server..."
    # Start the API server in the background
    cd src/api && uvicorn main:app --reload --port 8003 &
    API_PID=$!
    
    # Wait a moment for the API to start
    sleep 2
    
    # Check if API server started successfully
    if nc -z localhost 8003 2>/dev/null; then
        echo "✅ API server started at http://localhost:8003"
    else
        echo "❌ Failed to start API server"
        exit 1
    fi
fi

# Navigate to the web application directory
cd "$(dirname "$0")/webapp"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js to run the demo application."
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the web application
echo "🚀 Starting web application..."
npm start

# Clean up on exit
trap 'kill $API_PID 2>/dev/null' EXIT 