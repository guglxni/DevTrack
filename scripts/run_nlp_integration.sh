#!/bin/bash

# Script to run NLP integration and test enhanced functionality
set -e

echo "======================================================"
echo "  Running NLP Integration for ASD Assessment System"
echo "======================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found on your system."
    exit 1
fi

# Check if the required files exist
if [ ! -f "integrate_nlp.py" ]; then
    echo "Error: integrate_nlp.py not found in current directory."
    exit 1
fi

if [ ! -f "advanced_nlp.py" ]; then
    echo "Error: advanced_nlp.py not found in current directory."
    exit 1
fi

echo "Step 1: Making scripts executable..."
chmod +x advanced_nlp.py
chmod +x integrate_nlp.py
chmod +x test_response.sh
chmod +x start_api.sh

echo "Step 2: Running integration script..."
python3 integrate_nlp.py

echo "Step 3: Checking if API server is running..."
if ! curl -s http://localhost:8002/health &> /dev/null; then
    echo "API server is not running. Starting it now..."
    ./start_api.sh
    
    # Wait for server to start
    echo "Waiting for server to start..."
    sleep 10
fi

echo "Step 4: Testing problematic responses..."
echo "Testing 'no, not yet' response:"
./test_response.sh test "walks independently" "no, not yet"

echo "Testing 'not at all, he has never walked independently' response:"
./test_response.sh test "walks independently" "not at all, he has never walked independently"

echo "Testing simple 'not at all' response (should be correctly scored):"
./test_response.sh test "walks independently" "not at all"

echo "======================================================"
echo "  Integration Complete"
echo "======================================================"
echo "The enhanced NLP model is now integrated and should handle"
echo "complex responses with explanations correctly."
echo "Add additional patterns to handle_complex_response() in"
echo "advanced_nlp.py if you find other problematic cases."
echo "======================================================" 