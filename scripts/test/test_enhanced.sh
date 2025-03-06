#!/bin/bash

# Enhanced ASD Assessment API Testing Tool
# This script provides an easy way to test the enhanced scoring model

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MILESTONE="selects and brings familiar objects from another room when asked"
SERVER_PORT=8002
SERVER_PID=""
SERVER_STARTED=false

# Function to display help
show_help() {
    echo -e "${GREEN}ASD Assessment API Enhanced Testing Tool${NC}"
    echo "This tool provides simplified testing of the ASD Assessment API with enhanced scoring."
    echo 
    echo "Usage:"
    echo "  ./test_enhanced.sh [command] [options]"
    echo
    echo "Commands:"
    echo "  start          Start the API server with enhanced scoring"
    echo "  test [response]  Test a specific response to the current milestone"
    echo "  milestone [text]  Set a specific milestone to test against (in quotes)"
    echo "  stop           Stop the API server"
    echo "  help           Display this help message"
    echo
    echo "Examples:"
    echo "  ./test_enhanced.sh start                     # Start the enhanced API server"
    echo "  ./test_enhanced.sh test \"yes, always\"        # Test a positive response"
    echo "  ./test_enhanced.sh test \"no\"                 # Test a negative response"
    echo "  ./test_enhanced.sh test \"sometimes with help\" # Test a partial response"
    echo "  ./test_enhanced.sh stop                      # Stop the API server"
    echo
}

# Function to start the server
start_server() {
    echo -e "${BLUE}Starting enhanced API server...${NC}"
    
    # Run enhance_model.py script first if it exists
    if [ -f "enhance_model.py" ]; then
        echo -e "${BLUE}Applying model enhancements...${NC}"
        python3 enhance_model.py
    else
        echo -e "${YELLOW}Warning: enhance_model.py not found. Server will run without enhancements.${NC}"
    fi
    
    # Check if start_api.sh exists
    if [ -f "start_api.sh" ]; then
        echo -e "${BLUE}Using start_api.sh to launch server...${NC}"
        ./start_api.sh &
        SERVER_PID=$!
    else
        echo -e "${BLUE}Starting API with uvicorn directly...${NC}"
        # Python command check
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            echo -e "${RED}Error: No Python command found. Please install Python 3.${NC}"
            exit 1
        fi
        
        # Start the server
        $PYTHON_CMD -m uvicorn app:app --host 0.0.0.0 --port $SERVER_PORT &
        SERVER_PID=$!
    fi
    
    echo -e "${GREEN}Server started with PID: $SERVER_PID${NC}"
    echo -e "${GREEN}Server is running at http://localhost:$SERVER_PORT${NC}"
    SERVER_STARTED=true
    
    # Give the server time to start up
    sleep 3
}

# Function to stop the server
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo -e "${BLUE}Stopping server with PID: $SERVER_PID${NC}"
        kill -15 $SERVER_PID
        SERVER_PID=""
        SERVER_STARTED=false
        echo -e "${GREEN}Server stopped${NC}"
    else
        echo -e "${YELLOW}No server PID found. Trying to find and kill uvicorn processes...${NC}"
        pkill -f "uvicorn app:app"
        echo -e "${GREEN}Any running servers have been stopped${NC}"
    fi
}

# Function to test a response
test_response() {
    response="$1"
    
    if [ -z "$response" ]; then
        echo -e "${RED}Error: No response provided. Use: ./test_enhanced.sh test \"your response\"${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Testing response against milestone:${NC}"
    echo -e "${YELLOW}\"$MILESTONE\"${NC}"
    echo -e "${BLUE}Caregiver response:${NC}"
    echo -e "${YELLOW}\"$response\"${NC}"
    
    # Use curl to call the API
    echo -e "${BLUE}Sending request to API...${NC}"
    curl_result=$(curl -s -X POST "http://localhost:$SERVER_PORT/score-response" \
        -H "Content-Type: application/json" \
        -d "{\"milestone\": \"$MILESTONE\", \"response\": \"$response\"}")
    
    # Check if curl was successful
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Could not connect to the API server. Make sure it's running.${NC}"
        return 1
    fi
    
    # Parse the JSON response for better formatting
    if command -v jq &> /dev/null; then
        # Pretty print with jq if available
        echo -e "${GREEN}API Response:${NC}"
        echo "$curl_result" | jq
        
        # Extract score and label for highlighting
        score=$(echo "$curl_result" | jq -r '.score')
        label=$(echo "$curl_result" | jq -r '.score_label')
        
        # Display a more user-friendly summary based on the score
        echo -e "${BLUE}Summary:${NC}"
        case $score in
            4)
                echo -e "${GREEN}✓ INDEPENDENT (4):${NC} Child performs the skill independently without help"
                ;;
            3)
                echo -e "${BLUE}✓ WITH_SUPPORT (3):${NC} Child can perform with assistance or prompting"
                ;;
            2)
                echo -e "${YELLOW}⚠ EMERGING (2):${NC} Child is beginning to develop this skill, but inconsistently"
                ;;
            1)
                echo -e "${YELLOW}⚠ NOT_YET (1):${NC} Child is not yet demonstrating this skill"
                ;;
            0)
                echo -e "${RED}✗ CANNOT_DO (0):${NC} Child is definitely unable to perform this skill"
                ;;
            *)
                echo -e "${RED}Unknown score: $score${NC}"
                ;;
        esac
    else
        # Basic output without jq
        echo -e "${GREEN}API Response:${NC} $curl_result"
    fi
}

# Function to set a specific milestone
set_milestone() {
    new_milestone="$1"
    
    if [ -z "$new_milestone" ]; then
        echo -e "${RED}Error: No milestone provided. Use: ./test_enhanced.sh milestone \"milestone text\"${NC}"
        return 1
    fi
    
    MILESTONE="$new_milestone"
    echo -e "${GREEN}Milestone set to:${NC}"
    echo -e "${YELLOW}\"$MILESTONE\"${NC}"
}

# Main script logic
case "$1" in
    "start")
        start_server
        ;;
    "stop")
        stop_server
        ;;
    "test")
        test_response "$2"
        ;;
    "milestone")
        set_milestone "$2"
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        echo -e "${YELLOW}No command provided.${NC}"
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        ;;
esac 