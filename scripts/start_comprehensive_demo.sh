#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting ASD Assessment API Demo Application${NC}"
echo "-------------------------------------------------------------------------"

# Check if port 8003 is already in use
if lsof -i:8003 > /dev/null 2>&1; then
    echo -e "${YELLOW}Port 8003 is already in use.${NC}"
    echo -e "The API server may already be running. If you need to restart it, first run:"
    echo -e "${RED}pkill -f \"uvicorn app:app\"${NC}"
    echo ""
else
    echo -e "${GREEN}Starting API server...${NC}"
    # Start the API server in the background
    cd src/api && uvicorn app:app --reload --port 8003 &
    API_PID=$!
    echo -e "API server started with PID: ${YELLOW}$API_PID${NC}"
    echo -e "API will be available at: ${YELLOW}http://localhost:8003${NC}"
    
    # Give the API server a moment to start up
    sleep 2
fi

# Check if port 3000 is already in use
if lsof -i:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}Port 3000 is already in use.${NC}"
    echo -e "The web application may already be running. If you need to restart it, first run:"
    echo -e "${RED}pkill -f \"node server.js\"${NC}"
    echo ""
else
    echo -e "${GREEN}Starting web application...${NC}"
    # Start the web application in the background
    cd webapp && npm start &
    WEB_PID=$!
    echo -e "Web application started with PID: ${YELLOW}$WEB_PID${NC}"
    echo -e "Web application will be available at: ${YELLOW}http://localhost:3000${NC}"
fi

echo "-------------------------------------------------------------------------"
echo -e "${GREEN}Demo Application is now running!${NC}"
echo "To access the web application, open a browser and navigate to:"
echo -e "${YELLOW}http://localhost:3000${NC}"
echo ""
echo "The application includes a tab for testing the new comprehensive assessment endpoint."
echo ""
echo -e "To stop the servers, run: ${RED}pkill -f \"uvicorn app:app\" && pkill -f \"node server.js\"${NC}" 