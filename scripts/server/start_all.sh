#!/bin/bash

# Start both the API server and the web application

echo "Starting ASD Assessment System..."
echo "This script will start both the API server and the web application in separate terminals."

# Make the individual scripts executable
chmod +x start_api_server.sh
chmod +x start_webapp.sh

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use open command with Terminal app
    echo "Detected macOS, opening terminals..."
    open -a Terminal.app start_api_server.sh
    sleep 3  # Wait for the API server to start
    open -a Terminal.app start_webapp.sh
else
    # Linux - use gnome-terminal or xterm
    if command -v gnome-terminal &> /dev/null; then
        echo "Using gnome-terminal..."
        gnome-terminal -- ./start_api_server.sh
        sleep 3  # Wait for the API server to start
        gnome-terminal -- ./start_webapp.sh
    elif command -v xterm &> /dev/null; then
        echo "Using xterm..."
        xterm -e "./start_api_server.sh" &
        sleep 3  # Wait for the API server to start
        xterm -e "./start_webapp.sh" &
    else
        echo "Could not find a suitable terminal emulator."
        echo "Please run the scripts manually:"
        echo "  ./start_api_server.sh"
        echo "  ./start_webapp.sh"
        exit 1
    fi
fi

echo "Started both services. You can access:"
echo "  - API server at: http://localhost:8003"
echo "  - Web application at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to exit this script (the services will continue running in their own terminals)"

# Keep the script running to allow the user to Ctrl+C
while true; do
    sleep 1
done 