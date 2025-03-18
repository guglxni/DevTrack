#!/bin/bash

# Wrapper script to start the API server with the appropriate configuration

# Display available options
echo "DevTrack Assessment API Server Launcher"
echo "========================================"
echo "Select a startup mode:"
echo "1. Standard mode (CPU only)"
echo "2. Metal GPU accelerated mode (recommended for Apple Silicon)"
echo "3. Advanced Metal GPU accelerated mode (optimized for best performance)"
echo ""

# Prompt for selection
read -p "Enter your choice (1-3) [3]: " choice

# Set default if empty
if [ -z "$choice" ]; then
    choice=3
fi

# Execute the appropriate startup script
case $choice in
    1)
        echo "Starting in standard CPU-only mode..."
        ./restart_integrated_server.sh
        ;;
    2)
        echo "Starting with basic Metal GPU acceleration..."
        ./restart_with_metal.sh
        ;;
    3)
        echo "Starting with optimized Metal GPU acceleration..."
        ./restart_with_metal_advanced.sh
        ;;
    *)
        echo "Invalid choice. Defaulting to optimized Metal GPU acceleration..."
        ./restart_with_metal_advanced.sh
        ;;
esac

echo ""
echo "Server startup complete!"
echo "To access the dashboard, visit: http://localhost:8003"
echo "To stop the server, run: pkill -f 'uvicorn src.app:app'" 