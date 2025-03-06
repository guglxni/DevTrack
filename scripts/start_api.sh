#!/bin/bash

# ASD Assessment API Optimized Launcher
# This script provides an optimized way to start the API with better error handling

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PORT=8002
WORKERS=4
LOG_LEVEL="info"
RELOAD=false
MODEL_ENHANCEMENT=true
TIMEOUT=120
DEBUG_MODE=false

# Function to display help information
show_help() {
    echo -e "${GREEN}ASD Assessment API Launcher${NC}"
    echo "Usage: ./start_api.sh [options]"
    echo ""
    echo "Options:"
    echo "  -p, --port PORT        Port to run the API on (default: 8002)"
    echo "  -w, --workers NUM      Number of worker processes (default: 4)"
    echo "  -l, --log-level LEVEL  Log level (debug, info, warning, error) (default: info)"
    echo "  -r, --reload           Enable auto-reload on code changes (development only)"
    echo "  -t, --timeout SECONDS  Request timeout in seconds (default: 120)"
    echo "  --no-enhance           Skip model enhancement"
    echo "  --debug                Run in debug mode with more verbose output"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start_api.sh                          # Start with default settings"
    echo "  ./start_api.sh -p 8000 -w 2             # Run on port 8000 with 2 workers"
    echo "  ./start_api.sh --reload --log-level debug  # Development mode with debug logging"
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -r|--reload)
            RELOAD=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --no-enhance)
            MODEL_ENHANCEMENT=false
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed or not in your PATH.${NC}"
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking required packages...${NC}"
REQUIRED_PACKAGES=("fastapi" "uvicorn" "pandas" "numpy" "sentence-transformers" "torch")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

# Install missing packages if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing packages: ${MISSING_PACKAGES[*]}${NC}"
    python3 -m pip install "${MISSING_PACKAGES[@]}"
    
    # Check if installation was successful
    for package in "${MISSING_PACKAGES[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            echo -e "${RED}Failed to install $package. Please install it manually:${NC}"
            echo "python3 -m pip install $package"
            exit 1
        fi
    done
    echo -e "${GREEN}All required packages installed successfully.${NC}"
fi

# Check if the port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t &> /dev/null; then
    echo -e "${RED}Error: Port $PORT is already in use.${NC}"
    echo "Please choose a different port with --port option or stop the process using this port."
    exit 1
fi

# Function to check Apple Silicon and setup optimizations
check_apple_silicon() {
    if [[ $(uname) == "Darwin" && $(uname -m) == "arm64" ]]; then
        echo -e "${GREEN}✓ Apple Silicon detected${NC}"
        
        # Set environment variables for optimal performance
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        
        # Set for native ARM compilation and optimization
        export OPENBLAS_MAIN_FREE=1
        export OMP_NUM_THREADS=$WORKERS
        
        # Check if Metal Performance Shaders are available
        if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
            echo -e "${GREEN}✓ MPS (Metal Performance Shaders) are available${NC}"
            export USE_MPS=1
        else
            echo -e "${YELLOW}⚠ MPS not available on this device, using CPU${NC}"
        fi
        
        return 0
    else
        echo -e "${YELLOW}⚠ Not running on Apple Silicon${NC}"
        return 1
    fi
}

# Function to enhance the model if needed
enhance_model() {
    if [ "$MODEL_ENHANCEMENT" = true ]; then
        echo -e "${BLUE}Running model enhancement...${NC}"
        
        # Check if the enhancement script exists
        if [ -f "enhance_model.py" ]; then
            python3 enhance_model.py
            # Check return code
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Model enhancement completed successfully${NC}"
            else
                echo -e "${YELLOW}⚠ Model enhancement encountered issues${NC}"
                echo -e "${YELLOW}⚠ Continuing with startup...${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ Model enhancement script not found${NC}"
            echo -e "${YELLOW}⚠ Continuing with standard model...${NC}"
        fi
    else
        echo -e "${BLUE}Model enhancement skipped${NC}"
    fi
}

# Function to start the API server
start_server() {
    echo -e "${BLUE}Starting ASD Assessment API...${NC}"
    
    # Build command based on options
    CMD="python3 -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers $WORKERS --timeout-keep-alive $TIMEOUT --log-level $LOG_LEVEL"
    
    # Add reload flag if enabled
    if [ "$RELOAD" = true ]; then
        CMD="$CMD --reload"
    fi
    
    # Print the command in debug mode
    if [ "$DEBUG_MODE" = true ]; then
        echo -e "${YELLOW}Command: $CMD${NC}"
    fi
    
    # Execute the command
    echo -e "${GREEN}Server starting on http://localhost:$PORT${NC}"
    eval $CMD
}

# Main execution flow
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}   ASD Assessment API - Optimized Launcher   ${NC}"
echo -e "${GREEN}=============================================${NC}"

check_apple_silicon
enhance_model
start_server 