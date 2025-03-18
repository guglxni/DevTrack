#!/bin/bash

# Enhanced Startup Script with Auto-detection and Optimization
# -----------------------------------------------------------
# This script automatically detects Apple Silicon, applies
# chip-specific optimizations, and includes memory monitoring.

# Set text colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}DevTrack Assessment API - Enhanced Startup${NC}"
echo "================================================="

# Function to detect Apple Silicon and chip model
detect_chip() {
    if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        echo -e "${GREEN}✅ Apple Silicon detected${NC}"
        
        # Determine specific chip model
        local chip_info=$(sysctl -n machdep.cpu.brand_string)
        if [[ $chip_info == *"M1"* ]]; then
            echo -e "${BLUE}→ Apple M1 chip detected${NC}"
            return 1
        elif [[ $chip_info == *"M2"* ]]; then
            echo -e "${BLUE}→ Apple M2 chip detected${NC}"
            return 2
        elif [[ $chip_info == *"M3"* ]]; then
            echo -e "${BLUE}→ Apple M3 chip detected${NC}"
            return 3
        elif [[ $chip_info == *"M4"* ]]; then
            echo -e "${BLUE}→ Apple M4 chip detected${NC}"
            return 4
        else
            echo -e "${YELLOW}⚠️ Unknown Apple Silicon variant detected${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️ Not running on Apple Silicon${NC}"
        return 0
    fi
}

# Function to optimize settings based on chip model
apply_optimizations() {
    local chip=$1
    
    # Common settings for all chips
    export ENABLE_R2R=true
    export ENABLE_ACTIVE_LEARNING=true
    export ENABLE_LLM_SCORING=true
    export ENABLE_SMART_SCORING=true
    export ENABLE_CONTINUOUS_LEARNING=true
    
    # Set the path to the Mistral model
    export LLM_MODEL_PATH="$(pwd)/models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"
    echo -e "${BLUE}Using Mistral model at:${NC} $LLM_MODEL_PATH"
    
    # Set CPU thread count
    cpu_count=$(sysctl -n hw.ncpu)
    logical_cpus=$(sysctl -n hw.logicalcpu)
    physical_cpus=$(sysctl -n hw.physicalcpu)
    
    echo -e "${BLUE}CPU Configuration:${NC}"
    echo " - Total CPU Count: $cpu_count"
    echo " - Logical CPUs: $logical_cpus"
    echo " - Physical CPUs: $physical_cpus"
    
    # Chip-specific optimizations
    if [[ $chip -eq 0 ]]; then
        # Non-Apple Silicon - CPU only mode
        echo -e "${YELLOW}Running in CPU-only mode${NC}"
        export N_GPU_LAYERS=0
        export OMP_NUM_THREADS=$cpu_count
        
    elif [[ $chip -eq 1 || $chip -eq 2 ]]; then
        # M1/M2 - Conservative GPU usage
        echo -e "${GREEN}Applying optimized settings for M1/M2 chip${NC}"
        export USE_METAL=true
        export N_GPU_LAYERS=24  # Offload most layers but not all to avoid memory pressure
        export LLAMA_N_GPU_LAYERS=24
        export GGML_N_GPU_LAYERS=24
        export GGML_METAL_FULL_OFFLOAD=0
        export F16_KV=true
        
        # Thread optimization for M1/M2
        export OMP_NUM_THREADS=$((physical_cpus / 2))
        export MKL_NUM_THREADS=$((physical_cpus / 2))
        export NUMEXPR_NUM_THREADS=$((physical_cpus / 2))
        
        # Performance tuning
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        
    else
        # M3/M4 - Aggressive GPU usage
        echo -e "${GREEN}Applying optimized settings for M3/M4 chip${NC}"
        export USE_METAL=true
        export N_GPU_LAYERS=32  # Offload ALL layers to GPU
        export LLAMA_N_GPU_LAYERS=32
        export GGML_N_GPU_LAYERS=32
        export GGML_METAL_FULL_OFFLOAD=1
        export F16_KV=true
        
        # Thread optimization for M3/M4
        export OMP_NUM_THREADS=$((physical_cpus / 2))
        export MKL_NUM_THREADS=$((physical_cpus / 2))
        export NUMEXPR_NUM_THREADS=$((physical_cpus / 2))
        
        # Additional performance tuning
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        export TOKENIZERS_PARALLELISM=true
        export GGML_METAL_PATH_RESOURCES="$(pwd)/models"
    fi
    
    # Weight configuration
    export SCORE_WEIGHT_KEYWORD=0.2
    export SCORE_WEIGHT_EMBEDDING=0.3
    export SCORE_WEIGHT_TRANSFORMER=0.2
    export SCORE_WEIGHT_LLM=0.5
    
    echo -e "${BLUE}Applied optimizations:${NC}"
    echo " - GPU Layers: $N_GPU_LAYERS"
    echo " - Thread Count: $OMP_NUM_THREADS"
    echo " - Metal Enabled: ${USE_METAL:-false}"
}

# Function to monitor GPU memory
setup_monitoring() {
    # Create directory for monitoring data if it doesn't exist
    mkdir -p "monitoring"
    
    # Set up monitoring log file
    monitoring_log="monitoring/gpu_memory_$(date +%Y%m%d_%H%M%S).log"
    echo "timestamp,gpu_memory_used_mb,gpu_memory_total_mb,cpu_usage_percent" > "$monitoring_log"
    
    echo -e "${BLUE}Monitoring enabled:${NC} $monitoring_log"
    
    # Return the log file path
    echo "$monitoring_log"
}

# Function to collect memory stats
collect_memory_stats() {
    local log_file=$1
    local pid=$2
    
    # Start background monitoring (runs every 5 seconds)
    (
        while ps -p $pid > /dev/null; do
            timestamp=$(date +%Y-%m-%d_%H:%M:%S)
            
            # Get GPU memory usage if possible
            if command -v system_profiler &>/dev/null; then
                gpu_info=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -i "metal" -A 5)
                if [[ -n "$gpu_info" ]]; then
                    gpu_memory=$(echo "$gpu_info" | grep -i "VRAM" | awk '{print $2}' | sed 's/[^0-9]*//g')
                    echo "$timestamp,$gpu_memory,N/A,N/A" >> "$log_file"
                fi
            fi
            
            # Get process CPU usage
            if command -v ps &>/dev/null; then
                cpu_usage=$(ps -p $pid -o %cpu | tail -n 1 | tr -d ' ')
                echo "$timestamp,N/A,N/A,$cpu_usage" >> "$log_file"
            fi
            
            sleep 5
        done
    ) &
    
    echo $!  # Return the monitoring process PID
}

# Function to start the server
start_server() {
    echo -e "${BLUE}Starting API server...${NC}"
    
    python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8003 > api_server.log 2>&1 &
    SERVER_PID=$!
    
    echo -e "${GREEN}API server started with PID: ${NC}$SERVER_PID"
    
    # Wait for the server to start
    echo "Waiting for the server to start..."
    MAX_RETRIES=20
    RETRY_COUNT=0
    SERVER_READY=false
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:8003/health > /dev/null; then
            SERVER_READY=true
            echo -e "${GREEN}✅ API server is running!${NC}"
            break
        fi
        echo "API server not responding yet, retrying in 3 seconds... (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
        sleep 3
        RETRY_COUNT=$((RETRY_COUNT+1))
        
        # Check if the process is still running
        if ! ps -p $SERVER_PID > /dev/null; then
            echo -e "${RED}❌ Error: API server process has terminated.${NC}"
            echo "Showing last 50 lines of the log:"
            tail -n 50 api_server.log
            return 1
        fi
    done
    
    if [ "$SERVER_READY" = false ]; then
        echo -e "${RED}❌ Error: API server failed to respond after $MAX_RETRIES attempts.${NC}"
        echo "Showing last 50 lines of the log:"
        tail -n 50 api_server.log
        return 1
    fi
    
    # Wait a bit to ensure LLM is fully loaded
    sleep 3
    
    # Check LLM scoring availability
    if curl -s http://localhost:8003/llm-scoring/health > /dev/null; then
        LLM_STATUS=$(curl -s http://localhost:8003/llm-scoring/health)
        echo -e "${GREEN}✅ LLM scoring is available:${NC} $LLM_STATUS"
        
        # Test GPU usage by running a quick inference
        echo -e "${BLUE}Performing quick test to verify GPU acceleration...${NC}"
        curl -s -X POST -H "Content-Type: application/json" -d '{
            "question": "Does your child recognize familiar people?",
            "milestone": "Recognizes familiar people",
            "response": "Yes, she recognizes all family members easily"
        }' http://localhost:8003/llm-scoring/direct-test > /dev/null
        
        # Check if model is using Metal as expected
        if grep -q "Metal" api_server.log; then
            echo -e "${GREEN}✅ Metal GPU acceleration confirmed!${NC}"
            if grep -q "offloaded.*layers to GPU" api_server.log; then
                layers=$(grep "offloaded.*layers to GPU" api_server.log | tail -n 1)
                echo -e "${GREEN}→ $layers${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️ Metal GPU acceleration not detected in logs${NC}"
        fi
    else
        echo -e "${RED}❌ Warning: LLM scoring endpoint is not available.${NC}"
        echo "Showing last 50 lines of the log:"
        tail -n 50 api_server.log
    fi
    
    return 0
}

# Function to display performance tips
display_tips() {
    echo -e "\n${BLUE}Performance Tips:${NC}"
    echo " 1. The first inference will be slower as the model warms up"
    echo " 2. Keep the server running to avoid cold starts"
    echo " 3. For best performance, ensure no other intensive apps are running"
    echo " 4. Check 'monitoring/' directory for performance logs"
    echo -e "\n${BLUE}Troubleshooting:${NC}"
    echo " - If you see memory errors, try reducing N_GPU_LAYERS"
    echo " - If inference is slow, ensure Metal acceleration is working"
    echo " - To stop the server: pkill -f 'uvicorn src.app:app'"
}

# Main execution flow
detect_chip
CHIP_TYPE=$?

apply_optimizations $CHIP_TYPE
MONITORING_LOG=$(setup_monitoring)

# Start the server
start_server
SERVER_STATUS=$?

if [ $SERVER_STATUS -eq 0 ]; then
    # Start the monitoring process
    MONITOR_PID=$(collect_memory_stats "$MONITORING_LOG" $SERVER_PID)
    echo -e "${GREEN}Monitoring process started with PID: ${NC}$MONITOR_PID"
    
    echo -e "\n${GREEN}DevTrack Assessment API is now running with optimized settings!${NC}"
    echo -e "API server PID: $SERVER_PID"
    echo -e "You can access the web interface at: ${BLUE}http://localhost:8003${NC}"
    echo -e "API documentation is available at: ${BLUE}http://localhost:8003/docs${NC}"
    
    # Display tips
    display_tips
else
    echo -e "${RED}Failed to start the server with optimized settings.${NC}"
fi

echo -e "\n${YELLOW}To stop the server and monitoring:${NC} pkill -f 'uvicorn src.app:app' && kill $MONITOR_PID" 