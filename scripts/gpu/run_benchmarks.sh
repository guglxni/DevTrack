#!/bin/bash

# Script to run all Metal GPU acceleration benchmarks and tests

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1;m'
NC='\033[0m' # No Color

# Create output directories
mkdir -p monitoring
mkdir -p results

# Current timestamp for naming files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${YELLOW}${BOLD}Metal GPU Acceleration Benchmarks${NC}"
echo -e "${BLUE}Running comprehensive benchmarks and tests...${NC}"
echo ""

# Step 1: Ensure the monitoring directory exists
echo -e "${BLUE}Step 1: Creating monitoring directory...${NC}"
./create_monitoring_dir.sh
echo ""

# Step 2: Display system information
echo -e "${BLUE}Step 2: Collecting system information...${NC}"
./show_system_info.py > "results/system_info_${TIMESTAMP}.txt"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}System information saved to results/system_info_${TIMESTAMP}.txt${NC}"
else
    echo -e "${RED}Error collecting system information${NC}"
fi
echo ""

# Step 3: Run GPU acceleration test
echo -e "${BLUE}Step 3: Testing GPU acceleration setup...${NC}"
./test_gpu_acceleration.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}GPU acceleration test passed!${NC}"
else
    echo -e "${RED}GPU acceleration test failed.${NC}"
    echo -e "${YELLOW}You may need to set environment variables:${NC}"
    echo "  export PYTORCH_ENABLE_MPS_FALLBACK=1"
    echo "  export METAL_DEVICE_WRAPPING_ENABLED=1" 
    echo "  export MPS_FALLBACK_ENABLED=1"
    echo ""
    echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
    read -p "Continue with benchmarks? " CONTINUE
    if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
        echo -e "${RED}Exiting benchmarks.${NC}"
        exit 1
    fi
fi
echo ""

# Step 4: Run basic GPU operations benchmark
echo -e "${BLUE}Step 4: Running GPU operations benchmark...${NC}"
./gpu_benchmark.py \
    --matrix-sizes 1000 2000 3000 \
    --conv-sizes 128 256 512 \
    --iterations 3 \
    --output "results/gpu_benchmark_${TIMESTAMP}.json" \
    --plot "results/gpu_benchmark_${TIMESTAMP}.png"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}GPU operations benchmark completed successfully!${NC}"
    echo -e "${GREEN}Results saved to results/gpu_benchmark_${TIMESTAMP}.json${NC}"
    echo -e "${GREEN}Plot saved to results/gpu_benchmark_${TIMESTAMP}.png${NC}"
else
    echo -e "${RED}Error in GPU operations benchmark${NC}"
fi
echo ""

# Step 5: Run LLM benchmark (if available)
echo -e "${BLUE}Step 5: Running LLM benchmark...${NC}"
if [ -f "benchmark_llm.py" ]; then
    python3 benchmark_llm.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}LLM benchmark completed successfully!${NC}"
    else
        echo -e "${RED}Error in LLM benchmark${NC}"
    fi
else
    echo -e "${YELLOW}benchmark_llm.py not found, skipping LLM benchmark${NC}"
fi
echo ""

# Step 6: Analyze performance data
echo -e "${BLUE}Step 6: Analyzing performance data...${NC}"
if [ -d "monitoring" ] && [ "$(ls -A monitoring 2>/dev/null)" ]; then
    ./analyze_performance.py "monitoring/" \
        --output "results/performance_analysis_${TIMESTAMP}.png"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Performance analysis completed successfully!${NC}"
        echo -e "${GREEN}Analysis saved to results/performance_analysis_${TIMESTAMP}.png${NC}"
    else
        echo -e "${RED}Error in performance analysis${NC}"
    fi
else
    echo -e "${YELLOW}No monitoring data found to analyze${NC}"
fi
echo ""

echo -e "${GREEN}${BOLD}All benchmarks and tests completed!${NC}"
echo -e "${BLUE}Results are saved in the results/ directory${NC}"
echo -e "${YELLOW}To apply Metal GPU optimizations to the server, run:${NC}"
echo -e "  ${BLUE}./start_optimized.sh${NC}"
echo "" 