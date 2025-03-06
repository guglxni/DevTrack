#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
API_URL="http://localhost:8003"
DATA_FILE="test_data/comprehensive_test.json"

# Check if at least one parameter (data file) is provided
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage:${NC} $0 [data_file] [API URL]"
    echo -e "  data_file: JSON file with comprehensive test data (default: test_data/comprehensive_test.json)"
    echo -e "  API URL: URL for the API (default: http://localhost:8003)"
    echo -e "\nUsing default values..."
else
    DATA_FILE="$1"
    # If second parameter is provided, use it as API URL
    if [ $# -ge 2 ]; then
        API_URL="$2"
    fi
fi

# Verify data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error:${NC} Data file '$DATA_FILE' not found!"
    exit 1
fi

echo -e "${GREEN}Testing comprehensive assessment endpoint${NC}"
echo -e "API URL: ${YELLOW}$API_URL${NC}"
echo -e "Data file: ${YELLOW}$DATA_FILE${NC}"
echo -e "Endpoint: ${YELLOW}/comprehensive-assessment${NC}"
echo "-------------------------------------------------------------------------"

# Execute the curl command with timing
echo -e "${GREEN}Sending comprehensive assessment request...${NC}"
echo -e "${YELLOW}Request data:${NC}"
cat "$DATA_FILE"
echo ""

start_time=$(date +%s.%N)
response=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d @"$DATA_FILE" \
  "$API_URL/comprehensive-assessment")
end_time=$(date +%s.%N)

# Calculate elapsed time
elapsed=$(echo "$end_time - $start_time" | bc)
elapsed_ms=$(echo "$elapsed * 1000" | bc)

# Check for errors
if [[ "$response" == *"\"detail\":"* ]]; then
    echo -e "${RED}Error in response:${NC}"
    echo "$response" | python3 -m json.tool
    exit 1
else
    echo -e "${GREEN}Response received successfully in ${YELLOW}${elapsed_ms}ms${NC}${GREEN}:${NC}"
    echo "$response" | python3 -m json.tool
fi

echo -e "\n${GREEN}Test completed.${NC}" 