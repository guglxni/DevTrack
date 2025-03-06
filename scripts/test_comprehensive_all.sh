#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default API URL
API_URL="http://localhost:8003"

# Check if API URL is provided as an argument
if [ $# -ge 1 ]; then
    API_URL="$1"
fi

echo -e "${GREEN}Running all comprehensive endpoint tests${NC}"
echo -e "API URL: ${YELLOW}$API_URL${NC}"
echo "-------------------------------------------------------------------------"

# Array of test files
TEST_FILES=(
    "test_data/comprehensive_test.json"
    "test_data/comprehensive_test_emerging.json"
    "test_data/comprehensive_test_with_support.json"
)

# Counter for successful tests
SUCCESS_COUNT=0
TOTAL_TESTS=${#TEST_FILES[@]}

# Run each test
for test_file in "${TEST_FILES[@]}"; do
    echo -e "\n${YELLOW}Running test with file:${NC} $test_file"
    
    if ./scripts/test_comprehensive_endpoint.sh "$test_file" "$API_URL"; then
        echo -e "${GREEN}Test passed!${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}Test failed!${NC}"
    fi
    
    echo "-------------------------------------------------------------------------"
done

# Print summary
echo -e "\n${GREEN}Test Summary:${NC}"
echo -e "Total tests: ${YELLOW}$TOTAL_TESTS${NC}"
echo -e "Successful tests: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failed tests: ${RED}$((TOTAL_TESTS - SUCCESS_COUNT))${NC}"

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed!${NC}"
    exit 1
fi 