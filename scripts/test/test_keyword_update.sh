#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
API_URL="http://localhost:8003"
DATA_FILE="test_data/comprehensive_keyword_test.json"

echo -e "${YELLOW}Testing Comprehensive Assessment Endpoint with Keyword Updates${NC}"
echo -e "API URL: ${GREEN}$API_URL${NC}"
echo -e "Data file: ${GREEN}$DATA_FILE${NC}"

# Check if file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: Data file $DATA_FILE not found${NC}"
    exit 1
fi

# Check if API server is running
echo -e "\n${YELLOW}Checking if API server is running...${NC}"
if ! curl -s "$API_URL" > /dev/null; then
    echo -e "${RED}Error: API server not running at $API_URL${NC}"
    echo -e "Please start the API server with: cd src/api && uvicorn app:app --reload --port 8003"
    exit 1
fi
echo -e "${GREEN}API server is running!${NC}"

# First test - with original keywords
echo -e "\n${YELLOW}Test 1: Sending original request${NC}"
echo -e "Request data:"
cat "$DATA_FILE" | jq

RESPONSE=$(curl -s -X POST "$API_URL/comprehensive-assessment" \
    -H "Content-Type: application/json" \
    -d @"$DATA_FILE")

echo -e "\nResponse:"
echo "$RESPONSE" | jq || echo "$RESPONSE"

SCORE=$(echo $RESPONSE | grep -o '"score":[^,]*' | cut -d ':' -f2)
SCORE_LABEL=$(echo $RESPONSE | grep -o '"score_label":"[^"]*"' | cut -d '"' -f4)

echo -e "\nOriginal response score: ${GREEN}$SCORE_LABEL ($SCORE)${NC}"

# Second test - modifying keywords to change expected outcome
echo -e "\n${YELLOW}Test 2: Modifying keywords to move INDEPENDENT keywords to EMERGING${NC}"

# Create a modified JSON with "understands" and "responds" moved from INDEPENDENT to EMERGING
TMP_FILE=$(mktemp)
cat "$DATA_FILE" | sed 's/"INDEPENDENT": \["understands", "responds", "follows"\]/"INDEPENDENT": \["follows"\]/' > "$TMP_FILE"
# Add the keywords to EMERGING
cat "$TMP_FILE" | sed 's/"EMERGING": \["sometimes", "beginning to", "inconsistent"\]/"EMERGING": \["sometimes", "beginning to", "inconsistent", "understands", "responds"\]/' > "$TMP_FILE.2"
mv "$TMP_FILE.2" "$TMP_FILE"

echo -e "${YELLOW}Modified test data to move 'understands' and 'responds' from INDEPENDENT to EMERGING category${NC}"
echo -e "Modified request data:"
cat "$TMP_FILE" | jq

RESPONSE2=$(curl -s -X POST "$API_URL/comprehensive-assessment" \
    -H "Content-Type: application/json" \
    -d @"$TMP_FILE")

echo -e "\nResponse:"
echo "$RESPONSE2" | jq || echo "$RESPONSE2"

SCORE2=$(echo $RESPONSE2 | grep -o '"score":[^,]*' | cut -d ':' -f2)
SCORE_LABEL2=$(echo $RESPONSE2 | grep -o '"score_label":"[^"]*"' | cut -d '"' -f4)

echo -e "\nModified response score: ${GREEN}$SCORE_LABEL2 ($SCORE2)${NC}"

# Clean up temp files
rm -f "$TMP_FILE"

echo -e "\n${YELLOW}Testing complete!${NC}"
if [ "$SCORE" != "$SCORE2" ]; then
    echo -e "${GREEN}Success: Keyword updates changed the scoring result!${NC}"
    echo -e "Original score: $SCORE_LABEL ($SCORE)"
    echo -e "Modified score: $SCORE_LABEL2 ($SCORE2)"
else
    echo -e "${RED}Possible issue: Keyword updates did not change scoring result${NC}"
    echo -e "Both tests returned: $SCORE_LABEL ($SCORE)"
fi 