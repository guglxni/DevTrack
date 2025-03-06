#!/bin/bash
# Test Script for Individual Endpoints
# This script tests each of the new API endpoints with multiple requests

ENDPOINT=$1
COUNT=$2
DATA_FILE=$3

if [ -z "$ENDPOINT" ] || [ -z "$COUNT" ] || [ -z "$DATA_FILE" ]; then
  echo "Usage: $0 <endpoint> <count> <data_file>"
  echo "Example: $0 /question 30 test_data/single_question.json"
  exit 1
fi

echo "=== Testing $ENDPOINT endpoint ($COUNT iterations) ==="
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint $ENDPOINT --method POST --data $DATA_FILE --count $COUNT

echo "=== Testing Complete ==="
echo "Check test_results/api_test_report.html for detailed results" 