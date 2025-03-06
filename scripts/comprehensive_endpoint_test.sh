#!/bin/bash
# Comprehensive Endpoint Testing Script
# This script tests the new API endpoints with multiple requests

echo "=== Starting Comprehensive Endpoint Testing ==="
echo ""

# Reset the assessment engine first
echo "Resetting assessment engine..."
python3 src/testing/comprehensive_api_tester.py --verbose basic --age 24 > /dev/null 2>&1

# Test the /question endpoint
echo "Testing /question endpoint (30 iterations)..."
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint /question --method POST --data test_data/various_questions.json --count 30

# Test the /keywords endpoint with all categories
echo ""
echo "Testing /keywords endpoint (25 iterations)..."
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint /keywords --method POST --data test_data/all_keywords_data.json --count 25 --concurrent

# Test the /send-score endpoint
echo ""
echo "Testing /send-score endpoint (25 iterations)..."
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint /send-score --method POST --data test_data/various_scores.json --count 25 --concurrent

echo ""
echo "=== Comprehensive Endpoint Testing Complete ==="
echo "Check test_results/api_test_report.html for detailed results" 