#!/bin/bash
# Comprehensive Test Runner for ASD Assessment API
# This script runs tests for all endpoints and generates a consolidated report

echo "=== Starting Comprehensive API Testing ==="

# Create a directory for individual endpoint results
mkdir -p test_results/endpoints

# Test Question Endpoint
echo "Testing /question endpoint..."
./scripts/test_single_endpoint.sh /question 30 test_data/single_question.json
# Save the results for this endpoint
cp test_results/api_test_results.json test_results/endpoints/question_results.json

# Test Keywords Endpoint
echo "Testing /keywords endpoint..."
./scripts/test_single_endpoint.sh /keywords 30 test_data/single_category_keywords.json
# Save the results for this endpoint
cp test_results/api_test_results.json test_results/endpoints/keywords_results.json

# Test Send-Score Endpoint
echo "Testing /send-score endpoint..."
./scripts/test_single_endpoint.sh /send-score 30 test_data/single_score.json
# Save the results for this endpoint
cp test_results/api_test_results.json test_results/endpoints/send_score_results.json

# Test Score-Response Endpoint
echo "Testing /score-response endpoint..."
./scripts/test_single_endpoint.sh /score-response 30 test_data/sample_response.json
# Save the results for this endpoint
cp test_results/api_test_results.json test_results/endpoints/score_response_results.json

# Combine all endpoint results
echo "Combining test results and generating consolidated report..."
python3 -c "
import json
import os

# Initialize combined results
combined_results = {
    'summary': {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'success_rate': 0,
        'avg_response_time': 0,
        'endpoints': {}
    },
    'results': []
}

# Directory with individual endpoint results
results_dir = 'test_results/endpoints'

# Process each endpoint result file
for filename in os.listdir(results_dir):
    if filename.endswith('.json'):
        with open(os.path.join(results_dir, filename), 'r') as f:
            endpoint_results = json.load(f)
            
        # Add these results to the combined results
        combined_results['results'].extend(endpoint_results['results'])
        
        # Update summary metrics
        summary = endpoint_results['summary']
        combined_results['summary']['total_tests'] += summary['total_tests']
        combined_results['summary']['passed_tests'] += summary['passed_tests']
        combined_results['summary']['failed_tests'] += summary['failed_tests']
        
        # Add endpoint data
        for endpoint, data in summary['endpoints'].items():
            combined_results['summary']['endpoints'][endpoint] = data

# Calculate overall avg response time and success rate
if combined_results['results']:
    combined_results['summary']['avg_response_time'] = sum(
        r['response_time'] for r in combined_results['results']
    ) / len(combined_results['results'])
    
    if combined_results['summary']['total_tests'] > 0:
        combined_results['summary']['success_rate'] = (
            combined_results['summary']['passed_tests'] / 
            combined_results['summary']['total_tests']
        ) * 100

# Save the combined results
with open('test_results/combined_results.json', 'w') as f:
    json.dump(combined_results, f, indent=2)

print('Combined test results saved to test_results/combined_results.json')
"

# Generate consolidated report using combined results
echo "Generating consolidated test report..."
cp test_results/combined_results.json test_results/api_test_results.json
python3 scripts/generate_test_report.py

echo "=== Testing Complete ==="
echo "Check test_results/consolidated_report.html for detailed results" 