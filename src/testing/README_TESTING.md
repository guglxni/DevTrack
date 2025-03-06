# Comprehensive API Testing Framework for ASD Assessment API

This directory contains a comprehensive testing framework for the ASD Assessment API. The framework allows you to test all endpoints, including the standard ones and the newly added endpoints for questions, keywords, and manual scoring.

## Key Features

- **Complete endpoint coverage**: Tests all API endpoints including the new ones
- **Comprehensive reporting**: Generates detailed HTML reports with charts
- **Performance metrics**: Tracks response times and success rates
- **Load testing**: Simulates multiple concurrent requests to test performance
- **Complex workflows**: Tests complete assessment flows and keyword updates
- **Flexible configuration**: Supports various testing scenarios through command-line options

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install pytest requests matplotlib pandas tqdm jinja2
```

### Running the Tests

The testing framework provides several commands for different testing scenarios:

#### Basic Tests

Run basic API tests to verify that the core endpoints are working:

```bash
python3 src/testing/comprehensive_api_tester.py --verbose basic --age 24
```

#### Complete Assessment Flow

Test a complete assessment flow, from setting age to generating a report:

```bash
python3 src/testing/comprehensive_api_tester.py --verbose assessment --age 30
```

You can also provide custom responses:

```bash
python3 src/testing/comprehensive_api_tester.py --verbose assessment --age 24 --responses responses.json
```

Where `responses.json` is a JSON file mapping milestone behaviors to response texts:

```json
{
  "Shows empathy": "Yes, she always notices when I am sad and tries to comfort me.",
  "Makes eye contact": "She rarely makes eye contact, even when I call her name."
}
```

#### Keywords Workflow

Test the complete keywords update workflow:

```bash
python3 src/testing/comprehensive_api_tester.py --verbose keywords
```

This will:
1. Update keywords for each scoring category
2. Test scoring with various responses containing those keywords
3. Verify the scoring matches the expected categories

#### Load Testing

Test how an endpoint performs under load:

```bash
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint /health --count 100 --concurrent
```

For POST endpoints, you can provide request data:

```bash
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint /score-response --method POST --data score_data.json --count 50 --concurrent
```

Where `score_data.json` contains the request data:

```json
{
  "milestone_behavior": "Shows empathy",
  "response": "Yes, my child always shows empathy when others are upset."
}
```

### Configuration Options

The framework supports the following configuration options:

- `--url`: API base URL (default: http://localhost:8003)
- `--verbose`: Enable verbose output
- `--age`: Child's age in months for testing (default: 24)
- `--responses`: JSON file with milestone responses for assessment testing
- `--endpoint`: Endpoint to test for load testing
- `--method`: HTTP method to use for load testing (default: GET)
- `--data`: JSON file with request data for load testing
- `--count`: Number of requests to make for load testing (default: 10)
- `--concurrent`: Run requests concurrently for load testing

## Test Report

After running tests, the framework generates:

1. A JSON file with detailed test results (`test_results/api_test_results.json`)
2. An HTML report with visualizations (`test_results/api_test_report.html`)
3. Performance charts for response times and success rates

The HTML report includes:
- Overall success rate and response time metrics
- Endpoint-specific performance statistics
- Detailed results for each test
- Interactive elements to view response details

## Testing New Endpoints

The framework specifically includes tests for the newly added endpoints:

### 1. Question Endpoint Test

Tests submitting questions about a child's behavior:

```python
tester.test_question(
    question_text="Does the child respond when called by name?",
    milestone_id="Recognizes familiar people"
)
```

### 2. Keywords Endpoint Test

Tests updating keywords for scoring categories:

```python
tester.test_update_keywords(
    category="CANNOT_DO",
    keywords=["no", "not", "never", "doesn't", "does not", "cannot"]
)
```

### 3. Manual Scoring Endpoint Test

Tests manually setting a score for a milestone:

```python
tester.test_send_score(
    milestone_id="Shows empathy",
    score=4,
    score_label="INDEPENDENT"
)
```

## Extending the Framework

You can extend the framework with additional tests by:

1. Adding new test methods to the `ComprehensiveAPITester` class
2. Creating new complex test scenarios like `run_complete_assessment`
3. Adding new command-line options to the `main` function

## Troubleshooting

If you encounter issues:

1. Check that the API server is running (`python3 main.py --api`)
2. Verify the API URL (`--url` option)
3. Check the logs in `logs/api_tests.log`
4. Try running with `--verbose` for more detailed output 