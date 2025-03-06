# ASD Assessment System Testing Framework

This comprehensive testing framework allows for easy demonstration and testing of the ASD (Autism Spectrum Disorder) assessment system. It provides both command-line and web-based interfaces to interact with the API, test responses, run assessments, and visualize results.

## Overview

The ASD assessment system evaluates developmental milestones based on caregiver responses, scoring them on a 5-point scale:

| Score | Label | Description |
|-------|-------|-------------|
| 0 | CANNOT_DO | Skill not acquired |
| 1 | LOST_SKILL | Acquired but lost |
| 2 | EMERGING | Emerging and inconsistent |
| 3 | WITH_SUPPORT | Acquired but consistent in specific situations only |
| 4 | INDEPENDENT | Acquired and present in all situations |

This testing framework provides tools to demonstrate and validate this scoring system.

## Installation

### Prerequisites

- Python 3.7 or higher (tested with Python 3.12)
- pip (Python package manager)
- macOS, Linux, or Windows with bash/zsh support

### Environment Setup

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd asd-assessment-system
   ```

2. Install the required dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Make the script files executable:
   ```bash
   chmod +x asd_test_cli.py
   chmod +x start_api.sh
   chmod +x test_response.sh
   chmod +x run_tests.sh
   ```

4. Verify your Python executable path:
   ```bash
   python3 -c "import sys; print(sys.executable)"
   ```
   Take note of this path, as you may need it for troubleshooting.

5. Install specialized dependencies if needed:
   ```bash
   python3 -m pip install sentence-transformers
   ```

## Testing Framework Components

This testing framework consists of two main components:

1. **Command-Line Interface (CLI)** - For power users, automation, and scripting
2. **Web Application** - For interactive testing and visualization

## 1. Command-Line Interface (CLI)

The CLI tool (`asd_test_cli.py`) provides a comprehensive set of commands to test the assessment system.

### Usage

```bash
python3 asd_test_cli.py [COMMAND] [OPTIONS]
```

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `start-api` | Start the API server | `python3 asd_test_cli.py start-api` |
| `stop-api` | Stop the API server | `python3 asd_test_cli.py stop-api` |
| `health` | Check if the API server is running | `python3 asd_test_cli.py health` |
| `set-age` | Set the child's age in months | `python3 asd_test_cli.py set-age 24` |
| `milestone` | Get the next milestone for assessment | `python3 asd_test_cli.py milestone` |
| `test` | Test a specific response for a milestone | `python3 asd_test_cli.py test "walks independently" "yes, she can walk"` |
| `batch-test` | Run batch tests with various responses | `python3 asd_test_cli.py batch-test --count 10` |
| `demo` | Run a demonstration with various responses | `python3 asd_test_cli.py demo` |
| `report` | Generate a developmental report | `python3 asd_test_cli.py report` |
| `benchmark` | Run a performance benchmark | `python3 asd_test_cli.py benchmark --iterations 100` |
| `interactive` | Start an interactive assessment session | `python3 asd_test_cli.py interactive` |

### CLI Examples

#### Running the Demo Mode

The demo mode provides an excellent introduction to the system:

```bash
python3 asd_test_cli.py demo
```

This will:
1. Start a new assessment for a 24-month-old child
2. Get a milestone
3. Generate example responses for each score level (0-4)
4. Score each response and display the results

#### Interactive Assessment

For a guided assessment experience:

```bash
python3 asd_test_cli.py interactive
```

This interactive session will:
1. Prompt for child's age
2. Present milestones one by one
3. Provide response options or allow custom responses
4. Score responses and show results
5. Generate a comprehensive report at the end

## 2. Web Application

The web application (`asd_test_webapp.py`) provides a user-friendly interface for testing the assessment system.

### Starting the Web App

```bash
streamlit run asd_test_webapp.py
```

This will launch a local web server and open the application in your default browser.

### Web App Features

The web application includes the following pages:

- **Interactive Assessment**: Complete a full assessment by responding to milestones
- **Test Individual Responses**: Test specific responses for any milestone
- **Benchmark**: Run performance tests and view metrics
- **Response Examples**: Explore example responses for each score level
- **Test History**: View a history of all tests run in the session
- **Report**: Generate and view a developmental report

### Web App Workflow

1. Use the sidebar to start the API server if it's not running
2. Set the child's age
3. Navigate to the desired testing page
4. Interact with the system and view results

## API Server Management

The API server must be running for the testing framework to function. Here are direct commands for managing the API server:

### Starting the API Server Directly

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8002
```

### Running the API Server in the Background

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8002 &
```

### Checking If the API Server Is Running

```bash
curl -s http://localhost:8002/health
```

or

```bash
lsof -i :8002
```

### Stopping the API Server

```bash
pkill -f "python3 -m uvicorn"
```

### Starting the API Server with the Helper Script

```bash
./start_api.sh
```

### Testing Using the Helper Script

```bash
./test_response.sh test "walks independently" "yes, he walks independently"
```

## Common Testing Scenarios

### Scenario 1: Validating Response Scoring

To verify that responses are scored correctly:

```bash
# CLI approach
python3 asd_test_cli.py test "walks independently" "no, not yet"                   # Should score 0 - CANNOT_DO
python3 asd_test_cli.py test "walks independently" "he used to walk but regressed" # Should score 1 - LOST_SKILL
python3 asd_test_cli.py test "walks independently" "sometimes, but not consistently" # Should score 2 - EMERGING
python3 asd_test_cli.py test "walks independently" "with help, he can walk"        # Should score 3 - WITH_SUPPORT
python3 asd_test_cli.py test "walks independently" "yes, he walks independently"   # Should score 4 - INDEPENDENT

# Or use the web app's "Test Individual Responses" page
```

### Scenario 2: Running a Complete Assessment

For a complete assessment experience:

```bash
# CLI approach - interactive mode
python3 asd_test_cli.py interactive

# Or use the web app's "Interactive Assessment" page
```

### Scenario 3: Performance Testing

To evaluate system performance:

```bash
# CLI approach
python3 asd_test_cli.py benchmark --iterations 100

# Or use the web app's "Benchmark" page
```

### Scenario 4: Quick Testing with Shell Script

The `test_response.sh` script provides quick access to the API for testing:

```bash
# Get help on usage
./test_response.sh help

# Set a child's age
./test_response.sh age 24

# Get the next milestone
./test_response.sh milestone

# Test a response
./test_response.sh test "walks independently" "not at all"

# Generate a report
./test_response.sh report
```

## Troubleshooting

### API Server Issues

#### Address Already in Use Error

If you see an error like "address already in use" when starting the API server:

```bash
# Check what's using port 8002
lsof -i :8002

# Kill the processes using that port
kill -9 [PID1] [PID2]  # Replace with actual PIDs

# Alternatively, kill all uvicorn processes
pkill -f "python3 -m uvicorn"

# Restart the API server
python3 -m uvicorn app:app --host 0.0.0.0 --port 8002
```

#### Command Not Found for Python

If you see "command not found: python" but you know Python is installed:

1. Use `python3` instead of `python` (especially on macOS)
2. Verify your Python path:
   ```bash
   which python3
   ```
3. Use the full path if needed:
   ```bash
   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8002
   ```

#### Dependency Issues

If you encounter problems with sentence-transformers or other dependencies:

```bash
# Install directly from pip
python3 -m pip install sentence-transformers

# If you still have issues, try:
python3 -m pip install --upgrade pip
python3 -m pip install --force-reinstall sentence-transformers
```

#### Permission Denied for Scripts

If you get "permission denied" when running scripts:

```bash
chmod +x script_name.sh  # Make the script executable
```

#### Verifying API Functionality

To check if the API is running and responding correctly:

```bash
curl -s http://localhost:8002/health | jq .
curl -s http://localhost:8002/next-milestone | jq .
```

### Scoring Inconsistencies

If you notice unexpected scoring results:

1. Use simple, clear phrases for more consistent scoring
2. Avoid complex negations like "not at all, he has never walked independently" as they can be misinterpreted
3. For the most reliable results, use these standard response templates:
   - CANNOT_DO: "not at all" or "no, not yet"
   - LOST_SKILL: "used to but lost it" or "he had this skill but lost it"
   - EMERGING: "sometimes" or "occasionally" 
   - WITH_SUPPORT: "with help" or "with assistance"
   - INDEPENDENT: "yes" or "consistently"

### Integration Test Failures

If you encounter integration test failures:

1. Check for syntax errors in the assessment engine:
   ```bash
   python3 -m py_compile assessment_engine.py
   ```

2. Run the integration test with more detailed output:
   ```bash
   python3 integrate_nlp.py --verbose
   ```

3. Manually fix any syntax issues in the relevant files

## Advanced API Server Management

### Checking for Running API Processes

```bash
ps aux | grep -i "app.py\|uvicorn" | grep -v grep
```

### Running the API Server with Different Port

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8003
```

### Running the API Server with Reload Option (for Development)

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

## Response Scoring Results

The API scoring system provides a 0-4 scale score based on caregiver responses. Here's what each score means:

0. **CANNOT_DO**: The skill is not acquired
1. **LOST_SKILL**: The skill was acquired but has been lost
2. **EMERGING**: The skill is emerging but inconsistent
3. **WITH_SUPPORT**: The skill is consistent but needs support
4. **INDEPENDENT**: The skill is acquired and present in all situations

The scoring engine analyzes the response text to determine the appropriate score. For example:

| Response | Likely Score | Explanation |
|----------|--------------|-------------|
| "not at all" | 0 | Indicates skill is not present |
| "used to but lost it" | 1 | Indicates regression |
| "sometimes" | 2 | Indicates inconsistent ability |
| "with help" | 3 | Indicates need for support |
| "yes consistently" | 4 | Indicates independent ability |

## Extensions and Customization

The testing framework is designed to be extensible:

1. **Adding New Test Types**: Modify `asd_test_cli.py` or `asd_test_webapp.py` to add new commands or pages
2. **Customizing Response Templates**: Edit the `RESPONSE_TEMPLATES` dictionary in either file
3. **Modifying Visualization**: Update the charts and displays in `asd_test_webapp.py`

## Requirements

All required dependencies are listed in `requirements.txt`. The key packages include:

- click>=8.1.3
- pandas>=1.5.3
- requests>=2.28.1
- streamlit>=1.22.0
- termcolor>=2.3.0
- inquirer>=3.1.3
- tabulate>=0.9.0
- matplotlib>=3.7.1
- plotly>=5.14.1
- sentence-transformers>=2.2.2
- uvicorn>=0.21.1
- fastapi>=0.95.0
- python-multipart>=0.0.6
- nltk>=3.8.1
- jinja2>=3.1.2
- scikit-learn>=1.3.0

## Environment Variables

The testing framework uses the following environment variables that can be customized:

- `API_PORT`: Port for the API server (default: 8002)
- `API_HOST`: Host for the API server (default: 0.0.0.0)

## Contributing

Contributions to improve the testing framework are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Specify your license information here]

## ASD Assessment API Testing Framework

## Overview

This document provides comprehensive information about the ASD Assessment API testing framework. The framework is designed to test the performance, reliability, and functionality of the API endpoints. It includes tools for single endpoint testing, comprehensive test runs, and detailed report generation.

## Components

The testing framework consists of the following components:

1. **Test Scripts**:
   - `scripts/test_single_endpoint.sh`: Tests a single API endpoint with configurable iterations and data.
   - `scripts/run_all_tests.sh`: Runs tests for all API endpoints and generates a consolidated report.
   - `scripts/generate_test_report.py`: Generates HTML reports from test results.

2. **Test Data**:
   - `test_data/sample_questions.json`: Sample data for testing the `/question` endpoint.
   - `test_data/sample_keywords.json`: Sample data for testing the `/keywords` endpoint.
   - `test_data/sample_scores.json`: Sample data for testing the `/send-score` endpoint.
   - `test_data/sample_response.json`: Sample data for testing the `/score-response` endpoint.

3. **Test Results**:
   - `test_results/api_test_results.json`: Raw JSON results from the most recent test run.
   - `test_results/api_test_report.html`: HTML report for a single endpoint test.
   - `test_results/consolidated_report.html`: Comprehensive HTML report for all endpoints.
   - Various `.png` files: Performance charts generated during testing.

## API Endpoints Tested

The framework tests the following API endpoints:

1. **Question Endpoint** (`/question`):
   - Tests the API's ability to process questions related to developmental milestones.
   - Validates milestone matching functionality.
   - Measures response time and success rate.

2. **Keywords Endpoint** (`/keywords`):
   - Tests the API's ability to process categorized keywords.
   - Validates keyword management functionality.
   - Measures response time and success rate.

3. **Send Score Endpoint** (`/send-score`):
   - Tests the API's ability to record scores for specific milestones.
   - Validates score recording functionality.
   - Measures response time and success rate.

4. **Score Response Endpoint** (`/score-response`):
   - Tests the API's ability to analyze parent/caregiver responses.
   - Validates score determination functionality, including the fallback pattern detection mechanism.
   - Measures response time and success rate.

## Running Tests

### Single Endpoint Testing

To test a single API endpoint:

```bash
./scripts/test_single_endpoint.sh /endpoint_path iterations data_file
```

Parameters:
- `/endpoint_path`: The path of the endpoint to test (e.g., `/question`, `/keywords`, `/send-score`, `/score-response`).
- `iterations`: The number of test iterations to run (recommended: 10-100).
- `data_file`: The path to the JSON file containing test data for the endpoint.

Example:
```bash
./scripts/test_single_endpoint.sh /question 30 test_data/sample_questions.json
```

This command will:
1. Send 30 requests to the `/question` endpoint using data from `test_data/sample_questions.json`.
2. Record response times and success/failure status for each request.
3. Generate performance charts.
4. Create an HTML report at `test_results/api_test_report.html`.

### Comprehensive Testing

To test all API endpoints:

```bash
./scripts/run_all_tests.sh
```

This command will:
1. Test each endpoint with a predefined number of iterations (default: 30).
2. Use the appropriate sample data file for each endpoint.
3. Save individual test results and generate performance charts.
4. Create a consolidated HTML report at `test_results/consolidated_report.html`.

## Test Data Format

Each endpoint requires a specific JSON format for test data:

### Question Endpoint

```json
{
  "text": "Does the child respond when called by name?",
  "milestone_id": "Recognizes familiar people"
}
```

### Keywords Endpoint

```json
{
  "category": "CANNOT_DO",
  "keywords": [
    "no", "not", "never", "doesn't", "does not", 
    "cannot", "can't", "unable", "hasn't", "has not", 
    "not able", "not at all", "not yet started", "not capable"
  ]
}
```

### Send Score Endpoint

```json
{
  "milestone_id": "Shows empathy",
  "score": 4,
  "score_label": "INDEPENDENT"
}
```

### Score Response Endpoint

```json
{
  "milestone_behavior": "Recognizes familiar people",
  "response": "My child is very good at recognizing familiar faces. He always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well."
}
```

## Generated Reports

### Single Endpoint Report

The single endpoint report (`test_results/api_test_report.html`) includes:

1. **Test Summary**:
   - Endpoint tested
   - Number of iterations
   - Success rate
   - Average response time
   - Timestamp of the test

2. **Performance Charts**:
   - Response time distribution
   - Success/failure ratio
   - Response time trend

3. **Detailed Results**:
   - Request data
   - Response data
   - Status code
   - Response time

### Consolidated Report

The consolidated report (`test_results/consolidated_report.html`) includes:

1. **Overall Summary**:
   - Total tests run
   - Overall success rate
   - Average response time across all endpoints
   - Timestamp of the test

2. **Endpoint Comparison**:
   - Comparative performance charts
   - Success rates by endpoint
   - Average response times by endpoint

3. **Individual Endpoint Summaries**:
   - Detailed results for each endpoint
   - Links to individual test results

## Interpreting Results

When analyzing test results, consider the following metrics:

1. **Success Rate**:
   - 100%: All requests successful, ideal scenario.
   - 95-99%: Acceptable for most use cases.
   - <95%: May indicate issues with the API or test data.

2. **Response Time**:
   - <5ms: Excellent performance.
   - 5-20ms: Good performance.
   - >20ms: May require optimization.

3. **Response Time Variability**:
   - Low variability indicates consistent performance.
   - High variability may indicate intermittent issues.

## Customizing Tests

### Modifying Test Data

To customize test data, edit the JSON files in the `test_data` directory. Ensure the data follows the required format for each endpoint.

### Adjusting Test Parameters

To modify the number of iterations or other test parameters:

1. For single endpoint tests, adjust the parameters when calling `test_single_endpoint.sh`.
2. For comprehensive tests, edit the `run_all_tests.sh` script to change the number of iterations or data files used.

### Adding New Test Data

To add new test data files:

1. Create a new JSON file in the `test_data` directory.
2. Follow the required format for the endpoint you're testing.
3. Use the file path when calling `test_single_endpoint.sh`.

## Troubleshooting

### API Server Issues

If tests fail with connection errors:

1. Ensure the API server is running (`uvicorn main:app --reload --port 8003`).
2. Check if the port (8003) is already in use by another process.
3. Verify network connectivity if testing a remote API.

### Data Format Issues

If tests fail with 400 Bad Request errors:

1. Check the test data format matches the API requirements.
2. Verify JSON syntax and field names.
3. Ensure required fields are present.

### Report Generation Issues

If report generation fails:

1. Ensure Python 3.8+ is installed.
2. Check that matplotlib and other required libraries are installed.
3. Verify the `test_results` directory exists and is writable.

## Best Practices

1. **Regular Testing**:
   - Run comprehensive tests after any API code changes.
   - Perform regular performance monitoring.

2. **Progressive Load Testing**:
   - Start with a small number of iterations and gradually increase.
   - Monitor system resources during high-load tests.

3. **Data Variety**:
   - Use diverse test data to cover different scenarios.
   - Include both common and edge cases.

4. **Results Analysis**:
   - Compare results over time to identify trends.
   - Investigate any significant performance changes.

## Conclusion

The ASD Assessment API testing framework provides comprehensive tools for evaluating the API's performance, reliability, and functionality. By following the guidelines in this document, you can effectively test the API and identify areas for improvement.