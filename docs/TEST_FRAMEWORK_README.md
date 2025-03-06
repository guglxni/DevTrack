# ASD Developmental Milestone Assessment API Testing Framework

This testing framework provides a comprehensive solution for testing the ASD Developmental Milestone Assessment API. It includes tools for generating realistic test data based on various developmental profiles, running API tests, and generating detailed test reports.

## Overview

The testing framework consists of the following components:

1. **Test Configurations (`test_configs.py`)**: Defines various developmental profiles for different ages and domains.
2. **Test Data Generator (`generate_test_data.py`)**: Creates realistic caregiver responses based on milestone characteristics and developmental profiles.
3. **API Tester (`test_api.py`)**: Tests the API endpoints and generates performance metrics.
4. **Test Runner (`run_tests.sh`)**: Shell script that automates the entire testing process.

## Requirements

- Python 3.6+
- Required Python packages (see `requirements.txt`)
- Bash shell (for running the automation script)

## Getting Started

### Installation

1. Ensure all required Python packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Make the test runner script executable:
   ```bash
   chmod +x run_tests.sh
   ```

### Basic Usage

To run a basic test with default settings:

```bash
./run_tests.sh
```

This will:
1. Generate test data for a 24-month-old with neurotypical development profile
2. Run API tests against the default API URL (http://localhost:8002)
3. Generate an HTML report with test results

### Advanced Usage

The framework offers numerous configuration options:

```bash
./run_tests.sh --age 30 --num-tests 50 --domains GM,FM --profile asd --start-server --concurrent
```

This example will:
1. Start the API server on the default port (8002)
2. Generate test data for a 30-month-old with an ASD development profile
3. Focus on Gross Motor (GM) and Fine Motor (FM) domains
4. Run 50 test cases concurrently
5. Generate a detailed HTML report

For a complete list of options:

```bash
./run_tests.sh --help
```

## Test Data Generation

### Available Developmental Profiles

The framework includes several pre-defined developmental profiles:

- **Neurotypical**: Typical development patterns
- **Delay**: General developmental delays across domains
- **ASD**: Autism Spectrum Disorder profile with regression patterns
- **Uneven Motor**: Strong motor skills, weak language and social skills
- **Uneven Cognitive**: Strong cognitive/language skills, weak motor/social skills
- **Random**: Randomly selects from the above profiles

### Response Types

The generator can create responses of different lengths:

- **Short**: Basic responses (1-2 sentences)
- **Medium**: More detailed responses (3-4 sentences)
- **Long**: Very detailed responses with contextual information (5+ sentences)

## API Testing

The API tester checks the following endpoints:

- `/` (Health check)
- `/set-child-age`
- `/next-milestone`
- `/score-response`
- `/batch-score`
- `/generate-report`

For each endpoint, the tester records:
- Success/failure status
- Response time
- HTTP status code
- Expected vs. actual values (where applicable)

## Test Reports

The framework generates detailed HTML reports that include:

- Overall test summary (pass/fail counts, success rate)
- Visualizations (success rate pie chart, response time bar chart)
- Per-endpoint performance metrics
- Detailed results for each test case

Reports are saved to the specified output file (default: `api_test_report.html`).

## Example Workflow

1. **Generate Test Data**: Create realistic test data for specific age and profile
   ```bash
   python3 generate_test_data.py --age 24 --profile neurotypical --count 30 --output test_data.json
   ```

2. **Run API Tests**: Test the API using the generated data
   ```bash
   python3 test_api.py --url http://localhost:8002 --age 24 --data test_data.json --tests 30
   ```

3. **Automated Testing**: Or run everything with a single command
   ```bash
   ./run_tests.sh --age 24 --profile neurotypical --num-tests 30 --start-server
   ```

## Advanced Configuration

### Test Data Configuration

You can customize the test data generation with various parameters:

```bash
python3 generate_test_data.py --age 18 --profile asd --count 50 --domains GM,FM,COG --response_length long --output test_data.json
```

### API Test Configuration

You can customize the API tests with various parameters:

```bash
python3 test_api.py --url http://localhost:8002 --age 24 --data test_data.json --tests 30 --domains GM,FM --report detailed_report.html --concurrent --edge-cases --verbose
```

## Troubleshooting

- **API Connection Issues**: If the API is not responsive, check if the server is running and the correct port is being used.
- **Test Data Errors**: Ensure the test data file is properly formatted JSON with the expected structure.
- **Script Permissions**: Make sure the shell script has execution permissions (`chmod +x run_tests.sh`).
- **Port Conflicts**: If the API server won't start, check for processes already using the specified port.

## Contributing

This testing framework can be extended with additional features:

- New developmental profiles in `test_configs.py`
- Enhanced response templates in `generate_test_data.py`
- Additional metrics or visualizations in `test_api.py`
- New test automation options in `run_tests.sh` 