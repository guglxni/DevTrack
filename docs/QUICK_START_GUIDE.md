# ASD Assessment API Quick Start Guide

This guide provides step-by-step instructions to quickly get started with the ASD Assessment API and supporting tools.

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm 6 or higher
- Git (for cloning the repository)

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/asd-assessment-api.git
cd asd-assessment-api
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Install Node.js dependencies (for web demo)**

```bash
cd webapp
npm install
cd ..
```

## Starting the API Server

1. **Start the API server**

```bash
cd src/api
uvicorn main:app --reload --port 8003
```

2. **Verify the server is running**

Open a web browser and navigate to `http://localhost:8003/docs`. You should see the FastAPI Swagger documentation page with all available endpoints.

## API Endpoints Overview

The API provides four main endpoints:

1. **Question Endpoint (`/question`)**
   - Processes questions related to developmental milestones
   - Example request:
     ```json
     {
       "text": "Does the child respond when called by name?",
       "milestone_id": "Recognizes familiar people"
     }
     ```

2. **Keywords Endpoint (`/keywords`)**
   - Processes categorized keywords used for analyzing responses
   - Example request:
     ```json
     {
       "category": "CANNOT_DO",
       "keywords": ["no", "not", "never", "doesn't"]
     }
     ```

3. **Send Score Endpoint (`/send-score`)**
   - Records scores for specific developmental milestones
   - Example request:
     ```json
     {
       "milestone_id": "Shows empathy",
       "score": 4,
       "score_label": "INDEPENDENT"
     }
     ```

4. **Score Response Endpoint (`/score-response`)**
   - Analyzes parent/caregiver responses to determine appropriate scores
   - Example request:
     ```json
     {
       "milestone_behavior": "Recognizes familiar people",
       "response": "My child always smiles when he sees family members."
     }
     ```

5. **Comprehensive Assessment Endpoint (`/comprehensive-assessment`)**
   - Combines the functionality of all other endpoints in a single call
   - Example request:
     ```json
     {
       "question": "Does the child recognize familiar people?",
       "milestone_behavior": "Recognizes familiar people",
       "parent_response": "My child always smiles when he sees grandparents or his favorite babysitter.",
       "keywords": {
         "CANNOT_DO": ["no", "not", "never"],
         "INDEPENDENT": ["always", "consistently", "definitely"]
       }
     }
     ```

## Running Tests

### Testing a Single Endpoint

Use the provided script to test a single endpoint:

```bash
./scripts/test_single_endpoint.sh /endpoint_path iterations data_file
```

Example:
```bash
./scripts/test_single_endpoint.sh /question 10 test_data/sample_questions.json
```

### Testing the Comprehensive Endpoint

Use the dedicated script to test the comprehensive endpoint:

```bash
./scripts/test_comprehensive_endpoint.sh test_data/comprehensive_test.json
```

To run all comprehensive endpoint test cases:

```bash
./scripts/test_comprehensive_all.sh
```

### Running All Tests

To run tests for all endpoints:

```bash
./scripts/run_all_tests.sh
```

### Viewing Test Results

After running tests, results are available in the `test_results` directory:

- Single endpoint test report: `test_results/api_test_report.html`
- Consolidated report: `test_results/consolidated_report.html`
- Performance charts: Various `.png` files in `test_results`

## Using the Web Demo

1. **Start the web demo (separate terminal)**

```bash
cd webapp
npm start
```

2. **Access the web demo**

Open a web browser and navigate to `http://localhost:3000`

3. **Use the tabbed interface**

The web demo provides tabs for testing each API endpoint:

- Question tab: Test the `/question` endpoint
- Keywords tab: Test the `/keywords` endpoint
- Send Score tab: Test the `/send-score` endpoint
- Score Response tab: Test the `/score-response` endpoint

## Quick Start with Demo Script

For convenience, you can use the demo script to start both the API server and web application:

```bash
./scripts/start_demo.sh
```

This script starts both servers in the background. Access the web demo at `http://localhost:3000`.

## Basic Troubleshooting

### API Server Issues

1. **Port already in use**

If you see an error like `address already in use`, another process is already using port 8003. Find and terminate the process:

```bash
lsof -i :8003
kill -9 [PID]
```

2. **Module not found errors**

If you see import errors, ensure you've installed all dependencies:

```bash
pip install -r requirements.txt
```

3. **NLP Module Fallback**

If you see logs mentioning "Advanced NLP module not available", the system is using keyword pattern matching instead. This is normal and doesn't affect functionality, but may reduce accuracy.

### Web Demo Issues

1. **Cannot access localhost:3000**

Try using `http://127.0.0.1:3000` instead, or check for any errors when starting the Node.js server.

2. **API connection errors**

Ensure the API server is running on port 8003. Check browser console for specific error messages.

3. **Node.js errors**

Ensure you have the correct Node.js version and have installed all dependencies:

```bash
cd webapp
npm install
```

## Next Steps

After getting started:

1. **Explore the API Documentation**: Review `docs/API_DOCUMENTATION.md` for detailed API usage information.

2. **Run Performance Tests**: Use the testing framework to assess API performance.

3. **Review Test Reports**: Study the HTML reports to understand API behavior.

4. **Customize the Application**: Modify test data or web demo to suit your specific needs.

For more detailed information, refer to the following documentation:

- [API Documentation](API_DOCUMENTATION.md)
- [Testing Documentation](README_TESTING.md)
- [Web Demo Documentation](../webapp/README.md)

## Support

If you encounter issues not covered in this guide, please:

1. Check the detailed documentation
2. Review the troubleshooting sections
3. Check for existing issues in the repository
4. Contact the project maintainer 