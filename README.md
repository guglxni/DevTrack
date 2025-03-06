# ASD Assessment API and Demo Application

## Project Overview

This project provides a comprehensive API and demonstration web application for processing and analyzing developmental milestone assessments for Autism Spectrum Disorder (ASD) screening. The system supports question processing, keyword management, score recording, and response analysis to facilitate the assessment of developmental milestones in children.

## Components

The project consists of three main components:

1. **API Service**: A FastAPI-based API that provides endpoints for processing developmental milestone assessments.
2. **Testing Framework**: A suite of tools for testing the API endpoints and generating performance reports.
3. **Web Demo Application**: A web-based interface for demonstrating the API capabilities to stakeholders.

## API Service

The API service is a FastAPI application that provides the following endpoints:

- `/question`: Processes questions related to developmental milestones
- `/keywords`: Manages keywords used for analyzing parent/caregiver responses
- `/send-score`: Records scores for specific developmental milestones
- `/score-response`: Analyzes parent/caregiver responses to determine appropriate scores
- `/comprehensive-assessment`: Combines the functionality of all other endpoints in a single call, processing a question, updating keywords, analyzing a response, and recording a score

For detailed API documentation, see [API Documentation](docs/API_DOCUMENTATION.md).

### Running the API Service

To start the API service:

```bash
cd src/api
uvicorn main:app --reload --port 8003
```

The API will be available at `http://localhost:8003`.

## Testing Framework

The testing framework provides tools for testing the API endpoints and generating performance reports. It includes:

- Single endpoint testing script: `scripts/test_single_endpoint.sh`
- Comprehensive test runner: `scripts/run_all_tests.sh`
- Test report generator: `scripts/generate_test_report.py`

### Running Tests

To run tests for a single endpoint:

```bash
./scripts/test_single_endpoint.sh /endpoint_path iterations data_file
```

For example:
```bash
./scripts/test_single_endpoint.sh /question 10 test_data/sample_questions.json
```

To test the comprehensive endpoint:

```bash
./scripts/test_comprehensive_endpoint.sh test_data/comprehensive_test.json
```

To run tests for all comprehensive endpoint test cases:

```bash
./scripts/test_comprehensive_all.sh
```

To run tests for all endpoints:

```bash
./scripts/run_all_tests.sh
```

Test results are saved to the `test_results` directory, including:
- Raw JSON results: `api_test_results.json`
- Performance charts: PNG files in the `test_results` directory
- HTML reports: `api_test_report.html` and `consolidated_report.html`

For detailed testing documentation, see [Testing Documentation](docs/README_TESTING.md).

## Web Demo Application

The web demo application provides a user-friendly interface for demonstrating the API capabilities. It includes tabs for each API endpoint and displays responses in a structured format.

### Running the Web Demo

To start the web demo application:

```bash
cd webapp
npm install
npm start
```

The web application will be available at `http://localhost:3000`.

Alternatively, you can use the convenience script to start both the API service and web demo:

```bash
./scripts/start_demo.sh
```

**Note**: This script starts both servers in the background. If port 8003 is already in use by another process, the API server may not start properly. Check for existing processes with `lsof -i :8003`.

### Web Demo Screenshots

Screenshots of the web demo application are available in the `webapp/img` directory. The application provides a tabbed interface for testing each API endpoint.

## Milestone Domains

The API includes milestones across the following developmental domains:

1. **Cognitive**: Mental processes like thinking, learning, problem-solving
2. **Gross Motor**: Large muscle movements (walking, running, jumping)
3. **Fine Motor**: Small muscle movements (grasping, writing, cutting)
4. **Expressive Language**: Communication through speech, gestures, or writing
5. **Activities of Daily Living**: Self-care skills (eating, dressing, hygiene)
6. **Social**: Interactions with others, relationships, play skills
7. **Emotional**: Understanding and expressing feelings, self-regulation
8. **Receptive Language**: Understanding and processing language

## Technical Details

### API Implementation

- Built with FastAPI
- Uses Uvicorn as the ASGI server
- Implements NLP processing with a fallback mechanism
- Includes CORS support for cross-origin requests

### Testing Framework

- Uses Python for API interaction
- Generates performance metrics (response time, success rate)
- Creates visual charts with matplotlib
- Produces HTML reports for easy viewing

### Web Demo Application

- Built with HTML, CSS, and JavaScript
- Uses a Node.js server for local hosting
- Implements a proxy to handle CORS issues
- Includes tabbed interface for testing different endpoints

## Troubleshooting

### API Service Issues

- If port 8003 is already in use, check for existing processes with `lsof -i :8003`
- If you see "Advanced NLP module not available" in logs, the system is using the fallback pattern matching

### Web Demo Issues

- If you cannot access the web demo, ensure that both the API service and web server are running
- Try using `http://127.0.0.1:3000` instead of `localhost` if you encounter connectivity issues
- Check the browser console for any JavaScript errors

## Additional Resources

- [Quick Start Guide](docs/QUICK_START_GUIDE.md) - Get started quickly with the basic functionality
- [Tutorial](docs/TUTORIAL.md) - Step-by-step guide to using the system
- [Enhanced Documentation](docs/README_ENHANCED.md) - Advanced usage and configuration options

## License

This project is licensed under the MIT License - see the LICENSE file for details. 