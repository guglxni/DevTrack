# ASD Assessment System

This project provides a comprehensive system for assessing Autism Spectrum Disorder (ASD) in children through natural language processing and developmental milestone tracking.

## Project Structure

The project is organized into the following directories:

- `src/`: Main source code
  - `core/`: Core assessment engine and NLP components
  - `api/`: FastAPI server for the backend
  - `web/`: Streamlit web application
  - `testing/`: Testing frameworks and utilities
  - `utils/`: Utility scripts and tools
- `data/`: Data files for milestones and assessment criteria
- `docs/`: Documentation files
- `scripts/`: Shell scripts for automation
- `logs/`: Log files
- `test_results/`: Test results and reports

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

You can run the application using the main.py script:

```bash
# Start the API server
python3 main.py --api

# Start the web application
python3 main.py --web

# Show help
python3 main.py --help
```

Alternatively, you can run the components separately:

```bash
# Start the API server
python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8003

# Start the web application
python3 -m streamlit run src/web/asd_test_webapp.py
```

## Features

- Interactive assessment of developmental milestones
- Natural language processing for analyzing caregiver responses
- Comprehensive reporting and visualization
- Benchmark testing for system performance
- Individual response testing for specific milestones
- Customizable keywords for scoring categories
- Manual scoring capabilities for special cases

## API Endpoints

The API server provides the following endpoints:

- `/set-child-age`: Set the child's age for appropriate milestone filtering
- `/next-milestone`: Get the next milestone to assess
- `/score-response`: Score a caregiver response for a specific milestone
- `/batch-score`: Score multiple responses in parallel
- `/generate-report`: Generate a comprehensive assessment report
- `/reset`: Reset the assessment engine for a new assessment
- `/health`: Health check endpoint
- `/all-milestones`: Get all available milestone behaviors
- `/question`: Receive and process questions about a child's behavior
- `/keywords`: Update keywords for specific scoring categories
- `/send-score`: Manually set a score for a specific milestone

## Web Application

The web application provides a user-friendly interface for:

1. Running interactive assessments
2. Testing individual responses
3. Viewing detailed reports
4. Benchmarking the system
5. Exploring milestone behaviors

## Testing Framework

The project includes a comprehensive API testing framework located in `src/testing/comprehensive_api_tester.py`. This framework allows you to:

1. Test all API endpoints, including the new ones
2. Run complex test scenarios and workflows
3. Perform load testing
4. Generate detailed reports with visualizations

### Running Tests

```bash
# Run basic API tests
python3 src/testing/comprehensive_api_tester.py --verbose basic

# Run a complete assessment flow
python3 src/testing/comprehensive_api_tester.py --verbose assessment --age 24

# Test the keywords update workflow
python3 src/testing/comprehensive_api_tester.py --verbose keywords

# Perform load testing
python3 src/testing/comprehensive_api_tester.py --verbose load --endpoint /health --count 100 --concurrent
```

For more details, see [Testing Framework Documentation](src/testing/README_TESTING.md).

## API Usage Guide

Below is a comprehensive guide for using the API endpoints:

### Basic Assessment Flow

1. **Set Child Age**
   ```bash
   curl -X POST "http://localhost:8003/set-child-age" \
     -H "Content-Type: application/json" \
     -d '{"age": 24, "name": "Alex"}'
   ```

2. **Get Next Milestone**
   ```bash
   curl -X GET "http://localhost:8003/next-milestone"
   ```

3. **Score Response**
   ```bash
   curl -X POST "http://localhost:8003/score-response" \
     -H "Content-Type: application/json" \
     -d '{
       "milestone_behavior": "Shows empathy",
       "response": "Yes, she always notices when I am sad and tries to comfort me."
     }'
   ```

4. **Generate Report**
   ```bash
   curl -X GET "http://localhost:8003/generate-report"
   ```

5. **Reset Assessment**
   ```bash
   curl -X POST "http://localhost:8003/reset"
   ```

### Advanced Features

#### Submit a Question
Use this endpoint to submit questions about a child's behavior:

```bash
curl -X POST "http://localhost:8003/question" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Does the child respond when called by name?",
    "milestone_id": "Recognizes familiar people"
  }'
```

#### Update Keywords for Scoring Categories
Customize the keywords used for automatic scoring:

```bash
curl -X POST "http://localhost:8003/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "CANNOT_DO",
    "keywords": [
      "no", "not", "never", "doesn't", "does not", 
      "cannot", "can't", "unable", "hasn't", "has not", 
      "not able", "not at all", "not yet started", "not capable"
    ]
  }'
```

Available categories:
- `NOT_RATED`: For unrated milestones (-1)
- `CANNOT_DO`: For skills not acquired (0)
- `LOST_SKILL`: For skills acquired but lost (1)
- `EMERGING`: For emerging and inconsistent skills (2)
- `WITH_SUPPORT`: For skills acquired with support (3)
- `INDEPENDENT`: For skills fully acquired (4)

#### Manual Scoring
Manually score a milestone when automatic scoring is not sufficient:

```bash
curl -X POST "http://localhost:8003/send-score" \
  -H "Content-Type: application/json" \
  -d '{
    "milestone_id": "Shows empathy",
    "score": 4,
    "score_label": "INDEPENDENT"
  }'
```

Score values and their meanings:
- -1: `NOT_RATED` - No rating provided
- 0: `CANNOT_DO` - Skill not acquired
- 1: `LOST_SKILL` - Skill was acquired but lost
- 2: `EMERGING` - Skill is emerging but inconsistent
- 3: `WITH_SUPPORT` - Skill is acquired but needs support
- 4: `INDEPENDENT` - Skill is fully acquired

#### Batch Scoring
Score multiple responses at once:

```bash
curl -X POST "http://localhost:8003/batch-score" \
  -H "Content-Type: application/json" \
  -d '{
    "responses": [
      {
        "milestone_behavior": "Shows empathy",
        "response": "Yes, he always notices when someone is upset."
      },
      {
        "milestone_behavior": "Makes eye contact",
        "response": "She rarely makes eye contact, even when I call her name."
      }
    ]
  }'
```

#### Get All Milestones
Retrieve all available milestones:

```bash
curl -X GET "http://localhost:8003/all-milestones"
```

### Response Format

Most API endpoints return JSON responses with the following structure:

```json
{
  "status": "success",
  "message": "Operation completed successfully",
  "data": { ... }
}
```

In case of errors, the response will include an appropriate HTTP status code and an error message:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Documentation

For more detailed information, see:

- [API Documentation](docs/API_DOCUMENTATION.md) - Detailed API endpoint documentation
- [Testing Framework](src/testing/README_TESTING.md) - Guide to using the testing framework
- [Quick Start Guide](docs/QUICK_START_GUIDE.md) - Quick introduction to using the system

## License

This project is licensed under the MIT License - see the LICENSE file for details. 