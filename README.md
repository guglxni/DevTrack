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

## Web Application

The web application provides a user-friendly interface for:

1. Running interactive assessments
2. Testing individual responses
3. Viewing detailed reports
4. Benchmarking the system
5. Exploring milestone behaviors

## License

This project is licensed under the MIT License - see the LICENSE file for details. 