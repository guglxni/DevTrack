# ASD Assessment API - Quick Start Guide

This guide will help you start and use the ASD Assessment API with reliable scoring.

## Starting the API Server

To start the API server correctly, use the following command from the project root:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m uvicorn src.api.app:app --port 8003
```

Or use the provided script:

```bash
./scripts/start/start_api.sh
```

Important notes:
- Use the full path to your Python interpreter
- Run the command from the project root (not from src/api)
- Make sure port 8003 is not already in use

### Starting with R2R Enabled (Recommended)

For enhanced scoring with R2R (Reason to Retrieve) capabilities:

```bash
# Using the provided script
./start_with_r2r.sh

# Or manually
export ENABLE_R2R=true
python3 -m uvicorn src.api.app:app --port 8003
```

The R2R dashboard will be available at: http://localhost:8003/r2r-dashboard/

## Starting the Web UI

To start the web interface, use the following command from the project root:

```bash
cd webapp && npm start
```

Or use the provided script:

```bash
./scripts/start/start_demo.sh
```

This will start the web server on port 3000. You can then access the web interface at http://localhost:3000.

Important notes:
- Make sure the API server is running before starting the web interface
- Make sure port 3000 is not already in use
- The web interface proxies requests to the API server at http://localhost:8003

## Using the API

### Option 1: Using the Reliable Client (Recommended)

The simplest way to use the API is through the `ReliableASDClient` class:

```python
from src.reliable_api_client import ReliableASDClient

# Initialize the client
client = ReliableASDClient(api_url="http://localhost:8003")

# Score a response
result = client.score_response(
    "Recognizes familiar people", 
    "My child always smiles when he sees grandparents and recognizes all family members."
)

# Print the result
print(f"Score: {result['score_label']} ({result['score']})")
```

This client ensures consistent, accurate scoring for all milestones.

### Option 2: Direct API Calls

If you need to make direct API calls, make sure to include keywords in your requests:

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:8003"

# Define reliable keywords
keywords = {
    "INDEPENDENT": ["always recognizes", "knows everyone"],
    "CANNOT_DO": ["doesn't recognize anyone", "unable to recognize"]
    # Add other categories as needed
}

# Assessment data
data = {
    "question": "Does your child Recognizes familiar people?",
    "milestone_behavior": "Recognizes familiar people",
    "parent_response": "My response text",
    "keywords": keywords  # Always include keywords
}

# Make the API call
response = requests.post(
    f"{API_URL}/comprehensive-assessment",
    json=data
)

if response.status_code == 200:
    result = response.json()
    print(f"Score: {result['score_label']} ({result['score']})")
```

## Using the Web UI

The web interface provides an easy way to test different API endpoints:

1. **Comprehensive Assessment:** This is the most complete endpoint that processes a question, scores a response, and provides detailed results.
2. **Score Response:** A simpler endpoint that just scores a response for a specific milestone.
3. **Keywords:** Use this to update the keywords used for scoring.
4. **Send Score:** Manually send a score for a milestone.

For reliable scoring, use the "Comprehensive Assessment" tab which includes options to use the enhanced hybrid scoring.

## Hybrid Scoring Approach

The API uses a hybrid approach for scoring responses, which combines multiple NLP techniques:

1. **Word boundary-aware keyword matching:** Ensures keywords like "knows" aren't matched within words like "acknowledges"
2. **Negation detection:** Understands phrases like "doesn't recognize" vs "recognizes"
3. **Milestone-specific pattern matching:** Uses custom patterns for each milestone
4. **Special phrase handling:** Correctly interprets phrases like "Sometimes, but only when we encourage him"

This hybrid approach provides more accurate scoring compared to simple keyword matching.

## Testing the API

Several test scripts are provided to verify that the API is working correctly:

1. Run all tests at once:
   ```bash
   ./run_tests.sh
   ```

2. Test the comprehensive assessment endpoint:
   ```bash
   ./tests/test_comprehensive_endpoint.py
   ```

3. Test the reliable client:
   ```bash
   ./tests/test_api_client.py
   ```

4. Test specific hybrid scoring cases:
   ```bash
   ./tests/test_hybrid_scoring.py
   ```

5. Run end-to-end tests:
   ```bash
   ./tests/test_end_to_end.py
   ```

## Example Usage

Check the `examples/` directory for example scripts showing how to use the API:

1. Integration example:
   ```bash
   python examples/integration_example.py
   ```

2. Submit assessment example:
   ```bash
   python examples/submit_assessment.py
   ```

3. Direct keywords usage:
   ```bash
   python examples/use_direct_keywords.py
   ```

## Troubleshooting

If you encounter issues:

1. **Port conflicts**:
   - If port 8003 is in use, find and kill the process:
     ```bash
     lsof -i :8003  # Find the process ID
     kill -9 <PID>  # Kill the process
     ```
   - For port 3000 (web UI):
     ```bash
     lsof -i :3000  # Find the process ID
     kill -9 <PID>  # Kill the process
     ```

2. **Python not found**: Use the full path to your Python interpreter:
   ```bash
   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
   ```

3. **Module not found errors**: 
   - Make sure to run the server from the project root
   - If you see "No module named 'advanced_nlp'", it's okay - the system will fall back to basic scoring

4. **API Error: 0 -**:
   - This generic error usually means the web UI can't connect to the API server
   - Make sure the API server is running on port 8003
   - Try restarting both the API server and web UI

5. **Inconsistent scoring**: 
   - Always use the reliable client or provide keywords with your API calls
   - For direct API calls, use the comprehensive-assessment endpoint for best results

For more detailed information, refer to the `docs/RELIABLE_SCORING.md` documentation. 