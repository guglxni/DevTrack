# ASD Assessment Testing Framework: Quick Start Guide

This guide provides the essential steps to quickly get started with the ASD assessment testing framework.

## Installation in 3 Steps

1. **Clone & Install**
   ```bash
   git clone [repository-url]
   cd asd-assessment-system
   python3 -m pip install -r requirements.txt
   ```

2. **Make Scripts Executable**
   ```bash
   chmod +x *.sh
   chmod +x asd_test_cli.py
   ```

3. **Start the API Server**
   ```bash
   ./start_api.sh
   ```
   If the port is already in use:
   ```bash
   pkill -f "python3 -m uvicorn"
   ./start_api.sh
   ```

## Quick Testing (CLI)

### Run the Demo
```bash
python3 asd_test_cli.py demo
```

### Test Individual Responses
```bash
# Format: python3 asd_test_cli.py test "milestone" "response"
python3 asd_test_cli.py test "walks independently" "yes, consistently"
```

### Run Interactive Session
```bash
python3 asd_test_cli.py interactive
```

## Quick Testing (Shell Script)

```bash
# Set a child's age
./test_response.sh age 24

# Get a milestone
./test_response.sh milestone

# Test a response
./test_response.sh test "walks independently" "not at all"

# Generate a report
./test_response.sh report
```

## Web Interface

```bash
streamlit run asd_test_webapp.py
```

## Common Scoring Patterns

For consistent scoring, use these response patterns:

| Score | Label | Example Response |
|-------|-------|------------------|
| 0 | CANNOT_DO | "not at all" |
| 1 | LOST_SKILL | "used to but lost it" |
| 2 | EMERGING | "sometimes" |
| 3 | WITH_SUPPORT | "with help" |
| 4 | INDEPENDENT | "yes, consistently" |

## Troubleshooting

### API Server Issues
```bash
# Check if server is running
curl -s http://localhost:8002/health | jq .

# Check for processes using port 8002
lsof -i :8002

# Kill processes and restart
pkill -f "python3 -m uvicorn"
./start_api.sh
```

### Dependency Issues
```bash
python3 -m pip install --upgrade pip
python3 -m pip install --force-reinstall sentence-transformers
```

## Next Steps

For more detailed information, refer to the following documents:
- `README_TESTING.md`: Comprehensive testing documentation
- `API_DOCUMENTATION.md`: API endpoints and usage details
- `TUTORIAL.md`: Step-by-step tutorial 