#!/bin/bash
# ASD Developmental Milestone Assessment API Testing Runner
# This script automates the test data generation and API testing process

# Default configurations
API_URL="http://localhost:8002"
PORT=8002
AGE=24
NUM_TESTS=20
REPORT_FILE="api_test_report.html"
DOMAINS="all"
RESPONSE_LENGTH="medium"
CONCURRENT=false
START_SERVER=false
EDGE_CASES=false
VERBOSE=false
TEST_DATA_FILE="test_data.json"
PROFILE="neurotypical"

# Display help information
show_help() {
    echo "ASD Developmental Milestone Assessment API Testing Runner"
    echo ""
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -u, --url URL             API base URL (default: $API_URL)"
    echo "  -p, --port PORT           Port for the API server (default: $PORT)"
    echo "  -a, --age AGE             Child's age in months (default: $AGE)"
    echo "  -n, --num-tests COUNT     Number of test cases to run (default: $NUM_TESTS)"
    echo "  -r, --report FILE         Output file for HTML report (default: $REPORT_FILE)"
    echo "  -d, --domains DOMAINS     Comma-separated list of domains to test (default: all)"
    echo "                           Examples: GM,FM,COG,RL,EL,SOC,EMO,ADL"
    echo "  -l, --response-length LEN Length of generated responses: short, medium, long (default: $RESPONSE_LENGTH)"
    echo "  -c, --concurrent          Run tests concurrently where possible"
    echo "  -s, --start-server        Start the API server before testing"
    echo "  -e, --edge-cases          Include edge cases in the tests"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -t, --test-data FILE      Specify test data file (default: $TEST_DATA_FILE)"
    echo "  -f, --profile PROFILE     Development profile for generated data: neurotypical, delay, asd,"
    echo "                           uneven_motor, uneven_cognitive, random (default: $PROFILE)"
    echo ""
    echo "Example:"
    echo "  ./run_tests.sh --age 30 --num-tests 30 --domains GM,FM --start-server --concurrent"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -u|--url)
            API_URL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            API_URL="http://localhost:$PORT"
            shift 2
            ;;
        -a|--age)
            AGE="$2"
            shift 2
            ;;
        -n|--num-tests)
            NUM_TESTS="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        -d|--domains)
            DOMAINS="$2"
            shift 2
            ;;
        -l|--response-length)
            RESPONSE_LENGTH="$2"
            shift 2
            ;;
        -c|--concurrent)
            CONCURRENT=true
            shift
            ;;
        -s|--start-server)
            START_SERVER=true
            shift
            ;;
        -e|--edge-cases)
            EDGE_CASES=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test-data)
            TEST_DATA_FILE="$2"
            shift 2
            ;;
        -f|--profile)
            PROFILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "ASD Developmental Milestone Assessment API Testing"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Child Age: $AGE months"
echo "  Number of Tests: $NUM_TESTS"
echo "  Domains: $DOMAINS"
echo "  Test Data File: $TEST_DATA_FILE"
echo "  Profile: $PROFILE"
echo "  Response Length: $RESPONSE_LENGTH"
echo "  Report File: $REPORT_FILE"
echo "  Start Server: $START_SERVER"
echo "  Run Concurrently: $CONCURRENT"
echo "  Include Edge Cases: $EDGE_CASES"
echo "  Verbose Output: $VERBOSE"
echo ""

# Start API server if requested
if [ "$START_SERVER" = true ]; then
    echo "Starting API server on port $PORT..."
    
    # First kill any existing server process
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo "Port $PORT is in use. Terminating the existing process..."
        kill $(lsof -t -i:$PORT) 2>/dev/null || echo "Could not kill existing process"
        sleep 1
    fi
    
    # Start server in background
    python3 app.py --port $PORT &
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    echo "Waiting for server to initialize..."
    sleep 3
    
    # Check if server started successfully
    if ! curl -s "$API_URL" > /dev/null; then
        echo "Failed to start server. Please check logs."
        exit 1
    fi
    
    echo "Server is running."
fi

# Generate test data
echo ""
echo "Generating test data..."
DOMAIN_ARG=""
if [ "$DOMAINS" != "all" ]; then
    DOMAIN_ARG="--domains $DOMAINS"
fi

python3 generate_test_data.py --age $AGE --profile $PROFILE --count $NUM_TESTS \
    --output $TEST_DATA_FILE --response_length $RESPONSE_LENGTH \
    $DOMAIN_ARG

# Prepare domain arguments if needed
DOMAIN_ARGS=""
if [ "$DOMAINS" != "all" ]; then
    DOMAIN_ARGS="--domains $DOMAINS"
fi

# Run tests
echo ""
echo "Running API tests..."
CONCURRENT_ARG=""
if [ "$CONCURRENT" = true ]; then
    CONCURRENT_ARG="--concurrent"
fi

EDGE_CASES_ARG=""
if [ "$EDGE_CASES" = true ]; then
    EDGE_CASES_ARG="--edge-cases"
fi

VERBOSE_ARG=""
if [ "$VERBOSE" = true ]; then
    VERBOSE_ARG="--verbose"
fi

python3 test_api.py --url $API_URL --age $AGE --tests $NUM_TESTS \
    --data $TEST_DATA_FILE --report $REPORT_FILE \
    $DOMAIN_ARGS $CONCURRENT_ARG $EDGE_CASES_ARG $VERBOSE_ARG

TEST_RESULT=$?

# Shut down server if we started it
if [ "$START_SERVER" = true ]; then
    echo ""
    echo "Shutting down API server..."
    kill $SERVER_PID 2>/dev/null || echo "Could not kill server process"
fi

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "Tests completed successfully."
    echo "Report saved to: $REPORT_FILE"
    # Open report in browser on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$REPORT_FILE"
    fi
else
    echo "Tests failed with exit code $TEST_RESULT"
fi

exit $TEST_RESULT 