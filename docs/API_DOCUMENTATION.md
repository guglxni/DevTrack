# ASD Assessment API Documentation

## Overview

The ASD Assessment API provides endpoints for processing and analyzing developmental milestone assessments. The API supports question processing, keyword management, score recording, and response analysis to facilitate the assessment of developmental milestones in children.

## Base URL

All API endpoints are available at the base URL:

```
http://localhost:8003
```

## Authentication

The API currently does not require authentication for development and demonstration purposes. In a production environment, appropriate authentication mechanisms should be implemented.

## Endpoints

### 1. Question Endpoint

**Endpoint:** `/question`

**Method:** POST

**Description:** Processes questions related to developmental milestones and matches them to appropriate milestone behaviors.

**Request Format:**
```json
{
  "text": "String - The question text",
  "milestone_id": "String - Optional milestone ID"
}
```

**Sample Request:**
```json
{
  "text": "Does the child respond when called by name?",
  "milestone_id": "Recognizes familiar people"
}
```

**Response Format:**
```json
{
  "success": true,
  "milestone_matched": true,
  "milestone_id": "String - ID of the matched milestone",
  "confidence": 0.92
}
```

**Sample Response:**
```json
{
  "success": true,
  "milestone_matched": true,
  "milestone_id": "Recognizes familiar people",
  "confidence": 0.92
}
```

**Error Responses:**
- `400 Bad Request`: Invalid request format
- `404 Not Found`: Milestone not found
- `500 Internal Server Error`: Server processing error

### 2. Keywords Endpoint

**Endpoint:** `/keywords`

**Method:** POST

**Description:** Processes categorized keywords used for analyzing parent/caregiver responses.

**Request Format:**
```json
{
  "category": "String - Keyword category (CANNOT_DO, LOST_SKILL, EMERGING, WITH_SUPPORT, INDEPENDENT)",
  "keywords": ["String Array - List of keywords"]
}
```

**Sample Request:**
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

**Response Format:**
```json
{
  "success": true,
  "message": "Keywords updated successfully",
  "category": "String - Category updated",
  "keywords_count": 14
}
```

**Sample Response:**
```json
{
  "success": true,
  "message": "Keywords updated successfully",
  "category": "CANNOT_DO",
  "keywords_count": 14
}
```

**Error Responses:**
- `400 Bad Request`: Invalid request format or category
- `500 Internal Server Error`: Server processing error

**Available Categories:**
- `CANNOT_DO`: Keywords indicating the child cannot perform the behavior (score: 0)
- `LOST_SKILL`: Keywords indicating a skill was previously acquired but lost (score: 1)
- `EMERGING`: Keywords indicating a skill is emerging but inconsistent (score: 2)
- `WITH_SUPPORT`: Keywords indicating a skill is acquired with support (score: 3)
- `INDEPENDENT`: Keywords indicating a skill is fully acquired (score: 4)

### 3. Send Score Endpoint

**Endpoint:** `/send-score`

**Method:** POST

**Description:** Sends scores for specific developmental milestones.

**Request Format:**
```json
{
  "milestone_id": "String - Milestone ID",
  "score": "Integer - Score value (0-4)",
  "score_label": "String - Score category label"
}
```

**Sample Request:**
```json
{
  "milestone_id": "Shows empathy",
  "score": 4,
  "score_label": "INDEPENDENT"
}
```

**Response Format:**
```json
{
  "success": true,
  "message": "Score recorded successfully",
  "milestone_id": "String - Milestone ID",
  "score": "Integer - Score value",
  "score_label": "String - Score category label"
}
```

**Sample Response:**
```json
{
  "success": true,
  "message": "Score recorded successfully",
  "milestone_id": "Shows empathy",
  "score": 4,
  "score_label": "INDEPENDENT"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid request format or score
- `404 Not Found`: Milestone not found
- `500 Internal Server Error`: Server processing error

**Score Values and Labels:**
- `0`: `CANNOT_DO` - Skill not acquired
- `1`: `LOST_SKILL` - Skill was acquired but lost
- `2`: `EMERGING` - Skill is emerging but inconsistent
- `3`: `WITH_SUPPORT` - Skill is acquired but needs support
- `4`: `INDEPENDENT` - Skill is fully acquired

### 4. Score Response Endpoint

**Endpoint:** `/score-response`

**Method:** POST

**Description:** Analyzes parent/caregiver responses about specific milestone behaviors to determine appropriate scores.

**Request Format:**
```json
{
  "milestone_behavior": "String - Milestone behavior description",
  "response": "String - Parent/caregiver response"
}
```

**Sample Request:**
```json
{
  "milestone_behavior": "Recognizes familiar people",
  "response": "My child is very good at recognizing familiar faces. He always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well."
}
```

**Response Format:**
```json
{
  "success": true,
  "milestone_behavior": "String - Milestone behavior",
  "score": "Integer - Determined score (0-4)",
  "score_label": "String - Score category label",
  "confidence": "Float - Confidence level (0-1)"
}
```

**Sample Response:**
```json
{
  "success": true,
  "milestone_behavior": "Recognizes familiar people",
  "score": 4,
  "score_label": "INDEPENDENT",
  "confidence": 0.95
}
```

**Error Responses:**
- `400 Bad Request`: Invalid request format
- `404 Not Found`: Milestone behavior not recognized
- `422 Unprocessable Entity`: Unable to analyze response
- `500 Internal Server Error`: Server processing error

**Note on NLP Processing:**
The API uses a fallback pattern detection mechanism when the advanced NLP module is not available. In such cases, the response analysis will rely on keyword pattern matching to determine scores. The log message "Advanced NLP module not available or error during analysis" indicates that the system is using the fallback mechanism.

### 5. Comprehensive Assessment Endpoint

**Endpoint:** `/comprehensive-assessment`

**Method:** POST

**Description:** Combines the functionality of all other endpoints in a single call, processing a question, updating keywords, analyzing a response, and recording a score.

**Request Format:**
```json
{
  "question": "String - The question text about the child's behavior",
  "milestone_behavior": "String - The milestone behavior being assessed",
  "parent_response": "String - Parent/caregiver response describing the child's behavior",
  "keywords": {
    "CATEGORY_NAME": ["String array - List of keywords for this category"],
    ...
  }
}
```

**Sample Request:**
```json
{
  "question": "Does the child recognize familiar people?",
  "milestone_behavior": "Recognizes familiar people",
  "parent_response": "My child always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well.",
  "keywords": {
    "CANNOT_DO": ["no", "not", "never", "doesn't", "does not", "cannot"],
    "LOST_SKILL": ["used to", "lost it", "previously", "before", "stopped", "regressed"],
    "EMERGING": ["sometimes", "occasionally", "not consistent", "trying", "beginning to"],
    "WITH_SUPPORT": ["with help", "if guided", "with assistance", "prompted", "when reminded"],
    "INDEPENDENT": ["always", "consistently", "definitely", "very good at", "excellent", "mastered"]
  }
}
```

**Response Format:**
```json
{
  "question_processed": true,
  "milestone_found": true,
  "milestone_details": {
    "behavior": "String - The milestone behavior",
    "criteria": "String - The criteria for this milestone",
    "domain": "String - The developmental domain",
    "age_range": "String - The age range for this milestone"
  },
  "keywords_updated": ["String array - Categories that were updated with new keywords"],
  "score": 0-4,
  "score_label": "String - The score category (e.g., INDEPENDENT)",
  "confidence": 0.0-1.0,
  "domain": "String - The developmental domain"
}
```

**Sample Response:**
```json
{
  "question_processed": true,
  "milestone_found": true,
  "milestone_details": {
    "behavior": "Recognizes familiar people",
    "criteria": "Recognizes familiar people",
    "domain": "SOC",
    "age_range": "0-6"
  },
  "keywords_updated": [
    "CANNOT_DO",
    "LOST_SKILL",
    "EMERGING",
    "WITH_SUPPORT",
    "INDEPENDENT"
  ],
  "score": 4,
  "score_label": "INDEPENDENT",
  "confidence": 0.85,
  "domain": "SOC"
}
```

**Error Responses:**

- `404 Not Found`: Milestone not found
- `400 Bad Request`: Missing required fields or invalid data
- `422 Unprocessable Entity`: Unable to analyze response
- `500 Internal Server Error`: Server error processing the request

## Error Handling

All API endpoints return standardized error responses with appropriate HTTP status codes. Error responses follow this format:

```json
{
  "detail": "Description of what went wrong"
}
```

## Performance

The API is designed for high performance with most responses returned in milliseconds. Response times are typically:

- `/question`: 1-5ms
- `/keywords`: 1-3ms
- `/send-score`: 1-3ms
- `/score-response`: 1-5ms

## Implementation Notes

- The API is built using FastAPI
- The server runs on Uvicorn with auto-reload enabled for development
- Default port is 8003
- Cross-Origin Resource Sharing (CORS) is enabled for all origins in development mode

## Troubleshooting

1. **Server Already Running**: If you see the error "address already in use" when starting the server, it means the API server is already running on port 8003.

2. **NLP Module Issues**: If you see "Advanced NLP module not available" in logs, the system is using the fallback keyword pattern matching instead of advanced NLP techniques. This doesn't affect functionality but may reduce accuracy of response analysis.

3. **Request Validation Errors**: Ensure all request fields match the expected format and types.

## Child Information

#### Set Child Age

Sets the child's age to filter appropriate milestones.

- **URL**: `/set-child-age`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "age": 24,
    "name": "Alex"
  }
  ```
- **Parameters**:
  - `age` (integer, required): Child's age in months (0-36)
  - `name` (string, optional): Child's name

- **Success Response**:
  ```json
  {
    "message": "Child age set to 24 months",
    "total_milestones": 55
  }
  ```

- **Error Response**:
  ```json
  {
    "detail": "Error setting child age: age must be between 0 and 36 months"
  }
  ```

### Milestone Assessment

#### Get Next Milestone

Retrieves the next milestone to assess.

- **URL**: `/next-milestone`
- **Method**: `GET`
- **Success Response** (if more milestones are available):
  ```json
  {
    "behavior": "Shows empathy",
    "criteria": "Child shows concern when others are distressed",
    "domain": "SOC",
    "age_range": "24-30",
    "complete": false
  }
  ```

- **Success Response** (if all milestones have been assessed):
  ```json
  {
    "message": "No more milestones to assess",
    "complete": true
  }
  ```

#### Batch Score

Scores multiple responses in parallel.

- **URL**: `/batch-score`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
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
  }
  ```

- **Success Response**:
  ```json
  [
    {
      "milestone": "Shows empathy",
      "domain": "SOC",
      "score": 4,
      "score_label": "INDEPENDENT"
    },
    {
      "milestone": "Makes eye contact",
      "domain": "SOC",
      "score": 0,
      "score_label": "CANNOT_DO"
    }
  ]
  ```

### Report Generation

#### Generate Report

Generates a comprehensive assessment report.

- **URL**: `/generate-report`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "scores": [
      {
        "milestone": "Shows empathy",
        "domain": "SOC",
        "score": 4,
        "score_label": "INDEPENDENT",
        "age_range": "24-30"
      },
      {
        "milestone": "Makes eye contact",
        "domain": "SOC",
        "score": 0,
        "score_label": "CANNOT_DO",
        "age_range": "0-6"
      }
    ],
    "domain_quotients": {
      "SOC": 68.75,
      "Cog": 75.0,
      "GM": 82.5,
      "FM": 85.0,
      "EL": 77.78,
      "ADL": 80.0,
      "Emo": 91.67,
      "RL": 83.33
    }
  }
  ```

### Assessment Management

#### Reset Assessment

Resets the assessment engine for a new assessment.

- **URL**: `/reset`
- **Method**: `POST`
- **Success Response**:
  ```json
  {
    "message": "Assessment engine reset"
  }
  ```

### System Information

#### Health Check

Checks if the API server is running.

- **URL**: `/health`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "status": "ok",
    "message": "API server is running"
  }
  ```

#### Get All Milestones

Retrieves all available milestone behaviors.

- **URL**: `/all-milestones`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "milestones": [
      {
        "behavior": "Lifts head when on tummy",
        "criteria": "Child can lift head when placed on tummy",
        "domain": "GM",
        "age_range": "0-6"
      },
      {
        "behavior": "Shows empathy",
        "criteria": "Child shows concern when others are distressed",
        "domain": "SOC",
        "age_range": "24-30"
      }
    ]
  }
  ```

### New Endpoints

#### Submit Question

Receives and processes questions about a child's behavior.

- **URL**: `/question`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Does the child respond when called by name?",
    "milestone_id": "Recognizes familiar people"
  }
  ```
- **Parameters**:
  - `text` (string, required): The question text
  - `milestone_id` (string, optional): Associated milestone ID (behavior)

- **Success Response**:
  ```json
  {
    "status": "success",
    "message": "Question received successfully",
    "question": "Does the child respond when called by name?",
    "milestone_found": true,
    "milestone_details": {
      "behavior": "Recognizes familiar people",
      "criteria": "Child shows recognition of familiar individuals",
      "domain": "SOC",
      "age_range": "0-6",
      "keywords": ["recognize", "familiar", "people", "faces"],
      "scoring_rules": {
        "cannot": 0,
        "lost": 1,
        "emerging": 2,
        "support": 3,
        "independent": 4
      }
    }
  }
  ```

#### Update Keywords

Updates keywords for specific scoring categories.

- **URL**: `/keywords`
- **Method**: `POST`
- **Request Body**:
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
- **Parameters**:
  - `category` (string, required): Scoring category (e.g., "CANNOT_DO")
  - `keywords` (array of strings, required): List of keywords for this category

- **Success Response**:
  ```json
  {
    "status": "success",
    "message": "Keywords for category CANNOT_DO updated successfully",
    "category": "CANNOT_DO",
    "keywords": ["no", "not", "never", "doesn't", "does not", "cannot", "can't", "unable", "hasn't", "has not", "not able", "not at all", "not yet started", "not capable"]
  }
  ```

#### Send Score

Manually sets a score for a specific milestone.

- **URL**: `/send-score`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "milestone_id": "Shows empathy",
    "score": 4,
    "score_label": "INDEPENDENT"
  }
  ```
- **Parameters**:
  - `milestone_id` (string, required): The milestone ID (behavior)
  - `score` (integer, required): The numeric score value (0-4)
  - `score_label` (string, required): The score label (e.g., "INDEPENDENT")

- **Success Response**:
  ```json
  {
    "status": "success",
    "message": "Score for milestone 'Shows empathy' set successfully",
    "milestone": "Shows empathy",
    "domain": "SOC",
    "score": 4,
    "score_label": "INDEPENDENT"
  }
  ```

## Score Values and Labels

| Score Value | Label         | Meaning                                    |
|-------------|---------------|-------------------------------------------|
| -1          | NOT_RATED     | No rating provided                         |
| 0           | CANNOT_DO     | Skill not acquired                         |
| 1           | LOST_SKILL    | Skill was acquired but lost                |
| 2           | EMERGING      | Skill is emerging but inconsistent         |
| 3           | WITH_SUPPORT  | Skill is acquired but needs support        |
| 4           | INDEPENDENT   | Skill is fully acquired                    |

## Developmental Domains

| Domain Code | Description                      |
|-------------|----------------------------------|
| SOC         | Social Development               |
| Cog         | Cognitive Development            |
| GM          | Gross Motor Skills               |
| FM          | Fine Motor Skills                |
| EL          | Expressive Language              |
| ADL         | Activities of Daily Living       |
| Emo         | Emotional Development            |
| RL          | Receptive Language               |

## Error Handling

All endpoints return appropriate HTTP status codes for different error scenarios:

- `400`: Bad Request - Invalid input data
- `404`: Not Found - Resource not found
- `500`: Internal Server Error - Server-side issues

Error responses include a detailed message in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Future Enhancements

Planned enhancements for the API include:

1. OAuth2 authentication
2. Rate limiting
3. Pagination for large result sets
4. Enhanced error handling
5. WebSocket support for real-time assessment 

## GPU Acceleration API

These endpoints provide access to Metal GPU acceleration features for Apple Silicon Macs.

### System Information

#### Get System Information

Retrieves basic system information including GPU capabilities.

- **URL**: `/gpu-acceleration/system-info`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "os_version": "macOS-15.3.1-arm64-arm-64bit",
    "python_version": "3.12.6 (v3.12.6:a4a2d2b0d85, Sep 6 2024, 16:08:03)",
    "memory_info": "16.00 GB",
    "is_apple_silicon": true,
    "chip_model": "Apple M4",
    "metal_enabled": true,
    "gpu_mode": "Enabled",
    "gpu_layers": "32/32",
    "metal_family": null,
    "gpu_memory": null
  }
  ```

#### Get Detailed System Information

Retrieves detailed system information including Python packages and environment variables.

- **URL**: `/gpu-acceleration/detailed-system-info`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "os_version": "macOS-15.3.1-arm64-arm-64bit",
    "python_version": "3.12.6 (v3.12.6:a4a2d2b0d85, Sep 6 2024, 16:08:03)",
    "memory_info": "16.00 GB",
    "is_apple_silicon": true,
    "chip_model": "Apple M4",
    "metal_enabled": true,
    "gpu_mode": "Enabled",
    "gpu_layers": "32/32",
    "metal_family": null,
    "gpu_memory": null,
    "python_packages": {
      "torch": "2.1.0",
      "pandas": "2.1.1",
      "numpy": "1.24.3",
      "fastapi": "0.104.1"
    },
    "environment_variables": {
      "ENABLE_METAL": "1",
      "METAL_LAYERS": "32"
    },
    "gpu_settings": {
      "ENABLE_METAL": "1",
      "METAL_LAYERS": "32",
      "METAL_MEMORY_LIMIT": "8192",
      "METAL_BUFFER_SIZE": "2048",
      "OPTIMIZE_FOR_APPLE": "1",
      "DEFAULT_CONTEXT_SIZE": "8192"
    },
    "server_pid": 45215
  }
  ```

### Server Control

#### Get Server Status

Checks if the server is running and its current mode.

- **URL**: `/gpu-acceleration/server-status`
- **Method**: `GET`
- **Success Response (Running)**:
  ```json
  {
    "status": "running",
    "message": "Server is running with PID 45215",
    "pid": 45215,
    "running": true,
    "mode": "gpu_accelerated"
  }
  ```
- **Success Response (Not Running)**:
  ```json
  {
    "status": "not_running",
    "message": "Server is not running",
    "pid": null,
    "running": false
  }
  ```

#### Restart Server

Restarts the server with the specified GPU acceleration mode.

- **URL**: `/gpu-acceleration/restart-server`
- **Method**: `POST`
- **Query Parameters**:
  - `mode` (string, optional): One of "cpu", "basic_gpu", or "advanced_gpu". Default is "advanced_gpu".
- **Success Response**:
  ```json
  {
    "status": "restarting",
    "message": "Server is restarting with advanced_gpu mode",
    "previous_pid": 45215
  }
  ```

#### Stop Server

Stops the running server.

- **URL**: `/gpu-acceleration/stop-server`
- **Method**: `POST`
- **Success Response**:
  ```json
  {
    "status": "stopped",
    "message": "Server stopped (PID: 45215)"
  }
  ```

### GPU Settings

#### Update GPU Settings

Updates GPU acceleration settings and optionally restarts the server.

- **URL**: `/gpu-acceleration/settings`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "mode": "optimal",
    "restart_server": false
  }
  ```
- **Parameters**:
  - `mode` (string, required): GPU acceleration mode. One of "optimal", "basic", "advanced", "custom", or "disabled".
  - `restart_server` (boolean, optional): Whether to restart the server with new settings. Default is false.
- **Success Response**:
  ```json
  {
    "status": "success",
    "message": "GPU acceleration enabled with optimal mode",
    "settings": {
      "ENABLE_METAL": "1",
      "METAL_LAYERS": "32",
      "METAL_MEMORY_LIMIT": "8192",
      "METAL_BUFFER_SIZE": "2048",
      "OPTIMIZE_FOR_APPLE": "1",
      "DEFAULT_CONTEXT_SIZE": "8192"
    },
    "restart_scheduled": false
  }
  ```

### Testing and Benchmarking

#### Test GPU

Runs a simple test to verify GPU acceleration is working.

- **URL**: `/gpu-acceleration/test-gpu`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "status": "success",
    "message": "Metal GPU test completed. Speedup: 1.15x",
    "cpu_time": 0.0165,
    "gpu_time": 0.0143,
    "speedup": 1.15,
    "matrix_size": 2000
  }
  ```

#### Get Benchmarks

Retrieves historical benchmark results.

- **URL**: `/gpu-acceleration/benchmarks`
- **Method**: `GET`
- **Success Response**:
  ```json
  [
    {
      "benchmark_id": "5967248e-6eeb-4f13-8b2d-18ac2345a7e9",
      "timestamp": "2025-03-18T13:14:55.208553",
      "duration": 0.021274924278259277,
      "operations_per_second": 2538199398208,
      "speedup_vs_cpu": 1.1183299993836395,
      "model_name": "Apple M4",
      "status": "success",
      "message": "Benchmark completed with 5 iterations. Matrix size: 3000x3000"
    }
  ]
  ```

#### Run Benchmark

Runs a benchmark to measure GPU acceleration performance.

- **URL**: `/gpu-acceleration/run-benchmark`
- **Method**: `POST`
- **Query Parameters**:
  - `matrix_size` (integer, optional): Size of the matrix for multiplication benchmark. Default is 2000.
  - `iterations` (integer, optional): Number of iterations to run. Default is 5.
- **Success Response**:
  ```json
  {
    "benchmark_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "timestamp": "2025-03-18T14:30:15.123456",
    "duration": 0.01859,
    "operations_per_second": 2150435982151,
    "speedup_vs_cpu": 1.23,
    "model_name": "Apple M4",
    "status": "success",
    "message": "Benchmark completed with 5 iterations. Matrix size: 2000x2000"
  }
  ```

#### Run Benchmarks (Alternative Endpoint)

Alternative endpoint for running benchmarks with default parameters.

- **URL**: `/gpu-acceleration/run-benchmarks`
- **Method**: `POST`
- **Success Response**: Same as `/gpu-acceleration/run-benchmark`

### Monitoring

#### Get Monitoring Data

Retrieves performance monitoring data for visualization.

- **URL**: `/gpu-acceleration/monitoring-data`
- **Method**: `GET`
- **Success Response**:
  ```json
  {
    "timestamps": [
      "2025-03-18T13:11:18.484343",
      "2025-03-18T13:12:18.484343"
    ],
    "memory_usage": [
      0.3137,
      0.2372
    ],
    "cpu_usage": [
      35.66,
      38.00
    ]
  }
  ```

### Dashboard

#### Get GPU Acceleration Dashboard

Returns the HTML page for the GPU Acceleration Dashboard.

- **URL**: `/gpu-acceleration/`
- **Method**: `GET`
- **Response**: HTML page
- **Alternative Routes**:
  - `/gpu-acceleration/index.html`
  - `/gpu-acceleration/dashboard` 