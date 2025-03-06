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