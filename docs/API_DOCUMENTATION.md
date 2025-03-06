# ASD Assessment API Documentation

This document provides detailed information about all API endpoints available in the ASD Assessment System.

## Base URL

The API is available at: `http://localhost:8003`

## Authentication

Currently, the API does not require authentication. For production environments, you should implement appropriate authentication mechanisms.

## Endpoints

### Child Information

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

#### Score Response

Scores a caregiver's response for a specific milestone.

- **URL**: `/score-response`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "milestone_behavior": "Shows empathy",
    "response": "Yes, she always notices when I am sad and tries to comfort me."
  }
  ```
- **Parameters**:
  - `milestone_behavior` (string, required): The behavior being assessed
  - `response` (string, required): Caregiver's response describing the child's behavior

- **Success Response**:
  ```json
  {
    "milestone": "Shows empathy",
    "domain": "SOC",
    "score": 4,
    "score_label": "INDEPENDENT"
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