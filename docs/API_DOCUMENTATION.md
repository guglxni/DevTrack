# ASD Developmental Milestone Assessment API

This API allows you to process textual descriptions of a child's behavior and automatically score them against developmental milestones. The system uses NLP-based analysis to evaluate responses and generate comprehensive developmental assessments.

## Base URL

The API is hosted at: `http://localhost:8000`

## Authentication

Currently, the API does not require authentication for local usage. For production deployment, proper authentication should be implemented.

## API Endpoints

### Set Child Age

Sets the child's age to filter age-appropriate milestones.

**Endpoint:** `/set-child-age`  
**Method:** `POST`  
**Request Body:**
```json
{
  "age": 24  // Child's age in months (0-36)
}
```

**Response:**
```json
{
  "message": "Child age set to 24 months",
  "total_milestones": 120
}
```

### Get Next Milestone

Retrieves the next milestone to assess.

**Endpoint:** `/next-milestone`  
**Method:** `GET`  
**Response:**
```json
{
  "behavior": "Opens mouth for spoon",
  "criteria": "Child opens mouth when spoon approaches",
  "domain": "Activities of Daily Living",
  "age_range": "12-18 months",
  "complete": false
}
```

When all milestones have been assessed, you'll receive:
```json
{
  "message": "No more milestones to assess",
  "complete": true
}
```

### Score Response

Scores a single response for a specific milestone using NLP analysis.

**Endpoint:** `/score-response`  
**Method:** `POST`  
**Request Body:**
```json
{
  "response": "My child always opens their mouth when they see the spoon coming. They recognize it's time to eat.",
  "milestone_behavior": "Opens mouth for spoon"
}
```

**Response:**
```json
{
  "milestone": "Opens mouth for spoon",
  "domain": "Activities of Daily Living",
  "score": 3,
  "score_label": "INDEPENDENT"
}
```

### Batch Score Responses

Scores multiple responses in parallel for better performance.

**Endpoint:** `/batch-score`  
**Method:** `POST`  
**Request Body:**
```json
{
  "responses": [
    {
      "response": "My child always opens their mouth when they see the spoon coming.",
      "milestone_behavior": "Opens mouth for spoon"
    },
    {
      "response": "My child enjoys playing with a variety of toys.",
      "milestone_behavior": "Plays with toys"
    }
  ]
}
```

**Response:**
```json
[
  {
    "milestone": "Opens mouth for spoon",
    "domain": "Activities of Daily Living",
    "score": 3,
    "score_label": "INDEPENDENT"
  },
  {
    "milestone": "Plays with toys",
    "domain": "Play",
    "score": 3,
    "score_label": "INDEPENDENT"
  }
]
```

### Generate Report

Generates a comprehensive assessment report based on all scored milestones.

**Endpoint:** `/generate-report`  
**Method:** `GET`  
**Response:**
```json
{
  "scores": [
    {
      "milestone": "Opens mouth for spoon",
      "domain": "Activities of Daily Living",
      "score": 3,
      "score_label": "INDEPENDENT",
      "age_range": "12-18 months"
    },
    {
      "milestone": "Plays with toys",
      "domain": "Play",
      "score": 3,
      "score_label": "INDEPENDENT",
      "age_range": "12-18 months"
    }
  ],
  "domain_quotients": {
    "Activities of Daily Living": 100.0,
    "Play": 100.0,
    "Communication": 75.0,
    "Motor": 80.0,
    "Social": 85.0,
    "Cognitive": 90.0
  }
}
```

### Reset Assessment

Resets the assessment engine for a new assessment.

**Endpoint:** `/reset`  
**Method:** `POST`  
**Response:**
```json
{
  "message": "Assessment engine reset"
}
```

## Score Labels

The system uses the following scoring scale:

- `NOT_OBSERVED` (0): Child does not demonstrate the skill
- `LOST_SKILL` (1): Child previously had the skill but has lost it
- `EMERGING` (2): Child is beginning to show the skill sometimes
- `WITH_SUPPORT` (3): Child can do this with help and support
- `INDEPENDENT` (4): Child does this independently and consistently

## Domain Quotients

Domain quotients represent the percentage of achieved developmental milestones in each domain relative to the expected skills for the child's age. The domains include:

- Activities of Daily Living (ADL)
- Play
- Communication
- Motor
- Social
- Cognitive
- Sensory Processing
- Emotional Regulation

## Using the API Client

A command-line API client is included to facilitate interaction with the API without requiring a frontend. The client supports:

1. Interactive assessments
2. Processing responses from CSV or JSON files
3. Scoring individual responses

### Examples

Run an interactive assessment:
```bash
python3 api_client.py --age 24 interactive
```

Process responses from a file:
```bash
python3 api_client.py --age 24 file sample_responses.json
```

Score a single response:
```bash
python3 api_client.py --age 24 score "Opens mouth for spoon" "My child always opens their mouth when they see the spoon coming."
```

## Input File Format

The API supports processing responses from CSV or JSON files:

### JSON Format
```json
[
  {
    "milestone_behavior": "Opens mouth for spoon",
    "response": "My child always opens mouth when spoon approaches."
  }
]
```

### CSV Format
```csv
milestone_behavior,response
"Opens mouth for spoon","My child always opens mouth when spoon approaches."
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid input parameters
- `404 Not Found`: Milestone not found
- `500 Internal Server Error`: Server-side processing error 