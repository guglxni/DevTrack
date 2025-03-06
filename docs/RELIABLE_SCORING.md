# Reliable Scoring for ASD Assessment API

This document explains how to use the new reliable scoring functionality that ensures consistent, accurate scoring of developmental milestone assessments.

## Background

The scoring system used by the API has been enhanced to provide more reliable results, especially for problematic milestones like "Recognizes familiar people". The improved scoring system uses:

1. **More specific keyword phrases** instead of short words that might cause false matches
2. **Direct keyword provision** with each API call to ensure consistent scoring
3. **Custom milestone handling** for problematic milestones
4. **Hybrid scoring approach** that combines multiple NLP techniques

## Advanced Hybrid Scoring

The API now includes an advanced hybrid scoring approach that combines multiple natural language processing techniques:

### Word Boundary-Aware Keyword Matching

The system now respects word boundaries when matching keywords. This prevents false matches where a substring appears within a larger word. For example:
- "knows" won't match "acknowledges"
- "notices" won't match "unnoticed"

### Negation Detection

The system can identify negations and understand the difference between phrases like:
- "recognizes family members" (positive)
- "doesn't recognize family members" (negated)

### Milestone-Specific Pattern Matching

Each milestone has specific patterns tailored to its unique language requirements:
- "Recognizes familiar people" has patterns to detect recognition-related expressions
- "Makes eye contact" has patterns to detect descriptions of eye contact behavior

### Special Phrase Handling

Common phrases that might be difficult to score are handled specifically:
- "Sometimes, but only when we encourage him" → EMERGING
- "She notices when her grandparents come to visit" → INDEPENDENT for recognition milestones

## Integration Options

### Option 1: Use the Python Client Class

The simplest approach is to use the `ReliableASDClient` class which handles everything for you:

```python
from src.api.reliable_client import ReliableASDClient

# Initialize the client
client = ReliableASDClient(api_url="http://localhost:8003")

# Score a response
result = client.score_response("Recognizes familiar people", "My child always smiles when he sees grandparents.")

# Print the result
print(f"Score: {result['score_label']} ({result['score']})")
```

The client class includes all the same methods as the original API client:

- `set_child_age(age)` - Set the child's age in months
- `get_next_milestone()` - Get the next milestone to assess
- `score_response(milestone, response)` - Score a response with reliable scoring
- `comprehensive_assessment(milestone, response, keywords=None)` - Perform a comprehensive assessment
- `generate_report()` - Generate an assessment report
- `batch_analyze_responses(responses)` - Analyze a batch of responses

### Option 2: Use the Comprehensive Assessment Endpoint with Keywords

If you can't modify your code to use the client class, you can directly call the `/comprehensive-assessment` endpoint with keywords:

```python
import requests
import json

# API endpoint
API_URL = "http://localhost:8003"

# Reliable keywords for the problematic milestone
keywords = {
    "INDEPENDENT": ["always recognizes", "knows everyone", "distinguishes between strangers"],
    "CANNOT_DO": ["doesn't recognize anyone", "unable to recognize"]
    # Add other categories as needed
}

# Assessment data
assessment_data = {
    "question": "Does your child Recognizes familiar people?",
    "milestone_behavior": "Recognizes familiar people",
    "parent_response": "My child always smiles when he sees grandparents.",
    "keywords": keywords  # Include keywords in the request
}

# Make the API call
response = requests.post(
    f"{API_URL}/comprehensive-assessment",
    json=assessment_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Score: {result['score_label']} ({result['score']})")
```

### Option 3: Use the Reliable API Client in Web Applications

For JavaScript/web applications, add the reliable keywords to your API calls:

```javascript
// Default keywords for reliable scoring
const reliableKeywords = {
    "Recognizes familiar people": {
        "INDEPENDENT": [
            "always recognizes",
            "consistently recognizes",
            "knows everyone",
            // Add more keywords
        ],
        "CANNOT_DO": [
            "doesn't recognize anyone",
            "cannot recognize",
            // Add more keywords
        ]
        // Add other categories
    }
};

// Generic keywords for other milestones
const genericKeywords = {
    "INDEPENDENT": ["yes", "always", "consistently"],
    "CANNOT_DO": ["no", "doesn't", "cannot"]
    // Add other categories
};

// Helper function to get reliable keywords for a milestone
function getReliableKeywords(milestone) {
    if (reliableKeywords[milestone]) {
        return reliableKeywords[milestone];
    }
    return genericKeywords;
}

// Make API call with reliable keywords
function scoreResponse(milestone, response) {
    const data = {
        milestone_behavior: milestone,
        parent_response: response,
        keywords: getReliableKeywords(milestone)
    };
    
    // Make API call with the data
    fetch('/comprehensive-assessment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    }).then(/* handle response */);
}
```

## Testing the Integration

Run the included test scripts to verify that the reliable scoring works correctly:

```bash
./run_tests.sh  # Runs all tests
./test_hybrid_scoring.py  # Tests specifically the hybrid scoring approach
./test_api_client.py  # Tests the client integration
```

These scripts help you verify that the integration is working correctly and that problematic cases are handled properly.

## Previously Problematic Cases Now Fixed

The hybrid scoring approach resolves several previously problematic cases:

1. **Substring matching issues**: 
   - "My child knows family members" now correctly scores as INDEPENDENT
   - "She notices when grandparents visit" now correctly scores as INDEPENDENT

2. **Complex phrases**:
   - "Sometimes, but only when we encourage him" now correctly scores as EMERGING
   - "No, he doesn't recognize anyone" now correctly scores as CANNOT_DO

3. **Negation handling**:
   - "He never looks at people" now correctly scores as CANNOT_DO
   - "She doesn't make eye contact with strangers" now correctly scores as context-appropriate

## Benefits of Reliable Scoring

1. **Consistent results** - The same response will always receive the same score
2. **Accurate scoring** - Improved keyword matching ensures correct scoring
3. **Special handling for problematic milestones** - Known issues with certain milestones are fixed
4. **Minimal changes required** - No need to modify the API server code
5. **Advanced NLP techniques** - Hybrid scoring provides more accurate analysis
6. **Fallback mechanisms** - If advanced scoring isn't available, the system falls back to basic scoring

## Troubleshooting

If you encounter scoring issues:

1. Check that you're providing keywords with your API calls
2. Verify that the API server is running and accessible
3. Ensure that you're using the correct milestone behavior text
4. Check the logs for any error messages
5. If you see "No module named 'advanced_nlp'" in logs, this is normal - the system will fall back to basic scoring

For further assistance, contact the API development team. 