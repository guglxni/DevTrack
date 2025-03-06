# Comprehensive Assessment Endpoint Demo

## Overview

The Comprehensive Assessment Endpoint (`/comprehensive-assessment`) is a powerful addition to the ASD Assessment API that combines the functionality of multiple endpoints into a single API call. This document explains how to use the demo application to test this endpoint.

## What is the Comprehensive Assessment Endpoint?

The Comprehensive Assessment Endpoint streamlines the assessment process by allowing you to:

1. Process a question about a child's behavior
2. Match it to an appropriate milestone
3. Update keywords used for scoring (optional)
4. Analyze a parent/caregiver response
5. Record the score

All of this happens in a single API call, making the assessment process more efficient.

## Using the Demo Application

### Starting the Demo

To start the demo application:

```bash
./scripts/start_comprehensive_demo.sh
```

This script will:
1. Start the API server (if not already running)
2. Start the web application (if not already running)
3. Provide URLs for accessing the web interface

Once the servers are running, open your browser and navigate to:
```
http://localhost:3000
```

### Accessing the Comprehensive Assessment Tab

1. In the web interface, click on the "Comprehensive" tab in the navigation tabs.
2. You'll see a form with the following fields:
   - **Question**: Enter a question about the child's behavior
   - **Milestone Behavior**: Enter the milestone behavior to assess
   - **Parent/Caregiver Response**: Enter the parent's description of the child's behavior
   - **Include Keywords**: Toggle to include or exclude predefined keywords
   - **Keywords by Category**: If "Include Keywords" is checked, you can customize keywords for each category

### Sample Test Cases

Here are some sample test cases you can use:

#### Case 1: INDEPENDENT Score

- **Question**: Does the child recognize familiar people?
- **Milestone Behavior**: Recognizes familiar people
- **Parent Response**: My child always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well.

#### Case 2: EMERGING Score

- **Question**: Does the child walk independently?
- **Milestone Behavior**: Walks independently
- **Parent Response**: My child is sometimes able to take a few steps on their own, but they're not consistent yet. They're beginning to walk but still need to hold onto furniture most of the time. They're trying to walk more each day.

#### Case 3: WITH_SUPPORT Score

- **Question**: Can the child use a spoon?
- **Milestone Behavior**: able to hold spoon with fingers appropriately
- **Parent Response**: My child can use a spoon with help from me. If guided, they can get food to their mouth, but they need assistance to hold the spoon properly. When reminded about how to hold it, they can do it for a few bites before reverting to their old grip.

### Understanding the Response

When you submit the form, the API will return a response containing:

- **question_processed**: Whether the question was successfully processed
- **milestone_found**: Whether the milestone was found
- **milestone_details**: Details about the milestone if found
- **keywords_updated**: Categories that were updated with new keywords (if keywords were included)
- **score**: The determined score (0-4)
- **score_label**: The score category (e.g., INDEPENDENT)
- **confidence**: Confidence level of the score determination (0-1)
- **domain**: Developmental domain of the milestone

The response will be formatted as JSON and displayed in the response area.

## Benefits of the Comprehensive Endpoint

- **Efficiency**: Reduces multiple API calls to a single request
- **Consistency**: Ensures all components of the assessment process are completed
- **Simplicity**: Provides a streamlined interface for the assessment process
- **Flexibility**: Allows optional keyword updates as part of the assessment

## Technical Details

The endpoint accepts a JSON request with the following structure:

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

The `keywords` field is optional. If provided, it should contain keyword lists for one or more of these categories:
- `CANNOT_DO` (score: 0)
- `LOST_SKILL` (score: 1)
- `EMERGING` (score: 2)
- `WITH_SUPPORT` (score: 3)
- `INDEPENDENT` (score: 4)

## Troubleshooting

If you encounter issues with the demo:

1. **API Server Not Starting**: Check if port 8003 is already in use with `lsof -i:8003`
2. **Web Application Not Starting**: Check if port 3000 is already in use with `lsof -i:3000` 
3. **CORS Issues**: The demo uses a proxy to avoid CORS problems, but if you see CORS errors, ensure the API server is running
4. **Milestone Not Found**: Verify the milestone behavior matches exactly with one in the database
5. **Error Responses**: The demo will display detailed error messages from the API 