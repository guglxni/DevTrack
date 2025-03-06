#!/usr/bin/env python3
import requests
import json

# API endpoint
API_URL = "http://localhost:8003"

# Define the milestone we want to fix
MILESTONE = "Recognizes familiar people"

# Reset the assessment engine
print("Resetting assessment engine...")
response = requests.post(f"{API_URL}/reset")
if response.status_code != 200:
    print(f"Failed to reset: {response.text}")
    exit(1)

# Define specific negative patterns
negative_keywords = [
    "doesn't recognize",
    "does not recognize",
    "unable to recognize",
    "never recognizes",
    "can't recognize",
    "cannot recognize",
    "no recognition",
    "fails to recognize",
    "shows no recognition",
    "doesn't know anyone",
    "doesn't recognize anyone",
    "doesn't know who",
    "no, he doesn't",
    "no, she doesn't",
    "no, they don't",
    "no he does not",
    "no she does not",
    "no they do not"
]

# Update the CANNOT_DO category with these keywords
update_data = {
    "category": "CANNOT_DO",
    "keywords": negative_keywords
}

print(f"Updating negative keywords for milestone: {MILESTONE}")
response = requests.post(
    f"{API_URL}/keywords",
    json=update_data
)

if response.status_code == 200:
    print(f"✅ Updated {len(negative_keywords)} keywords for CANNOT_DO")
else:
    print(f"❌ Failed to update CANNOT_DO keywords: {response.text}")

# Test with negative responses
test_responses = [
    "No, he doesn't recognize anyone, even his parents",
    "He doesn't recognize familiar people at all",
    "She can't recognize family members",
    "No, they don't show any sign of recognition"
]

print("\nTesting negative responses:")
for response_text in test_responses:
    # First test with score-response endpoint
    data = {
        "milestone_behavior": MILESTONE,
        "response": response_text
    }
    response = requests.post(
        f"{API_URL}/score-response",
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: '{response_text}'")
        print(f"Score: {result['score_label']} ({result['score']})")
        
        if result['score_label'] == "CANNOT_DO":
            print("✅ CORRECTLY scored as CANNOT_DO")
        else:
            print("❌ INCORRECTLY scored - should be CANNOT_DO")
        print("-" * 40)
    else:
        print(f"❌ Failed to score response: {response.text}")

    # Now test with comprehensive assessment
    assessment_data = {
        "question": f"Does your child {MILESTONE}?",
        "milestone_behavior": MILESTONE,
        "parent_response": response_text,
        "keywords": None
    }
    
    response = requests.post(
        f"{API_URL}/comprehensive-assessment",
        json=assessment_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Comprehensive Score: {result['score_label']} ({result['score']})")
        print("-" * 40)
    else:
        print(f"❌ Failed to process comprehensive assessment: {response.text}")
        
print("\nFix complete!") 