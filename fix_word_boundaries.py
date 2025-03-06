#!/usr/bin/env python3
import requests
import json

# API endpoint
API_URL = "http://localhost:8003"

# Reset the assessment engine
print("Resetting assessment engine...")
response = requests.post(f"{API_URL}/reset")
if response.status_code != 200:
    print(f"Failed to reset: {response.text}")
    exit(1)

# Define the milestone we want to fix
MILESTONE = "Recognizes familiar people"

# Define better keywords with word boundaries to prevent partial matches
keywords = {
    "INDEPENDENT": [
        "always recognizes",
        "consistently recognizes", 
        "easily recognizes",
        "immediately recognizes",
        "recognizes instantly",
        "definitely recognizes",
        "clearly recognizes",
        "recognizes without any issues",
        "knows all family members",
        "recognizes everyone",
        "knows everyone",
        "distinguishes between strangers",
        "smiles at familiar people"
    ],
    "WITH_SUPPORT": [
        "recognizes with help", 
        "sometimes recognizes", 
        "recognizes when prompted",
        "recognizes with assistance",
        "recognizes with support",
        "recognizes with guidance",
        "recognizes when reminded"
    ],
    "EMERGING": [
        "starting to recognize", 
        "beginning to recognize",
        "occasionally recognizes",
        "recognizes inconsistently",
        "sometimes seems to recognize",
        "might recognize",
        "recognizes rarely"
    ],
    "LOST_SKILL": [
        "used to recognize",
        "previously recognized",
        "recognized before",
        "no longer recognizes",
        "stopped recognizing",
        "lost ability to recognize"
    ],
    "CANNOT_DO": [
        "doesn't recognize",
        "does not recognize",
        "unable to recognize",
        "never recognizes",
        "can't recognize",
        "cannot recognize",
        "fails to recognize",
        "shows no recognition",
        "treats everyone as strangers",
        "doesn't know who people are"
    ]
}

# Note: We're removing simple "no" as it causes false positives

# Update keywords for the milestone
print(f"Updating keywords for milestone: {MILESTONE}")
for category, category_keywords in keywords.items():
    update_data = {
        "category": category,
        "keywords": category_keywords
    }
    response = requests.post(
        f"{API_URL}/keywords",
        json=update_data
    )
    
    if response.status_code == 200:
        print(f"✅ Updated {len(category_keywords)} keywords for {category}")
    else:
        print(f"❌ Failed to update {category} keywords: {response.text}")

# Test with the problematic response
test_response = "my child always smiles when he sees grandparents or his favorite babysitter. he knows all his family members and distinguishes between strangers and people he knows well."

print(f"\nTesting response: '{test_response}'")

# Test with score-response endpoint
data = {
    "milestone_behavior": MILESTONE,
    "response": test_response
}

response = requests.post(
    f"{API_URL}/score-response",
    json=data
)

if response.status_code == 200:
    result = response.json()
    print(f"Score: {result['score_label']} ({result['score']})")
    
    if result['score_label'] == "INDEPENDENT":
        print("✅ CORRECTLY scored as INDEPENDENT")
    else:
        print("❌ INCORRECTLY scored - should be INDEPENDENT")
else:
    print(f"❌ Failed to score response: {response.text}")

# Test with comprehensive assessment
assessment_data = {
    "question": f"Does your child {MILESTONE}?",
    "milestone_behavior": MILESTONE,
    "parent_response": test_response,
    "keywords": None
}

response = requests.post(
    f"{API_URL}/comprehensive-assessment",
    json=assessment_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Comprehensive Score: {result['score_label']} ({result['score']})")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"❌ Failed to process comprehensive assessment: {response.text}")

print("\nFix complete!") 