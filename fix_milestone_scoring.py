#!/usr/bin/env python3
import requests
import json

# API endpoint
API_URL = "http://localhost:8003"

# Define the milestone behavior we want to fix
MILESTONE = "Recognizes familiar people"

# First, let's reset the assessment engine
print("Resetting assessment engine...")
response = requests.post(f"{API_URL}/reset")
if response.status_code == 200:
    print("✅ Reset successful")
else:
    print(f"❌ Failed to reset: {response.text}")
    exit(1)

# Define better keywords for each score category
keywords = {
    "INDEPENDENT": [
        "always recognizes", 
        "consistently recognizes", 
        "easily recognizes",
        "immediately recognizes",
        "recognizes instantly",
        "no problem recognizing",
        "definitely recognizes",
        "clearly recognizes",
        "recognizes without any issues"
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
        "no recognition",
        "fails to recognize",
        "shows no recognition"
    ]
}

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
        print(f"✅ Updated keywords for {category}")
    else:
        print(f"❌ Failed to update {category} keywords: {response.text}")

# Test the milestone with a sample response
test_responses = [
    "Yes, she recognizes her parents and siblings easily",
    "He struggles to recognize familiar people, even parents",
    "She's starting to recognize her grandparents but it's inconsistent",
    "He recognizes mom and dad with some prompting",
    "He used to recognize family members but doesn't anymore"
]

print("\nTesting milestone responses:")
for response_text in test_responses:
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
        print("-" * 40)
    else:
        print(f"❌ Failed to score response: {response.text}")

print("\nCompleted keyword updates and testing!") 