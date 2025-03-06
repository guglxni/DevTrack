#!/usr/bin/env python3
import requests
import json
import re

# API endpoint
API_URL = "http://localhost:8003"

# Reset the assessment engine
print("Resetting assessment engine...")
response = requests.post(f"{API_URL}/reset")
if response.status_code != 200:
    print(f"Failed to reset: {response.text}")
    exit(1)

# Define the milestone we want to debug
MILESTONE = "Recognizes familiar people"

# Get the current keyword map to understand what's happening
print("\nFetching all milestones to analyze current system...")
response = requests.get(f"{API_URL}/all-milestones")
if response.status_code != 200:
    print(f"Failed to get milestones: {response.text}")
    exit(1)

# Set up a completely different approach with very specific keywords
keywords = {
    "INDEPENDENT": [
        "definitely recognizes everyone",
        "excellent at recognizing people",
        "always knows who people are",
        "identifies familiar people easily",
        "knows all family members without fail",
        "has mastered recognizing people"
    ],
    "WITH_SUPPORT": [
        "needs some help recognizing people",
        "recognizes with assistance",
        "can identify if prompted"
    ],
    "EMERGING": [
        "beginning to show signs of recognition",
        "sometimes appears to recognize"
    ],
    "LOST_SKILL": [
        "used to recognize but no longer does",
        "previously recognized but lost this ability"
    ],
    "CANNOT_DO": [
        "absolutely does not recognize anyone",
        "definitely cannot recognize familiar people",
        "unable to identify even parents",
        "completely lacks recognition ability"
    ]
}

# First, let's print out the problematic response character by character to see if there's anything hidden
test_response = "my child always smiles when he sees grandparents or his favorite babysitter. he knows all his family members and distinguishes between strangers and people he knows well."
print("\nAnalyzing response character by character:")
print("Character codes:")
for i, char in enumerate(test_response):
    print(f"Position {i}: '{char}' (ASCII/Unicode: {ord(char)})")

# Update keywords with our extreme approach
print(f"\nUpdating keywords for milestone: {MILESTONE} with extremely specific phrases")
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

# Create a simpler, unambiguous test response
simple_response = "My child definitely recognizes everyone in our family perfectly."

print(f"\nTesting very simple response: '{simple_response}'")
data = {
    "milestone_behavior": MILESTONE,
    "response": simple_response
}
response = requests.post(f"{API_URL}/score-response", json=data)
if response.status_code == 200:
    result = response.json()
    print(f"Score: {result['score_label']} ({result['score']})")
else:
    print(f"❌ Failed to score response: {response.text}")

# Try the original problematic response again
print(f"\nTesting original problem response again: '{test_response}'")
data = {
    "milestone_behavior": MILESTONE,
    "response": test_response
}
response = requests.post(f"{API_URL}/score-response", json=data)
if response.status_code == 200:
    result = response.json()
    print(f"Score: {result['score_label']} ({result['score']})")
else:
    print(f"❌ Failed to score response: {response.text}")

# Try with keywords directly in the comprehensive assessment endpoint
print("\nTrying direct keyword update through comprehensive assessment")

direct_keywords = {
    "INDEPENDENT": [
        "knows",
        "identifies",
        "recognizes",
        "distinguishes",
        "smiles when he sees"
    ],
    "CANNOT_DO": [
        "absolutely not",
        "totally unable",
        "completely fails"
    ]
}

assessment_data = {
    "question": f"Does your child {MILESTONE}?",
    "milestone_behavior": MILESTONE,
    "parent_response": test_response,
    "keywords": direct_keywords
}

response = requests.post(
    f"{API_URL}/comprehensive-assessment",
    json=assessment_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Comprehensive Score with direct keywords: {result['score_label']} ({result['score']})")
    print(f"Confidence: {result['confidence']}")
else:
    print(f"❌ Failed to process comprehensive assessment: {response.text}")

print("\nDebugging complete!") 