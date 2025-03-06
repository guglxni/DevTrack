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

# Define the milestone
MILESTONE = "Recognizes familiar people"

# Direct keywords for each category
direct_keywords = {
    "INDEPENDENT": [
        "always smiles when he sees",
        "knows all family members",
        "distinguishes between strangers",
        "recognizes everyone",
        "identifies people easily",
        "knows who people are",
        "recognizes immediately",
        "definitely recognizes"
    ],
    "WITH_SUPPORT": [
        "recognizes with help",
        "recognizes when prompted",
        "needs assistance to recognize",
        "recognizes with support",
        "can identify with guidance"
    ],
    "EMERGING": [
        "starting to recognize",
        "beginning to show recognition",
        "occasionally recognizes",
        "sometimes seems to recognize",
        "inconsistently recognizes"
    ],
    "LOST_SKILL": [
        "used to recognize",
        "previously recognized",
        "no longer recognizes",
        "stopped recognizing",
        "lost ability to recognize"
    ],
    "CANNOT_DO": [
        "doesn't recognize anyone",
        "does not recognize people",
        "cannot identify anyone",
        "never recognizes people",
        "shows no recognition",
        "unable to recognize"
    ]
}

print("\nTesting with direct keywords in comprehensive assessment")

# Test responses
test_responses = [
    # INDEPENDENT responses
    "My child always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well.",
    "Yes, she definitely recognizes everyone in the family and even remembers people she's only met a few times.",
    
    # WITH_SUPPORT responses
    "He recognizes with help - if I say 'who's that?' and give him a hint, he can identify family members.",
    "She needs assistance to recognize extended family members, but does well with immediate family.",
    
    # EMERGING responses
    "He's just starting to recognize grandparents, but it's inconsistent.",
    "She's beginning to show recognition of close family members, but it's not reliable yet.",
    
    # LOST_SKILL responses
    "He used to recognize us but stopped after his developmental regression.",
    "She previously recognized family members, but has lost this ability in the past few months.",
    
    # CANNOT_DO responses
    "No, he doesn't recognize anyone, even his parents.",
    "She cannot identify anyone, not even people she sees every day."
]

expected_scores = [
    "INDEPENDENT", "INDEPENDENT",
    "WITH_SUPPORT", "WITH_SUPPORT",
    "EMERGING", "EMERGING",
    "LOST_SKILL", "LOST_SKILL",
    "CANNOT_DO", "CANNOT_DO"
]

for i, response_text in enumerate(test_responses):
    # Prepare the comprehensive assessment data with direct keywords
    assessment_data = {
        "question": f"Does your child {MILESTONE}?",
        "milestone_behavior": MILESTONE,
        "parent_response": response_text,
        "keywords": direct_keywords
    }
    
    # Make the request
    response = requests.post(
        f"{API_URL}/comprehensive-assessment",
        json=assessment_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTest #{i+1}: '{response_text}'")
        print(f"Score: {result['score_label']} ({result['score']})")
        print(f"Expected: {expected_scores[i]}")
        
        if result['score_label'] == expected_scores[i]:
            print("✅ PASS")
        else:
            print("❌ FAIL")
    else:
        print(f"❌ Failed to process assessment: {response.text}")

print("\nTesting complete!") 