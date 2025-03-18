#!/usr/bin/env python3
import requests
import json

# API endpoint
API_URL = "http://localhost:8003"

# Define the milestone we want to test
MILESTONE = "Recognizes familiar people"

# Define test responses
test_responses = [
    "Yes, he recognizes all family members easily",
    "He's just starting to recognize grandparents, but it's inconsistent",
    "She only recognizes me with assistance and prompting",
    "He used to recognize us but stopped after his developmental regression",
    "No, he doesn't recognize anyone, even his parents"
]

# Expected scores (for reference)
expected_scores = [
    "INDEPENDENT",  # Should recognize all family members
    "EMERGING",     # Starting to recognize but inconsistent
    "WITH_SUPPORT", # Recognizes with assistance/prompting  
    "LOST_SKILL",   # Used to recognize but stopped
    "CANNOT_DO"     # Doesn't recognize anyone
]

print(f"Testing comprehensive assessment for milestone: {MILESTONE}\n")

for i, response_text in enumerate(test_responses):
    # Prepare the comprehensive assessment data
    assessment_data = {
        "parent_responses": [{
            "id": f"test{i+1}",
            "question": f"Does your child {MILESTONE}?",
            "milestone_behavior": MILESTONE,
            "response": response_text
        }]
    }
    
    # Make the request to the comprehensive assessment endpoint
    response = requests.post(
        f"{API_URL}/smart-scoring/smart-comprehensive-assessment",
        json=assessment_data
    )
    
    if response.status_code == 200:
        results = response.json()
        if results and len(results) > 0:
            result = results[0]  # Get the first result
            print(f"Test #{i+1}: '{response_text}'")
            print(f"Score: {result['label']} ({result['score']})")
            print(f"Confidence: {result['confidence']}")
            print(f"Expected: {expected_scores[i]}")
            
            if result['label'] == expected_scores[i]:
                print("✅ PASS")
            else:
                print("❌ FAIL")
            print("-" * 60)
        else:
            print(f"❌ No results returned for: '{response_text}'")
    else:
        print(f"❌ Failed to process assessment: {response.text}")
        
print("\nTesting complete!") 