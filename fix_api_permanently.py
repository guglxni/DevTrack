#!/usr/bin/env python3
import requests
import json
import time
import sys

# API endpoint
API_URL = "http://localhost:8003"

print("=============================================")
print("PERMANENT FIX FOR ASD ASSESSMENT API SCORING")
print("=============================================")

# Check if API is running
try:
    health_check = requests.get(f"{API_URL}/health", timeout=5)
    if health_check.status_code != 200:
        print("❌ API server is not responding correctly.")
        print(f"Response: {health_check.status_code}")
        sys.exit(1)
    print("✅ API server is running.")
except Exception as e:
    print(f"❌ Could not connect to API server: {str(e)}")
    print("Please make sure the server is running at http://localhost:8003")
    sys.exit(1)

# First reset the assessment engine
print("\nResetting assessment engine...")
response = requests.post(f"{API_URL}/reset")
if response.status_code != 200:
    print(f"❌ Failed to reset: {response.text}")
    sys.exit(1)
print("✅ Assessment engine reset successfully.")

# Get available milestones
print("\nFetching all milestones...")
response = requests.get(f"{API_URL}/all-milestones")
if response.status_code != 200:
    print(f"❌ Failed to get milestones: {response.text}")
    sys.exit(1)

milestones = response.json()["milestones"]
print(f"✅ Found {len(milestones)} milestones")

# Define improved keyword maps for problematic milestones
special_milestone_keywords = {
    "Recognizes familiar people": {
        "INDEPENDENT": [
            "always recognizes",
            "consistently recognizes", 
            "easily recognizes",
            "immediately recognizes",
            "recognizes instantly",
            "definitely recognizes",
            "clearly recognizes",
            "recognizes without issues",
            "knows family members",
            "recognizes everyone",
            "knows everyone",
            "distinguishes between strangers",
            "smiles at familiar people",
            "yes he recognizes",
            "yes she recognizes",
            "yes they recognize"
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
            "doesn't recognize anyone",
            "does not recognize anyone",
            "unable to recognize",
            "never recognizes",
            "can't recognize",
            "cannot recognize anyone",
            "fails to recognize",
            "shows no recognition",
            "treats everyone as strangers",
            "doesn't know who people are"
        ]
    }
}

# Define generic keyword templates for all milestones
generic_keyword_templates = {
    "INDEPENDENT": [
        "always {verb}",
        "consistently {verb}",
        "easily {verb}",
        "definitely {verb}",
        "clearly {verb}",
        "{verb} without any issues",
        "{verb} independently",
        "{verb} well",
        "has mastered {gerund}",
        "regularly {verb}",
        "yes {pronoun} {verb}"
    ],
    "WITH_SUPPORT": [
        "{verb} with help",
        "{verb} with assistance",
        "{verb} with support",
        "{verb} with guidance",
        "{verb} when prompted",
        "{verb} when reminded",
        "needs help to {verb}",
        "needs assistance {gerund}",
        "{verb} with encouragement"
    ],
    "EMERGING": [
        "starting to {verb}",
        "beginning to {verb}",
        "occasionally {verb}",
        "{verb} inconsistently",
        "sometimes {verb}",
        "might {verb}",
        "trying to {verb}",
        "learning to {verb}"
    ],
    "LOST_SKILL": [
        "used to {verb}",
        "previously {past}",
        "{past} before",
        "no longer {verb}",
        "stopped {gerund}",
        "lost ability to {verb}",
        "could {verb} before",
        "regressed in {gerund}"
    ],
    "CANNOT_DO": [
        "doesn't {verb}",
        "does not {verb}",
        "unable to {verb}",
        "never {verb}",
        "can't {verb}",
        "cannot {verb}",
        "fails to {verb}",
        "not able to {verb}",
        "hasn't {past}",
        "has not {past}"
    ]
}

# Process all milestones
count_updated = 0
print("\nUpdating keyword maps for all milestones...")

for milestone in milestones:
    behavior = milestone["behavior"]
    domain = milestone["domain"]
    
    # Check if this is a special milestone with predefined keywords
    if behavior in special_milestone_keywords:
        print(f"\nProcessing special milestone: {behavior} (Domain: {domain})")
        
        for category, keywords in special_milestone_keywords[behavior].items():
            update_data = {
                "category": category,
                "keywords": keywords
            }
            
            response = requests.post(
                f"{API_URL}/keywords",
                json=update_data
            )
            
            if response.status_code == 200:
                print(f"  ✅ Updated {len(keywords)} keywords for {category}")
                count_updated += 1
            else:
                print(f"  ❌ Failed to update {category} keywords: {response.text}")
    else:
        # For regular milestones, just update the CANNOT_DO keywords to prevent false positives
        print(f"\nProcessing milestone: {behavior} (Domain: {domain})")
        
        # Extract main verb from behavior
        words = behavior.lower().split()
        verb = words[0] if len(words) > 0 else behavior.lower()
        
        # Common verb forms
        verb_forms = {
            "verb": verb,
            "gerund": verb + "ing",
            "past": verb + "ed",
            "pronoun": "they"
        }
        
        # Update only CANNOT_DO category to remove simple keywords and use more specific phrases
        cannot_do_keywords = []
        for template in generic_keyword_templates["CANNOT_DO"]:
            keyword = template.format(**verb_forms)
            cannot_do_keywords.append(keyword)
        
        update_data = {
            "category": "CANNOT_DO",
            "keywords": cannot_do_keywords
        }
        
        response = requests.post(
            f"{API_URL}/keywords",
            json=update_data
        )
        
        if response.status_code == 200:
            print(f"  ✅ Updated CANNOT_DO keywords")
            count_updated += 1
        else:
            print(f"  ❌ Failed to update CANNOT_DO keywords: {response.text}")

print(f"\nCompleted updating {count_updated} keyword maps.")

# Test the most problematic milestone
test_response = "my child always smiles when he sees grandparents or his favorite babysitter. he knows all his family members and distinguishes between strangers and people he knows well."

print("\nTesting the fixed API with a known problematic response...")
assessment_data = {
    "question": "Does your child Recognizes familiar people?",
    "milestone_behavior": "Recognizes familiar people",
    "parent_response": test_response,
    "keywords": None  # Now we don't need to provide keywords directly
}

response = requests.post(
    f"{API_URL}/comprehensive-assessment",
    json=assessment_data
)

if response.status_code == 200:
    result = response.json()
    print(f"Response: '{test_response}'")
    print(f"Score: {result['score_label']} ({result['score']})")
    
    if result['score_label'] == "INDEPENDENT":
        print("✅ SUCCESS! The API is now correctly scoring positive responses.")
    else:
        print("❓ The API is still not scoring correctly. Consider using the utility script for your API calls.")
else:
    print(f"❌ Failed to process assessment: {response.text}")

print("\n=============================================")
print("HOW TO USE THE FIXED API")
print("=============================================")
print("Option 1: Use the submit_assessment.py utility for reliable scoring:")
print("  python3 submit_assessment.py \"Recognizes familiar people\" \"Your response text\"")
print("\nOption 2: Import the function in your code:")
print("  from submit_assessment import submit_milestone_assessment")
print("  result = submit_milestone_assessment(\"Recognizes familiar people\", \"Your response\")")
print("\nOption 3: Always include keywords in your API requests")
print("=============================================") 