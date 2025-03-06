#!/usr/bin/env python3
import requests
import json
import time

# API endpoint
API_URL = "http://localhost:8003"

print("Fetching all milestones...")
response = requests.get(f"{API_URL}/all-milestones")
if response.status_code != 200:
    print(f"Failed to get milestones: {response.text}")
    exit(1)

milestones = response.json()["milestones"]
print(f"Found {len(milestones)} milestones")

# Reset the assessment engine
print("Resetting assessment engine...")
response = requests.post(f"{API_URL}/reset")
if response.status_code != 200:
    print(f"Failed to reset: {response.text}")
    exit(1)

# Define common keyword templates for each score category
keyword_templates = {
    "INDEPENDENT": [
        "always {verb}",
        "consistently {verb}",
        "easily {verb}",
        "definitely {verb}",
        "clearly {verb}",
        "{verb} without any issues",
        "{verb} independently",
        "{verb} well",
        "no problem {gerund}",
        "has mastered {gerund}",
        "regularly {verb}",
        "yes, {verb}"
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
        "{verb} in familiar situations",
        "{verb} with encouragement"
    ],
    "EMERGING": [
        "starting to {verb}",
        "beginning to {verb}",
        "occasionally {verb}",
        "{verb} inconsistently",
        "sometimes {verb}",
        "might {verb}",
        "rarely {verb}",
        "trying to {verb}",
        "learning to {verb}",
        "developing ability to {verb}"
    ],
    "LOST_SKILL": [
        "used to {verb}",
        "previously {past}",
        "{past} before",
        "no longer {verb}",
        "stopped {gerund}",
        "lost ability to {verb}",
        "could {verb} before",
        "regressed in {gerund}",
        "regressed"
    ],
    "CANNOT_DO": [
        "doesn't {verb}",
        "does not {verb}",
        "unable to {verb}",
        "never {verb}",
        "can't {verb}",
        "cannot {verb}",
        "no {noun}",
        "fails to {verb}",
        "shows no {noun}",
        "not able to {verb}",
        "hasn't {past}",
        "has not {past}",
        "doesn't know how to {verb}",
        "not interested in {gerund}"
    ]
}

# Define verb forms for each milestone
milestone_verbs = {}

# Process each milestone
for milestone in milestones:
    behavior = milestone["behavior"]
    
    # Extract verb and noun from behavior
    words = behavior.lower().split()
    
    # Default verb forms
    verb_forms = {
        "verb": behavior.lower(),
        "gerund": behavior.lower() + "ing",
        "past": behavior.lower() + "ed",
        "noun": behavior.lower()
    }
    
    # Try to extract a more specific verb
    if len(words) > 1:
        # First word is often a verb
        main_verb = words[0]
        
        # Common verbs and their forms
        if main_verb in ["recognize", "recognizes", "recognise", "recognises"]:
            verb_forms = {
                "verb": "recognize",
                "gerund": "recognizing",
                "past": "recognized",
                "noun": "recognition"
            }
        elif main_verb in ["respond", "responds"]:
            verb_forms = {
                "verb": "respond",
                "gerund": "responding",
                "past": "responded",
                "noun": "response"
            }
        elif main_verb in ["use", "uses"]:
            verb_forms = {
                "verb": "use",
                "gerund": "using",
                "past": "used",
                "noun": "use"
            }
        elif main_verb in ["play", "plays"]:
            verb_forms = {
                "verb": "play",
                "gerund": "playing",
                "past": "played",
                "noun": "play"
            }
        elif main_verb in ["show", "shows"]:
            verb_forms = {
                "verb": "show",
                "gerund": "showing",
                "past": "shown",
                "noun": "showing"
            }
        elif main_verb in ["take", "takes"]:
            verb_forms = {
                "verb": "take",
                "gerund": "taking",
                "past": "taken",
                "noun": "taking"
            }
    
    milestone_verbs[behavior] = verb_forms

# Generate custom keywords for each milestone
for milestone in milestones:
    behavior = milestone["behavior"]
    domain = milestone["domain"]
    
    print(f"\nProcessing milestone: {behavior} (Domain: {domain})")
    
    verb_forms = milestone_verbs[behavior]
    
    # Generate keywords for each category
    for category, templates in keyword_templates.items():
        keywords = []
        
        for template in templates:
            keyword = template.format(
                verb=verb_forms["verb"],
                gerund=verb_forms["gerund"],
                past=verb_forms["past"],
                noun=verb_forms["noun"]
            )
            keywords.append(keyword)
        
        # Add milestone-specific keywords
        if category == "INDEPENDENT" and "recognize" in behavior.lower():
            keywords.extend([
                "knows everyone",
                "recognizes immediately",
                "always knows who is who",
                "easily identifies people"
            ])
        elif category == "CANNOT_DO" and "recognize" in behavior.lower():
            keywords.extend([
                "doesn't know anyone",
                "confused by familiar faces",
                "treats everyone as strangers"
            ])
        
        # Update the keywords
        update_data = {
            "category": category,
            "keywords": keywords
        }
        
        response = requests.post(
            f"{API_URL}/keywords",
            json=update_data
        )
        
        if response.status_code == 200:
            print(f"✅ Updated {len(keywords)} keywords for {category}")
        else:
            print(f"❌ Failed to update {category} keywords: {response.text}")
    
    # Test with some common responses
    positive_response = f"Yes, they {verb_forms['verb']} very well."
    negative_response = f"No, they don't {verb_forms['verb']} at all."
    emerging_response = f"They're just starting to {verb_forms['verb']}."
    
    test_responses = [
        positive_response,
        negative_response,
        emerging_response
    ]
    
    print("\nTesting responses:")
    for response_text in test_responses:
        data = {
            "milestone_behavior": behavior,
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
    
    # Allow some time between processing milestones to not overwhelm the server
    time.sleep(0.5)

print("\nAll milestones processed successfully!")
print("The scoring system has been updated with better keywords.")
print("Try using the API again with your responses.") 