#!/usr/bin/env python3
"""
Example of how to integrate the reliable ASD Assessment API client
into your existing code.
"""

from reliable_api_client import ASDAsmessmentAPI
import json

def process_milestone_assessment(milestone, response_text):
    """Example function processing a milestone assessment."""
    
    # Initialize the API client
    api_client = ASDAsmessmentAPI(api_url="http://localhost:8003")
    
    # Check if the API is running
    if not api_client.health_check():
        print("Error: API server is not running")
        return None
    
    # Submit the assessment with reliable scoring
    result = api_client.submit_milestone_assessment(milestone, response_text)
    
    # Process the result
    if "error" in result:
        print(f"Error: {result['message']}")
        return None
    
    # Extract important information
    score_label = result.get("score_label")
    score = result.get("score")
    domain = result.get("domain")
    
    # Do something with the results
    print(f"Assessment for milestone '{milestone}':")
    print(f"  Score: {score_label} ({score})")
    print(f"  Domain: {domain}")
    
    return result

def run_examples():
    """Run examples of milestone assessment scoring."""
    
    # Example 1: Positive response
    print("\n--- Example 1: Positive Response ---")
    positive_response = "My child always smiles when he sees grandparents and recognizes all family members."
    process_milestone_assessment("Recognizes familiar people", positive_response)
    
    # Example 2: Negative response
    print("\n--- Example 2: Negative Response ---")
    negative_response = "No, he doesn't recognize anyone, not even his parents."
    process_milestone_assessment("Recognizes familiar people", negative_response)
    
    # Example 3: Emerging response
    print("\n--- Example 3: Emerging Response ---")
    emerging_response = "She's just starting to recognize grandparents, but it's inconsistent."
    process_milestone_assessment("Recognizes familiar people", emerging_response)

if __name__ == "__main__":
    print("ASD Assessment API Integration Example")
    print("======================================")
    run_examples()
    print("\n======================================")
    print("USAGE GUIDE:")
    print("1. Import the API client: from reliable_api_client import ASDAsmessmentAPI")
    print("2. Initialize the client: api_client = ASDAsmessmentAPI()")
    print("3. Call the assessment method: result = api_client.submit_milestone_assessment(milestone, response)")
    print("4. Process the result in your application")
    print("=======================================") 