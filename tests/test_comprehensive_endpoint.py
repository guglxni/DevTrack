#!/usr/bin/env python3
"""
Test script to directly test the comprehensive assessment endpoint.
"""

import requests
import json
import sys

# Configuration
API_URL = "http://localhost:8003"

def test_comprehensive_assessment():
    """Test the comprehensive assessment endpoint with properly formatted data."""
    print("Testing comprehensive assessment endpoint...")
    
    # Test data for the problematic milestone
    milestone = "Recognizes familiar people"
    response_text = "My child always smiles when he sees grandparents and recognizes all family members."
    
    # Reliable keywords for this milestone
    keywords = {
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
            "yes they recognize",
            "always smiles when he sees"
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
    
    # Create the request data
    data = {
        "question": f"Does your child {milestone}?",
        "milestone_behavior": milestone,
        "parent_response": response_text,
        "keywords": keywords
    }
    
    # Print the request data for debugging
    print("\nRequest data:")
    print(json.dumps(data, indent=2))
    
    # Make the request
    try:
        response = requests.post(
            f"{API_URL}/comprehensive-assessment",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse data:")
            print(json.dumps(result, indent=2))
            
            if result.get("score_label") == "INDEPENDENT":
                print("\n✅ SUCCESS: The response was correctly scored as INDEPENDENT")
            else:
                print(f"\n❌ ERROR: The response was scored as {result.get('score_label')}, expected INDEPENDENT")
        else:
            print("\nError response:")
            print(response.text)
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}")

if __name__ == "__main__":
    # Check if the API is running
    try:
        health_check = requests.get(f"{API_URL}/health", timeout=5)
        if health_check.status_code != 200:
            print(f"Error: API server returned status code {health_check.status_code}")
            print("Make sure the API server is running.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Could not connect to API server: {str(e)}")
        print(f"Make sure the API server is running at {API_URL}")
        sys.exit(1)
    
    test_comprehensive_assessment()
