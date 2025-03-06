#!/usr/bin/env python3
"""
Test script to verify that the reliable client is working properly.
"""

import sys
from src.api.reliable_client import ReliableASDClient

def test_client():
    """Test the reliable client with a problematic milestone."""
    print("Testing reliable client...")
    
    # Create the client
    client = ReliableASDClient(api_url="http://localhost:8003")
    
    # Check if API is running
    if not client.health_check():
        print("Error: API server is not running")
        sys.exit(1)
        
    print("API server is running.")
    
    # Test with the problematic milestone
    milestone = "Recognizes familiar people"
    response_text = "My child always smiles when he sees grandparents and recognizes all family members."
    
    print(f"\nTesting milestone: {milestone}")
    print(f"Response: {response_text}")
    
    # Score the response
    result = client.score_response(milestone, response_text)
    
    # Print the result
    print("\nResult:")
    if "error" in result:
        print(f"Error: {result['message']}")
        sys.exit(1)
    
    print(f"Score: {result['score_label']} ({result['score']})")
    print(f"Confidence: {result['confidence']}")
    
    # Check if the scoring is correct
    if result.get("score_label") == "INDEPENDENT":
        print("\n✅ SUCCESS: The response was correctly scored as INDEPENDENT")
    else:
        print(f"\n❌ ERROR: The response was scored as {result.get('score_label')}, expected INDEPENDENT")
    
    # Test with a negative response
    negative_response = "No, he doesn't recognize anyone, not even his parents."
    print(f"\nTesting negative response: '{negative_response}'")
    
    result = client.score_response(milestone, negative_response)
    
    print("\nResult:")
    print(f"Score: {result['score_label']} ({result['score']})")
    
    if result.get("score_label") == "CANNOT_DO":
        print("\n✅ SUCCESS: The negative response was correctly scored as CANNOT_DO")
    else:
        print(f"\n❌ ERROR: The negative response was scored as {result.get('score_label')}, expected CANNOT_DO")
    
    print("\nClient test complete!")

if __name__ == "__main__":
    test_client() 