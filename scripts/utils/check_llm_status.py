#!/usr/bin/env python3
"""
Script to check the LLM status and see if there's any issue.
"""

import requests
import json
import sys

def check_llm_status():
    """Check the LLM status and print the response."""
    try:
        # Check the LLM health endpoint
        llm_response = requests.get("http://localhost:8003/llm-scoring/health")
        llm_data = llm_response.json()
        
        print("LLM Health Response:")
        print(json.dumps(llm_data, indent=2))
        
        # Check if the LLM is available
        if llm_data.get("status") == "available":
            print("\nLLM is available!")
            print(f"Model: {llm_data.get('model', 'Unknown')}")
            print(f"Mode: {llm_data.get('mode', 'Unknown')}")
        else:
            print("\nLLM is not available!")
            print(f"Status: {llm_data.get('status', 'Unknown')}")
            print(f"Message: {llm_data.get('message', 'Unknown')}")
        
        # Test the LLM with a simple request
        print("\nTesting LLM with a simple request...")
        test_data = {
            "question": "Does your child recognize familiar people?",
            "milestone": "recognizes familiar people",
            "response": "Yes, my child recognizes all family members easily."
        }
        
        test_response = requests.post("http://localhost:8003/llm-scoring/direct-test", json=test_data)
        test_data = test_response.json()
        
        print("LLM Test Response:")
        print(json.dumps(test_data, indent=2))
        
        return True
    except Exception as e:
        print(f"Error checking LLM status: {e}")
        return False

if __name__ == "__main__":
    success = check_llm_status()
    sys.exit(0 if success else 1) 