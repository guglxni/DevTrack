#!/usr/bin/env python3
"""
Test script to verify that the API integration with reliable scoring works correctly.
"""

import requests
import json
import sys
from src.api.reliable_client import ReliableASDClient

# Configuration
API_URL = "http://localhost:8003"

def print_header(text):
    """Print a header with the given text."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_result(label, result):
    """Print a result with a label."""
    print(f"\n{label}:")
    print(json.dumps(result, indent=2))

def test_both_methods(milestone, response):
    """Test both the direct API call and the reliable client with the same input."""
    print_header(f"Testing: '{milestone}' with response: '{response}'")
    
    # Test 1: Direct API call
    print("\n1. Using direct API call:")
    direct_data = {
        "milestone_behavior": milestone,
        "response": response
    }
    
    try:
        direct_response = requests.post(
            f"{API_URL}/score-response",
            json=direct_data
        )
        
        if direct_response.status_code == 200:
            direct_result = direct_response.json()
            print_result("Direct API Call Result", direct_result)
        else:
            print(f"Error with direct API call: {direct_response.status_code}")
            print(direct_response.text)
    except Exception as e:
        print(f"Exception with direct API call: {str(e)}")
    
    # Test 2: Using reliable client
    print("\n2. Using reliable client:")
    client = ReliableASDClient(api_url=API_URL)
    
    try:
        reliable_result = client.score_response(milestone, response)
        print_result("Reliable Client Result", reliable_result)
    except Exception as e:
        print(f"Exception with reliable client: {str(e)}")
    
    print("\n" + "-" * 60)

def run_tests():
    """Run a series of tests with different milestones and responses."""
    # Test 1: The problematic milestone with a positive response
    test_both_methods(
        "Recognizes familiar people",
        "My child always smiles when he sees grandparents or his favorite babysitter. He knows all his family members and distinguishes between strangers and people he knows well."
    )
    
    # Test 2: The problematic milestone with a negative response
    test_both_methods(
        "Recognizes familiar people",
        "No, he doesn't recognize anyone, not even his parents."
    )
    
    # Test 3: A different milestone
    test_both_methods(
        "Makes eye contact",
        "Yes, she makes eye contact consistently when interacting."
    )

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
    
    print_header("API INTEGRATION TESTS")
    print("This script will test both direct API calls and the reliable client.")
    print("The reliable client should produce consistent, correct results.")
    
    run_tests()
    
    print_header("TESTS COMPLETE")
    print("If the reliable client shows consistent, correct scoring while")
    print("direct API calls show inconsistent or incorrect scoring, then")
    print("the integration is working correctly.") 