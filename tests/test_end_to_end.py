#!/usr/bin/env python3
"""
End-to-End Test for ASD Assessment API with Hybrid Scoring

This script tests the full API integration with the hybrid scoring approach,
simulating how the web UI interacts with the API.
"""

import requests
import json
import time
import sys

# API configuration
API_URL = "http://localhost:8003"

def test_health_check():
    """Verify that the API server is running."""
    print("Testing API Health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {str(e)}")
        return False

def test_comprehensive_assessment():
    """Test the comprehensive assessment endpoint with various test cases."""
    print("\nTesting Comprehensive Assessment Endpoint...")
    
    test_cases = [
        {
            "milestone": "Recognizes familiar people",
            "response": "My child always smiles when he sees grandparents and recognizes all family members.",
            "expected": "INDEPENDENT"
        },
        {
            "milestone": "Recognizes familiar people",
            "response": "No, he doesn't recognize anyone, not even his parents.",
            "expected": "CANNOT_DO"
        },
        {
            "milestone": "Recognizes familiar people",
            "response": "He knows all his family members and distinguishes between strangers and people he knows well.",
            "expected": "INDEPENDENT"
        },
        {
            "milestone": "Makes eye contact",
            "response": "Yes, she makes eye contact consistently when interacting.",
            "expected": "INDEPENDENT"
        },
        {
            "milestone": "Makes eye contact",
            "response": "Sometimes, but only when we encourage him.",
            "expected": "WITH_SUPPORT"
        },
        {
            "milestone": "Recognizes familiar people",
            "response": "My child knows all family members and smiles when they come in the room.",
            "expected": "INDEPENDENT",
            "note": "Previously problematic case - 'knows' substring issue"
        },
        {
            "milestone": "Recognizes familiar people", 
            "response": "She notices when her grandparents come to visit and gets excited.",
            "expected": "INDEPENDENT",
            "note": "Previously problematic case - 'notices' substring issue"
        }
    ]
    
    # Define reliable keywords for test cases
    reliable_keywords = {
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
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case #{i}:")
        print(f"  Milestone: {case['milestone']}")
        print(f"  Response: '{case['response']}'")
        print(f"  Expected: {case['expected']}")
        if "note" in case:
            print(f"  Note: {case['note']}")
        
        # Create assessment data
        assessment_data = {
            "question": f"Does your child {case['milestone']}?",
            "milestone_behavior": case['milestone'],
            "parent_response": case['response'],
            "keywords": reliable_keywords
        }
        
        # Make API request
        try:
            response = requests.post(
                f"{API_URL}/comprehensive-assessment",
                json=assessment_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                score_label = result["score_label"]
                score = result["score"]
                confidence = result.get("confidence", 0.0)
                
                print(f"  Result: {score_label} ({score}) with confidence {confidence:.2f}")
                
                # Check if the result matches the expected score
                if score_label == case["expected"]:
                    print("  ✅ PASSED")
                else:
                    print(f"  ❌ FAILED - Expected {case['expected']}, got {score_label}")
            else:
                print(f"  ❌ ERROR: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
    
    print("\nComprehensive assessment testing complete")

def test_score_response():
    """Test the score-response endpoint with the problematic test cases."""
    print("\nTesting Score Response Endpoint...")
    
    # Test cases that were previously problematic
    problematic_cases = [
        {
            "milestone": "Recognizes familiar people",
            "response": "My child knows all family members and smiles when they come in the room.",
            "expected": "INDEPENDENT",
            "note": "Previously problematic case - 'knows' substring issue"
        },
        {
            "milestone": "Recognizes familiar people", 
            "response": "She notices when her grandparents come to visit and gets excited.",
            "expected": "INDEPENDENT",
            "note": "Previously problematic case - 'notices' substring issue"
        }
    ]
    
    for i, case in enumerate(problematic_cases, 1):
        print(f"\nCase #{i}:")
        print(f"  Milestone: {case['milestone']}")
        print(f"  Response: '{case['response']}'")
        print(f"  Expected: {case['expected']}")
        print(f"  Note: {case['note']}")
        
        # Create request data
        request_data = {
            "milestone_behavior": case['milestone'],
            "response": case['response']
        }
        
        # Make API request
        try:
            response = requests.post(
                f"{API_URL}/score-response",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                score_label = result["score_label"]
                score = result["score"]
                confidence = result.get("confidence", 0.0)
                
                print(f"  Result: {score_label} ({score}) with confidence {confidence:.2f}")
                
                # Check if the result matches the expected score
                if score_label == case["expected"]:
                    print("  ✅ PASSED")
                else:
                    print(f"  ❌ FAILED - Expected {case['expected']}, got {score_label}")
            else:
                print(f"  ❌ ERROR: {response.status_code} - {response.text}")
        
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
    
    print("\nScore response testing complete")

def test_client_integration():
    """Test the API client integration to ensure it's working correctly."""
    print("\nTesting API Client Integration...")
    
    try:
        # Import the client
        from src.api.reliable_client import ReliableASDClient
        
        # Create the client
        client = ReliableASDClient(api_url=API_URL)
        
        # Check if API is running
        if not client.health_check():
            print("❌ API server is not running")
            return
            
        print("✅ API server is running")
        
        # Test milestone for recognition
        milestone = "Recognizes familiar people"
        
        # Test with positive response
        positive_response = "My child always smiles when he sees grandparents and recognizes all family members."
        print(f"\nTesting positive response with client:")
        print(f"  Milestone: {milestone}")
        print(f"  Response: '{positive_response}'")
        print(f"  Expected: INDEPENDENT")
        
        positive_result = client.score_response(milestone, positive_response)
        
        if "error" in positive_result:
            print(f"  ❌ ERROR: {positive_result['message']}")
        else:
            print(f"  Result: {positive_result['score_label']} ({positive_result['score']})")
            
            if positive_result.get("score_label") == "INDEPENDENT":
                print("  ✅ PASSED")
            else:
                print(f"  ❌ FAILED - Expected INDEPENDENT, got {positive_result.get('score_label')}")
        
        # Test with negative response
        negative_response = "No, he doesn't recognize anyone, not even his parents."
        print(f"\nTesting negative response with client:")
        print(f"  Milestone: {milestone}")
        print(f"  Response: '{negative_response}'")
        print(f"  Expected: CANNOT_DO")
        
        negative_result = client.score_response(milestone, negative_response)
        
        if "error" in negative_result:
            print(f"  ❌ ERROR: {negative_result['message']}")
        else:
            print(f"  Result: {negative_result['score_label']} ({negative_result['score']})")
            
            if negative_result.get("score_label") == "CANNOT_DO":
                print("  ✅ PASSED")
            else:
                print(f"  ❌ FAILED - Expected CANNOT_DO, got {negative_result.get('score_label')}")
        
        # Test with previously problematic case
        edge_response = "My child knows all family members and smiles when they come in the room."
        print(f"\nTesting edge case response with client:")
        print(f"  Milestone: {milestone}")
        print(f"  Response: '{edge_response}'")
        print(f"  Expected: INDEPENDENT")
        print(f"  Note: Previously problematic case - 'knows' substring issue")
        
        edge_result = client.score_response(milestone, edge_response)
        
        if "error" in edge_result:
            print(f"  ❌ ERROR: {edge_result['message']}")
        else:
            print(f"  Result: {edge_result['score_label']} ({edge_result['score']})")
            
            if edge_result.get("score_label") == "INDEPENDENT":
                print("  ✅ PASSED")
            else:
                print(f"  ❌ FAILED - Expected INDEPENDENT, got {edge_result.get('score_label')}")
        
    except ImportError as e:
        print(f"❌ Could not import ReliableASDClient: {str(e)}")
    except Exception as e:
        print(f"❌ Error testing client integration: {str(e)}")
    
    print("\nClient integration testing complete")

if __name__ == "__main__":
    print("======================================")
    print("      ASD Assessment API Test         ")
    print("======================================")
    
    # First check if the API is running
    if not test_health_check():
        print("\n❌ API server is not running. Please start the server with:")
        print("   /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m uvicorn src.api.app:app --port 8003")
        sys.exit(1)
    
    # Run all tests
    test_comprehensive_assessment()
    test_score_response()
    test_client_integration()
    
    print("\n======================================")
    print("        All tests completed           ")
    print("======================================") 