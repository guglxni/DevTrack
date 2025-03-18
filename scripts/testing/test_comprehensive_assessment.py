#!/usr/bin/env python3
"""
Script to test the comprehensive assessment functionality with the LLM.
"""

import requests
import json
import sys

def test_comprehensive_assessment():
    """Test the comprehensive assessment functionality with the LLM."""
    try:
        # Define test cases
        test_cases = [
            {
                "question": "Does your child recognize familiar people?",
                "milestone_behavior": "Recognizes familiar people",
                "parent_response": "Yes, my child recognizes all family members easily."
            },
            {
                "question": "Does your child walk independently?",
                "milestone_behavior": "Walks independently",
                "parent_response": "She's just starting to take a few steps on her own but still needs support sometimes."
            },
            {
                "question": "Does your child use words to communicate?",
                "milestone_behavior": "Uses words to communicate",
                "parent_response": "No, he doesn't say any words yet, just makes sounds."
            }
        ]
        
        print("Testing comprehensive assessment with LLM...\n")
        
        for i, test_case in enumerate(test_cases):
            print(f"Test Case #{i+1}:")
            print(f"Question: {test_case['question']}")
            print(f"Milestone: {test_case['milestone_behavior']}")
            print(f"Response: {test_case['parent_response']}")
            
            # Call the comprehensive assessment endpoint
            response = requests.post(
                "http://localhost:8003/api/comprehensive-assessment",
                json=test_case
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\nResult:")
                print(f"Score: {result['score_label']} ({result['score']})")
                print(f"Confidence: {result['confidence']}")
                print(f"Domain: {result['domain']}")
                print(f"Milestone Found: {result['milestone_found']}")
                
                # Check if domain information is included
                if result['domain']:
                    print("✅ Domain information is included")
                else:
                    print("❌ Domain information is missing")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
            
            print("-" * 60)
        
        return True
    except Exception as e:
        print(f"Error testing comprehensive assessment: {e}")
        return False

if __name__ == "__main__":
    success = test_comprehensive_assessment()
    sys.exit(0 if success else 1) 