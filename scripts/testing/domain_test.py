#!/usr/bin/env python3
"""
Script to test the comprehensive assessment with domain information.
"""

import requests
import json
import sys

def test_domain_info():
    """Test the comprehensive assessment with domain information."""
    try:
        # Define test cases with expected domains
        test_cases = [
            {
                "question": "Does your child recognize familiar people?",
                "milestone_behavior": "Recognizes familiar people",
                "parent_response": "Yes, my child recognizes all family members easily.",
                "expected_domain": "SOC"  # Social domain
            },
            {
                "question": "Does your child walk independently?",
                "milestone_behavior": "Walks independently",
                "parent_response": "She's just starting to take a few steps on her own but still needs support sometimes.",
                "expected_domain": "GM"  # Gross Motor domain
            },
            {
                "question": "Does your child use words to communicate?",
                "milestone_behavior": "Uses words to communicate",
                "parent_response": "No, he doesn't say any words yet, just makes sounds.",
                "expected_domain": "EL"  # Expressive Language domain
            }
        ]
        
        print("Testing comprehensive assessment with domain information...\n")
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"Test Case #{i+1}:")
            print(f"Question: {test_case['question']}")
            print(f"Milestone: {test_case['milestone_behavior']}")
            print(f"Response: {test_case['parent_response']}")
            print(f"Expected Domain: {test_case['expected_domain']}")
            
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
                
                # Check if domain information is included
                if result['domain']:
                    print("✅ Domain information is included")
                    
                    # Check if the domain matches the expected domain
                    if result['domain'] == test_case['expected_domain']:
                        print("✅ Domain matches expected domain")
                        success_count += 1
                    else:
                        print(f"❌ Domain does not match expected domain (got {result['domain']}, expected {test_case['expected_domain']})")
                else:
                    print("❌ Domain information is missing")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
            
            print("-" * 60)
        
        print(f"\nSummary: {success_count}/{len(test_cases)} test cases passed")
        
        return success_count == len(test_cases)
    except Exception as e:
        print(f"Error testing domain information: {e}")
        return False

if __name__ == "__main__":
    success = test_domain_info()
    sys.exit(0 if success else 1) 