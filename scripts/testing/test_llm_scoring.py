#!/usr/bin/env python3
"""
Script to test the LLM scoring with different responses.
"""

import requests
import json
import sys
import time

def test_llm_scoring():
    """Test the LLM scoring with different responses."""
    try:
        # Define test cases with expected scores
        test_cases = [
            {
                "milestone_behavior": "Recognizes familiar people",
                "response": "Yes, my child recognizes all family members easily.",
                "expected_score": "INDEPENDENT"
            },
            {
                "milestone_behavior": "Walks independently",
                "response": "She's just starting to take a few steps on her own but still needs support sometimes.",
                "expected_score": "EMERGING"
            },
            {
                "milestone_behavior": "Uses words to communicate",
                "response": "No, he doesn't say any words yet, just makes sounds.",
                "expected_score": "CANNOT_DO"
            },
            {
                "milestone_behavior": "Points to ask for things",
                "response": "He used to point but has stopped doing that recently.",
                "expected_score": "LOST_SKILL"
            },
            {
                "milestone_behavior": "Follows simple directions",
                "response": "She can follow directions but only when I help her understand what to do.",
                "expected_score": "WITH_SUPPORT"
            }
        ]
        
        print("Testing LLM scoring with different responses...\n")
        
        success_count = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"Test Case #{i+1}:")
            print(f"Milestone: {test_case['milestone_behavior']}")
            print(f"Response: {test_case['response']}")
            print(f"Expected Score: {test_case['expected_score']}")
            
            # Call the LLM scoring endpoint
            response = requests.post(
                "http://localhost:8003/llm-scoring/score",
                json={
                    "response": test_case['response'],
                    "milestone_behavior": test_case['milestone_behavior'],
                    "domain": None,
                    "age_range": None
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\nResult:")
                print(f"Score: {result['score_label']} ({result['score']})")
                print(f"Confidence: {result['confidence']}")
                print(f"Reasoning: {result['reasoning']}")
                
                # Check if the score matches the expected score
                if result['score_label'] == test_case['expected_score']:
                    print("✅ Score matches expected score")
                    success_count += 1
                else:
                    print(f"❌ Score does not match expected score")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
            
            print("-" * 60)
            
            # Add a small delay to avoid overwhelming the LLM
            time.sleep(1)
        
        print(f"\nSummary: {success_count}/{len(test_cases)} test cases passed")
        
        return success_count == len(test_cases)
    except Exception as e:
        print(f"Error testing LLM scoring: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_scoring()
    sys.exit(0 if success else 1) 