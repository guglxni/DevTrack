#!/usr/bin/env python3
"""
Test script to verify that the hybrid scorer correctly handles word boundaries
and provides reliable scoring results.
"""

import sys
import json
from src.api.hybrid_scorer import score_response

def test_hybrid_scoring():
    """Test the hybrid scoring system with various test cases."""
    print("Testing Hybrid Scoring System")
    print("=============================\n")
    
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
            "milestone": "Walking",
            "response": "She's just starting to take a few steps with support.",
            "expected": "EMERGING"
        },
        {
            "milestone": "Recognizes familiar people",
            "response": "Knows all of his family members",
            "expected": "INDEPENDENT"
        },
        {
            "milestone": "Recognizes familiar people",
            "response": "He used to recognize people but now he doesn't anymore.",
            "expected": "LOST_SKILL"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case #{i}:")
        print(f"  Milestone: {case['milestone']}")
        print(f"  Response: '{case['response']}'")
        print(f"  Expected: {case['expected']}")
        
        # Score the response
        result = score_response(case['milestone'], case['response'])
        
        # Check if the result is as expected
        score_label = result['score_label']
        confidence = result['confidence']
        passed = score_label == case['expected']
        
        print(f"  Result: {score_label} (confidence: {confidence:.2f})")
        
        if passed:
            print("  ✅ PASSED")
        else:
            print(f"  ❌ FAILED - Expected {case['expected']}, got {score_label}")
        
        # Print detailed matching info
        if 'detail' in result and 'matches' in result['detail']:
            matches = result['detail']['matches']
            # Show only categories with matches
            for category, keywords in matches.items():
                if keywords:
                    print(f"    Matched {category} keywords: {', '.join(keywords)}")
        
        print()
    
    print("Testing complete!\n")

def test_problematic_cases():
    """Test specific cases that were problematic in the original system."""
    print("Testing Previously Problematic Cases")
    print("===================================\n")
    
    problematic_cases = [
        {
            "milestone": "Recognizes familiar people",
            "response": "My child knows all family members and smiles when they come in the room.",
            "issue": "Original system matched 'no' in 'knows' incorrectly classifying as CANNOT_DO",
            "expected": "INDEPENDENT"
        },
        {
            "milestone": "Recognizes familiar people",
            "response": "She notices when her grandparents come to visit and gets excited.",
            "issue": "Original system matched 'not' in 'notices' incorrectly classifying as CANNOT_DO",
            "expected": "INDEPENDENT"
        }
    ]
    
    for i, case in enumerate(problematic_cases, 1):
        print(f"Problematic Case #{i}:")
        print(f"  Issue: {case['issue']}")
        print(f"  Milestone: {case['milestone']}")
        print(f"  Response: '{case['response']}'")
        print(f"  Expected: {case['expected']}")
        
        # Score the response
        result = score_response(case['milestone'], case['response'])
        
        # Check if the result is as expected
        score_label = result['score_label']
        confidence = result['confidence']
        passed = score_label == case['expected']
        
        print(f"  Result: {score_label} (confidence: {confidence:.2f})")
        
        if passed:
            print("  ✅ FIXED - Correct classification!")
        else:
            print(f"  ❌ STILL PROBLEMATIC - Expected {case['expected']}, got {score_label}")
        
        # Print detailed matching info
        if 'detail' in result and 'matches' in result['detail']:
            matches = result['detail']['matches']
            # Show only categories with matches
            for category, keywords in matches.items():
                if keywords:
                    print(f"    Matched {category} keywords: {', '.join(keywords)}")
        
        print()
    
    print("Problematic case testing complete!\n")

def test_with_custom_keywords():
    """Test with custom keywords to verify they work correctly."""
    print("Testing with Custom Keywords")
    print("===========================\n")
    
    milestone = "Recognizes familiar people"
    response = "My child shows recognition when family members are present."
    
    # Custom keywords for this test
    custom_keywords = {
        "INDEPENDENT": ["shows recognition", "recognizes when", "knows family"],
        "CANNOT_DO": ["doesn't show recognition", "no recognition"]
    }
    
    print(f"Milestone: {milestone}")
    print(f"Response: '{response}'")
    print(f"Custom Keywords: {json.dumps(custom_keywords, indent=2)}")
    
    # First test without custom keywords
    print("\nScoring WITHOUT custom keywords:")
    result_without = score_response(milestone, response)
    print(f"  Result: {result_without['score_label']} (confidence: {result_without['confidence']:.2f})")
    
    # Print matched keywords
    if 'detail' in result_without and 'matches' in result_without['detail']:
        matches = result_without['detail']['matches']
        for category, keywords in matches.items():
            if keywords:
                print(f"  Matched {category} keywords: {', '.join(keywords)}")
    
    # Now test with custom keywords
    print("\nScoring WITH custom keywords:")
    result_with = score_response(milestone, response, custom_keywords)
    print(f"  Result: {result_with['score_label']} (confidence: {result_with['confidence']:.2f})")
    
    # Print matched keywords
    if 'detail' in result_with and 'matches' in result_with['detail']:
        matches = result_with['detail']['matches']
        for category, keywords in matches.items():
            if keywords:
                print(f"  Matched {category} keywords: {', '.join(keywords)}")
    
    print("\nCustom keyword testing complete!\n")

if __name__ == "__main__":
    # Run the tests
    test_hybrid_scoring()
    test_problematic_cases()
    test_with_custom_keywords()
    
    print("All tests completed!") 