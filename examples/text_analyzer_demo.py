#!/usr/bin/env python3
"""
Text Analyzer Demonstration

This script demonstrates the text analysis capabilities for developmental assessments.
"""

import sys
import os
import logging
from typing import Dict, List, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Import the text analyzer
from src.core.knowledge.text_analyzer import (
    analyze_text_for_category,
    get_best_category_match,
    extract_key_details,
    generate_analysis_explanation,
    CATEGORY_PHRASES,
    DOMAIN_PHRASES
)

def print_available_phrases():
    """Print all available phrases used for text analysis."""
    print("\n===== AVAILABLE CATEGORY PHRASES =====\n")
    
    for category, phrases in CATEGORY_PHRASES.items():
        print(f"Category: {category}")
        for phrase in phrases:
            print(f"  - {phrase}")
        print()
    
    print("\n===== DOMAIN-SPECIFIC PHRASES =====\n")
    
    for domain, categories in DOMAIN_PHRASES.items():
        print(f"Domain: {domain}")
        for category, phrases in categories.items():
            print(f"  Category: {category}")
            for phrase in phrases:
                print(f"    - {phrase}")
        print()

def analyze_single_response():
    """Analyze a single response with detailed output."""
    response = input("\nEnter a response to analyze: ")
    domain = input("Enter domain (MOTOR, COMMUNICATION, SOCIAL, COGNITIVE) or leave blank: ")
    
    if not domain:
        domain = None
    
    print("\n===== ANALYSIS RESULTS =====\n")
    
    # Analyze for each category
    print("Category-by-Category Analysis:")
    for category in CATEGORY_PHRASES.keys():
        confidence, matched_phrases = analyze_text_for_category(response, category, domain)
        if confidence > 0:
            explanation = generate_analysis_explanation(category, matched_phrases, domain)
            print(f"  {category}: {confidence:.2f} - {explanation}")
        else:
            print(f"  {category}: No matches")
    
    # Get best match
    best_category, confidence, matched_phrases = get_best_category_match(response, domain)
    
    print("\nBest Category Match:")
    if best_category != "NOT_RATED":
        explanation = generate_analysis_explanation(best_category, matched_phrases, domain)
        print(f"  {best_category}: {confidence:.2f} - {explanation}")
    else:
        print("  No clear category match found")
    
    # Extract key details
    details = extract_key_details(response)
    
    print("\nExtracted Details:")
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")
    else:
        print("  No specific details extracted")

def batch_analyze_responses():
    """Analyze a batch of predefined responses."""
    test_responses = [
        {
            "domain": "MOTOR",
            "response": "My child is starting to crawl but needs help. She can push up on her hands but struggles to move forward on her own. I usually need to position her and guide her movements.",
            "expected": "WITH_SUPPORT"
        },
        {
            "domain": "MOTOR",
            "response": "He crawls quickly across the room without any help. He's very mobile and can get to anything he wants by crawling.",
            "expected": "INDEPENDENT"
        },
        {
            "domain": "COMMUNICATION",
            "response": "He has started saying a few words consistently like 'mama' and 'dada' in the right context. He can also point to things he wants.",
            "expected": "INDEPENDENT"
        },
        {
            "domain": "COMMUNICATION",
            "response": "She doesn't say any words yet. She makes some sounds but nothing that seems like real words.",
            "expected": "CANNOT_DO"
        },
        {
            "domain": "SOCIAL",
            "response": "Sometimes she plays with other kids, but it's hit or miss. Some days she'll interact and share toys, other days she prefers to play alone.",
            "expected": "EMERGING"
        },
        {
            "domain": "COGNITIVE",
            "response": "He used to be able to complete simple puzzles on his own, but in the last few months he's stopped doing them altogether. When I try to engage him with puzzles now, he gets frustrated and walks away.",
            "expected": "LOST_SKILL"
        }
    ]
    
    print("\n===== BATCH ANALYSIS RESULTS =====\n")
    
    correct = 0
    total = len(test_responses)
    
    for i, test in enumerate(test_responses, 1):
        domain = test["domain"]
        response = test["response"]
        expected = test["expected"]
        
        # Get best match
        best_category, confidence, matched_phrases = get_best_category_match(response, domain)
        explanation = generate_analysis_explanation(best_category, matched_phrases, domain)
        
        # Check if correct
        is_correct = best_category == expected
        if is_correct:
            correct += 1
            result_marker = "✓"
        else:
            result_marker = "✗"
        
        print(f"Test {i}: {domain} Domain")
        print(f"Response: \"{response}\"")
        print(f"Expected: {expected}")
        print(f"Result: {best_category} (confidence: {confidence:.2f})")
        print(f"Analysis: {explanation}")
        print(f"Outcome: {result_marker} {'Correct' if is_correct else 'Incorrect'}")
        print()
    
    # Print accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2f} ({correct}/{total})")

def main():
    """Main function to run the demonstration."""
    print("\nTEXT ANALYZER DEMONSTRATION")
    print("This script demonstrates text analysis for developmental assessments.\n")
    
    while True:
        print("\nOptions:")
        print("1. View available phrases")
        print("2. Analyze a single response")
        print("3. Run batch analysis")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            print_available_phrases()
        elif choice == "2":
            analyze_single_response()
        elif choice == "3":
            batch_analyze_responses()
        elif choice == "4":
            print("\nExiting demonstration.\n")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 