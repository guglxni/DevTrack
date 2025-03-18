#!/usr/bin/env python3
"""
Age-Specific Knowledge Integration Demo

This script demonstrates the use of age-specific knowledge for more accurate
developmental milestone assessment.
"""

import sys
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Import the knowledge module
from src.core.knowledge import (
    get_age_expectations,
    get_category_guidance,
    get_expected_skills,
    get_confidence_adjustment,
    get_age_bracket,
    adjust_category_for_age,
    get_best_category_match
)

# Define some test cases with different ages
TEST_CASES = [
    {
        "age_months": 8,
        "domain": "motor",
        "response": "My baby is just starting to sit without support. He wobbles a bit but can stay up for a few seconds before falling over. When I put him in a sitting position, he tries to stay there.",
        "milestone": "Sitting without support"
    },
    {
        "age_months": 18,
        "domain": "communication",
        "response": "She says about 10 words consistently and understands simple commands. She points to things she wants and is starting to put two words together sometimes.",
        "milestone": "Using words to communicate"
    },
    {
        "age_months": 30,
        "domain": "social",
        "response": "He plays alongside other children but rarely interacts directly. He'll watch what they're doing and sometimes copy them, but he doesn't really join in the play.",
        "milestone": "Interactive play with peers"
    }
]

def print_age_expectations(age_months: int):
    """Print age-specific developmental expectations"""
    expectations = get_age_expectations(age_months)
    if not expectations:
        print(f"No age-specific expectations found for {age_months} months")
        return
    
    print(f"\n===== EXPECTATIONS FOR {age_months} MONTHS ({expectations['age_range_str']}) =====")
    print(f"Description: {expectations['description']}")
    
    print("\nExpected Skills by Domain:")
    for domain, skills in expectations['expected_skills'].items():
        print(f"\n  {domain.upper()}:")
        for skill in skills:
            print(f"    - {skill}")
    
    print("\nAssessment Considerations:")
    for consideration in expectations['assessment_considerations']:
        print(f"  - {consideration}")
    
    print("\nCommon Misconceptions:")
    for misconception in expectations['common_misconceptions']:
        print(f"  - {misconception}")

def print_category_guidance_for_age(category: str, age_months: int):
    """Print age-specific guidance for a scoring category"""
    guidance = get_category_guidance(category, age_months)
    if not guidance:
        print(f"No age-specific guidance found for {category} at {age_months} months")
        return
    
    print(f"\n===== {category} AT {age_months} MONTHS ({guidance['age_range_str']}) =====")
    print(f"Description: {guidance['description']}")
    
    print("\nTypical Indicators:")
    for indicator in guidance['typical_indicators']:
        print(f"  - {indicator}")
    
    print(f"\nConfidence Adjustment: {guidance['confidence_adjustment']:.2f}")
    
    print("\nBoundary Considerations:")
    for boundary, note in guidance['boundary_considerations'].items():
        print(f"  - {boundary}: {note}")
    
    print("\nDomain-Specific Notes:")
    for domain, note in guidance['domain_specific_notes'].items():
        print(f"  - {domain.upper()}: {note}")

def analyze_with_age_adjustment(response: str, age_months: int, domain: str):
    """Analyze a response with age-specific adjustments"""
    # First get the category without age adjustment
    category, confidence, phrases = get_best_category_match(response, domain)
    
    # Then adjust based on age-specific knowledge
    adjusted_category, adjusted_confidence = adjust_category_for_age(category, confidence, age_months)
    
    print(f"\n===== ANALYSIS WITH AGE ADJUSTMENT ({age_months} months) =====")
    print(f"Response: \"{response}\"")
    print(f"Domain: {domain}")
    print(f"Age bracket: {get_age_bracket(age_months)}")
    
    print("\nWithout Age Adjustment:")
    print(f"  Category: {category}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Matched phrases: {', '.join(phrases)}")
    
    print("\nWith Age Adjustment:")
    print(f"  Category: {adjusted_category}")
    print(f"  Confidence: {adjusted_confidence:.2f}")
    print(f"  Adjustment: {adjusted_confidence - confidence:.2f}")
    
    # Get domain-specific expected skills for this age
    expected_skills = get_expected_skills(domain, age_months)
    if expected_skills:
        print("\nExpected skills for this age and domain:")
        for skill in expected_skills:
            print(f"  - {skill}")

def compare_across_ages():
    """Compare the same response analyzed at different ages"""
    response = "He tries to stand on his own but needs to hold onto furniture for balance. Sometimes he can let go for a few seconds."
    domain = "motor"
    
    print("\n===== COMPARING ANALYSIS ACROSS AGES =====")
    print(f"Response: \"{response}\"")
    print(f"Domain: {domain}")
    
    age_ranges = [9, 12, 15, 18]
    
    print("\nResults by Age:")
    for age in age_ranges:
        # Get category and confidence
        category, confidence, _ = get_best_category_match(response, domain)
        
        # Adjust for age
        adjusted_category, adjusted_confidence = adjust_category_for_age(category, confidence, age)
        
        # Display results
        print(f"\n  At {age} months ({get_age_bracket(age)}):")
        print(f"    Category: {adjusted_category}")
        print(f"    Confidence: {adjusted_confidence:.2f}")
        print(f"    Adjustment: {adjusted_confidence - confidence:.2f}")
        
        # Get relevant milestone for this age
        skills = get_expected_skills(domain, age)
        if skills:
            relevant_skills = [s for s in skills if "stand" in s.lower()]
            if relevant_skills:
                print(f"    Relevant expected skill: {relevant_skills[0]}")
            else:
                print(f"    Expected skills at this age: {skills[0]}")

def run_test_cases():
    """Run analysis on the predefined test cases"""
    print("\n===== ANALYZING TEST CASES WITH AGE-SPECIFIC KNOWLEDGE =====")
    
    for i, case in enumerate(TEST_CASES, 1):
        age_months = case["age_months"]
        domain = case["domain"]
        response = case["response"]
        milestone = case["milestone"]
        
        print(f"\nTest Case {i}: {milestone} ({domain}, {age_months} months)")
        print(f"Response: \"{response}\"")
        
        # Get category without age adjustment
        category, confidence, phrases = get_best_category_match(response, domain)
        
        # Adjust based on age
        adjusted_category, adjusted_confidence = adjust_category_for_age(category, confidence, age_months)
        
        # Display results
        print(f"\nBasic Analysis:")
        print(f"  Category: {category}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Phrases: {', '.join(phrases) if phrases else 'None'}")
        
        print(f"\nAge-Adjusted Analysis ({age_months} months, {get_age_bracket(age_months)}):")
        print(f"  Category: {adjusted_category}")
        print(f"  Confidence: {adjusted_confidence:.2f}")
        print(f"  Adjustment: {adjusted_confidence - confidence:.2f}")
        
        # Get age-specific guidance
        guidance = get_category_guidance(adjusted_category, age_months)
        if guidance:
            print("\nAge-Specific Guidance:")
            relevant_note = guidance["domain_specific_notes"].get(domain, "")
            if relevant_note:
                print(f"  {relevant_note}")
            
            # Get boundary consideration if confidence is borderline
            if 0.55 <= adjusted_confidence <= 0.75:
                for boundary, note in guidance["boundary_considerations"].items():
                    print(f"  Boundary with {boundary}: {note}")

def main():
    """Main function to demonstrate age-specific knowledge integration"""
    print("AGE-SPECIFIC KNOWLEDGE INTEGRATION DEMO")
    print("=======================================")
    
    while True:
        print("\nOptions:")
        print("1. View age-specific expectations")
        print("2. View category guidance for a specific age")
        print("3. Analyze a response with age adjustment")
        print("4. Compare the same response across ages")
        print("5. Run predefined test cases")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            age = int(input("Enter age in months: "))
            print_age_expectations(age)
        elif choice == "2":
            category = input("Enter category (CANNOT_DO, WITH_SUPPORT, EMERGING, INDEPENDENT, LOST_SKILL): ").upper()
            age = int(input("Enter age in months: "))
            print_category_guidance_for_age(category, age)
        elif choice == "3":
            response = input("Enter response to analyze: ")
            age = int(input("Enter age in months: "))
            domain = input("Enter domain (motor, communication, social, cognitive): ").lower()
            analyze_with_age_adjustment(response, age, domain)
        elif choice == "4":
            compare_across_ages()
        elif choice == "5":
            run_test_cases()
        elif choice == "6":
            print("\nExiting the demo.")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main() 