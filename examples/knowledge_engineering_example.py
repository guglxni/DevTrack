#!/usr/bin/env python3
"""
Knowledge Engineering Example

This script demonstrates how to use the Knowledge Engineering module
and domain-specific prompts for developmental milestone assessment.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.append('.')

from src.core.knowledge import (
    get_domain_by_name, 
    get_all_domains, 
    load_prompt, 
    format_prompt_with_context
)
from src.core.scoring.llm_scorer import LLMBasedScorer
from src.core.scoring.base import Score

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("knowledge_example")

def print_domain_info(domain_name: str) -> None:
    """Print information about a developmental domain."""
    domain = get_domain_by_name(domain_name)
    if not domain:
        logger.error(f"Domain '{domain_name}' not found")
        return
    
    print(f"\n{'=' * 50}")
    print(f"Domain: {domain.name} ({domain.code})")
    print(f"{'=' * 50}")
    print(f"Description: {domain.description}")
    print("\nMilestone Types:")
    for milestone_type in domain.milestone_types:
        print(f"  - {milestone_type}")
    
    print("\nAssessment Considerations:")
    for consideration in domain.assessment_considerations:
        print(f"  - {consideration}")
    
    print("\nCategory Rubrics:")
    for category_name, rubric in domain.category_rubrics.items():
        print(f"  {category_name}:")
        print(f"    Description: {rubric.description}")
        print(f"    Criteria: {', '.join(rubric.criteria[:2])}...")
        print(f"    Keywords: {', '.join(rubric.keywords[:5])}...")
        print()

def demonstrate_prompt_templates() -> None:
    """Demonstrate the use of domain-specific prompt templates."""
    domains = ["motor", "communication", "social", "cognitive"]
    
    print("\n\n" + "=" * 70)
    print("DOMAIN-SPECIFIC PROMPT TEMPLATES")
    print("=" * 70)
    
    for domain in domains:
        template = load_prompt(domain)
        if not template:
            logger.warning(f"No template found for domain: {domain}")
            continue
        
        print(f"\n{'-' * 50}")
        print(f"Domain: {template['domain_name']}")
        print(f"{'-' * 50}")
        print("Domain Guidance:")
        print(template["domain_guidance"])
        print("\nCategory Descriptions:")
        for category, desc in template["category_descriptions"].items():
            print(f"  {category}: {desc}")
        print()

def score_with_domain_specific_prompts() -> None:
    """Score responses using domain-specific prompts."""
    # Test cases for different domains
    test_cases = [
        {
            "domain": "motor",
            "response": "My child is starting to crawl but needs help sometimes. She can push up on her hands but struggles to move forward consistently.",
            "milestone_context": {
                "behavior": "Crawling",
                "criteria": "Child moves on hands and knees across the floor",
                "age_range": "6-10 months",
                "domain": "motor"
            }
        },
        {
            "domain": "communication",
            "response": "He babbles a lot and sometimes seems to say 'mama' or 'dada' but I'm not sure if he knows what they mean yet.",
            "milestone_context": {
                "behavior": "First words",
                "criteria": "Child says first meaningful words",
                "age_range": "10-14 months",
                "domain": "communication"
            }
        },
        {
            "domain": "social",
            "response": "She loves playing peek-a-boo with me and her dad. She laughs and tries to hide her face with a blanket when we play.",
            "milestone_context": {
                "behavior": "Interactive play",
                "criteria": "Child engages in back-and-forth play with caregivers",
                "age_range": "8-12 months",
                "domain": "social"
            }
        },
        {
            "domain": "cognitive",
            "response": "He can stack two blocks but they usually fall over when he tries to add a third one.",
            "milestone_context": {
                "behavior": "Block stacking",
                "criteria": "Child stacks blocks to build a tower",
                "age_range": "12-18 months",
                "domain": "cognitive"
            }
        }
    ]
    
    print("\n\n" + "=" * 70)
    print("SCORING WITH DOMAIN-SPECIFIC PROMPTS")
    print("=" * 70)
    
    # Initialize the LLM scorer with domain-specific prompts enabled
    print("\nInitializing LLM scorer with domain-specific prompts enabled...")
    scorer = LLMBasedScorer({
        "use_domain_specific_prompts": True,
        "custom_templates_dir": "config/prompt_templates"
    })
    
    # Score each test case
    for i, test_case in enumerate(test_cases):
        domain = test_case["domain"]
        response = test_case["response"]
        milestone_context = test_case["milestone_context"]
        
        print(f"\n{'-' * 50}")
        print(f"Test Case {i+1}: {domain.upper()} domain")
        print(f"{'-' * 50}")
        print(f"Milestone: {milestone_context['behavior']}")
        print(f"Criteria: {milestone_context['criteria']}")
        print(f"Age Range: {milestone_context['age_range']}")
        print(f"\nResponse: \"{response}\"")
        
        try:
            # Score the response
            result = scorer.score(response, milestone_context)
            
            # Print the result
            print(f"\nScore: {result.score.name}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Reasoning: {result.reasoning}")
        except Exception as e:
            logger.error(f"Error scoring response: {e}")
            print(f"Error: {str(e)}")

def compare_with_and_without_domain_prompts() -> None:
    """Compare scoring with and without domain-specific prompts."""
    # Test case
    test_case = {
        "domain": "cognitive",
        "response": "He can stack two blocks but they usually fall over when he tries to add a third one.",
        "milestone_context": {
            "behavior": "Block stacking",
            "criteria": "Child stacks blocks to build a tower",
            "age_range": "12-18 months",
            "domain": "cognitive"
        }
    }
    
    print("\n\n" + "=" * 70)
    print("COMPARING WITH AND WITHOUT DOMAIN-SPECIFIC PROMPTS")
    print("=" * 70)
    
    # Initialize scorers
    standard_scorer = LLMBasedScorer({
        "use_domain_specific_prompts": False
    })
    
    domain_scorer = LLMBasedScorer({
        "use_domain_specific_prompts": True,
        "custom_templates_dir": "config/prompt_templates"
    })
    
    # Get the response and context
    response = test_case["response"]
    milestone_context = test_case["milestone_context"]
    
    print(f"\nMilestone: {milestone_context['behavior']}")
    print(f"Criteria: {milestone_context['criteria']}")
    print(f"Age Range: {milestone_context['age_range']}")
    print(f"Domain: {milestone_context['domain']}")
    print(f"\nResponse: \"{response}\"")
    
    # Score with standard prompt
    print("\n1. Using Standard Prompt:")
    try:
        standard_result = standard_scorer.score(response, milestone_context)
        print(f"Score: {standard_result.score.name}")
        print(f"Confidence: {standard_result.confidence:.2f}")
        print(f"Reasoning: {standard_result.reasoning}")
    except Exception as e:
        logger.error(f"Error with standard prompt: {e}")
        print(f"Error: {str(e)}")
    
    # Score with domain-specific prompt
    print("\n2. Using Domain-Specific Prompt:")
    try:
        domain_result = domain_scorer.score(response, milestone_context)
        print(f"Score: {domain_result.score.name}")
        print(f"Confidence: {domain_result.confidence:.2f}")
        print(f"Reasoning: {domain_result.reasoning}")
    except Exception as e:
        logger.error(f"Error with domain-specific prompt: {e}")
        print(f"Error: {str(e)}")

def main():
    """Main function to demonstrate the Knowledge Engineering module."""
    print("\nKNOWLEDGE ENGINEERING MODULE DEMONSTRATION")
    print("=" * 50)
    
    # Demonstrate domain knowledge
    print("\nDEMONSTRATING DOMAIN KNOWLEDGE")
    print_domain_info("motor")
    
    # Demonstrate prompt templates
    demonstrate_prompt_templates()
    
    # Ask user if they want to run the LLM-based examples
    run_llm = input("\nDo you want to run the LLM-based examples? (y/n): ").lower() == 'y'
    
    if run_llm:
        # Score with domain-specific prompts
        score_with_domain_specific_prompts()
        
        # Compare with and without domain-specific prompts
        compare_with_and_without_domain_prompts()
    else:
        print("\nSkipping LLM-based examples.")
    
    print("\nDemonstration completed.")

if __name__ == "__main__":
    main() 