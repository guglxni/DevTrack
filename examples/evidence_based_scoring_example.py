#!/usr/bin/env python3
"""
Evidence-Based Scoring Example

This script demonstrates how to use the evidence-based category knowledge
in scoring decisions, showing how research-backed criteria enhance the
accuracy and consistency of developmental assessments.
"""

import sys
import os
import logging
from typing import Dict, Any, List
import json
from pathlib import Path
from colorama import init, Fore, Style

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.scoring.base import Score, ScoringResult
from src.core.knowledge.category_knowledge import (
    get_category_evidence,
    get_category_boundary,
    get_all_categories
)
from src.core.knowledge.category_helper import (
    get_research_based_definition,
    get_domain_indicators,
    analyze_response_for_category,
    determine_category_from_response,
    refine_category_with_research,
    get_citation_for_category
)
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine

# Initialize colorama
init()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evidence_example")


def print_category_definitions():
    """Print the research-based definitions for all categories"""
    categories = get_all_categories()
    
    print(f"\n{Fore.CYAN}===== RESEARCH-BASED CATEGORY DEFINITIONS ====={Style.RESET_ALL}\n")
    
    for category in categories:
        print(f"{Fore.YELLOW}Category: {category.name}{Style.RESET_ALL}")
        print(f"Definition: {category.description}")
        print(f"\nResearch-Based Indicators:")
        for i, indicator in enumerate(category.research_based_indicators, 1):
            print(f"  {i}. {indicator}")
        
        print(f"\nFramework Mappings:")
        for framework, mapping in category.framework_mappings.items():
            print(f"  {framework}: {mapping}")
        
        print(f"\nThreshold Indicators:")
        for key, value in category.threshold_indicators.items():
            print(f"  {key}: {value}")
        
        print(f"\nCitations:")
        for i, citation in enumerate(category.citations, 1):
            print(f"  {i}. {citation}")
        
        print("\n" + "-" * 80 + "\n")


def print_domain_specific_indicators():
    """Print domain-specific indicators for each category"""
    domains = ["motor", "communication", "social", "cognitive"]
    
    print(f"\n{Fore.CYAN}===== DOMAIN-SPECIFIC INDICATORS ====={Style.RESET_ALL}\n")
    
    for domain in domains:
        print(f"{Fore.GREEN}Domain: {domain.upper()}{Style.RESET_ALL}\n")
        
        for score in [Score.CANNOT_DO, Score.WITH_SUPPORT, Score.EMERGING, Score.INDEPENDENT, Score.LOST_SKILL]:
            indicators = get_domain_indicators(score, domain)
            
            print(f"{Fore.YELLOW}Category: {score.name}{Style.RESET_ALL}")
            if indicators:
                for i, indicator in enumerate(indicators, 1):
                    print(f"  {i}. {indicator}")
            else:
                print("  No domain-specific indicators defined.")
            print()
        
        print("-" * 80 + "\n")


def demonstrate_response_analysis():
    """Demonstrate analyzing responses for category indicators"""
    test_cases = [
        {
            "domain": "motor",
            "response": "My child is starting to crawl but needs help. She can push up on her hands but struggles to move forward on her own. I usually need to position her and guide her movements.",
            "expected_category": "WITH_SUPPORT"
        },
        {
            "domain": "communication",
            "response": "He has started saying a few words consistently like 'mama' and 'dada' in the right context. He can also point to things he wants and uses different sounds to communicate different needs.",
            "expected_category": "INDEPENDENT"
        },
        {
            "domain": "social",
            "response": "Sometimes she plays with other kids, but it's hit or miss. Some days she'll interact and share toys, other days she prefers to play alone. It really depends on her mood and who the other children are.",
            "expected_category": "EMERGING"
        },
        {
            "domain": "cognitive",
            "response": "He used to be able to complete simple puzzles on his own, but in the last few months he's stopped doing them altogether. When I try to engage him with puzzles now, he gets frustrated and walks away.",
            "expected_category": "LOST_SKILL"
        }
    ]
    
    print(f"\n{Fore.CYAN}===== RESPONSE ANALYSIS WITH RESEARCH-BASED INDICATORS ====={Style.RESET_ALL}\n")
    
    for i, case in enumerate(test_cases, 1):
        domain = case["domain"]
        response = case["response"]
        expected = case["expected_category"]
        
        print(f"{Fore.GREEN}Test Case {i}: {domain.upper()} Domain{Style.RESET_ALL}")
        print(f"Response: \"{response}\"")
        print(f"Expected Category: {expected}")
        print()
        
        # Analyze for the expected category
        expected_score = Score[expected]
        analysis = analyze_response_for_category(response, expected_score, domain)
        
        print(f"{Fore.YELLOW}Analysis for {expected}:{Style.RESET_ALL}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        print(f"Analysis: {analysis['analysis']}")
        
        if analysis["indicators_found"]:
            print(f"Indicators Found:")
            for ind in analysis["indicators_found"]:
                print(f"  • {ind}")
        
        # Determine category from response
        determined_score, confidence, reasoning = determine_category_from_response(response, domain)
        
        print(f"\n{Fore.YELLOW}Automated Category Determination:{Style.RESET_ALL}")
        print(f"Determined Category: {determined_score.name}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reasoning: {reasoning}")
        
        # Refine with research
        refined_score, refined_conf, refined_reason = refine_category_with_research(
            determined_score, confidence, domain
        )
        
        print(f"\n{Fore.YELLOW}Research-Refined Determination:{Style.RESET_ALL}")
        print(f"Refined Category: {refined_score.name}")
        print(f"Refined Confidence: {refined_conf:.2f}")
        print(f"Research-Based Reasoning: {refined_reason}")
        
        # Get research citations
        citations = get_citation_for_category(determined_score)
        if citations:
            print(f"\n{Fore.YELLOW}Supporting Research:{Style.RESET_ALL}")
            for i, citation in enumerate(citations[:2], 1):  # Limit to 2 citations for brevity
                print(f"  {i}. {citation}")
        
        print("\n" + "-" * 80 + "\n")


def compare_with_standard_scoring():
    """Compare evidence-based scoring with standard scoring engine"""
    # Initialize the standard scoring engine
    standard_engine = ImprovedDevelopmentalScoringEngine()
    
    test_cases = [
        {
            "domain": "motor",
            "behavior": "Crawling",
            "criteria": "Child moves on hands and knees across the floor",
            "age_range": "6-10 months",
            "response": "My child is starting to crawl but needs help. She can push up on her hands but struggles to move forward on her own. I usually need to position her and guide her movements."
        },
        {
            "domain": "communication",
            "behavior": "First words",
            "criteria": "Child says first meaningful words",
            "age_range": "10-14 months",
            "response": "He has started saying a few words consistently like 'mama' and 'dada' in the right context. He can also point to things he wants and uses different sounds to communicate different needs."
        },
        {
            "domain": "social",
            "behavior": "Peer interaction",
            "criteria": "Child plays cooperatively with peers",
            "age_range": "24-36 months",
            "response": "Sometimes she plays with other kids, but it's hit or miss. Some days she'll interact and share toys, other days she prefers to play alone. It really depends on her mood and who the other children are."
        }
    ]
    
    print(f"\n{Fore.CYAN}===== COMPARISON: STANDARD VS. EVIDENCE-BASED SCORING ====={Style.RESET_ALL}\n")
    
    for i, case in enumerate(test_cases, 1):
        domain = case["domain"]
        response = case["response"]
        milestone_context = {
            "domain": domain,
            "behavior": case["behavior"],
            "criteria": case["criteria"],
            "age_range": case["age_range"]
        }
        
        print(f"{Fore.GREEN}Test Case {i}: {domain.upper()} Domain{Style.RESET_ALL}")
        print(f"Milestone: {case['behavior']}")
        print(f"Criteria: {case['criteria']}")
        print(f"Age Range: {case['age_range']}")
        print(f"Response: \"{response}\"")
        print()
        
        # Standard scoring
        standard_result = standard_engine.score_response(response, milestone_context, detailed=True)
        
        print(f"{Fore.YELLOW}Standard Scoring Result:{Style.RESET_ALL}")
        print(f"Category: {standard_result['score_name']}")
        print(f"Confidence: {standard_result['confidence']:.2f}")
        if 'reasoning' in standard_result and standard_result['reasoning']:
            print(f"Reasoning: {standard_result['reasoning']}")
        
        # Evidence-based scoring
        determined_score, confidence, reasoning = determine_category_from_response(response, domain)
        refined_score, refined_conf, refined_reason = refine_category_with_research(
            determined_score, confidence, domain
        )
        
        print(f"\n{Fore.YELLOW}Evidence-Based Scoring Result:{Style.RESET_ALL}")
        print(f"Category: {refined_score.name}")
        print(f"Confidence: {refined_conf:.2f}")
        print(f"Reasoning: {refined_reason}")
        
        # Comparison
        print(f"\n{Fore.YELLOW}Comparison:{Style.RESET_ALL}")
        if standard_result['score_name'] == refined_score.name:
            print(f"{Fore.GREEN}✓ Categories match{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Category difference: {standard_result['score_name']} vs. {refined_score.name}{Style.RESET_ALL}")
        
        conf_diff = abs(standard_result['confidence'] - refined_conf)
        if conf_diff < 0.1:
            print(f"{Fore.GREEN}✓ Confidence levels similar (diff: {conf_diff:.2f}){Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}! Confidence difference: {conf_diff:.2f}{Style.RESET_ALL}")
        
        print("\n" + "-" * 80 + "\n")


def main():
    """Main function to run the examples"""
    print(f"\n{Fore.MAGENTA}EVIDENCE-BASED DEVELOPMENTAL SCORING EXAMPLES{Style.RESET_ALL}")
    print(f"This script demonstrates using research-backed knowledge for developmental assessments.\n")
    
    # Run the examples
    print_category_definitions()
    print_domain_specific_indicators()
    demonstrate_response_analysis()
    
    # Ask if the user wants to run the comparison (which requires the scoring engine)
    run_comparison = input("Run comparison with standard scoring engine? (y/n): ").lower().strip() == 'y'
    if run_comparison:
        compare_with_standard_scoring()
    
    print(f"\n{Fore.MAGENTA}Example completed successfully!{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main() 