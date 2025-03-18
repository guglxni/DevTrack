#!/usr/bin/env python3
"""
Test script for domain-specific prompts in the LLM scorer.

This script demonstrates the use of domain-specific prompts for different
developmental domains in the LLM-based scorer.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append('.')

from src.core.scoring.llm_scorer import LLMBasedScorer
from src.core.scoring.base import Score

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("domain_prompt_test")

def test_domain_specific_prompts():
    """Test domain-specific prompts for different domains"""
    
    # Initialize the LLM scorer with domain-specific prompts enabled
    config = {
        "use_domain_specific_prompts": True,
        "custom_templates_dir": "config/prompt_templates"
    }
    
    scorer = LLMBasedScorer(config)
    
    # Test data for different domains
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
    
    # Test each domain
    for i, test_case in enumerate(test_cases):
        domain = test_case["domain"]
        response = test_case["response"]
        milestone_context = test_case["milestone_context"]
        
        logger.info(f"\n\n===== Testing {domain.upper()} domain =====")
        
        # Format the prompt
        prompt = scorer._format_prompt(response, milestone_context)
        
        # Print the formatted prompt
        logger.info(f"Formatted prompt for {domain}:\n{prompt}\n")
        
        # Score the response (if model is available)
        try:
            result = scorer.score(response, milestone_context)
            logger.info(f"Score: {result.score.name}")
            logger.info(f"Confidence: {result.confidence:.2f}")
            logger.info(f"Reasoning: {result.reasoning}")
        except Exception as e:
            logger.error(f"Error scoring response: {e}")
            logger.info("Skipping scoring due to error")
        
        logger.info(f"===== End of {domain.upper()} test =====\n")

if __name__ == "__main__":
    test_domain_specific_prompts() 