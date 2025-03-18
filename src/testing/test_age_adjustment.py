#!/usr/bin/env python3
"""
Test script for age-specific category adjustments.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Tuple

# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.core.knowledge import adjust_category_for_age, get_age_bracket, get_category_guidance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("age_adjustment_test")

def test_category_adjustment():
    """Test the adjust_category_for_age function with various inputs."""
    test_cases = [
        # Category, confidence, age_months, domain
        ("EMERGING", 0.65, 6, "motor"),
        ("EMERGING", 0.65, 6, "communication"),
        ("EMERGING", 0.65, 6, "social"),
        ("EMERGING", 0.65, 6, "cognitive"),
        ("EMERGING", 0.65, 18, "motor"),
        ("EMERGING", 0.65, 18, "communication"),
        ("EMERGING", 0.65, 18, "social"),
        ("EMERGING", 0.65, 18, "cognitive"),
        ("EMERGING", 0.65, 36, "motor"),
        ("EMERGING", 0.65, 36, "communication"),
        ("EMERGING", 0.65, 36, "social"),
        ("EMERGING", 0.65, 36, "cognitive"),
        ("WITH_SUPPORT", 0.65, 6, "motor"),
        ("WITH_SUPPORT", 0.65, 6, "communication"),
        ("WITH_SUPPORT", 0.65, 6, "social"),
        ("WITH_SUPPORT", 0.65, 6, "cognitive"),
        ("INDEPENDENT", 0.85, 6, "motor"),
        ("INDEPENDENT", 0.85, 6, "communication"),
        ("INDEPENDENT", 0.85, 6, "social"),
        ("INDEPENDENT", 0.85, 6, "cognitive"),
    ]
    
    logger.info("Testing adjust_category_for_age function")
    logger.info("=" * 80)
    
    for category, confidence, age_months, domain in test_cases:
        # Get the age bracket
        age_bracket = get_age_bracket(age_months)
        
        # Get the guidance for this category and age
        guidance = get_category_guidance(category, age_months)
        
        # Print the guidance
        logger.info(f"Category: {category}, Age: {age_months} months ({age_bracket}), Domain: {domain}")
        if guidance:
            logger.info(f"  Description: {guidance['description']}")
            if domain in guidance['domain_specific_notes']:
                logger.info(f"  Domain note: {guidance['domain_specific_notes'][domain]}")
            logger.info(f"  Confidence adjustment: {guidance['confidence_adjustment']}")
        else:
            logger.info("  No guidance found")
        
        # Apply the adjustment
        adjusted_category, adjusted_confidence = adjust_category_for_age(category, confidence, age_months, domain)
        
        # Print the results
        logger.info(f"  Original: {category} (confidence: {confidence:.2f})")
        logger.info(f"  Adjusted: {adjusted_category} (confidence: {adjusted_confidence:.2f})")
        
        # Highlight changes
        if category != adjusted_category:
            logger.info(f"  CATEGORY CHANGED: {category} -> {adjusted_category}")
        if abs(confidence - adjusted_confidence) > 0.001:
            logger.info(f"  CONFIDENCE CHANGED: {confidence:.2f} -> {adjusted_confidence:.2f}")
        
        logger.info("-" * 80)

if __name__ == "__main__":
    test_category_adjustment() 