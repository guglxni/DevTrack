#!/usr/bin/env python3
"""
Direct fix for LLM scorer to handle response parsing
"""

import sys
import os
import logging
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_patch")

# Import the necessary classes from the project
try:
    # Add the project root to the path
    sys.path.insert(0, os.getcwd())
    
    # Import the necessary classes
    from src.core.scoring.llm_scorer import LLMBasedScorer
    from src.core.scoring.models import Score, ScoringResult
    from src.core.scoring.ensemble_scorer import ImprovedScoringEngine
    import re
    
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

def patch_llm_scorer():
    """
    Directly patch the LLMBasedScorer class to improve response parsing
    """
    original_parse = LLMBasedScorer._parse_llm_response
    
    def improved_parse_llm_response(self, response_text: str) -> Tuple[Score, float, str]:
        """
        Enhanced version of the LLM response parser that handles more formats
        """
        logger.info(f"Parsing LLM response with improved parser: {response_text[:100]}...")
        
        # First try the original parsing logic
        score, confidence, reasoning = original_parse(self, response_text)
        
        # If it worked, return the result
        if score != Score.NOT_RATED:
            return score, confidence, reasoning
            
        # Otherwise, try our enhanced patterns
        # Look for text-based category labels
        patterns = [
            r"milestone status is ([A-Z_]+)",
            r"([A-Z_]+) \(\d+\)",
            r"([A-Z_]+) category",
            r"classified as ([A-Z_]+)",
            r"score of ([A-Z_]+)",
            r"child is ([A-Z_]+)",
            r"child's milestone status is ([A-Z_]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                category = match.group(1).upper()
                logger.info(f"Found category: {category}")
                
                # Map to scores
                if "INDEPENDENT" in category:
                    return Score.INDEPENDENT, 0.85, "Score extracted from category label"
                if "EMERGING" in category:
                    return Score.EMERGING, 0.85, "Score extracted from category label"
                if "WITH_SUPPORT" in category or "SUPPORT" in category:
                    return Score.WITH_SUPPORT, 0.85, "Score extracted from category label"
                if "LOST_SKILL" in category or "LOST" in category:
                    return Score.LOST_SKILL, 0.85, "Score extracted from category label"
                if "CANNOT_DO" in category or "CANNOT" in category:
                    return Score.CANNOT_DO, 0.85, "Score extracted from category label"
        
        # Try to extract numbers
        number_patterns = [
            r"score.*?(\d+)",
            r"^(\d+)$",
            r"\((\d+)\)"
        ]
        
        for pattern in number_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    num = int(match.group(1))
                    logger.info(f"Found number: {num}")
                    
                    # Map numbers to scores
                    if num in [6, 7]:
                        return Score.INDEPENDENT, 0.75, f"Numeric score {num} mapped to INDEPENDENT"
                    if num in [4, 5]:
                        return Score.EMERGING, 0.75, f"Numeric score {num} mapped to EMERGING"
                    if num == 3:
                        return Score.WITH_SUPPORT, 0.75, f"Numeric score {num} mapped to WITH_SUPPORT"
                    if num == 2:
                        return Score.LOST_SKILL, 0.75, f"Numeric score {num} mapped to LOST_SKILL"
                    if num in [0, 1]:
                        return Score.CANNOT_DO, 0.75, f"Numeric score {num} mapped to CANNOT_DO"
                except ValueError:
                    pass
        
        # Text analysis for key phrases
        text_lower = response_text.lower()
        
        if "frequently" in text_lower or "regularly" in text_lower or "often" in text_lower:
            logger.info("Found frequency indicators suggesting INDEPENDENT skill")
            return Score.INDEPENDENT, 0.8, "Frequency indicators suggest independent skill"
            
        if "sometimes" in text_lower or "occasionally" in text_lower:
            logger.info("Found frequency indicators suggesting EMERGING skill")
            return Score.EMERGING, 0.7, "Frequency indicators suggest emerging skill"
        
        # Final fallback
        logger.warning(f"All parsing attempts failed for: {response_text[:100]}")
        return Score.NOT_RATED, 0.5, "Could not extract score from response"
    
    # Replace the method
    LLMBasedScorer._parse_llm_response = improved_parse_llm_response
    logger.info("Successfully patched LLMBasedScorer._parse_llm_response method")
    return True

def patch_ensemble_scorer():
    """
    Modify ImprovedScoringEngine to prioritize LLM results
    """
    original_combine = ImprovedScoringEngine._combine_results
    
    def improved_combine_results(self, results: List[ScoringResult]) -> ScoringResult:
        """
        Enhanced version that gives priority to high-confidence LLM results
        """
        # Check for high-confidence LLM result first
        for result in results:
            if result.method == "llm_based" and result.score != Score.NOT_RATED and result.confidence > 0.7:
                logger.info(f"Using high-confidence LLM result: {result.score.name}, confidence: {result.confidence}")
                return result
        
        # Fall back to original implementation
        return original_combine(self, results)
    
    # Replace the method
    ImprovedScoringEngine._combine_results = improved_combine_results
    logger.info("Successfully patched ImprovedScoringEngine._combine_results method")
    return True

if __name__ == "__main__":
    success1 = patch_llm_scorer()
    success2 = patch_ensemble_scorer()
    
    if success1 and success2:
        logger.info("All patches successfully applied!")
        sys.exit(0)
    else:
        logger.error("Failed to apply all patches")
        sys.exit(1) 