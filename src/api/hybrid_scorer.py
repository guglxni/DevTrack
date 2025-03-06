#!/usr/bin/env python3
"""
Hybrid Scoring Module for ASD Assessment API

This module implements a hybrid approach to scoring developmental milestone responses,
combining word boundary-aware keyword matching, negation detection, and advanced
NLP techniques for more accurate and reliable scoring.
"""

import sys
import logging
import re
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("hybrid-scorer")

# Try to import our new improved advanced_nlp_hybrid module
try:
    from src.core.advanced_nlp_hybrid import score_response as advanced_score_response
    ADVANCED_HYBRID_AVAILABLE = True
    logger.info("Advanced NLP Hybrid module available - using enhanced scoring")
except ImportError as e:
    ADVANCED_HYBRID_AVAILABLE = False
    logger.warning(f"Advanced NLP Hybrid module not available: {str(e)}")

# Try to import the original advanced NLP module as fallback
try:
    from src.core.advanced_nlp import HybridScorer
    HYBRID_SCORER_AVAILABLE = True and not ADVANCED_HYBRID_AVAILABLE  # Only use if advanced_nlp_hybrid is not available
    logger.info("Original HybridScorer available as fallback")
    # Initialize with basic features only for performance
    hybrid_scorer = HybridScorer(
        use_transformer=False,  # Disabled by default for performance
        use_spacy=False,        # Disabled by default for performance
        use_sentence_embeddings=False  # Disabled by default for performance
    )
except ImportError as e:
    HYBRID_SCORER_AVAILABLE = False
    logger.warning(f"Original HybridScorer not available: {str(e)}")


class ReliableScorer:
    """
    Implementation of the hybrid scoring approach for reliable scoring of
    developmental milestone responses.
    """
    def __init__(self):
        """Initialize the reliable scorer with default settings."""
        self.enable_advanced_hybrid = ADVANCED_HYBRID_AVAILABLE
        self.enable_original_hybrid = HYBRID_SCORER_AVAILABLE
        
        # Word boundary-aware regex patterns for each score category
        self.score_regexes = {
            "CANNOT_DO": re.compile(r'\b(no|not|never|doesn\'t|does not|cannot|can\'t|unable|hasn\'t|has not|not able|not at all|not yet started|not capable)\b'),
            "LOST_SKILL": re.compile(r'\b(used to|previously|before|no longer|stopped|regressed|lost ability|could before|forgotten how|used to \w+ but now|but now he doesn\'t|but now she doesn\'t)\b'),
            "EMERGING": re.compile(r'\b(sometimes|occasionally|beginning to|starting to|trying to|inconsistently|might|rarely|not consistent|learning to)\b'),
            "WITH_SUPPORT": re.compile(r'\b(with help|when assisted|with support|with guidance|needs help|when prompted|specific situations|only when|if guided|with assistance|when we|when encouraged|encourage)\b'),
            "INDEPENDENT": re.compile(r'\b(yes|always|consistently|definitely|independently|without help|on own|mastered|very good at|excellent|regularly|all situations|recognizes|distinguishes between|knows all|knows family|family members|smiles when)\b')
        }
        
        # Milestone-specific keyword patterns to improve accuracy
        self.milestone_keywords = {
            "Recognizes familiar people": {
                "INDEPENDENT": re.compile(r'\b(knows|recognizes|identifies|distinguishes|smiles when|familiar faces|family members)\b'),
                "CANNOT_DO": re.compile(r'\b(doesn\'t recognize|does not recognize|unable to recognize|never recognizes|can\'t recognize|fails to recognize)\b')
            },
            "Makes eye contact": {
                "INDEPENDENT": re.compile(r'\b(makes eye contact|maintains eye contact|good eye contact|looks in the eyes|eye gaze|direct gaze)\b'),
                "CANNOT_DO": re.compile(r'\b(no eye contact|doesn\'t make eye contact|avoids eye contact|poor eye contact|won\'t look)\b')
            }
        }
    
    def score_response(self, milestone: str, response: str, keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Score a response for a given milestone using the hybrid approach.
        
        Args:
            milestone: The milestone behavior text
            response: The response text to analyze
            keywords: Optional dictionary of keywords by category
            
        Returns:
            Dictionary with scoring results, including score, score_label, and confidence
        """
        # Try to use the advanced hybrid module first (best option)
        if self.enable_advanced_hybrid:
            logger.info(f"Using advanced_nlp_hybrid for milestone: {milestone}")
            return advanced_score_response(milestone, response, keywords)
        
        # Then try the original hybrid scorer if available
        elif self.enable_original_hybrid:
            logger.info(f"Using original HybridScorer for milestone: {milestone}")
            return hybrid_scorer.score(response, milestone, keywords)
        
        # Otherwise use the simple fallback approach
        logger.info(f"Using fallback scorer for milestone: {milestone}")
        return self._score_with_word_boundaries(response, milestone, keywords)
    
    def _score_with_word_boundaries(self, response_text: str, milestone: Optional[str] = None, 
                                   keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Score response text using regex patterns with proper word boundaries.
        
        Args:
            response_text: The text response to analyze
            milestone: Optional milestone for context
            keywords: Optional custom keywords for each score category
        
        Returns:
            Dict with score category and count of matched patterns
        """
        response_lower = response_text.lower()
        logger.info(f"Analyzing with word boundaries: '{response_lower}'")
        
        # Initialize counters for each score category
        score_counts = {"CANNOT_DO": 0, "LOST_SKILL": 0, "EMERGING": 0, 
                        "WITH_SUPPORT": 0, "INDEPENDENT": 0}
        
        # Track matched keywords for debugging
        matched_keywords = {category: [] for category in score_counts.keys()}
        
        # Use custom keywords if provided
        if keywords:
            for category, keyword_list in keywords.items():
                if category in score_counts:
                    # Create a regex pattern with word boundaries for each keyword
                    for keyword in keyword_list:
                        keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = re.findall(keyword_pattern, response_lower)
                        if matches:
                            score_counts[category] += len(matches)
                            matched_keywords[category].extend(matches)
        else:
            # Use milestone-specific keywords if available
            if milestone in self.milestone_keywords:
                for category, regex in self.milestone_keywords[milestone].items():
                    matches = regex.findall(response_lower)
                    if matches:
                        score_counts[category] += len(matches)
                        matched_keywords[category].extend(matches)
            
            # Also use the default regexes
            for category, regex in self.score_regexes.items():
                matches = regex.findall(response_lower)
                if matches:
                    score_counts[category] += len(matches)
                    matched_keywords[category].extend(matches)
        
        # Special handling for "used to" phrases to ensure LOST_SKILL overrides CANNOT_DO
        # when appropriate
        if "used to" in response_lower and "doesn't" in response_lower or "does not" in response_lower:
            score_counts["LOST_SKILL"] += 1
            score_counts["CANNOT_DO"] -= 1
            if score_counts["CANNOT_DO"] < 0:
                score_counts["CANNOT_DO"] = 0
        
        # Special handling for family-recognition milestones
        if milestone and "recognizes familiar people" in milestone.lower():
            if "knows all" in response_lower or "knows family members" in response_lower or "family members" in response_lower:
                score_counts["INDEPENDENT"] += 2
        
        # Log the matches for debugging
        for category, matched in matched_keywords.items():
            if matched:
                logger.info(f"Matched {category} keywords: {', '.join(matched)}")
        
        # Find the category with the highest score
        if all(count == 0 for count in score_counts.values()):
            best_category = None
            confidence = 0.0
        else:
            best_category = max(score_counts.items(), key=lambda x: x[1])[0]
            total_matches = sum(score_counts.values())
            confidence = score_counts[best_category] / total_matches if total_matches > 0 else 0.0
        
        # Convert category name to score value
        score_values = {"CANNOT_DO": 0, "LOST_SKILL": 1, "EMERGING": 2, 
                        "WITH_SUPPORT": 3, "INDEPENDENT": 4, "NOT_RATED": -1}
        
        score = score_values.get(best_category, -1) if best_category else -1
        score_label = best_category or "NOT_RATED"
        
        return {
            "score": score,
            "score_label": score_label,
            "confidence": confidence,
            "detail": {"matches": matched_keywords}
        }


# Singleton instance for reuse
reliable_scorer = ReliableScorer()

def score_response(milestone: str, response: str, keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Public function to score a response for a given milestone.
    
    Args:
        milestone: The milestone behavior text
        response: The response text to analyze
        keywords: Optional dictionary of keywords by category
        
    Returns:
        Dictionary with scoring results, including score, score_label, and confidence
    """
    return reliable_scorer.score_response(milestone, response, keywords)


if __name__ == "__main__":
    # Simple test if run directly
    test_milestone = "Recognizes familiar people"
    test_response = "My child always smiles when he sees grandparents and recognizes all family members."
    result = score_response(test_milestone, test_response)
    print(f"Score: {result['score_label']} ({result['score']})")
    print(f"Confidence: {result['confidence']:.2f}")
    
    # Test with negative response
    negative_response = "No, he doesn't recognize anyone, not even his parents."
    result = score_response(test_milestone, negative_response)
    print(f"Negative Score: {result['score_label']} ({result['score']})")
    print(f"Confidence: {result['confidence']:.2f}") 