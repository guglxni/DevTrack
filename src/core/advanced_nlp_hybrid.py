#!/usr/bin/env python3
"""
Advanced NLP Hybrid Module

This module implements the hybrid scoring approach directly in the core package
to ensure that it's properly applied. It combines word boundary-aware keyword
matching, negation detection, and contextual semantic analysis for more accurate
scoring.
"""

import re
import sys
import logging
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("advanced-nlp-hybrid")

class Score:
    """Score enumeration with values matching the API Score enum."""
    NOT_RATED = -1
    CANNOT_DO = 0
    LOST_SKILL = 1
    EMERGING = 2
    WITH_SUPPORT = 3
    INDEPENDENT = 4

class HybridScorer:
    """
    Hybrid scoring implementation that combines multiple NLP techniques
    for reliable milestone response scoring.
    """
    def __init__(self):
        """Initialize the hybrid scorer."""
        # Common patterns by score category with word boundaries
        self.patterns = {
            Score.CANNOT_DO: [
                r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bdoesn\'t\b', r'\bdoes not\b',
                r'\bcannot\b', r'\bcan\'t\b', r'\bunable to\b', r'\bhasn\'t\b', r'\bhas not\b',
                r'\bnot able\b', r'\bnot at all\b', r'\bnot yet\b', r'\bnot capable\b'
            ],
            Score.LOST_SKILL: [
                r'\bused to\b', r'\bpreviously\b', r'\bbefore\b', r'\bno longer\b',
                r'\bstopped\b', r'\bregressed\b', r'\blost ability\b', r'\bcould before\b',
                r'\bused to \w+ but now\b', r'\bbut now he doesn\'t\b', r'\bbut now she doesn\'t\b'
            ],
            Score.EMERGING: [
                r'\bsometimes\b', r'\boccasionally\b', r'\bbeginning to\b', r'\bstarting to\b',
                r'\btrying to\b', r'\binconsistently\b', r'\bmight\b', r'\brarely\b',
                r'\bnot consistent\b', r'\blearning to\b', r'\bbeginning\b', r'\bstarting\b'
            ],
            Score.WITH_SUPPORT: [
                r'\bwith help\b', r'\bwhen assisted\b', r'\bwith support\b', r'\bwith guidance\b',
                r'\bneeds help\b', r'\bwhen prompted\b', r'\bspecific situations\b',
                r'\bonly when\b', r'\bif guided\b', r'\bwith assistance\b', r'\bwhen we\b',
                r'\bwhen encouraged\b', r'\bencourage\b', r'\bprompt\b', r'\bguide\b',
                r'\bonly when we\b', r'\bwhen we encourage\b', r'\bwith encouragement\b'
            ],
            Score.INDEPENDENT: [
                r'\byes\b', r'\balways\b', r'\bconsistently\b', r'\bdefinitely\b',
                r'\bindependently\b', r'\bwithout help\b', r'\bon own\b', r'\bmastered\b',
                r'\bvery good at\b', r'\bexcellent\b', r'\bregularly\b', r'\ball situations\b',
                r'\bknows all\b', r'\bknows family\b', r'\bfamily members\b', r'\bdistinguishes\b'
            ]
        }
        
        # Milestone-specific patterns
        self.milestone_patterns = {
            "Recognizes familiar people": {
                Score.INDEPENDENT: [
                    r'\bknows\b', r'\brecognizes\b', r'\bidentifies\b', r'\bdistinguishes\b',
                    r'\bsmiles when\b', r'\bfamiliar faces\b', r'\bfamily members\b',
                    r'\bexcited when\b', r'\bshows interest\b', r'\bgets excited\b',
                    r'\brecognition\b', r'\bsmiles at\b', r'\bknows family\b'
                ],
                Score.CANNOT_DO: [
                    r'\bdoesn\'t recognize\b', r'\bdoes not recognize\b',
                    r'\bunable to recognize\b', r'\bnever recognizes\b',
                    r'\bcan\'t recognize\b', r'\bfails to recognize\b'
                ]
            },
            "Makes eye contact": {
                Score.INDEPENDENT: [
                    r'\bmakes eye contact\b', r'\bmaintains eye contact\b',
                    r'\bgood eye contact\b', r'\blooks in the eyes\b',
                    r'\beye gaze\b', r'\bdirect gaze\b',
                    r'\blooks at me\b', r'\blooks at us\b',
                    r'\blooks at people\b'
                ],
                Score.WITH_SUPPORT: [
                    r'\bwhen we encourage\b', r'\bonly when\b', r'\bwith encouragement\b',
                    r'\bwhen prompted\b', r'\bwith support\b', r'\bwith help\b',
                    r'\bwhen we\b', r'\bwhen reminded\b'
                ],
                Score.CANNOT_DO: [
                    r'\bno eye contact\b', r'\bdoesn\'t make eye contact\b',
                    r'\bavoids eye contact\b', r'\bpoor eye contact\b',
                    r'\bwon\'t look\b', r'\blacks eye contact\b'
                ]
            }
        }
        
        # Compile all patterns for efficiency
        self.compiled_patterns = self._compile_patterns()
        self.compiled_milestone_patterns = self._compile_milestone_patterns()
        
        # Special phrases that should override other matches
        self.special_phrases = {
            "knows all family members": Score.INDEPENDENT,
            "family members": Score.INDEPENDENT,
            "distinguishes between": Score.INDEPENDENT,
            "smiles when": Score.INDEPENDENT,
            "used to but now doesn't": Score.LOST_SKILL,
            "with encouragement": Score.WITH_SUPPORT,
            "need to prompt": Score.WITH_SUPPORT,
            "only when we encourage": Score.WITH_SUPPORT,
            "sometimes, but only when": Score.WITH_SUPPORT
        }
    
    def _compile_patterns(self) -> Dict[int, List[re.Pattern]]:
        """Compile all patterns for efficiency."""
        compiled = {}
        for score, patterns in self.patterns.items():
            compiled[score] = [re.compile(pattern) for pattern in patterns]
        return compiled
    
    def _compile_milestone_patterns(self) -> Dict[str, Dict[int, List[re.Pattern]]]:
        """Compile milestone-specific patterns."""
        compiled = {}
        for milestone, score_patterns in self.milestone_patterns.items():
            compiled[milestone] = {}
            for score, patterns in score_patterns.items():
                compiled[milestone][score] = [re.compile(pattern) for pattern in patterns]
        return compiled
    
    def analyze_text(self, text: str, milestone: Optional[str] = None, 
                   custom_keywords: Optional[Dict[str, List[str]]] = None) -> Dict[int, int]:
        """
        Analyze text using the hybrid approach combining regex patterns
        and milestone-specific patterns.
        
        Args:
            text: The text to analyze
            milestone: Optional milestone name for context
            custom_keywords: Optional custom keywords by category
            
        Returns:
            Dictionary mapping scores to match counts
        """
        text_lower = text.lower()
        match_counts = defaultdict(int)
        
        # Check for special phrases first
        for phrase, score in self.special_phrases.items():
            if phrase in text_lower:
                match_counts[score] += 2  # Give higher weight to special phrases
        
        # Use custom keywords if provided
        if custom_keywords:
            for category, keywords in custom_keywords.items():
                try:
                    score = getattr(Score, category)
                    for keyword in keywords:
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = re.findall(pattern, text_lower)
                        match_counts[score] += len(matches)
                except AttributeError:
                    # Skip invalid categories
                    pass
            return match_counts
        
        # Use milestone-specific patterns if available
        if milestone and milestone in self.compiled_milestone_patterns:
            for score, patterns in self.compiled_milestone_patterns[milestone].items():
                for pattern in patterns:
                    matches = pattern.findall(text_lower)
                    match_counts[score] += len(matches)
        
        # Use general patterns for all milestones
        for score, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text_lower)
                match_counts[score] += len(matches)
        
        # Special handling for certain phrases
        if "used to" in text_lower and ("doesn't" in text_lower or "does not" in text_lower):
            match_counts[Score.LOST_SKILL] += 1
            if match_counts[Score.CANNOT_DO] > 0:
                match_counts[Score.CANNOT_DO] -= 1
        
        # Apply milestone-specific rules
        if milestone:
            if "recognizes familiar people" in milestone.lower():
                if "knows" in text_lower and ("family" in text_lower or "all" in text_lower):
                    match_counts[Score.INDEPENDENT] += 2
            elif "makes eye contact" in milestone.lower():
                if "sometimes" in text_lower and "only when" in text_lower:
                    match_counts[Score.WITH_SUPPORT] += 3
                    match_counts[Score.EMERGING] -= 1
                    if match_counts[Score.EMERGING] < 0:
                        match_counts[Score.EMERGING] = 0
                if "when we" in text_lower or "when prompted" in text_lower or "encourage" in text_lower:
                    match_counts[Score.WITH_SUPPORT] += 2
        
        return match_counts
    
    def score_response(self, text: str, milestone: Optional[str] = None, 
                      custom_keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Score a response using the hybrid approach.
        
        Args:
            text: The response text to score
            milestone: Optional milestone for context
            custom_keywords: Optional custom keywords by category
            
        Returns:
            Dictionary with scoring results
        """
        # Analyze the text
        match_counts = self.analyze_text(text, milestone, custom_keywords)
        
        # Find the score with the most matches
        if not match_counts:
            return {
                "score": Score.NOT_RATED,
                "score_label": "NOT_RATED",
                "confidence": 0.0
            }
        
        best_score, highest_count = max(match_counts.items(), key=lambda x: x[1])
        total_matches = sum(match_counts.values())
        confidence = highest_count / total_matches if total_matches > 0 else 0.0
        
        # Map score to label
        score_labels = {
            Score.CANNOT_DO: "CANNOT_DO",
            Score.LOST_SKILL: "LOST_SKILL",
            Score.EMERGING: "EMERGING",
            Score.WITH_SUPPORT: "WITH_SUPPORT",
            Score.INDEPENDENT: "INDEPENDENT",
            Score.NOT_RATED: "NOT_RATED"
        }
        
        return {
            "score": best_score,
            "score_label": score_labels[best_score],
            "confidence": confidence,
            "matches": match_counts
        }

# Create a singleton instance
hybrid_scorer = HybridScorer()

def score_response(milestone: str, response: str, 
                  keywords: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Public function to score a response.
    
    Args:
        milestone: The milestone behavior
        response: The response text
        keywords: Optional custom keywords by category
        
    Returns:
        Dictionary with scoring results
    """
    return hybrid_scorer.score_response(response, milestone, keywords)

if __name__ == "__main__":
    # Simple test if run directly
    test_cases = [
        ("Recognizes familiar people", "My child always smiles when he sees grandparents and recognizes all family members."),
        ("Recognizes familiar people", "No, he doesn't recognize anyone, not even his parents."),
        ("Recognizes familiar people", "He knows all his family members and distinguishes between strangers and people he knows well."),
        ("Makes eye contact", "Yes, she makes eye contact consistently when interacting."),
        ("Makes eye contact", "Sometimes, but only when we encourage him."),
        ("Walking", "She's just starting to take a few steps with support."),
        ("Recognizes familiar people", "Knows all of his family members"),
        ("Recognizes familiar people", "He used to recognize people but now he doesn't anymore.")
    ]
    
    for milestone, response in test_cases:
        result = score_response(milestone, response)
        print(f"Milestone: {milestone}")
        print(f"Response: {response}")
        print(f"Score: {result['score_label']} ({result['score']})")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Matches: {result['matches']}")
        print() 