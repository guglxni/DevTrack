"""
Basic Assessment Engine for Developmental Milestones
"""

import os
import re
import sys
import importlib.util
from enum import Enum

# Common stopwords to filter out from milestone keywords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'when', 'then', 'with', 'without', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'to', 'of', 'for', 'in', 
    'on', 'at', 'by', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 
    'them', 'their', 'has', 'have', 'had', 'do', 'does', 'did', 'can', 'could', 
    'will', 'would', 'should', 'may', 'might', 'must', 'about', 'also', 'very',
    'just', 'only', 'not', 'no', 'yes', 'more', 'most', 'some', 'any', 'all',
    'from', 'as', 'so', 'than', 'too', 'how', 'what', 'who', 'whom', 'where',
    'when', 'why', 'which'
}

# Define Score enum
class Score(Enum):
    NOT_RATED = -1
    CANNOT_DO = 0      # Skill not acquired
    LOST_SKILL = 1     # Acquired but lost
    EMERGING = 2       # Emerging and inconsistent
    WITH_SUPPORT = 3   # Acquired but consistent in specific situations only
    INDEPENDENT = 4    # Acquired and present in all situations

# Advanced NLP integration
def _load_advanced_nlp():
    """Load the advanced NLP module if available."""
    try:
        # Try to find the advanced_nlp.py file
        nlp_paths = [
            os.path.join(os.path.dirname(__file__), "advanced_nlp.py"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "advanced_nlp.py"),
            "advanced_nlp.py"
        ]
        
        for path in nlp_paths:
            if os.path.isfile(path):
                spec = importlib.util.spec_from_file_location("advanced_nlp", path)
                advanced_nlp = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(advanced_nlp)
                # Set up optimizations for Apple Silicon
                advanced_nlp.setup_optimizations()
                return advanced_nlp
        return None
    except Exception as e:
        print(f"Warning: Could not load advanced NLP module: {str(e)}")
        return None

# Try to load advanced NLP module
_advanced_nlp = _load_advanced_nlp()

# Global variable for the NLP analyzer
nlp_analyzer = None

class AssessmentEngine:
    """
    Basic assessment engine for developmental milestones.
    This is a simplified version that serves as a base for the enhanced engine.
    """
    
    def __init__(self):
        """Initialize the basic assessment engine"""
        self.milestones = []
        self.scores = {}
        
        # Try to import and initialize the advanced NLP analyzer
        try:
            from advanced_nlp import AdvancedResponseAnalyzer
            self.nlp_analyzer = AdvancedResponseAnalyzer()
            print("Successfully initialized advanced NLP analyzer")
        except ImportError:
            self.nlp_analyzer = None
            print("Advanced NLP module not available")
    
    def score_response(self, milestone_behavior, response_text):
        """
        Score a response for a specific milestone behavior
        
        Args:
            milestone_behavior: The behavior to score
            response_text: The caregiver's response text
            
        Returns:
            Score: The score for the response
        """
        # Try to use advanced NLP if available
        if self.nlp_analyzer:
            try:
                analysis = self.nlp_analyzer.analyze_response(milestone_behavior, response_text)
                if analysis and 'score' in analysis:
                    score_value = analysis['score']
                    score_mapping = {
                        0: Score.CANNOT_DO,
                        1: Score.LOST_SKILL,
                        2: Score.EMERGING,
                        3: Score.WITH_SUPPORT,
                        4: Score.INDEPENDENT,
                        -1: Score.NOT_RATED
                    }
                    return score_mapping.get(score_value, Score.NOT_RATED)
            except Exception as e:
                print(f"Error using advanced NLP: {e}")
                # Fall back to basic scoring
        
        # Basic scoring logic
        response_lower = response_text.lower()
        
        # CANNOT_DO (0) patterns
        if re.search(r"\b(no|not),?\s+(yet|yet started|started yet)", response_lower) or \
           re.search(r"not at all", response_lower) or \
           re.search(r"\bnever\b", response_lower) or \
           re.search(r"(doesn't|does not|can't|cannot|hasn't|has not)\s+([a-z]+\s){0,3}(do|show|perform|demonstrate)", response_lower) or \
           response_lower.strip() == "no":
            print("Enhanced NLP detected clear negation - scoring as CANNOT_DO")
            return Score.CANNOT_DO
            
        # LOST_SKILL (1) patterns
        if re.search(r"used to", response_lower) or \
           re.search(r"(was able to|could before|previously|before but)", response_lower) or \
           re.search(r"(lost|regressed|stopped|no longer|not anymore)", response_lower):
            print("Enhanced NLP detected regression pattern - scoring as LOST_SKILL")
            return Score.LOST_SKILL
            
        # EMERGING (2) patterns
        if re.search(r"\b(sometimes|occasionally)\b", response_lower) or \
           re.search(r"not (always|consistently)", response_lower) or \
           re.search(r"(trying|beginning|starting|learning) to", response_lower) or \
           re.search(r"(inconsistent|developing|in progress)", response_lower):
            print("Enhanced NLP detected emerging pattern - scoring as EMERGING")
            return Score.EMERGING
            
        # WITH_SUPPORT (3) patterns
        if re.search(r"with (help|support|assistance)", response_lower) or \
           re.search(r"when (prompted|reminded|guided|helped|assisted)", response_lower) or \
           re.search(r"needs (help|support|assistance|prompting|reminding)", response_lower) or \
           re.search(r"(if i help|if we help|if someone helps)", response_lower):
            print("Enhanced NLP detected support pattern - scoring as WITH_SUPPORT")
            return Score.WITH_SUPPORT
            
        # Default for positive responses - INDEPENDENT (4)
        if re.search(r"\b(yes|yeah|yep|sure|absolutely|definitely|always|consistently)\b", response_lower) or \
           re.search(r"(does|can|is able to|performs|demonstrates)", response_lower) or \
           re.search(r"(mastered|achieved|accomplished)", response_lower):
            print("Enhanced NLP detected positive pattern - scoring as INDEPENDENT")
            return Score.INDEPENDENT
            
        # If nothing matched, default to NOT_RATED
        print(f"No pattern matched for response: '{response_text}'. Defaulting to NOT_RATED")
        return Score.NOT_RATED 