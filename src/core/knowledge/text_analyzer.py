"""
Text analysis utilities for detecting category indicators in responses.
"""

from typing import List, Tuple, Dict, Set
import re
import logging

logger = logging.getLogger(__name__)

# Category-specific key phrases
CATEGORY_PHRASES = {
    "CANNOT_DO": [
        "can't", "cannot", "doesn't", "does not", "no attempt", "not able", 
        "unable", "never", "not yet", "hasn't", "has not", "struggles to", 
        "not interested", "won't try"
    ],
    "WITH_SUPPORT": [
        "with help", "needs help", "assistance", "support", "guide", "position", 
        "hand-over-hand", "when I", "help her", "help him", "I have to", 
        "requires", "needs me to", "only with", "if I", "when prompted"
    ],
    "EMERGING": [
        "sometimes", "starting to", "beginning to", "trying to", "inconsistent", 
        "some days", "hit or miss", "depends", "occasionally", "not always", 
        "working on", "learning to", "varies", "partial", "in progress"
    ],
    "INDEPENDENT": [
        "consistently", "always", "by himself", "by herself", "on his own", 
        "on her own", "without help", "independently", "mastered", "can do", 
        "does well", "no problem", "easily", "regularly", "every time",
        "without any help", "without assistance", "by himself", "by herself",
        "on their own", "all by himself", "all by herself"
    ],
    "LOST_SKILL": [
        "used to", "no longer", "stopped", "lost", "regression", "previously", 
        "before", "not anymore", "regressed", "went backward", "declined", 
        "deteriorated", "forgotten how", "used to be able"
    ]
}

# Domain-specific key phrases
DOMAIN_PHRASES = {
    "MOTOR": {
        "CANNOT_DO": ["no movement", "doesn't move", "physically unable", "no attempt to move"],
        "WITH_SUPPORT": ["physical support", "stabilize", "balance", "position", "guide movements"],
        "EMERGING": ["wobbly", "unsteady", "attempts", "tries to move", "some control"],
        "INDEPENDENT": ["coordinated", "smooth movements", "good balance", "strong", "controlled"],
        "LOST_SKILL": ["less coordinated", "weaker", "lost strength", "stopped moving"]
    },
    "COMMUNICATION": {
        "CANNOT_DO": ["doesn't communicate", "no words", "no sounds", "no gestures", "silent"],
        "WITH_SUPPORT": ["repeat", "model", "prompt", "when asked", "echo", "imitate"],
        "EMERGING": ["trying to say", "beginning to talk", "starting to communicate", "inconsistent sounds"],
        "INDEPENDENT": ["clear speech", "sentences", "conversations", "explains", "asks questions", "consistently", "right context", "few words consistently"],
        "LOST_SKILL": ["stopped talking", "fewer words", "less vocal", "lost words"]
    },
    "SOCIAL": {
        "CANNOT_DO": ["no interest in others", "ignores people", "avoids interaction", "prefers alone"],
        "WITH_SUPPORT": ["encourage interaction", "facilitate", "arrange playdates", "structured activities"],
        "EMERGING": ["some interest", "brief interactions", "watches others", "parallel play"],
        "INDEPENDENT": ["makes friends", "shares", "takes turns", "plays cooperatively", "empathy"],
        "LOST_SKILL": ["less social", "withdrawn", "stopped playing with others", "avoids friends"]
    },
    "COGNITIVE": {
        "CANNOT_DO": ["doesn't understand", "confused", "no problem solving", "no interest in learning"],
        "WITH_SUPPORT": ["explain steps", "demonstrate", "break down", "simplify", "show how"],
        "EMERGING": ["simple concepts", "basic understanding", "beginning to grasp", "sometimes solves"],
        "INDEPENDENT": ["understands", "solves problems", "figures out", "remembers", "applies knowledge"],
        "LOST_SKILL": ["forgotten", "confused now", "can't remember", "lost interest in learning"]
    }
}

def analyze_text_for_category(text: str, category: str, domain: str = None) -> Tuple[float, List[str]]:
    """
    Analyze text for indicators of a specific developmental category.
    
    Args:
        text: The response text to analyze
        category: The category to check for (e.g., "EMERGING")
        domain: Optional domain to include domain-specific phrases
        
    Returns:
        Tuple of (confidence score, list of matched phrases)
    """
    if category not in CATEGORY_PHRASES:
        return 0.0, []
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Get general category phrases
    phrases_to_check = CATEGORY_PHRASES[category]
    
    # Add domain-specific phrases if domain is provided
    if domain and domain in DOMAIN_PHRASES and category in DOMAIN_PHRASES[domain]:
        phrases_to_check.extend(DOMAIN_PHRASES[domain][category])
    
    # Find matches
    matched_phrases = []
    for phrase in phrases_to_check:
        if phrase in text_lower:
            matched_phrases.append(phrase)
    
    # Special case for INDEPENDENT category - check for negations of support
    if category == "INDEPENDENT" and not matched_phrases:
        # Check for phrases that indicate independence
        independence_indicators = [
            "without help", "without any help", "without support", 
            "by himself", "by herself", "on his own", "on her own"
        ]
        for phrase in independence_indicators:
            if phrase in text_lower:
                matched_phrases.append(phrase)
    
    # Calculate confidence based on number of matches
    if not matched_phrases:
        return 0.0, []
    
    # More matches = higher confidence, up to a maximum of 0.9
    confidence = min(0.6 + (len(matched_phrases) * 0.1), 0.9)
    
    return confidence, matched_phrases

def get_best_category_match(text: str, domain: str = None) -> Tuple[str, float, List[str]]:
    """
    Determine the best category match for a text response.
    
    Args:
        text: The response text to analyze
        domain: Optional domain for domain-specific analysis
        
    Returns:
        Tuple of (category, confidence, matched phrases)
    """
    results = []
    
    for category in CATEGORY_PHRASES.keys():
        confidence, matched_phrases = analyze_text_for_category(text, category, domain)
        if confidence > 0:
            results.append((category, confidence, matched_phrases))
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    if not results:
        return "NOT_RATED", 0.0, []
    
    # If we have multiple categories with similar confidence, apply priority rules
    if len(results) > 1:
        best_category, best_confidence, best_phrases = results[0]
        second_category, second_confidence, second_phrases = results[1]
        
        # If confidence levels are close (within 0.1)
        if abs(best_confidence - second_confidence) <= 0.1:
            # Priority rule 1: INDEPENDENT takes precedence over EMERGING
            if second_category == "INDEPENDENT" and best_category == "EMERGING":
                return second_category, second_confidence, second_phrases
            
            # Priority rule 2: LOST_SKILL takes precedence when it has strong evidence
            if second_category == "LOST_SKILL" and second_confidence >= 0.7:
                return second_category, second_confidence, second_phrases
            
            # Priority rule 3: WITH_SUPPORT takes precedence over CANNOT_DO when close
            if second_category == "WITH_SUPPORT" and best_category == "CANNOT_DO":
                return second_category, second_confidence, second_phrases
    
    return results[0]

def extract_key_details(text: str) -> Dict[str, str]:
    """
    Extract key details from a response about a child's development.
    
    Args:
        text: The response text to analyze
        
    Returns:
        Dictionary of extracted details
    """
    details = {}
    
    # Age indicators
    age_pattern = r'(\d+)\s*(month|months|mo|year|years|yr)'
    age_matches = re.findall(age_pattern, text, re.IGNORECASE)
    if age_matches:
        age_value, age_unit = age_matches[0]
        details['age_mentioned'] = f"{age_value} {age_unit}"
    
    # Support level indicators
    if any(phrase in text.lower() for phrase in [
        "a lot of help", "always help", "constant support", "hand over hand", 
        "full assistance", "completely dependent"
    ]):
        details['support_level'] = "high"
    elif any(phrase in text.lower() for phrase in [
        "some help", "occasional support", "a little help", "minimal assistance"
    ]):
        details['support_level'] = "moderate"
    elif any(phrase in text.lower() for phrase in [
        "no help", "without assistance", "independently", "by himself", "by herself"
    ]):
        details['support_level'] = "none"
    
    # Consistency indicators
    if any(phrase in text.lower() for phrase in [
        "always", "every time", "consistently", "never fails", "100%"
    ]):
        details['consistency'] = "high"
    elif any(phrase in text.lower() for phrase in [
        "sometimes", "occasionally", "hit or miss", "on and off", "varies"
    ]):
        details['consistency'] = "moderate"
    elif any(phrase in text.lower() for phrase in [
        "rarely", "almost never", "hardly ever", "once in a while"
    ]):
        details['consistency'] = "low"
    
    # Previous ability indicators
    if any(phrase in text.lower() for phrase in [
        "used to", "previously", "before", "no longer", "stopped", "lost"
    ]):
        details['previous_ability'] = "yes"
    
    return details

def generate_analysis_explanation(category: str, matched_phrases: List[str], domain: str = None) -> str:
    """
    Generate a human-readable explanation of the text analysis.
    
    Args:
        category: The determined category
        matched_phrases: List of matched phrases that led to this determination
        domain: Optional domain for domain-specific explanation
        
    Returns:
        Explanation string
    """
    if not matched_phrases:
        return f"No clear indicators of {category} were found."
    
    domain_str = f" in the {domain} domain" if domain else ""
    
    if len(matched_phrases) == 1:
        return f"Found 1 indicator of {category}{domain_str}: '{matched_phrases[0]}'"
    
    phrase_list = ", ".join([f"'{phrase}'" for phrase in matched_phrases])
    return f"Found {len(matched_phrases)} indicators of {category}{domain_str}: {phrase_list}" 