"""
Category Helper Module

This module provides utility functions for working with evidence-based
category knowledge in scoring decisions.
"""

import re
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from src.core.scoring.base import Score
from src.core.knowledge.category_knowledge import (
    get_category_evidence,
    get_category_boundary,
    get_domain_specific_evidence,
    CategoryEvidence,
    CategoryBoundary
)

logger = logging.getLogger(__name__)


def get_research_based_definition(score: Union[Score, str]) -> str:
    """
    Get the research-based definition for a scoring category.
    
    Args:
        score: Score enum or category name string
        
    Returns:
        Research-backed definition of the category
    """
    if isinstance(score, Score):
        category_name = score.name
    else:
        category_name = score
    
    evidence = get_category_evidence(category_name)
    if evidence:
        return evidence.definition
    return "No research-based definition available."


def get_research_indicators(score: Union[Score, str]) -> List[str]:
    """
    Get research-based indicators for a scoring category.
    
    Args:
        score: Score enum or category name string
        
    Returns:
        List of research-based indicators
    """
    if isinstance(score, Score):
        category_name = score.name
    else:
        category_name = score
    
    evidence = get_category_evidence(category_name)
    if evidence:
        return evidence.indicators
    return []


def get_domain_indicators(score: Union[Score, str], domain: str) -> List[str]:
    """
    Get domain-specific indicators for a scoring category.
    
    Args:
        score: Score enum or category name string
        domain: Domain name (motor, communication, social, cognitive)
        
    Returns:
        List of domain-specific indicators
    """
    if isinstance(score, Score):
        category_name = score.name
    else:
        category_name = score
    
    evidence = get_category_evidence(category_name)
    if evidence:
        return evidence.get_domain_indicators(domain)
    return []


def get_boundary_criteria(from_score: Union[Score, str], to_score: Union[Score, str]) -> List[str]:
    """
    Get boundary criteria between two scoring categories.
    
    Args:
        from_score: Starting category
        to_score: Target category
        
    Returns:
        List of boundary indicators
    """
    if isinstance(from_score, Score):
        from_category = from_score.name
    else:
        from_category = from_score
    
    if isinstance(to_score, Score):
        to_category = to_score.name
    else:
        to_category = to_score
    
    boundary = get_category_boundary(from_category, to_category)
    if boundary:
        return boundary.criteria
    return []


def get_domain_boundary_criteria(
    from_score: Union[Score, str], 
    to_score: Union[Score, str],
    domain: str
) -> List[str]:
    """
    Get domain-specific boundary criteria between two categories.
    
    Args:
        from_score: Starting category
        to_score: Target category
        domain: Domain name (motor, communication, social, cognitive)
        
    Returns:
        List of domain-specific boundary criteria
    """
    if isinstance(from_score, Score):
        from_category = from_score.name
    else:
        from_category = from_score
    
    if isinstance(to_score, Score):
        to_category = to_score.name
    else:
        to_category = to_score
    
    boundary = get_category_boundary(from_category, to_category)
    if boundary:
        return boundary.get_domain_criteria(domain)
    return []


def get_confidence_threshold(score: Union[Score, str]) -> float:
    """
    Get research-based confidence threshold for a category.
    
    Args:
        score: Score enum or category name string
        
    Returns:
        Confidence threshold from evidence-based sources
    """
    if isinstance(score, Score):
        category_name = score.name
    else:
        category_name = score
    
    evidence = get_category_evidence(category_name)
    if evidence and "confidence_minimum" in evidence.thresholds:
        return evidence.thresholds["confidence_minimum"]
    
    # Default thresholds based on literature
    default_thresholds = {
        "CANNOT_DO": 0.70,
        "WITH_SUPPORT": 0.65,
        "EMERGING": 0.60,
        "INDEPENDENT": 0.80,
        "LOST_SKILL": 0.75,
        "NOT_RATED": 0.90  # High threshold for unknown
    }
    
    return default_thresholds.get(category_name.upper(), 0.70)


def get_boundary_threshold(from_score: Union[Score, str], to_score: Union[Score, str]) -> float:
    """
    Get the threshold value for moving between categories.
    
    Args:
        from_score: Starting category
        to_score: Target category
        
    Returns:
        Threshold value from evidence-based sources
    """
    if isinstance(from_score, Score):
        from_category = from_score.name
    else:
        from_category = from_score
    
    if isinstance(to_score, Score):
        to_category = to_score.name
    else:
        to_category = to_score
    
    boundary = get_category_boundary(from_category, to_category)
    if boundary:
        return boundary.threshold_value
    
    # Default thresholds based on literature
    default_thresholds = {
        ("CANNOT_DO", "WITH_SUPPORT"): 0.75,
        ("WITH_SUPPORT", "EMERGING"): 0.25,
        ("EMERGING", "INDEPENDENT"): 0.80,
        ("INDEPENDENT", "LOST_SKILL"): 0.90,
    }
    
    key = (from_category.upper(), to_category.upper())
    return default_thresholds.get(key, 0.50)


def analyze_response_for_category(response: str, category: Union[Score, str], domain: str) -> Dict[str, Any]:
    """
    Analyze a response text for indicators of a specific category in a given domain.
    
    Args:
        response: The text response to analyze
        category: The category to check for
        domain: The developmental domain
        
    Returns:
        Dictionary with analysis results
    """
    if isinstance(category, Score):
        category_name = category.name
    else:
        category_name = category
    
    evidence = get_category_evidence(category_name)
    if not evidence:
        return {
            "category": category_name,
            "indicators_found": [],
            "confidence": 0.0,
            "analysis": "No evidence definition available for this category."
        }
    
    domain_indicators = evidence.get_domain_indicators(domain)
    
    # Create patterns from the indicators
    indicator_patterns = []
    for indicator in domain_indicators:
        # Convert the indicator to a regex pattern
        pattern = indicator.lower()
        # Replace some common phrases with regex alternatives
        pattern = re.sub(r'requires', r'(requires|needs|must have|only with)', pattern)
        pattern = re.sub(r'demonstrates', r'(demonstrates|shows|exhibits|performs)', pattern)
        pattern = re.sub(r'consistent', r'(consistent|regular|reliable|always)', pattern)
        pattern = re.sub(r'inconsistent', r'(inconsistent|sometimes|occasionally|variable)', pattern)
        pattern = re.sub(r'no ', r'(no |not |never |doesn\'t |does not |unable to )', pattern)
        # Add word boundaries where appropriate
        pattern = re.sub(r'(?<!\w)(\w+)(?!\w)', r'\b\1\b', pattern)
        # Compile the pattern
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            indicator_patterns.append((indicator, regex))
        except re.error:
            logger.warning(f"Invalid regex pattern from indicator: {pattern}")
    
    # Analyze the response
    indicators_found = []
    for indicator, pattern in indicator_patterns:
        if pattern.search(response):
            indicators_found.append(indicator)
    
    # Calculate a basic confidence based on indicators found
    confidence = len(indicators_found) / max(len(domain_indicators), 1) if domain_indicators else 0.0
    
    # Prepare analysis summary
    if indicators_found:
        analysis = f"Found {len(indicators_found)} indicators of {category_name} in this {domain} domain response."
    else:
        analysis = f"No clear indicators of {category_name} were found in this {domain} domain response."
    
    return {
        "category": category_name,
        "indicators_found": indicators_found,
        "confidence": min(confidence, 0.9),  # Cap at 0.9 as text analysis is not definitive
        "analysis": analysis
    }


def determine_category_from_response(response: str, domain: str) -> Tuple[Score, float, str]:
    """
    Determine the most likely category based on response text.
    
    Args:
        response: The text response to analyze
        domain: The developmental domain
        
    Returns:
        Tuple of (Score, confidence, reasoning)
    """
    categories = [Score.CANNOT_DO, Score.WITH_SUPPORT, Score.EMERGING, Score.INDEPENDENT, Score.LOST_SKILL]
    
    # Analyze for each category
    results = []
    for category in categories:
        analysis = analyze_response_for_category(response, category, domain)
        results.append((category, analysis))
    
    # Find the category with the highest confidence
    best_match = max(results, key=lambda x: x[1]["confidence"])
    category, analysis = best_match
    
    # If the confidence is too low, return NOT_RATED
    if analysis["confidence"] < 0.3:
        return (
            Score.NOT_RATED, 
            0.0,
            "Insufficient evidence to determine category from text alone."
        )
    
    # Prepare reasoning
    indicators_text = ", ".join(analysis["indicators_found"])
    reasoning = f"Category {category.name} determined with {analysis['confidence']:.2f} confidence. "
    if analysis["indicators_found"]:
        reasoning += f"Indicators found: {indicators_text}."
    
    return (category, analysis["confidence"], reasoning)


def refine_category_with_research(
    category: Score, 
    confidence: float,
    domain: str,
    age_months: Optional[int] = None
) -> Tuple[Score, float, str]:
    """
    Refine a category assignment based on research-based criteria.
    
    Args:
        category: The initially determined category
        confidence: Initial confidence level
        domain: The developmental domain
        age_months: Optional child's age in months
        
    Returns:
        Tuple of (Score, confidence, reasoning)
    """
    # Get evidence for the category
    evidence = get_category_evidence(category.name)
    if not evidence:
        return (
            category,
            confidence,
            "No research-based evidence available to refine this category."
        )
    
    # Check if confidence meets the research-based threshold
    threshold = evidence.thresholds.get("confidence_minimum", 0.7)
    
    # If confidence is significantly below threshold, consider alternatives
    if confidence < threshold - 0.1:
        # Look at categories adjacent to the current one in the developmental sequence
        adjacent_categories = []
        if category != Score.CANNOT_DO:
            adjacent_categories.append(Score(category.value - 1))
        if category != Score.INDEPENDENT and category != Score.LOST_SKILL:
            adjacent_categories.append(Score(category.value + 1))
        
        # Check boundaries with adjacent categories
        boundary_notes = []
        for adj_category in adjacent_categories:
            # Determine which is the "from" and which is the "to" category
            if adj_category.value < category.value:
                from_cat, to_cat = adj_category, category
            else:
                from_cat, to_cat = category, adj_category
            
            # Get boundary info
            boundary = get_category_boundary(from_cat.name, to_cat.name)
            if boundary:
                boundary_value = boundary.threshold_value
                boundary_notes.append(
                    f"Boundary between {from_cat.name} and {to_cat.name} has threshold {boundary_value:.2f}."
                )
        
        # If we're at a category boundary, note that
        boundary_info = " ".join(boundary_notes)
        
        return (
            category,
            confidence,
            f"Research indicates higher confidence ({threshold:.2f}) is needed for {category.name}. "
            f"Current confidence ({confidence:.2f}) is below threshold. {boundary_info}"
        )
    
    # If confidence meets or exceeds threshold, reinforce with research
    domain_indicators = evidence.get_domain_indicators(domain)
    indicator_summary = f"{len(domain_indicators)} research-based indicators support this category in the {domain} domain."
    
    # Add age-specific note if age is available
    age_note = ""
    if age_months is not None:
        # Simple age brackets - could be more sophisticated with milestone-specific data
        if age_months < 12:
            bracket = "infant (0-12 months)"
        elif age_months < 24:
            bracket = "toddler (12-24 months)"
        else:
            bracket = "preschool (24+ months)"
        
        age_note = f" For {bracket} development, confidence thresholds are particularly important."
    
    return (
        category,
        confidence,
        f"Research-based confidence threshold ({threshold:.2f}) is met. "
        f"{indicator_summary}{age_note}"
    )


def get_citation_for_category(category: Union[Score, str]) -> List[str]:
    """
    Get research citations for a category.
    
    Args:
        category: Score enum or category name
        
    Returns:
        List of citation strings
    """
    if isinstance(category, Score):
        category_name = category.name
    else:
        category_name = category
    
    evidence = get_category_evidence(category_name)
    if evidence:
        return evidence.citations
    return [] 