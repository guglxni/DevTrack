"""
Keyword Management Module

This module provides functionality for managing keywords used in scoring.
"""

from typing import Dict, Any, List, Optional
import logging

from .base import Score


class KeywordManager:
    """
    Manager for handling keyword-based scoring data
    
    This class provides methods for:
    1. Storing and retrieving keywords by milestone
    2. Updating keywords for scoring categories
    3. Managing keyword variations
    """
    
    def __init__(self):
        """Initialize the keyword manager"""
        self.keywords = {}  # Dictionary to store keywords by milestone
        self.logger = logging.getLogger("keyword_manager")
        self.logger.setLevel(logging.INFO)
    
    def update_keywords(self, milestone_key: str, category: str, keywords: List[str]):
        """
        Update keywords for a specific milestone and category
        
        Args:
            milestone_key: Unique identifier for the milestone
            category: Scoring category (e.g., "INDEPENDENT")
            keywords: List of keywords for this category
        """
        self.logger.info(f"Updating keywords for milestone {milestone_key}, category {category}")
        
        # Initialize milestone entry if it doesn't exist
        if milestone_key not in self.keywords:
            self.keywords[milestone_key] = {}
        
        # Find the Score enum from the category name
        score_enum = None
        for score in Score:
            if score.name == category:
                score_enum = score
                break
        
        if not score_enum:
            self.logger.error(f"Invalid category: {category}")
            return
        
        # Remove existing keywords for this category
        keys_to_remove = []
        for key, score in self.keywords[milestone_key].items():
            if score == score_enum:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.keywords[milestone_key][key]
        
        # Add new keywords
        for keyword in keywords:
            self.keywords[milestone_key][keyword.lower()] = score_enum
        
        self.logger.info(f"Updated {len(keywords)} keywords for {category}")
    
    def get_keywords(self, milestone_key: str) -> Dict[str, Score]:
        """
        Get keywords for a specific milestone
        
        Args:
            milestone_key: Unique identifier for the milestone
            
        Returns:
            Dictionary mapping keywords to Score enums
        """
        return self.keywords.get(milestone_key, {})
    
    def get_keywords_by_category(self, milestone_key: str) -> Dict[str, List[str]]:
        """
        Get keywords organized by category for a specific milestone
        
        Args:
            milestone_key: Unique identifier for the milestone
            
        Returns:
            Dictionary mapping category names to lists of keywords
        """
        result = {}
        keyword_map = self.keywords.get(milestone_key, {})
        
        for keyword, score in keyword_map.items():
            category = score.name
            if category not in result:
                result[category] = []
            result[category].append(keyword)
        
        return result 