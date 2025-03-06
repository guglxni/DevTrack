#!/usr/bin/env python3
"""
Reliable ASD Assessment API Client

This module provides a reliable client for the ASD Assessment API that ensures
proper scoring by providing direct keywords with every API call.
"""

import requests
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Union

# Try to import our hybrid scorer for local scoring when possible
try:
    from src.api.hybrid_scorer import score_response as hybrid_score_response
    HYBRID_SCORER_AVAILABLE = True
    logging.info("Hybrid scorer available for local scoring")
except ImportError:
    HYBRID_SCORER_AVAILABLE = False
    logging.info("Hybrid scorer not available for local scoring - will use API calls")

class ReliableASDClient:
    """
    A reliable client for the ASD Assessment API that ensures proper scoring by providing
    direct keywords with every API call.
    """
    
    def __init__(self, api_url="http://localhost:8003", enable_local_scoring=True):
        """
        Initialize the reliable ASD client
        
        Args:
            api_url: The base URL for the ASD Assessment API
            enable_local_scoring: Whether to use local scoring when possible
        """
        self.api_url = api_url
        self.enable_local_scoring = enable_local_scoring and HYBRID_SCORER_AVAILABLE
        self.milestone_keywords = self._initialize_milestone_keywords()
    
    def _initialize_milestone_keywords(self):
        """Initialize keyword maps for specific milestones that need special handling."""
        # Special keywords for the most problematic milestone
        milestone_keywords = {
            "Recognizes familiar people": {
                "INDEPENDENT": [
                    "always recognizes",
                    "consistently recognizes", 
                    "easily recognizes",
                    "immediately recognizes",
                    "recognizes instantly",
                    "definitely recognizes",
                    "clearly recognizes",
                    "recognizes without issues",
                    "knows family members",
                    "recognizes everyone",
                    "knows everyone",
                    "distinguishes between strangers",
                    "smiles at familiar people",
                    "yes he recognizes",
                    "yes she recognizes",
                    "yes they recognize",
                    "always smiles when he sees"
                ],
                "WITH_SUPPORT": [
                    "recognizes with help", 
                    "sometimes recognizes", 
                    "recognizes when prompted",
                    "recognizes with assistance",
                    "recognizes with support",
                    "recognizes with guidance",
                    "recognizes when reminded"
                ],
                "EMERGING": [
                    "starting to recognize", 
                    "beginning to recognize",
                    "occasionally recognizes",
                    "recognizes inconsistently",
                    "sometimes seems to recognize",
                    "might recognize",
                    "recognizes rarely"
                ],
                "LOST_SKILL": [
                    "used to recognize",
                    "previously recognized",
                    "recognized before",
                    "no longer recognizes",
                    "stopped recognizing",
                    "lost ability to recognize"
                ],
                "CANNOT_DO": [
                    "doesn't recognize anyone",
                    "does not recognize anyone",
                    "unable to recognize",
                    "never recognizes",
                    "can't recognize",
                    "cannot recognize anyone",
                    "fails to recognize",
                    "shows no recognition",
                    "treats everyone as strangers",
                    "doesn't know who people are"
                ]
            }
        }
        
        return milestone_keywords
    
    def _get_milestone_keywords(self, milestone):
        """
        Get appropriate keywords for a specific milestone
        
        Args:
            milestone: The milestone behavior to get keywords for
            
        Returns:
            Dict of keywords by category for this milestone
        """
        # Check if we have reliable keywords for this milestone
        if milestone in self.milestone_keywords:
            return self.milestone_keywords[milestone]
        
        # Otherwise use generic keywords
        generic_keywords = {
            "INDEPENDENT": [
                "definitely",
                "always",
                "consistently",
                "very well",
                "yes",
                "without any issues",
                "independently",
                "has mastered",
                "regularly"
            ],
            "WITH_SUPPORT": [
                "with help",
                "with assistance",
                "needs support",
                "with guidance",
                "when prompted",
                "when reminded",
                "needs help"
            ],
            "EMERGING": [
                "starting to",
                "beginning to",
                "occasionally",
                "sometimes",
                "inconsistently",
                "might",
                "trying to",
                "learning to"
            ],
            "LOST_SKILL": [
                "used to",
                "previously",
                "no longer",
                "stopped",
                "lost ability",
                "could before",
                "regressed"
            ],
            "CANNOT_DO": [
                "doesn't",
                "does not",
                "cannot",
                "never",
                "unable to",
                "fails to",
                "not able to",
                "hasn't",
                "has not"
            ]
        }
        
        return generic_keywords
    
    # API Methods
    
    def health_check(self):
        """Check if the API is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def reset(self):
        """Reset the assessment engine."""
        try:
            response = requests.post(f"{self.api_url}/reset")
            return response.status_code == 200
        except:
            return False
    
    def set_child_age(self, age: int):
        """Set the child's age in months to filter appropriate milestones"""
        url = f"{self.api_url}/set-child-age"
        payload = {"age": age}
        
        response = requests.post(url, json=payload)
        return response.json() if response.status_code == 200 else None
    
    def get_next_milestone(self):
        """Get the next milestone to assess"""
        url = f"{self.api_url}/next-milestone"
        
        response = requests.get(url)
        return response.json() if response.status_code == 200 else None
    
    def score_response(self, milestone_behavior: str, response_text: str):
        """
        Score a response for a specific milestone using reliable methods
        
        Args:
            milestone_behavior: The milestone behavior to assess
            response_text: The parent/caregiver response text
            
        Returns:
            Dict with the scoring result
        """
        # Try to use local scoring if available and enabled
        if self.enable_local_scoring and HYBRID_SCORER_AVAILABLE:
            try:
                # Get keywords for this milestone
                keywords = self._get_milestone_keywords(milestone_behavior)
                
                # Use the hybrid scorer for reliable scoring
                result = hybrid_score_response(milestone_behavior, response_text, keywords)
                
                # Add milestone info to the result
                result["milestone"] = milestone_behavior
                
                return result
            except Exception as e:
                print(f"Error in local scoring, falling back to API: {str(e)}")
        
        # Use the API for scoring
        keywords = self._get_milestone_keywords(milestone_behavior)
        return self.comprehensive_assessment(milestone_behavior, response_text, keywords)
    
    def comprehensive_assessment(self, milestone_behavior: str, response_text: str, keywords=None):
        """
        Submit a milestone assessment with proper keyword handling.
        
        Args:
            milestone_behavior (str): The developmental milestone behavior
            response_text (str): The parent/caregiver response text
            keywords (dict, optional): Custom keywords to use for scoring
            
        Returns:
            dict: The assessment result
        """
        # If no keywords provided, get them for this milestone
        if keywords is None:
            keywords = self._get_milestone_keywords(milestone_behavior)
        
        assessment_data = {
            "question": f"Does your child {milestone_behavior}?",
            "milestone_behavior": milestone_behavior,
            "parent_response": response_text,
            "keywords": keywords
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/comprehensive-assessment",
                json=assessment_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": True,
                    "message": f"API Error: {response.status_code}",
                    "details": response.text
                }
        
        except Exception as e:
            return {
                "error": True,
                "message": f"Request Error: {str(e)}"
            }
    
    def generate_report(self):
        """Generate assessment report"""
        url = f"{self.api_url}/generate-report"
        
        response = requests.get(url)
        return response.json() if response.status_code == 200 else None
    
    def batch_analyze_responses(self, responses_data: List[Dict]):
        """
        Analyze a batch of responses at once using reliable scoring
        
        This method analyzes each response individually using the comprehensive assessment
        endpoint to ensure reliable scoring.
        """
        results = []
        
        for resp in responses_data:
            milestone = resp.get("milestone_behavior")
            response_text = resp.get("response")
            
            if milestone and response_text:
                # Score each response using comprehensive assessment
                result = self.score_response(milestone, response_text)
                
                if result and "score" in result and "score_label" in result:
                    # Format result like the batch-score endpoint would
                    results.append({
                        "milestone": milestone,
                        "domain": result.get("domain", ""),
                        "score": result.get("score", -1),
                        "score_label": result.get("score_label", "NOT_RATED")
                    })
        
        return results 