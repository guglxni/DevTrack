#!/usr/bin/env python3
import requests
import json
import argparse

class ASDAsmessmentAPI:
    """
    A reliable client for the ASD Assessment API that ensures proper scoring by providing
    direct keywords with every API call.
    """
    
    def __init__(self, api_url="http://localhost:8003"):
        """Initialize the API client with the API URL."""
        self.api_url = api_url
        self.milestone_keywords = self._initialize_milestone_keywords()
    
    def _initialize_milestone_keywords(self):
        """Initialize keyword maps for all milestones."""
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
        """Get keywords for a specific milestone."""
        if milestone in self.milestone_keywords:
            return self.milestone_keywords[milestone]
        
        # For other milestones, generate generic keywords
        generic_keywords = {
            "INDEPENDENT": [
                "definitely",
                "always",
                "consistently",
                "very well",
                "yes"
            ],
            "WITH_SUPPORT": [
                "with help",
                "with assistance",
                "needs support"
            ],
            "EMERGING": [
                "starting to",
                "beginning to",
                "occasionally",
                "sometimes"
            ],
            "LOST_SKILL": [
                "used to",
                "previously",
                "no longer",
                "stopped",
                "lost ability"
            ],
            "CANNOT_DO": [
                "doesn't",
                "does not",
                "cannot",
                "never",
                "unable to"
            ]
        }
        
        return generic_keywords
    
    def submit_milestone_assessment(self, milestone, response_text):
        """
        Submit a milestone assessment with proper keyword handling.
        
        Args:
            milestone (str): The developmental milestone behavior
            response_text (str): The parent/caregiver response text
            
        Returns:
            dict: The assessment result
        """
        keywords = self._get_milestone_keywords(milestone)
        
        assessment_data = {
            "question": f"Does your child {milestone}?",
            "milestone_behavior": milestone,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASD Assessment API Client")
    parser.add_argument("milestone", help="The developmental milestone behavior")
    parser.add_argument("response", help="The parent/caregiver response text")
    parser.add_argument("--api-url", default="http://localhost:8003", help="The API endpoint URL")
    
    args = parser.parse_args()
    
    # Initialize the API client
    api_client = ASDAsmessmentAPI(args.api_url)
    
    # Check if the API is running
    if not api_client.health_check():
        print("Error: API server is not running")
        exit(1)
    
    # Submit the assessment
    result = api_client.submit_milestone_assessment(args.milestone, args.response)
    
    # Print the result
    print(json.dumps(result, indent=2)) 