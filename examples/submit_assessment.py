#!/usr/bin/env python3
import requests
import json
import argparse

def submit_milestone_assessment(milestone, response_text, api_url="http://localhost:8003"):
    """
    Submit a milestone assessment with proper keyword handling to ensure accurate scoring.
    
    Args:
        milestone (str): The developmental milestone behavior
        response_text (str): The parent/caregiver response text
        api_url (str): The API endpoint URL
        
    Returns:
        dict: The assessment result
    """
    # Define direct keywords for accurate scoring
    direct_keywords = {
        "INDEPENDENT": [
            "always smiles when he sees",
            "knows all family members",
            "distinguishes between strangers",
            "recognizes everyone",
            "identifies people easily",
            "knows who people are",
            "recognizes immediately",
            "definitely recognizes",
            "recognizes without any issues",
            "consistently recognizes",
            "easily recognizes",
            "clearly recognizes",
            "remembers people"
        ],
        "WITH_SUPPORT": [
            "recognizes with help",
            "recognizes when prompted",
            "needs assistance to recognize",
            "recognizes with support",
            "can identify with guidance",
            "recognizes with reminding",
            "identifies with assistance"
        ],
        "EMERGING": [
            "starting to recognize",
            "beginning to show recognition",
            "occasionally recognizes",
            "sometimes seems to recognize",
            "inconsistently recognizes",
            "trying to recognize",
            "learning to recognize"
        ],
        "LOST_SKILL": [
            "used to recognize",
            "previously recognized",
            "no longer recognizes",
            "stopped recognizing",
            "lost ability to recognize",
            "could recognize before",
            "recognized before regression"
        ],
        "CANNOT_DO": [
            "doesn't recognize anyone",
            "does not recognize people",
            "cannot identify anyone",
            "never recognizes people",
            "shows no recognition",
            "unable to recognize",
            "doesn't know who",
            "can't tell who people are"
        ]
    }
    
    # Prepare the comprehensive assessment data with direct keywords
    assessment_data = {
        "question": f"Does your child {milestone}?",
        "milestone_behavior": milestone,
        "parent_response": response_text,
        "keywords": direct_keywords
    }
    
    # Submit the assessment
    try:
        response = requests.post(
            f"{api_url}/comprehensive-assessment",
            json=assessment_data
        )
        
        # Check if the request was successful
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

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Submit a developmental milestone assessment")
    parser.add_argument("milestone", help="The developmental milestone behavior")
    parser.add_argument("response", help="The parent/caregiver response text")
    parser.add_argument("--api-url", default="http://localhost:8003", help="The API endpoint URL")
    
    args = parser.parse_args()
    
    # Submit the assessment
    result = submit_milestone_assessment(args.milestone, args.response, args.api_url)
    
    # Print the result
    print(json.dumps(result, indent=2)) 