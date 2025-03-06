#!/usr/bin/env python3
"""
ASD Assessment API Client

This script demonstrates how to use the ASD Assessment API without a frontend.
It can be used to process textual data and generate developmental assessments.
"""

import requests
import json
import argparse
import pandas as pd
from typing import Dict, List, Optional, Union
import sys
from src.api.reliable_client import ReliableASDClient

# API configuration
API_BASE_URL = "http://localhost:8003"  # Update this if your API is hosted elsewhere

# Initialize the reliable client
reliable_client = ReliableASDClient(api_url=API_BASE_URL)

def print_response(response):
    """Pretty print API response"""
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(f"Status code: {response.status_code}")
        print(response.text)

def set_child_age(age: int):
    """Set the child's age in months to filter appropriate milestones"""
    url = f"{API_BASE_URL}/set-child-age"
    payload = {"age": age}
    
    print(f"\n>>> Setting child age to {age} months...")
    response = requests.post(url, json=payload)
    print_response(response)
    return response.json() if response.status_code == 200 else None

def get_next_milestone():
    """Get the next milestone to assess"""
    url = f"{API_BASE_URL}/next-milestone"
    
    print("\n>>> Getting next milestone...")
    response = requests.get(url)
    print_response(response)
    return response.json() if response.status_code == 200 else None

def score_response(milestone_behavior: str, response_text: str):
    """
    Score a response for a specific milestone
    
    Now uses the reliable client for consistent scoring.
    """
    print(f"\n>>> Scoring response for milestone: {milestone_behavior}")
    print(f">>> Response text: '{response_text}'")
    
    # Use the reliable client for scoring
    result = reliable_client.score_response(milestone_behavior, response_text)
    
    # Print the result
    print(json.dumps(result, indent=2))
    
    return result

def generate_report():
    """Generate assessment report"""
    url = f"{API_BASE_URL}/generate-report"
    
    print("\n>>> Generating report...")
    response = requests.get(url)
    
    if response.status_code == 200:
        report_data = response.json()
        
        # Print domain quotients
        print("\n=== DOMAIN QUOTIENTS ===")
        for domain, score in report_data.get("domain_quotients", {}).items():
            print(f"{domain}: {score:.1f}%")
        
        # Print individual scores if present
        if "scores" in report_data and report_data["scores"]:
            print("\n=== INDIVIDUAL MILESTONE SCORES ===")
            df = pd.DataFrame(report_data["scores"])
            print(df.to_string(index=False))
        
        return report_data
    else:
        print(f"Error generating report: {response.status_code}")
        print(response.text)
        return None

def batch_analyze_responses(responses_data: List[Dict]):
    """
    Analyze a batch of responses at once
    
    Now uses the reliable client for consistent scoring.
    """
    print(f"\n>>> Analyzing batch of {len(responses_data)} responses...")
    
    # Use the reliable client for batch analysis
    results = reliable_client.batch_analyze_responses(responses_data)
    
    # Print the results
    print(json.dumps(results, indent=2))
    
    return results

def process_input_file(filename: str, age: int):
    """Process responses from an input file and generate an assessment"""
    try:
        # Read the input file (supports CSV and JSON)
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            responses = df.to_dict('records')
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                responses = json.load(f)
        else:
            print(f"Unsupported file format: {filename}")
            return
        
        # Set the child's age
        set_child_age(age)
        
        # Process each response in sequence
        for resp in responses:
            milestone = resp.get('milestone_behavior')
            response_text = resp.get('response')
            
            if milestone and response_text:
                score_response(milestone, response_text)
            else:
                print(f"Skipping invalid response entry: {resp}")
        
        # Generate the final report
        generate_report()
        
    except Exception as e:
        print(f"Error processing input file: {str(e)}")

def run_interactive_assessment(age: int):
    """Run an interactive assessment with the user"""
    try:
        # Set the child's age
        set_child_age(age)
        
        while True:
            # Get the next milestone
            milestone_data = get_next_milestone()
            
            if not milestone_data or milestone_data.get('complete', False):
                print("\n>>> Assessment complete!")
                break
            
            milestone = milestone_data.get('behavior')
            domain = milestone_data.get('domain')
            age_range = milestone_data.get('age_range')
            
            print(f"\n>>> Milestone: {milestone}")
            print(f">>> Domain: {domain}")
            print(f">>> Age Range: {age_range}")
            
            # Get caregiver response
            print("\nPlease describe how your child performs this behavior:")
            response_text = input("> ")
            
            # Score the response
            score_response(milestone, response_text)
        
        # Generate the final report
        generate_report()
        
    except KeyboardInterrupt:
        print("\nAssessment terminated by user.")
    except Exception as e:
        print(f"Error during interactive assessment: {str(e)}")

def main():
    """Main function to handle command-line arguments and run the client"""
    parser = argparse.ArgumentParser(description="ASD Assessment API Client")
    parser.add_argument("--age", type=int, help="Child's age in months")
    parser.add_argument("--interactive", action="store_true", help="Run interactive assessment")
    parser.add_argument("--file", type=str, help="Input file with responses (CSV or JSON)")
    parser.add_argument("--milestone", type=str, help="Specific milestone to score")
    parser.add_argument("--response", type=str, help="Response text to score")
    
    args = parser.parse_args()
    
    # Check if API is running
    health_check = reliable_client.health_check()
    if not health_check:
        print("Error: Cannot connect to the API server.")
        print(f"Make sure the API server is running at {API_BASE_URL}")
        sys.exit(1)
    
    # Interactive assessment
    if args.interactive and args.age:
        run_interactive_assessment(args.age)
    
    # Process input file
    elif args.file and args.age:
        process_input_file(args.file, args.age)
    
    # Score a single response
    elif args.milestone and args.response:
        score_response(args.milestone, args.response)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAssessment terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 