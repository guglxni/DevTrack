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

# API configuration
API_BASE_URL = "http://localhost:8002"  # Update this if your API is hosted elsewhere

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
    """Score a response for a specific milestone"""
    url = f"{API_BASE_URL}/score-response"
    payload = {
        "response": response_text,
        "milestone_behavior": milestone_behavior
    }
    
    print(f"\n>>> Scoring response for milestone: {milestone_behavior}")
    print(f">>> Response text: '{response_text}'")
    response = requests.post(url, json=payload)
    print_response(response)
    return response.json() if response.status_code == 200 else None

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
    """Analyze a batch of responses at once"""
    url = f"{API_BASE_URL}/batch-score"
    payload = {"responses": responses_data}
    
    print(f"\n>>> Analyzing batch of {len(responses_data)} responses...")
    response = requests.post(url, json=payload)
    print_response(response)
    return response.json() if response.status_code == 200 else None

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
    """Run an interactive assessment in the terminal"""
    # Set the child's age
    result = set_child_age(age)
    if not result:
        print("Failed to set child age. Exiting.")
        return
    
    total_milestones = result.get("total_milestones", 0)
    print(f"\nStarting assessment with {total_milestones} age-appropriate milestones.")
    
    # Process milestones one by one
    assessed = 0
    while True:
        milestone = get_next_milestone()
        
        if not milestone or milestone.get("complete", False):
            print("\nAssessment complete!")
            break
        
        assessed += 1
        print(f"\nMilestone {assessed}/{total_milestones}: {milestone['behavior']}")
        print(f"Criteria: {milestone['criteria']}")
        print(f"Domain: {milestone['domain']}")
        
        # Get user input for the response
        print("\nEnter caregiver's description (or type 'skip' to skip, 'quit' to exit):")
        response_text = input("> ")
        
        if response_text.lower() == 'skip':
            continue
        elif response_text.lower() == 'quit':
            break
        
        # Score the response
        score_response(milestone['behavior'], response_text)
    
    # Generate final report
    generate_report()

def main():
    parser = argparse.ArgumentParser(description="ASD Assessment API Client")
    parser.add_argument("--age", type=int, default=24, help="Child's age in months (default: 24)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Interactive assessment
    interactive_parser = subparsers.add_parser("interactive", help="Run interactive assessment")
    
    # Process input file
    file_parser = subparsers.add_parser("file", help="Process responses from a file")
    file_parser.add_argument("filename", help="Input file (CSV or JSON)")
    
    # Single response scoring
    score_parser = subparsers.add_parser("score", help="Score a single response")
    score_parser.add_argument("milestone", help="Milestone behavior to assess")
    score_parser.add_argument("response", help="Response text to score")
    
    args = parser.parse_args()
    
    if not args.command:
        # Default to interactive if no command specified
        run_interactive_assessment(args.age)
    elif args.command == "interactive":
        run_interactive_assessment(args.age)
    elif args.command == "file":
        process_input_file(args.filename, args.age)
    elif args.command == "score":
        set_child_age(args.age)
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