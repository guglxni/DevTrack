#!/usr/bin/env python3
"""
Test Data Generator for Developmental Milestone Scoring System

This script generates test data for benchmarking and testing the scoring system.
It creates realistic parent/caregiver responses for different developmental milestones
with known expected scores.

Usage:
  python generate_test_data.py --output test_data.json --count 100
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import scoring classes
from src.core.scoring.base import Score
from src.testing.test_scoring_framework import TestDataGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate test data for the scoring system")
    parser.add_argument("--output", default="test_data/scoring/benchmark_data.json",
                        help="Output file path")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of test cases to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--domain", choices=["motor", "communication", "social", "cognitive", "all"],
                        default="all", help="Domain to generate data for")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                        help="Output format")
    parser.add_argument("--include-edge-cases", action="store_true",
                        help="Include edge cases in the generated data")
    return parser.parse_args()


def generate_edge_cases() -> List[Dict[str, Any]]:
    """Generate edge case test data to challenge the scoring system."""
    edge_cases = []
    
    # Empty responses
    edge_cases.append({
        "response": "",
        "milestone_context": {
            "id": "motor_01",
            "domain": "motor",
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": "NOT_RATED",
        "expected_score_value": -1,
        "edge_case_type": "empty_response"
    })
    
    # Very short responses
    edge_cases.append({
        "response": "Yes",
        "milestone_context": {
            "id": "communication_02",
            "domain": "communication",
            "behavior": "Uses two-word sentences",
            "criteria": "Child combines two different words to express ideas",
            "age_range": "18-24 months"
        },
        "expected_score": "INDEPENDENT",
        "expected_score_value": 4,
        "edge_case_type": "very_short_response"
    })
    
    # Contradictory responses
    edge_cases.append({
        "response": "No, she can't do this independently yet, but she does it by herself all the time.",
        "milestone_context": {
            "id": "social_03",
            "domain": "social",
            "behavior": "Shows empathy",
            "criteria": "Child demonstrates concern when others are upset",
            "age_range": "24-36 months"
        },
        "expected_score": "INDEPENDENT",
        "expected_score_value": 4,
        "edge_case_type": "contradictory_response"
    })
    
    # Responses with spelling errors
    edge_cases.append({
        "response": "Somtimes she wlaks for a fw steps but needs halp a lot.",
        "milestone_context": {
            "id": "motor_04",
            "domain": "motor",
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": "EMERGING",
        "expected_score_value": 2,
        "edge_case_type": "spelling_errors"
    })
    
    # Responses with irrelevant information
    edge_cases.append({
        "response": "My child is 15 months old now. He loves playing with his toy cars and watching cartoons. His favorite food is applesauce. Our dog likes to follow him around the house. Oh about the question, yes he can do this with help.",
        "milestone_context": {
            "id": "cognitive_05",
            "domain": "cognitive",
            "behavior": "Identifies colors",
            "criteria": "Child can name at least three colors correctly",
            "age_range": "30-36 months"
        },
        "expected_score": "WITH_SUPPORT",
        "expected_score_value": 3,
        "edge_case_type": "irrelevant_information"
    })
    
    # Multilingual responses
    edge_cases.append({
        "response": "Si, el niño puede hacerlo independientemente. He can do it by himself.",
        "milestone_context": {
            "id": "communication_06",
            "domain": "communication",
            "behavior": "Follows simple directions",
            "criteria": "Child follows a one-step direction without gestures",
            "age_range": "12-18 months"
        },
        "expected_score": "INDEPENDENT",
        "expected_score_value": 4,
        "edge_case_type": "multilingual"
    })
    
    # Very long, complex responses
    long_response = "My child has been developing this skill gradually over the past few months. " + \
                   "At first, she couldn't do it at all, and would get frustrated when attempting it. " + \
                   "Then she started showing some interest and would try occasionally, but with very limited success. " + \
                   "Over time, she began to improve slightly, being able to do parts of the skill but not completely. " + \
                   "Now, with some assistance and encouragement, she can do most of it, but still needs help with " + \
                   "certain aspects. Her pediatrician mentioned that this is completely normal development for her age. " + \
                   "We practice this skill several times a week, and I've noticed steady improvement. " + \
                   "At daycare, her teachers also work on this with her, and they've reported similar progress. " + \
                   "I'd say she's definitely making good progress but isn't fully independent with it yet."
    
    edge_cases.append({
        "response": long_response,
        "milestone_context": {
            "id": "cognitive_07",
            "domain": "cognitive",
            "behavior": "Completes simple puzzles",
            "criteria": "Child can complete a simple puzzle of 3-4 pieces",
            "age_range": "24-30 months"
        },
        "expected_score": "WITH_SUPPORT",
        "expected_score_value": 3,
        "edge_case_type": "long_complex_response"
    })
    
    return edge_cases


def main():
    """Main function to generate and save test data."""
    args = parse_args()
    
    # Set up output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure test data generator
    config = {
        "output_dir": str(output_path.parent),
        "random_seed": args.seed,
        "num_samples_per_category": args.count // (len(Score.__members__) - 1)  # Exclude NOT_RATED
    }
    
    # Initialize generator
    generator = TestDataGenerator(config)
    
    # Generate test data
    test_data = generator.generate_test_data(args.count)
    
    # Add edge cases if requested
    if args.include_edge_cases:
        edge_cases = generate_edge_cases()
        test_data.extend(edge_cases)
        
        # Reshuffle after adding edge cases
        random.seed(args.seed)
        random.shuffle(test_data)
    
    # Filter by domain if specified
    if args.domain != "all":
        test_data = [
            item for item in test_data 
            if item.get("milestone_context", {}).get("domain") == args.domain
        ]
    
    # Save test data
    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=2)
    else:  # CSV format
        import csv
        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ["response", "domain", "behavior", "criteria", "age_range", "expected_score", "expected_score_value"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in test_data:
                milestone = item.get("milestone_context", {})
                writer.writerow({
                    "response": item.get("response", ""),
                    "domain": milestone.get("domain", ""),
                    "behavior": milestone.get("behavior", ""),
                    "criteria": milestone.get("criteria", ""),
                    "age_range": milestone.get("age_range", ""),
                    "expected_score": item.get("expected_score", ""),
                    "expected_score_value": item.get("expected_score_value", "")
                })
    
    # Print summary
    score_counts = {}
    for item in test_data:
        score = item.get("expected_score", "UNKNOWN")
        score_counts[score] = score_counts.get(score, 0) + 1
    
    print(f"Generated {len(test_data)} test samples:")
    for score, count in score_counts.items():
        print(f"  {score}: {count} samples")
    
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    main() 