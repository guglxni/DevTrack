#!/usr/bin/env python3
"""
R2R Test Data Generator

This script generates test data for benchmarking the R2R integration.
It creates a varied set of milestone scenarios and responses with expected scores.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.scoring.base import Score

# Domains for test cases
DOMAINS = ["MOTOR", "COMMUNICATION", "SOCIAL", "COGNITIVE"]

# Sample milestones for each domain
MILESTONES = {
    "MOTOR": [
        {
            "id": "motor-1",
            "behavior": "Crawls on hands and knees",
            "criteria": "Child moves forward on hands and knees for at least 3 feet",
            "age_range": "9-12 months"
        },
        {
            "id": "motor-2",
            "behavior": "Walks independently",
            "criteria": "Child takes at least 5 steps without support",
            "age_range": "12-18 months"
        },
        {
            "id": "motor-3",
            "behavior": "Runs with coordination",
            "criteria": "Child runs without frequent falling",
            "age_range": "24-30 months"
        }
    ],
    "COMMUNICATION": [
        {
            "id": "comm-1",
            "behavior": "Uses words to express needs",
            "criteria": "Child uses at least 5 different words to request objects or actions",
            "age_range": "18-24 months"
        },
        {
            "id": "comm-2",
            "behavior": "Forms short sentences",
            "criteria": "Child combines 2-3 words to form simple sentences",
            "age_range": "24-30 months"
        },
        {
            "id": "comm-3",
            "behavior": "Follows two-step instructions",
            "criteria": "Child follows instructions with two distinct actions",
            "age_range": "24-36 months"
        }
    ],
    "SOCIAL": [
        {
            "id": "social-1",
            "behavior": "Takes turns in games",
            "criteria": "Child waits for their turn and follows simple game rules",
            "age_range": "30-36 months"
        },
        {
            "id": "social-2",
            "behavior": "Shows interest in peers",
            "criteria": "Child engages with other children in parallel or interactive play",
            "age_range": "24-30 months"
        },
        {
            "id": "social-3",
            "behavior": "Engages in pretend play",
            "criteria": "Child uses objects to represent other items during play",
            "age_range": "24-36 months"
        }
    ],
    "COGNITIVE": [
        {
            "id": "cognitive-1",
            "behavior": "Sorts objects by shape",
            "criteria": "Child sorts at least 3 different shapes correctly",
            "age_range": "24-30 months"
        },
        {
            "id": "cognitive-2",
            "behavior": "Completes simple puzzles",
            "criteria": "Child completes 3-5 piece puzzles independently",
            "age_range": "24-36 months"
        },
        {
            "id": "cognitive-3",
            "behavior": "Understands concept of counting",
            "criteria": "Child counts at least 3 objects with one-to-one correspondence",
            "age_range": "30-36 months"
        }
    ]
}

# Template responses for each score category
RESPONSE_TEMPLATES = {
    Score.CANNOT_DO: [
        "She hasn't started {behavior_gerund} yet. We've tried to encourage her but she's not showing any signs of this skill.",
        "No, he doesn't {behavior_present} at all. He hasn't shown any progress in this area.",
        "My child can't {behavior_present} currently. I haven't seen any attempts at this behavior.",
        "This is something we've been working on, but she still can't {behavior_present}."
    ],
    Score.LOST_SKILL: [
        "She used to {behavior_present} a few months ago, but she doesn't anymore. I'm not sure why she stopped.",
        "He could {behavior_present} before, but he seems to have lost that ability recently.",
        "My child was {behavior_gerund} consistently, but in the last month has stopped doing it completely.",
        "This is concerning because she had mastered {behavior_gerund}, but now she refuses to even attempt it."
    ],
    Score.EMERGING: [
        "She's just beginning to {behavior_present}. We've seen a few attempts, but she's not consistent yet.",
        "He sometimes tries to {behavior_present}, but he needs a lot of encouragement and doesn't always succeed.",
        "My child is showing some interest in {behavior_gerund} and occasionally attempts it, but it's very inconsistent.",
        "We see early signs of this skill. She has attempted to {behavior_present} a few times in the last week."
    ],
    Score.WITH_SUPPORT: [
        "She can {behavior_present} when I help her. With my support, she's able to do it quite well.",
        "He does {behavior_present} but only when I'm guiding him through it step by step.",
        "My child needs assistance with {behavior_gerund}, but participates actively when supported.",
        "She's getting better at {behavior_gerund}, but still needs me to help her with certain parts."
    ],
    Score.INDEPENDENT: [
        "She can definitely {behavior_present} on her own. She does this consistently without any help.",
        "Yes, he {behavior_present} independently and has been doing so for several weeks now.",
        "My child is very good at {behavior_gerund}. This is something she mastered and does regularly.",
        "He's fully capable of {behavior_gerund} on his own and does it many times throughout the day."
    ]
}

# Edge case scenarios
EDGE_CASE_TEMPLATES = {
    "ambiguous": [
        "Sometimes she can {behavior_present}, but other times she refuses. It's hard to say if she can do it consistently.",
        "He goes back and forth with {behavior_gerund}. Some days he's very good at it, other days it's like he's never done it before.",
        "My child's ability to {behavior_present} varies greatly depending on her mood and the environment."
    ],
    "multilingual": [
        "She can {behavior_present} very well. Elle est très douée pour ça. (She is very good at it.)",
        "Mi hijo puede {behavior_present} cuando quiere, but sometimes he chooses not to.",
        "Sometimes わたしの子 (my child) will {behavior_present}, but only when he feels like it."
    ],
    "complex": [
        "Her {behavior_gerund} skills developed earlier than expected, but then regressed after her brother was born. Now she's slowly relearning.",
        "He has a unique approach to {behavior_gerund}. Instead of {conventional_approach}, he does {alternative_approach}, which achieves the same result.",
        "My child combines {behavior_gerund} with other skills in unexpected ways, demonstrating creativity beyond what I'd expect."
    ]
}

class R2RTestDataGenerator:
    """Generates test data for R2R benchmarking"""
    
    def __init__(self, output_dir: str = "data/r2r_benchmark"):
        """Initialize the generator with an output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _verb_form(self, behavior: str, form: str) -> str:
        """Convert a behavior description to different verb forms."""
        # Extract the verb from the behavior description
        words = behavior.lower().split()
        if len(words) == 0:
            return behavior
            
        verb = words[0]
        
        # Simple rule-based conversion (would need improvement for real use)
        if form == "present":
            return verb
        elif form == "gerund":
            if verb.endswith('e'):
                return verb[:-1] + 'ing'
            else:
                return verb + 'ing'
        return verb
        
    def _format_template(self, template: str, milestone: Dict[str, str]) -> str:
        """Format a template string with milestone information."""
        behavior = milestone['behavior']
        
        # Extract verb from behavior and convert to different forms
        behavior_present = self._verb_form(behavior, "present")
        behavior_gerund = self._verb_form(behavior, "gerund")
        
        # For complex templates, provide some conventional and alternative approaches
        conventional_approach = f"following the usual steps for {behavior.lower()}"
        alternative_approach = f"using a creative method that still accomplishes {behavior.lower()}"
        
        # Replace placeholders in template
        return template.format(
            behavior_present=behavior_present,
            behavior_gerund=behavior_gerund,
            conventional_approach=conventional_approach,
            alternative_approach=alternative_approach
        )
        
    def generate_standard_test_case(self, 
                                  milestone: Dict[str, str], 
                                  score: Score) -> Dict[str, Any]:
        """Generate a standard test case for the given milestone and score."""
        # Select a template for the score category
        template = random.choice(RESPONSE_TEMPLATES[score])
        
        # Format the template with milestone information
        response = self._format_template(template, milestone)
        
        # Add some randomness to length
        if random.random() < 0.3:
            extra_details = [
                f" I've noticed this {random.choice(['recently', 'for several weeks', 'for about a month'])}.",
                f" This is {random.choice(['new', 'something we\'ve been working on', 'exciting to see'])}.",
                f" The {random.choice(['doctor', 'therapist', 'teacher'])} has also noticed this.",
                f" We practice this {random.choice(['daily', 'often', 'when we can'])}."
            ]
            response += random.choice(extra_details)
        
        return {
            "milestone_context": milestone,
            "response": response,
            "expected_score": score.name,
            "expected_score_value": score.value,
            "test_type": "standard"
        }
        
    def generate_edge_case(self, 
                        milestone: Dict[str, str], 
                        edge_type: str) -> Dict[str, Any]:
        """Generate an edge case test for the given milestone."""
        if edge_type not in EDGE_CASE_TEMPLATES:
            edge_type = random.choice(list(EDGE_CASE_TEMPLATES.keys()))
            
        template = random.choice(EDGE_CASE_TEMPLATES[edge_type])
        response = self._format_template(template, milestone)
        
        # Assign expected scores for edge cases
        if edge_type == "ambiguous":
            # Ambiguous cases should challenge the model
            expected_score = random.choice([Score.EMERGING, Score.WITH_SUPPORT])
        elif edge_type == "multilingual":
            # Multilingual cases test language handling
            expected_score = random.choice([Score.CANNOT_DO, Score.EMERGING, Score.INDEPENDENT])
        else:  # complex
            # Complex cases test nuanced understanding
            expected_score = random.choice([Score.WITH_SUPPORT, Score.INDEPENDENT])
        
        return {
            "milestone_context": milestone,
            "response": response,
            "expected_score": expected_score.name,
            "expected_score_value": expected_score.value,
            "test_type": f"edge_{edge_type}"
        }
        
    def generate_test_data(self, 
                        num_standard_cases: int = 50,
                        num_edge_cases: int = 15) -> List[Dict[str, Any]]:
        """Generate a complete test dataset with various test cases."""
        test_data = []
        
        # Generate standard test cases
        for _ in range(num_standard_cases):
            domain = random.choice(DOMAINS)
            milestone = random.choice(MILESTONES[domain])
            score = random.choice(list(Score))
            
            # Skip NOT_RATED as it's not a valid expected score
            if score == Score.NOT_RATED:
                score = Score.EMERGING
                
            test_case = self.generate_standard_test_case(milestone, score)
            test_case["domain"] = domain
            test_data.append(test_case)
            
        # Generate edge cases
        for _ in range(num_edge_cases):
            domain = random.choice(DOMAINS)
            milestone = random.choice(MILESTONES[domain])
            edge_type = random.choice(list(EDGE_CASE_TEMPLATES.keys()))
            
            test_case = self.generate_edge_case(milestone, edge_type)
            test_case["domain"] = domain
            test_data.append(test_case)
            
        return test_data
        
    def save_test_data(self, test_data: List[Dict[str, Any]], filename: str = None) -> str:
        """Save test data to a JSON file and return the filepath."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"r2r_test_data_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Create metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "standard_cases": sum(1 for case in test_data if case["test_type"] == "standard"),
            "edge_cases": sum(1 for case in test_data if "edge" in case["test_type"]),
            "domains": {domain: sum(1 for case in test_data if case["domain"] == domain) for domain in DOMAINS},
            "scores": {score.name: sum(1 for case in test_data if case["expected_score"] == score.name) 
                      for score in list(Score) if score != Score.NOT_RATED}
        }
        
        # Create full data structure
        data_package = {
            "metadata": metadata,
            "test_cases": test_data
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data_package, f, indent=2)
            
        return str(output_path)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate test data for R2R benchmarking")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/r2r_benchmark",
        help="Directory to save test data"
    )
    parser.add_argument(
        "--standard-cases",
        type=int,
        default=50,
        help="Number of standard test cases to generate"
    )
    parser.add_argument(
        "--edge-cases",
        type=int,
        default=15,
        help="Number of edge test cases to generate"
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Filename for output (default: auto-generated)"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    generator = R2RTestDataGenerator(args.output_dir)
    test_data = generator.generate_test_data(args.standard_cases, args.edge_cases)
    output_path = generator.save_test_data(test_data, args.filename)
    
    print(f"Generated {len(test_data)} test cases")
    print(f"Saved test data to {output_path}")

if __name__ == "__main__":
    main() 