"""
Test Data Generator for developmental milestone scoring system.
"""

import os
import json
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

class Score(Enum):
    """Score categories for developmental milestones"""
    CANNOT_DO = 0
    LOST_SKILL = 1
    EMERGING = 2
    WITH_SUPPORT = 3
    INDEPENDENT = 4
    NOT_RATED = -1

class TestDataGenerator:
    """Generate test data for scoring system tests"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the test data generator."""
        self.config = config or self._default_config()
        self.domains = ["motor", "communication", "social", "cognitive"]
        self.seed_responses = self._load_seed_responses()
        
        # Set random seed for reproducibility
        random.seed(self.config.get("random_seed", 42))
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "output_dir": "test_data/scoring",
            "random_seed": 42,
            "num_samples_per_category": 20,
            "response_length_min": 5,
            "response_length_max": 50,
        }
    
    def _load_seed_responses(self) -> Dict[str, List[str]]:
        """Load seed responses for each scoring category."""
        seed_file = Path(self.config.get("seed_responses_file", "test_data/seed_responses.json"))
        
        # Default seed responses if file doesn't exist
        default_seeds = {
            "CANNOT_DO": [
                "No, she cannot do this yet.",
                "He doesn't show any ability to do this.",
                "My child has never demonstrated this behavior.",
                "Not at all, this is beyond her current abilities."
            ],
            "LOST_SKILL": [
                "She used to do this but has stopped.",
                "He could do this a few months ago but not anymore.",
                "My child has regressed in this skill.",
                "She did this before but now refuses or is unable."
            ],
            "EMERGING": [
                "Sometimes, but not consistently.",
                "He's just starting to show signs of this.",
                "Occasionally I see her attempt this.",
                "It's beginning to develop but still early stages."
            ],
            "WITH_SUPPORT": [
                "Yes, but only when I help him.",
                "She can do this with assistance.",
                "He needs prompting but then can do it.",
                "With guidance she manages this well."
            ],
            "INDEPENDENT": [
                "Yes, completely independently.",
                "He does this all by himself.",
                "My child is very good at this without help.",
                "She has mastered this skill fully."
            ]
        }
        
        # Try to load from file, fall back to defaults
        if seed_file.exists():
            try:
                with open(seed_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return default_seeds
        else:
            # Create the seed file with defaults
            os.makedirs(seed_file.parent, exist_ok=True)
            with open(seed_file, 'w') as f:
                json.dump(default_seeds, f, indent=2)
            return default_seeds
    
    def _generate_milestone_context(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Generate a random milestone context."""
        if domain is None:
            domain = random.choice(self.domains)
            
        milestone_id = f"{domain}_{random.randint(1, 50):02d}"
        
        # Dictionary of example behaviors by domain
        behaviors = {
            "motor": [
                "Walks independently",
                "Stacks blocks",
                "Climbs stairs with support",
                "Kicks a ball",
                "Draws simple shapes"
            ],
            "communication": [
                "Uses two-word sentences",
                "Follows two-step instructions",
                "Points to named objects",
                "Responds to their name",
                "Uses gestures to communicate"
            ],
            "social": [
                "Plays alongside other children",
                "Takes turns in simple games",
                "Shows empathy when others are upset",
                "Recognizes familiar people",
                "Engages in pretend play"
            ],
            "cognitive": [
                "Sorts objects by color or shape",
                "Completes simple puzzles",
                "Counts to five",
                "Understands object permanence",
                "Identifies basic colors"
            ]
        }
        
        behavior = random.choice(behaviors.get(domain, behaviors["motor"]))
        
        # Age ranges based on typical developmental milestones
        age_ranges = {
            "motor": ["6-12 months", "12-18 months", "18-24 months", "24-36 months"],
            "communication": ["12-18 months", "18-24 months", "24-30 months", "30-36 months"],
            "social": ["12-18 months", "18-24 months", "24-30 months", "30-36 months"],
            "cognitive": ["18-24 months", "24-30 months", "30-36 months", "36-48 months"]
        }
        
        age_range = random.choice(age_ranges.get(domain, age_ranges["motor"]))
        
        return {
            "id": milestone_id,
            "domain": domain,
            "behavior": behavior,
            "criteria": f"Child {behavior.lower()} regularly and consistently",
            "age_range": age_range
        }
    
    def _generate_response(self, score: Score, milestone_context: Dict[str, Any]) -> str:
        """Generate a response for the given score and milestone."""
        # Get seed responses for this score
        seeds = self.seed_responses.get(score.name, [""])
        if not seeds:
            seeds = [""]
            
        # Select a seed response
        base_response = random.choice(seeds)
        
        # Insert milestone-specific context
        behavior = milestone_context.get("behavior", "").lower()
        pronouns = ["he", "she", "they"]
        pronoun = random.choice(pronouns)
        
        # Replace placeholders
        response = base_response.replace("{behavior}", behavior)
        response = response.replace("{pronoun}", pronoun)
        response = response.replace("{action}", behavior)
        
        # Add some randomness to length
        if random.random() < 0.3:
            time_period = random.choice(["recently", "for several weeks", "for about a month"])
            description = random.choice(["new", "something we've been working on", "exciting to see"])
            observer = random.choice(["doctor", "therapist", "teacher"])
            frequency = random.choice(["daily", "often", "when we can"])
            
            extra_detail = random.choice([
                f" I've noticed this {time_period}.",
                f" This is {description}.",
                f" The {observer} has also noticed this.",
                f" We practice this {frequency}."
            ])
            
            response += extra_detail
        
        return response
    
    def generate_test_data(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate a dataset of test cases with known score labels."""
        if num_samples is None:
            num_samples = self.config.get("num_samples_per_category", 20) * len(Score.__members__)
        
        test_data = []
        
        # Generate samples for each score category
        for score in Score:
            if score == Score.NOT_RATED:
                continue
                
            samples_for_category = num_samples // (len(Score.__members__) - 1)  # Exclude NOT_RATED
            
            for _ in range(samples_for_category):
                milestone_context = self._generate_milestone_context()
                response = self._generate_response(score, milestone_context)
                
                test_data.append({
                    "response": response,
                    "milestone_context": milestone_context,
                    "expected_score": score.name,
                    "expected_score_value": score.value
                })
        
        # Shuffle the data
        random.shuffle(test_data)
        
        return test_data
    
    def save_test_data(self, test_data: List[Dict[str, Any]], filename: str) -> None:
        """Save generated test data to file."""
        output_dir = Path(self.config.get("output_dir", "test_data/scoring"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=2) 