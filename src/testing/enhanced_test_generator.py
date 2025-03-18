"""
Enhanced Test Data Generator for Developmental Milestone Scoring System

This module extends the base test data generator with additional features for
generating more diverse and challenging test cases, including:
- Complex edge cases
- Multilingual responses
- Ambiguous responses
- Domain-specific variations
- Time-based progression scenarios
"""

import random
import json
import spacy
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from src.testing.test_data_generator import TestDataGenerator, Score

class EnhancedTestDataGenerator(TestDataGenerator):
    """Enhanced test data generator with more diverse and challenging cases."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced test data generator."""
        super().__init__(config)
        self.nlp = None  # Lazy load spaCy
        self.ambiguity_patterns = self._load_ambiguity_patterns()
        self.time_patterns = self._load_time_patterns()
        
    def _load_ambiguity_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for generating ambiguous responses."""
        return {
            "uncertainty": [
                "I'm not entirely sure, but {response}",
                "It's hard to tell, though {response}",
                "Sometimes {response}, other times not",
                "We're still figuring out if {response}"
            ],
            "conditional": [
                "When well-rested, {response}",
                "Depending on the situation, {response}",
                "With certain people, {response}",
                "In familiar settings, {response}"
            ],
            "mixed_signals": [
                "{response}, but then struggles unexpectedly",
                "Usually {response}, except when overwhelmed",
                "Started {response}, then regressed slightly",
                "{response} one day, then seems unable the next"
            ]
        }
    
    def _load_time_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for time-based progression scenarios."""
        return {
            "recent_progress": [
                "Just this week, {response}",
                "In the past few days, {response}",
                "We've noticed lately that {response}",
                "Recently started to {response}"
            ],
            "regression": [
                "Used to {response} consistently, but stopped last month",
                "Was doing well with this, but lately {negative_response}",
                "Showed promise initially, now {negative_response}",
                "Had mastered this, but recently {negative_response}"
            ],
            "fluctuation": [
                "Some days {response}, other days {negative_response}",
                "Progress varies - sometimes {response}, sometimes not",
                "Inconsistent - can {response} one day, struggles the next",
                "Goes back and forth between {response} and {negative_response}"
            ]
        }
    
    def generate_complex_edge_case(self, score: Score, milestone_context: Dict[str, Any]) -> str:
        """Generate a complex edge case response."""
        base_response = self._generate_response(score, milestone_context)
        
        # Add complexity based on score type
        if score == Score.EMERGING:
            pattern = random.choice(self.ambiguity_patterns["uncertainty"])
            return pattern.format(response=base_response)
        elif score == Score.WITH_SUPPORT:
            pattern = random.choice(self.ambiguity_patterns["conditional"])
            return pattern.format(response=base_response)
        elif score == Score.LOST_SKILL:
            pattern = random.choice(self.time_patterns["regression"])
            negative_response = self._generate_response(Score.CANNOT_DO, milestone_context)
            return pattern.format(response=base_response, negative_response=negative_response)
        
        return base_response
    
    def generate_multilingual_response(self, response: str, target_language: str = "es") -> str:
        """Generate a multilingual version of the response."""
        # Note: In a real implementation, this would use a translation service
        # For now, we'll simulate by adding language markers
        return f"{response} [{target_language}: Simulated translation]"
    
    def generate_progression_scenario(self, milestone_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a series of responses showing skill progression over time."""
        progression = []
        base_date = datetime.now() - timedelta(days=90)
        
        # Generate progression stages
        stages = [
            (Score.CANNOT_DO, 0),
            (Score.EMERGING, 15),
            (Score.WITH_SUPPORT, 45),
            (Score.INDEPENDENT, 75)
        ]
        
        for score, days_offset in stages:
            date = base_date + timedelta(days=days_offset)
            response = self.generate_complex_edge_case(score, milestone_context)
            
            progression.append({
                "response": response,
                "milestone_context": milestone_context,
                "expected_score": score.name,
                "expected_score_value": score.value,
                "observation_date": date.isoformat()
            })
        
        return progression
    
    def generate_enhanced_test_data(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate an enhanced dataset with more diverse test cases."""
        base_data = self.generate_test_data(num_samples)
        enhanced_data = []
        
        for item in base_data:
            # 30% chance of generating a complex edge case
            if random.random() < 0.3:
                item["response"] = self.generate_complex_edge_case(
                    Score[item["expected_score"]],
                    item["milestone_context"]
                )
            
            # 10% chance of adding multilingual content
            if random.random() < 0.1:
                item["response"] = self.generate_multilingual_response(item["response"])
            
            enhanced_data.append(item)
        
        # Add progression scenarios for 20% of milestones
        milestone_contexts = {item["milestone_context"]["id"]: item["milestone_context"] 
                            for item in base_data}
        num_progression = len(milestone_contexts) // 5
        
        for milestone_context in random.sample(list(milestone_contexts.values()), num_progression):
            progression_data = self.generate_progression_scenario(milestone_context)
            enhanced_data.extend(progression_data)
        
        return enhanced_data
    
    def save_enhanced_test_data(self, test_data: List[Dict[str, Any]], filename: str) -> None:
        """Save enhanced test data with additional metadata."""
        output_dir = Path(self.config.get("output_dir", "test_data/scoring"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "generator_version": "2.0",
            "generation_date": datetime.now().isoformat(),
            "total_samples": len(test_data),
            "score_distribution": self._calculate_score_distribution(test_data),
            "feature_statistics": self._calculate_feature_statistics(test_data)
        }
        
        output_data = {
            "metadata": metadata,
            "test_data": test_data
        }
        
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    
    def _calculate_score_distribution(self, test_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate the distribution of scores in the test data."""
        distribution = {}
        for item in test_data:
            score = item["expected_score"]
            distribution[score] = distribution.get(score, 0) + 1
        return distribution
    
    def _calculate_feature_statistics(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the generated test data features."""
        stats = {
            "complex_edge_cases": 0,
            "multilingual_responses": 0,
            "progression_scenarios": 0,
            "average_response_length": 0
        }
        
        total_length = 0
        for item in test_data:
            response = item["response"]
            total_length += len(response.split())
            
            if any(pattern in response for patterns in self.ambiguity_patterns.values() 
                   for pattern in patterns):
                stats["complex_edge_cases"] += 1
            
            if "[" in response and "]" in response:  # Simple check for multilingual content
                stats["multilingual_responses"] += 1
            
            if "observation_date" in item:
                stats["progression_scenarios"] += 1
        
        stats["average_response_length"] = total_length / len(test_data)
        return stats 