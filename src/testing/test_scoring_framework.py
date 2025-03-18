#!/usr/bin/env python3
"""
Developmental Milestone Scoring System - Testing Framework

This module provides comprehensive testing tools for the milestone scoring system,
with a focus on unit testing, integration testing, and test data generation.

Features:
- Unit tests for individual scoring components
- Integration tests across the scoring pipeline
- Test data generators for different scoring scenarios
- Test fixtures and utilities
"""

import os
import sys
import json
import pytest
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import scoring system
from src.core.scoring.base import Score, BaseScorer, ScoringResult
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
from src.core.scoring.keyword_scorer import KeywordBasedScorer
from src.core.scoring.embedding_scorer import SemanticEmbeddingScorer
from src.core.scoring.transformer_scorer import TransformerBasedScorer
from src.core.scoring.confidence_tracker import ConfidenceTracker
from src.core.scoring.audit_logger import AuditLogger
from src.core.scoring.continuous_learning import ContinuousLearningEngine


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
            extra_details = [
                f" I've noticed this {random.choice(['recently', 'for several weeks', 'for about a month'])}.",
                f" This is {random.choice(['new', 'something we\'ve been working on', 'exciting to see'])}.",
                f" The {random.choice(['doctor', 'therapist', 'teacher'])} has also noticed this.",
                f" We practice this {random.choice(['daily', 'often', 'when we can'])}."
            ]
            response += random.choice(extra_details)
        
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


class TestUtils:
    """Utility functions for testing scoring components"""
    
    @staticmethod
    def get_test_milestone_contexts() -> List[Dict[str, Any]]:
        """Return a set of standard milestone contexts for testing."""
        return [
            {
                "id": "motor_01",
                "domain": "motor",
                "behavior": "Walks independently",
                "criteria": "Child walks without support for at least 10 steps",
                "age_range": "12-18 months"
            },
            {
                "id": "communication_02",
                "domain": "communication",
                "behavior": "Uses two-word sentences",
                "criteria": "Child combines two different words to express ideas",
                "age_range": "18-24 months"
            },
            {
                "id": "social_03",
                "domain": "social",
                "behavior": "Shows empathy",
                "criteria": "Child demonstrates concern when others are upset",
                "age_range": "24-36 months"
            },
            {
                "id": "cognitive_04",
                "domain": "cognitive",
                "behavior": "Sorts objects by color",
                "criteria": "Child can sort objects into groups based on color",
                "age_range": "30-36 months"
            }
        ]
    
    @staticmethod
    def get_test_responses() -> Dict[Score, List[str]]:
        """Return a set of standard test responses for each score category."""
        return {
            Score.CANNOT_DO: [
                "No, she cannot do this yet.",
                "He's not able to do this at all.",
                "My child hasn't developed this skill.",
                "No signs of this ability at all."
            ],
            Score.LOST_SKILL: [
                "She used to do this but has stopped completely.",
                "He could do this a few months ago but now doesn't.",
                "This is a skill that has regressed recently.",
                "She did this before but now refuses or can't."
            ],
            Score.EMERGING: [
                "Sometimes, but not consistently.",
                "He's just starting to show signs of this.",
                "I see beginning attempts at this skill.",
                "It's in the early stages of development."
            ],
            Score.WITH_SUPPORT: [
                "Yes, but only when I help.",
                "She can do this with assistance.",
                "He needs prompting but then can do it.",
                "With guidance she completes this task."
            ],
            Score.INDEPENDENT: [
                "Yes, completely independently.",
                "He does this all by himself every time.",
                "My child has mastered this fully.",
                "She does this independently and consistently."
            ]
        }
    
    @staticmethod
    def assert_scoring_result(result: ScoringResult, expected_score: Score, 
                             min_confidence: float = 0.0) -> None:
        """Assert that a scoring result matches expected values."""
        assert result is not None, "Scoring result should not be None"
        assert isinstance(result, ScoringResult), f"Expected ScoringResult, got {type(result)}"
        assert result.score == expected_score, f"Expected score {expected_score.name}, got {result.score.name}"
        assert result.confidence >= min_confidence, f"Confidence {result.confidence} below minimum {min_confidence}"
        assert result.method, "Method should not be empty"
    
    @staticmethod
    def create_mock_scorer(score_to_return: Score = Score.INDEPENDENT, 
                         confidence: float = 0.9) -> BaseScorer:
        """Create a mock scorer that returns a predetermined result."""
        mock_scorer = MagicMock(spec=BaseScorer)
        
        def mock_score(response, milestone_context=None):
            return ScoringResult(
                score=score_to_return,
                confidence=confidence,
                method="mock_scorer",
                reasoning="This is a mock result"
            )
        
        mock_scorer.score.side_effect = mock_score
        return mock_scorer


# Test fixtures that can be used across test files
@pytest.fixture
def test_milestone_contexts():
    """Fixture providing standard milestone contexts for tests."""
    return TestUtils.get_test_milestone_contexts()

@pytest.fixture
def test_responses():
    """Fixture providing standard responses for each score category."""
    return TestUtils.get_test_responses()

@pytest.fixture
def test_data_generator():
    """Fixture providing a test data generator."""
    return TestDataGenerator()

@pytest.fixture
def mock_scorer():
    """Fixture providing a mock scorer."""
    return TestUtils.create_mock_scorer()

@pytest.fixture
def improved_engine():
    """Fixture providing an instance of the ImprovedDevelopmentalScoringEngine."""
    # Use a test-specific configuration
    config = {
        "enable_transformer_scorer": False,  # Disable for faster tests
        "score_weights": {
            "keyword": 0.6,
            "embedding": 0.4
        }
    }
    return ImprovedDevelopmentalScoringEngine(config)


# Basic unit tests for the Score enum
def test_score_enum():
    """Test the Score enum values and ordering."""
    # Check values
    assert Score.CANNOT_DO.value == 0
    assert Score.LOST_SKILL.value == 1
    assert Score.EMERGING.value == 2
    assert Score.WITH_SUPPORT.value == 3
    assert Score.INDEPENDENT.value == 4
    assert Score.NOT_RATED.value == -1
    
    # Check ordering
    assert Score.CANNOT_DO < Score.LOST_SKILL
    assert Score.LOST_SKILL < Score.EMERGING
    assert Score.EMERGING < Score.WITH_SUPPORT
    assert Score.WITH_SUPPORT < Score.INDEPENDENT


# Basic unit tests for the ScoringResult class
def test_scoring_result():
    """Test the ScoringResult class functionality."""
    # Create a basic result
    result = ScoringResult(
        score=Score.INDEPENDENT,
        confidence=0.95,
        method="test_method",
        reasoning="Test reasoning"
    )
    
    # Check attributes
    assert result.score == Score.INDEPENDENT
    assert result.confidence == 0.95
    assert result.method == "test_method"
    assert result.reasoning == "Test reasoning"
    
    # Check dictionary conversion
    result_dict = result.to_dict()
    assert result_dict["score"] == Score.INDEPENDENT.value
    assert result_dict["score_label"] == "INDEPENDENT"
    assert result_dict["confidence"] == 0.95
    assert result_dict["method"] == "test_method"
    assert result_dict["reasoning"] == "Test reasoning"


# Generate the test suite with parametrized tests
class TestKeywordScorer:
    """Tests for the KeywordBasedScorer"""
    
    def test_initialization(self):
        """Test that the KeywordBasedScorer initializes correctly."""
        scorer = KeywordBasedScorer()
        assert scorer is not None
        
        # Test with custom config
        custom_config = {
            "confidence_threshold": 0.8,
            "min_keyword_matches": 2
        }
        scorer = KeywordBasedScorer(custom_config)
        assert scorer.config["confidence_threshold"] == 0.8
        assert scorer.config["min_keyword_matches"] == 2
    
    @pytest.mark.parametrize("score_category", [
        Score.CANNOT_DO, 
        Score.LOST_SKILL, 
        Score.EMERGING, 
        Score.WITH_SUPPORT, 
        Score.INDEPENDENT
    ])
    def test_keyword_detection(self, score_category, test_responses):
        """Test that the scorer correctly identifies keywords for each category."""
        scorer = KeywordBasedScorer()
        
        # Test with responses known to match this category
        for response in test_responses[score_category]:
            result = scorer.score(response, {"behavior": "test behavior"})
            TestUtils.assert_scoring_result(result, score_category, min_confidence=0.5)
    
    def test_negation_detection(self):
        """Test that the scorer correctly handles negations."""
        scorer = KeywordBasedScorer()
        
        # Test negated positive statement
        result = scorer.score("No, she does not do this independently.", {"behavior": "test behavior"})
        assert result.score != Score.INDEPENDENT, "Negated positive should not score as positive"
        
        # Test double negation
        result = scorer.score("It's not true that she cannot do this.", {"behavior": "test behavior"})
        assert result.score != Score.CANNOT_DO, "Double negation should not score as negative"


class TestEmbeddingScorer:
    """Tests for the SemanticEmbeddingScorer"""
    
    def test_initialization(self):
        """Test that the SemanticEmbeddingScorer initializes correctly."""
        try:
            scorer = SemanticEmbeddingScorer()
            assert scorer is not None
        except ImportError:
            pytest.skip("Embedding model dependencies not available")
    
    @pytest.mark.slow
    def test_embedding_scoring(self, test_responses):
        """Test that the embedding scorer works for various responses."""
        try:
            scorer = SemanticEmbeddingScorer()
            
            # Test a clear INDEPENDENT response
            for response in test_responses[Score.INDEPENDENT]:
                result = scorer.score(response, {"behavior": "test behavior"})
                assert result.score in [Score.INDEPENDENT, Score.WITH_SUPPORT], \
                    f"Expected INDEPENDENT or WITH_SUPPORT, got {result.score.name}"
                    
            # Test a clear CANNOT_DO response
            for response in test_responses[Score.CANNOT_DO]:
                result = scorer.score(response, {"behavior": "test behavior"})
                assert result.score in [Score.CANNOT_DO, Score.LOST_SKILL], \
                    f"Expected CANNOT_DO or LOST_SKILL, got {result.score.name}"
        except (ImportError, RuntimeError):
            pytest.skip("Embedding model unable to initialize")


class TestTransformerScorer:
    """Tests for the TransformerBasedScorer"""
    
    def test_initialization(self):
        """Test that the TransformerBasedScorer initializes correctly."""
        try:
            scorer = TransformerBasedScorer()
            assert scorer is not None
        except ImportError:
            pytest.skip("Transformer model dependencies not available")
    
    @pytest.mark.slow
    def test_zero_shot_classification(self, test_responses):
        """Test zero-shot classification on clear examples."""
        try:
            scorer = TransformerBasedScorer()
            
            # Test a clear INDEPENDENT response
            response = test_responses[Score.INDEPENDENT][0]
            result = scorer.score_with_zero_shot(response, {"behavior": "test behavior"})
            assert result.score in [Score.INDEPENDENT, Score.WITH_SUPPORT], \
                f"Expected INDEPENDENT or WITH_SUPPORT, got {result.score.name}"
                
            # Test a clear CANNOT_DO response
            response = test_responses[Score.CANNOT_DO][0]
            result = scorer.score_with_zero_shot(response, {"behavior": "test behavior"})
            assert result.score in [Score.CANNOT_DO, Score.LOST_SKILL], \
                f"Expected CANNOT_DO or LOST_SKILL, got {result.score.name}"
                
        except (ImportError, RuntimeError):
            pytest.skip("Transformer model unable to initialize")


class TestScoringEngine:
    """Tests for the ImprovedDevelopmentalScoringEngine"""
    
    def test_initialization(self):
        """Test that the engine initializes correctly with various configurations."""
        # Default configuration
        engine = ImprovedDevelopmentalScoringEngine()
        assert engine is not None
        
        # Custom configuration
        custom_config = {
            "enable_keyword_scorer": True,
            "enable_embedding_scorer": False,
            "enable_transformer_scorer": False,
            "enable_audit_logging": False
        }
        engine = ImprovedDevelopmentalScoringEngine(custom_config)
        assert engine is not None
    
    def test_ensemble_scoring(self, test_responses, test_milestone_contexts):
        """Test that the ensemble scoring combines results correctly."""
        # Create engine with only keyword scoring for simplicity
        engine = ImprovedDevelopmentalScoringEngine({
            "enable_keyword_scorer": True,
            "enable_embedding_scorer": False,
            "enable_transformer_scorer": False,
            "enable_audit_logging": False
        })
        
        # Test with a clear INDEPENDENT response
        response = test_responses[Score.INDEPENDENT][0]
        milestone = test_milestone_contexts[0]
        result = engine.score_response(response, milestone)
        
        # Should return ScoringResult for non-detailed results
        assert isinstance(result, ScoringResult)
        assert result.score == Score.INDEPENDENT
        
        # Test with detailed results
        detailed_result = engine.score_response(response, milestone, detailed=True)
        assert isinstance(detailed_result, dict)
        assert detailed_result["score_value"] == Score.INDEPENDENT.value
        assert "component_results" in detailed_result
    
    def test_fallback_behavior(self):
        """Test that the engine gracefully handles component failures."""
        # Create mock scorers - one will fail
        working_scorer = TestUtils.create_mock_scorer(Score.INDEPENDENT, 0.9)
        failing_scorer = MagicMock(spec=BaseScorer)
        failing_scorer.score.side_effect = Exception("Simulated failure")
        
        # Create engine with mock scorers
        engine = ImprovedDevelopmentalScoringEngine()
        engine._scorers = {
            "working": working_scorer,
            "failing": failing_scorer
        }
        
        # The engine should still return a result using only the working scorer
        result = engine.score_response("Test response", {"behavior": "test"})
        assert result is not None
        assert result.score == Score.INDEPENDENT
    
    def test_confidence_tracking(self, test_responses, test_milestone_contexts):
        """Test that confidence tracking works correctly."""
        # Create engine with mock confidence tracker
        mock_tracker = MagicMock(spec=ConfidenceTracker)
        mock_tracker.calculate_confidence.return_value = 0.85
        mock_tracker.should_request_review.return_value = False
        
        engine = ImprovedDevelopmentalScoringEngine({
            "enable_keyword_scorer": True,
            "enable_embedding_scorer": False,
            "enable_transformer_scorer": False
        })
        engine._confidence_tracker = mock_tracker
        
        # Test scoring
        response = test_responses[Score.INDEPENDENT][0]
        milestone = test_milestone_contexts[0]
        result = engine.score_response(response, milestone)
        
        # Confidence should come from the mock tracker
        assert result.confidence == 0.85
        assert mock_tracker.calculate_confidence.called


# Run the tests if this file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 