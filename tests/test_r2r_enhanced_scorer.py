"""
Unit tests for the R2R Enhanced Scorer.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.scoring.base import Score, ScoringResult
from src.core.scoring.r2r_enhanced_scorer import R2REnhancedScorer
from src.core.retrieval.r2r_client import R2RClient

class TestR2REnhancedScorer(unittest.TestCase):
    """Test cases for the R2R Enhanced Scorer."""

    def setUp(self):
        """Set up test fixtures for each test method."""
        # Mock the R2RClient
        self.mock_client_patcher = patch('src.core.retrieval.r2r_client.R2RClient')
        self.mock_client = self.mock_client_patcher.start()
        
        # Configure mock client responses
        self.mock_instance = self.mock_client.return_value
        self.mock_instance.search.return_value = {
            "documents": [
                {"text": "Children at this age typically begin crawling on hands and knees."},
                {"text": "Crawling forward on hands and knees is a milestone achieved between 9-12 months."}
            ]
        }
        
        self.mock_instance.generate.return_value = {
            "response": "EMERGING|0.75|The child is showing signs of attempting to crawl but hasn't yet mastered moving forward consistently.",
            "sources": [
                {"text": "Children at this age typically begin crawling on hands and knees."},
                {"text": "Crawling forward on hands and knees is a milestone achieved between 9-12 months."}
            ]
        }
        
        # Create a test instance with mock client
        self.scorer = R2REnhancedScorer()
        self.scorer._client = self.mock_instance
        
        # Test milestone and response
        self.milestone = {
            "id": "motor-1",
            "behavior": "Crawls on hands and knees",
            "domain": "MOTOR",
            "criteria": "Child moves forward on hands and knees for at least 3 feet",
            "age_range": "9-12 months"
        }
        
        self.response = "She has just started trying to crawl. She can get up on her hands and knees and rock back and forth, but she hasn't figured out how to move forward yet."

    def tearDown(self):
        """Clean up after each test method."""
        self.mock_client_patcher.stop()

    def test_initialization(self):
        """Test that the scorer initializes correctly."""
        # Test with default config
        scorer = R2REnhancedScorer()
        self.assertIsNotNone(scorer)
        
        # Test with custom config
        custom_config = {
            "r2r_config": {
                "mistral_api_key": "test-key",
                "enable_hybrid_search": False
            }
        }
        scorer = R2REnhancedScorer(custom_config)
        self.assertIsNotNone(scorer)

    def test_format_query(self):
        """Test the query formatting for milestone."""
        query = self.scorer._format_query(self.milestone)
        self.assertIn(self.milestone["behavior"], query)
        self.assertIn(self.milestone["criteria"], query)
        self.assertIn(self.milestone["domain"], query)

    def test_extract_score_from_string(self):
        """Test extracting score from LLM response string."""
        # Test valid score extraction
        test_cases = [
            ("INDEPENDENT|0.9|Child can consistently crawl.", 
             (Score.INDEPENDENT, 0.9, "Child can consistently crawl.")),
            ("EMERGING|0.75|Shows some signs of the skill.", 
             (Score.EMERGING, 0.75, "Shows some signs of the skill.")),
            ("CANNOT_DO|0.8|No evidence of the skill.", 
             (Score.CANNOT_DO, 0.8, "No evidence of the skill.")),
            ("WITH_SUPPORT|0.6|Needs assistance.", 
             (Score.WITH_SUPPORT, 0.6, "Needs assistance.")),
        ]
        
        for input_str, expected_output in test_cases:
            result = self.scorer._extract_score_from_string(input_str)
            self.assertEqual(result, expected_output)
        
        # Test invalid score extraction
        invalid_inputs = [
            "No valid format",
            "UNKNOWN_SCORE|0.8|Invalid score",
            "|0.7|Missing score",
            "EMERGING||Missing confidence",
            "EMERGING|invalid|Invalid confidence"
        ]
        
        for invalid_input in invalid_inputs:
            # Should return NOT_RATED for invalid inputs
            score, confidence, reasoning = self.scorer._extract_score_from_string(invalid_input)
            self.assertEqual(score, Score.NOT_RATED)

    def test_scoring_with_r2r(self):
        """Test scoring a response using R2R."""
        result = self.scorer.score(self.response, self.milestone)
        
        # Verify the result type
        self.assertIsInstance(result, ScoringResult)
        
        # Verify calls to R2R client
        self.mock_instance.search.assert_called_once()
        self.mock_instance.generate.assert_called_once()
        
        # Verify result properties
        self.assertEqual(result.score, Score.EMERGING)
        self.assertAlmostEqual(result.confidence, 0.75)
        self.assertEqual(result.method, "r2r_enhanced")
        self.assertIsNotNone(result.reasoning)

    def test_scoring_fallback_mechanism(self):
        """Test fallback mechanism when R2R fails."""
        # Make R2R client generate method raise an exception
        self.mock_instance.generate.side_effect = Exception("API Error")
        
        result = self.scorer.score(self.response, self.milestone)
        
        # Should still return a valid ScoringResult
        self.assertIsInstance(result, ScoringResult)
        self.assertEqual(result.score, Score.NOT_RATED)
        self.assertEqual(result.method, "r2r_enhanced_fallback")
        self.assertIsNotNone(result.reasoning)

    def test_handle_empty_response(self):
        """Test handling of empty or None responses."""
        empty_responses = ["", None, "   "]
        
        for empty_response in empty_responses:
            result = self.scorer.score(empty_response, self.milestone)
            self.assertEqual(result.score, Score.NOT_RATED)
            self.assertEqual(result.method, "r2r_enhanced_fallback")
            self.assertIn("empty response", result.reasoning.lower())

    @patch.dict(os.environ, {"MISTRAL_API_KEY": "test-api-key"})
    def test_api_key_from_environment(self):
        """Test that the API key is correctly loaded from environment."""
        scorer = R2REnhancedScorer()
        self.assertIsNotNone(scorer)
        # The actual API key verification would be done by the R2RClient initialization

if __name__ == '__main__':
    unittest.main()