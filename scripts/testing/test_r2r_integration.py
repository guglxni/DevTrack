#!/usr/bin/env python3
"""
Test R2R Integration

This script tests the R2R (Reason to Retrieve) integration to ensure it's working properly.
It tries to initialize the R2R client and test basic functionality.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("r2r_test")

def check_module_importability(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        logger.info(f"✓ Module '{module_name}' is importable")
        return True
    except ImportError as e:
        logger.error(f"✗ Module '{module_name}' cannot be imported: {str(e)}")
        return False

def test_r2r_client():
    """Test the R2R client initialization and basic functionality."""
    try:
        # First, make sure we can import the R2R client
        from src.core.retrieval.r2r_client import R2RClient
        logger.info("✓ Successfully imported R2RClient")
        
        # Try to initialize the client
        client = R2RClient(
            llm_provider="local",
            llm_config={
                "model_path": "models/mistral-7b-instruct-v0.2.Q3_K_S.gguf",
                "temperature": 0.2,
                "max_tokens": 1024
            }
        )
        logger.info("✓ Successfully initialized R2RClient")
        
        # Check if the model was loaded
        if client.model:
            logger.info("✓ Successfully loaded local model")
        else:
            logger.warning("⚠ Local model not loaded (this is expected if model file is missing)")
        
        # Test the list_collections method
        collections = client.list_collections()
        logger.info(f"✓ Successfully called list_collections(): {collections}")
        
        # Generate a test response
        test_prompt = "What is child development?"
        response = client.generate(test_prompt)
        if "error" in response:
            logger.warning(f"⚠ Generate response returned error: {response['error']}")
        else:
            logger.info("✓ Successfully generated response")
            logger.info(f"Response excerpt: {response['text'][:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error testing R2R client: {str(e)}")
        return False

def test_r2r_enhanced_scorer():
    """Test the R2R enhanced scorer initialization and basic functionality."""
    try:
        # Import the R2R enhanced scorer
        from src.core.scoring.r2r_enhanced_scorer import R2REnhancedScorer
        logger.info("✓ Successfully imported R2REnhancedScorer")
        
        # Try to initialize the scorer
        scorer = R2REnhancedScorer(config={
            "model_path": "models/mistral-7b-instruct-v0.2.Q3_K_S.gguf",
            "temperature": 0.2,
            "max_tokens": 1024
        })
        logger.info("✓ Successfully initialized R2REnhancedScorer")
        
        # Check if the client is available
        if scorer.client_available:
            logger.info("✓ R2R client is available in scorer")
        else:
            logger.warning("⚠ R2R client is not available in scorer (fallback mode)")
        
        # Test scoring a simple response
        milestone_context = {
            "name": "Walks independently",
            "domain": "GM",
            "age_range": "12-18",
            "description": "Child can walk without support"
        }
        test_response = "Yes, my child started walking independently at 14 months."
        result = scorer.score(test_response, milestone_context)
        
        logger.info(f"✓ Successfully scored response: {result.score.name} (Confidence: {result.confidence})")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error testing R2R enhanced scorer: {str(e)}")
        return False

def test_improved_engine():
    """Test the improved engine's integration with R2R."""
    try:
        # Import the improved engine
        from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
        logger.info("✓ Successfully imported ImprovedDevelopmentalScoringEngine")
        
        # Try to initialize the engine
        engine = ImprovedDevelopmentalScoringEngine()
        logger.info("✓ Successfully initialized ImprovedDevelopmentalScoringEngine")
        
        # Test finding a milestone
        milestone = engine.find_milestone_by_name("Walks independently")
        if milestone:
            logger.info(f"✓ Successfully found milestone: {milestone.behavior} (Domain: {milestone.domain})")
        else:
            logger.error("✗ Failed to find milestone 'Walks independently'")
        
        # Test finding another milestone
        milestone = engine.find_milestone_by_name("Eats mashed food")
        if milestone:
            logger.info(f"✓ Successfully found milestone: {milestone.behavior} (Domain: {milestone.domain})")
        else:
            logger.error("✗ Failed to find milestone 'Eats mashed food'")
        
        return True
    except Exception as e:
        logger.error(f"✗ Error testing improved engine: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_all_tests():
    """Run all tests and report results."""
    logger.info("=== Starting R2R Integration Tests ===")
    
    # Check important modules
    check_module_importability("src.core.scoring.base")
    check_module_importability("src.core.scoring.r2r_enhanced_scorer")
    check_module_importability("src.core.scoring.r2r_active_learning")
    check_module_importability("src.core.retrieval.r2r_client")
    
    # Run the tests
    client_result = test_r2r_client()
    scorer_result = test_r2r_enhanced_scorer()
    engine_result = test_improved_engine()
    
    # Report results
    logger.info("=== R2R Integration Test Results ===")
    logger.info(f"R2R Client Test: {'PASSED' if client_result else 'FAILED'}")
    logger.info(f"R2R Enhanced Scorer Test: {'PASSED' if scorer_result else 'FAILED'}")
    logger.info(f"Improved Engine Test: {'PASSED' if engine_result else 'FAILED'}")
    
    if client_result and scorer_result and engine_result:
        logger.info("✓ All tests PASSED")
        return True
    else:
        logger.warning("⚠ Some tests FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 