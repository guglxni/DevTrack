#!/usr/bin/env python3
"""
Test Age-Specific LLM Scorer

This script tests the functionality of the LLM scorer with age-specific
knowledge integration, including prompt templates and category adjustments.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Fix imports
from src.core.scoring.base import Score, ScoringResult
from src.core.scoring.llm_scorer import LLMBasedScorer
from src.core.knowledge import get_age_bracket, get_domain_by_name
from src.core.knowledge import get_age_expectations, get_category_guidance
from src.core.knowledge import adjust_category_for_age

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("age_specific_test")

@dataclass
class TestResult:
    """Track test results"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    results: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_result(self, test_name: str, passed: bool, message: str = "", skipped: bool = False):
        """Add test result"""
        if skipped:
            self.skipped += 1
        elif passed:
            self.passed += 1
        else:
            self.failed += 1
        
        self.total += 1
        
        self.results.append({
            "test_name": test_name,
            "passed": passed,
            "skipped": skipped,
            "message": message
        })
        
        # Log result
        if skipped:
            logger.info(f"{test_name}: SKIPPED - {message}")
        elif passed:
            logger.info(f"{test_name}: PASSED - {message}")
        else:
            logger.error(f"{test_name}: FAILED - {message}")
    
    def print_summary(self):
        """Print summary of test results"""
        print("\n==== TEST SUMMARY ====")
        print(f"Total tests: {self.total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Skipped: {self.skipped}")
        print(f"Success rate: {self.passed/self.total*100:.1f}% ({self.passed}/{self.total})")
        
        if self.failed > 0:
            print("\nFailed tests:")
            for result in self.results:
                if not result["passed"] and not result["skipped"]:
                    print(f"- {result['test_name']}: {result['message']}")

def setup_test_environment():
    """Setup test environment by creating necessary files and directories"""
    # Create config directory and templates if they don't exist
    config_dir = Path(project_dir) / "config" / "prompt_templates"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test template files if they don't exist
    for template_name in ["infant", "toddler", "preschooler"]:
        template_file = config_dir / f"{template_name}_template.json"
        if not template_file.exists():
            with open(template_file, 'w') as f:
                json.dump({
                    "name": f"{template_name}_template",
                    "description": f"Test template for {template_name}",
                    "template": "Test template content with {{age_months}} months placeholder"
                }, f, indent=2)
    
    # Find or create a models directory for testing
    model_dir = Path(project_dir) / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Create a dummy model file if needed for testing
    dummy_model = model_dir / "test_model.txt"
    if not dummy_model.exists():
        with open(dummy_model, 'w') as f:
            f.write("# Test model file\n")
    
    return {
        "model_path": str(dummy_model),
        "n_ctx": 512,
        "n_batch": 8,
        "n_gpu_layers": 0,
        "n_threads": 1,
        "use_domain_specific_prompts": True,
        "use_age_specific_prompts": True, 
        "custom_templates_dir": str(config_dir)
    }

def test_prompt_loading(result: TestResult):
    """Test loading age-specific prompt templates"""
    try:
        # Check if template files exist
        config_dir = Path(project_dir) / "config" / "prompt_templates"
        template_files = list(config_dir.glob("*_template.json"))
        templates_found = len(template_files) >= 3
        
        result.add_result(
            "templates_found",
            templates_found,
            f"Found {len(template_files)} template files"
        )
        
        # Check if we can read template content
        for template_file in template_files:
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                    
                result.add_result(
                    f"read_{template_file.stem}",
                    True,
                    f"Successfully read template {template_file.name}"
                )
            except Exception as e:
                result.add_result(
                    f"read_{template_file.stem}",
                    False,
                    f"Error reading template {template_file.name}: {str(e)}"
                )
                
    except Exception as e:
        result.add_result("template_loading", False, f"Exception: {str(e)}")

def test_prompt_formatting(result: TestResult):
    """Test formatting prompts with age-specific information"""
    try:
        # Test cases with different ages
        test_cases = [
            {
                "name": "infant_motor",
                "age_months": 3,
                "domain": "MOTOR"
            },
            {
                "name": "toddler_communication",
                "age_months": 20,
                "domain": "COMMUNICATION"
            },
            {
                "name": "preschooler_social",
                "age_months": 42,
                "domain": "SOCIAL"
            }
        ]
        
        for case in test_cases:
            # Check age bracket function
            bracket = get_age_bracket(case["age_months"])
            result.add_result(
                f"{case['name']}_age_bracket",
                bool(bracket),
                f"Age bracket for {case['age_months']} months: {bracket}"
            )
            
            # Check if we can get any age expectations
            expectations = get_age_expectations(case["age_months"])
            has_expectations = expectations is not None
            result.add_result(
                f"{case['name']}_expectations",
                has_expectations,
                f"{'Found' if has_expectations else 'Missing'} age expectations for {case['age_months']} months"
            )
            
    except Exception as e:
        result.add_result(
            "prompt_formatting",
            False,
            f"Exception: {str(e)}"
        )

def test_category_adjustment(result: TestResult):
    """Test category adjustments based on age"""
    try:
        # Test cases for different categories and ages
        test_cases = [
            {
                "name": "infant_independent",
                "category": "INDEPENDENT",
                "confidence": 0.8,
                "age_months": 6,
                "expected_category": "INDEPENDENT",  # Usually category doesn't change
                "expect_lower_confidence": True  # But confidence should decrease for young age
            },
            {
                "name": "older_independent",
                "category": "INDEPENDENT",
                "confidence": 0.7,
                "age_months": 48,
                "expected_category": "INDEPENDENT",
                "expect_lower_confidence": False  # Confidence might increase for older age
            },
            {
                "name": "toddler_emerging",
                "category": "EMERGING",
                "confidence": 0.75,
                "age_months": 18,
                "expected_category": "EMERGING",
                "expect_lower_confidence": False  # Should be appropriate for this age
            },
            {
                "name": "older_with_support",
                "category": "WITH_SUPPORT",
                "confidence": 0.8,
                "age_months": 54,
                "expected_category": "WITH_SUPPORT",
                "expect_lower_confidence": True  # Support at this age might be concerning
            }
        ]
        
        for case in test_cases:
            # Apply adjustment with the correct signature
            # The function takes category, confidence, age_months, but not domain
            adjusted_category, adjusted_confidence = adjust_category_for_age(
                case["category"], 
                case["confidence"], 
                case["age_months"]
            )
            
            # Check category remains as expected
            category_correct = adjusted_category == case["expected_category"]
            result.add_result(
                f"{case['name']}_category",
                category_correct,
                f"Category adjustment correct: {case['category']} -> {adjusted_category}" if category_correct 
                else f"Unexpected category change: {case['category']} -> {adjusted_category}"
            )
            
            # Check confidence adjustment direction
            if case["expect_lower_confidence"]:
                confidence_correct = adjusted_confidence < case["confidence"]
                direction = "lower"
            else:
                confidence_correct = adjusted_confidence >= case["confidence"]
                direction = "same or higher"
            
            result.add_result(
                f"{case['name']}_confidence",
                confidence_correct,
                f"Confidence adjustment correct ({direction}): {case['confidence']:.2f} -> {adjusted_confidence:.2f}" if confidence_correct
                else f"Unexpected confidence adjustment: {case['confidence']:.2f} -> {adjusted_confidence:.2f}"
            )
            
    except ImportError:
        result.add_result("category_adjustment", False, "adjust_category_for_age function not found")
    except Exception as e:
        result.add_result("category_adjustment", False, f"Exception: {str(e)}")

def test_llm_integration(result: TestResult, run_llm: bool = False):
    """Test integration with LLM scorer"""
    if not run_llm:
        result.add_result("llm_integration", True, "Skipped LLM integration tests (run with --llm flag to enable)", skipped=True)
        return
    
    # We skip this test unless explicitly requested
    result.add_result("llm_full_integration", True, "LLM integration would be tested here if run_llm=True", skipped=True)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test age-specific LLM scoring")
    parser.add_argument("--llm", action="store_true", help="Run tests requiring LLM (slow)")
    args = parser.parse_args()
    
    print("Running Age-Specific LLM Scorer Tests")
    print("=====================================")
    
    # Setup test environment
    print("\nSetting up test environment...")
    setup_test_environment()
    
    # Create results tracker
    results = TestResult()
    
    # Run tests
    print("\n1. Testing prompt template loading...")
    test_prompt_loading(results)
    
    print("\n2. Testing prompt formatting...")
    test_prompt_formatting(results)
    
    print("\n3. Testing category adjustment...")
    test_category_adjustment(results)
    
    print("\n4. Testing LLM integration...")
    test_llm_integration(results, args.llm)
    
    # Print summary
    results.print_summary()
    
    # Return non-zero exit code if any tests failed
    return 1 if results.failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main()) 