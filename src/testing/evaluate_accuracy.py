#!/usr/bin/env python3

"""
Accuracy Evaluation Script for ASD Assessment API

This script evaluates the accuracy of the scoring model by:
1. Testing a predefined set of responses with known expected scores
2. Comparing before/after enhancement scores
3. Generating accuracy metrics and suggestions for further improvement
4. Visualization of the scoring differences
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("accuracy-evaluator")

# Default test milestone
DEFAULT_MILESTONE = "selects and brings familiar objects from another room when asked"

# Test cases with expected scores
TEST_CASES = [
    # Structure: (response, expected_score, expected_label, description)
    ("yes", 4, "INDEPENDENT", "Simple positive response"),
    ("no", 1, "NOT_YET", "Simple negative response"),
    ("not at all", 1, "NOT_YET", "Clear negative response"),
    ("does not do this", 1, "NOT_YET", "Explicit negative"),
    ("doesn't perform the action", 1, "NOT_YET", "Negative with action reference"),
    ("sometimes", 2, "EMERGING", "Occasional performance"),
    ("rarely but trying", 2, "EMERGING", "Minimal but emerging"),
    ("with help", 3, "WITH_SUPPORT", "Clear support needed"),
    ("needs assistance", 3, "WITH_SUPPORT", "Support indicator"),
    ("consistently does this", 4, "INDEPENDENT", "Strong positive"),
    ("can do it independently", 4, "INDEPENDENT", "Independence indicator"),
    ("not sure", 1, "NOT_YET", "Uncertainty"),
    ("maybe sometimes with help", 2, "EMERGING", "Mixed uncertainty with support"),
    ("no, but trying occasionally", 2, "EMERGING", "Negative with emergence"),
    ("not without help", 3, "WITH_SUPPORT", "Negative requiring support"),
    ("no, he does not perform the action", 1, "NOT_YET", "Strong negative with explanation"),
    ("never tried", 1, "NOT_YET", "Lack of opportunity"),
    ("begins to try but needs significant help", 2, "EMERGING", "Emerging with high support"),
    ("inconsistently, sometimes needs prompting", 2, "EMERGING", "Mixed consistency with support"),
    ("always does this without help", 4, "INDEPENDENT", "Strong positive with independence")
]

class AccuracyEvaluator:
    def __init__(self, api_url="http://localhost:8002", milestone=DEFAULT_MILESTONE):
        """Initialize the evaluator with API details."""
        self.api_url = api_url
        self.milestone = milestone
        self.results = {
            "before": [],
            "after": [],
            "test_cases": len(TEST_CASES),
            "timestamp": datetime.now().isoformat(),
            "milestone": milestone
        }
        self.server_running = self._check_server()
    
    def _check_server(self):
        """Check if the API server is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            try:
                # Try the root path as a fallback
                response = requests.get(self.api_url, timeout=2)
                return response.status_code in (200, 404)  # 404 might be returned if no root handler
            except requests.RequestException:
                logger.warning(f"API server does not appear to be running at {self.api_url}")
                return False
    
    def set_milestone(self, milestone):
        """Set the milestone to test against."""
        self.milestone = milestone
        self.results["milestone"] = milestone
        logger.info(f"Set milestone to: {milestone}")
    
    def score_response(self, response_text):
        """Send a response to the API for scoring."""
        if not self.server_running:
            logger.error("Cannot score response: API server not running")
            return None
        
        try:
            response = requests.post(
                f"{self.api_url}/score-response",
                json={"milestone": self.milestone, "response": response_text},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
        except requests.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return None
    
    def run_evaluation(self):
        """Run through all test cases and record results."""
        if not self.server_running:
            logger.error("Cannot run evaluation: API server not running")
            return False
        
        logger.info(f"Starting evaluation with {len(TEST_CASES)} test cases")
        logger.info(f"Testing against milestone: {self.milestone}")
        
        for i, (response, expected_score, expected_label, description) in enumerate(TEST_CASES, 1):
            logger.info(f"Testing case {i}/{len(TEST_CASES)}: {description}")
            result = self.score_response(response)
            
            if result:
                # Record the result
                actual_score = result.get("score")
                actual_label = result.get("score_label")
                
                case_result = {
                    "response": response,
                    "expected_score": expected_score,
                    "expected_label": expected_label,
                    "actual_score": actual_score,
                    "actual_label": actual_label,
                    "correct_score": actual_score == expected_score,
                    "correct_label": actual_label == expected_label,
                    "description": description
                }
                
                # Determine which collection to add to (before/after)
                # This is simplistic - in practice we'd need a more robust way to determine this
                if "ENHANCED MODEL ACTIVE" in result.get("notes", ""):
                    self.results["after"].append(case_result)
                else:
                    self.results["before"].append(case_result)
                
                # Log the result
                score_match = "✓" if case_result["correct_score"] else "✗"
                label_match = "✓" if case_result["correct_label"] else "✗"
                
                logger.info(f"  Response: '{response}'")
                logger.info(f"  Expected: Score={expected_score}, Label={expected_label}")
                logger.info(f"  Actual  : Score={actual_score}, Label={actual_label}")
                logger.info(f"  Match   : Score={score_match}, Label={label_match}")
            else:
                logger.warning(f"Failed to get result for response: '{response}'")
        
        return True
    
    def calculate_metrics(self, collection="after"):
        """Calculate accuracy metrics for the specified collection."""
        results = self.results.get(collection, [])
        if not results:
            return {
                "accuracy_score": 0,
                "accuracy_label": 0,
                "total_cases": 0,
                "correct_scores": 0,
                "correct_labels": 0,
                "errors_by_expected_score": {},
                "errors_by_actual_score": {}
            }
        
        correct_scores = sum(1 for r in results if r["correct_score"])
        correct_labels = sum(1 for r in results if r["correct_label"])
        total = len(results)
        
        # Calculate errors by expected and actual scores
        errors_by_expected = defaultdict(lambda: {"total": 0, "errors": 0, "accuracy": 0})
        errors_by_actual = defaultdict(lambda: {"total": 0, "errors": 0, "accuracy": 0})
        
        for r in results:
            expected = r["expected_score"]
            actual = r["actual_score"]
            
            # Count by expected score
            errors_by_expected[expected]["total"] += 1
            if not r["correct_score"]:
                errors_by_expected[expected]["errors"] += 1
            
            # Count by actual score
            errors_by_actual[actual]["total"] += 1
            if not r["correct_score"]:
                errors_by_actual[actual]["errors"] += 1
        
        # Calculate accuracy for each category
        for score in errors_by_expected:
            total = errors_by_expected[score]["total"]
            errors = errors_by_expected[score]["errors"]
            errors_by_expected[score]["accuracy"] = (total - errors) / total if total > 0 else 0
        
        for score in errors_by_actual:
            total = errors_by_actual[score]["total"]
            errors = errors_by_actual[score]["errors"]
            errors_by_actual[score]["accuracy"] = (total - errors) / total if total > 0 else 0
        
        return {
            "accuracy_score": correct_scores / total if total > 0 else 0,
            "accuracy_label": correct_labels / total if total > 0 else 0,
            "total_cases": total,
            "correct_scores": correct_scores,
            "correct_labels": correct_labels,
            "errors_by_expected_score": {k: v for k, v in errors_by_expected.items()},
            "errors_by_actual_score": {k: v for k, v in errors_by_actual.items()}
        }
    
    def visualize_results(self):
        """Generate visualizations of the results."""
        # Get metrics
        before_metrics = self.calculate_metrics("before")
        after_metrics = self.calculate_metrics("after")
        
        if not self.results["before"] and not self.results["after"]:
            logger.warning("No results to visualize")
            return
        
        # Determine which metrics to visualize
        metrics_to_use = after_metrics if self.results["after"] else before_metrics
        
        # Create confusion matrix
        expected_scores = sorted(list(set(r["expected_score"] for r in TEST_CASES)))
        actual_scores = list(range(0, 5))  # Scores 0-4
        
        # Initialize confusion matrix
        confusion = np.zeros((len(expected_scores), len(actual_scores)))
        
        # Fill confusion matrix
        results_to_use = self.results["after"] if self.results["after"] else self.results["before"]
        for r in results_to_use:
            # Find indices
            expected_idx = expected_scores.index(r["expected_score"])
            actual_idx = r["actual_score"] if r["actual_score"] is not None else 0
            confusion[expected_idx, actual_idx] += 1
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot accuracy by category
        categories = ["NOT_YET (1)", "EMERGING (2)", "WITH_SUPPORT (3)", "INDEPENDENT (4)"]
        accuracies = []
        
        for score in range(1, 5):
            if score in metrics_to_use["errors_by_expected_score"]:
                accuracy = metrics_to_use["errors_by_expected_score"][score]["accuracy"] * 100
            else:
                accuracy = 0
            accuracies.append(accuracy)
        
        ax1.bar(categories, accuracies, color=['#ff9999', '#ffcc99', '#99ccff', '#99ff99'])
        ax1.set_title('Accuracy by Expected Category')
        ax1.set_ylim([0, 105])
        ax1.set_ylabel('Accuracy (%)')
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 3, f"{v:.1f}%", ha='center')
        
        # Plot confusion matrix
        im = ax2.imshow(confusion, cmap='Blues')
        
        # Add labels
        ax2.set_xticks(np.arange(len(actual_scores)))
        ax2.set_yticks(np.arange(len(expected_scores)))
        ax2.set_xticklabels(actual_scores)
        ax2.set_yticklabels(expected_scores)
        ax2.set_xlabel('Actual Score')
        ax2.set_ylabel('Expected Score')
        ax2.set_title('Confusion Matrix')
        
        # Annotate confusion matrix cells
        for i in range(len(expected_scores)):
            for j in range(len(actual_scores)):
                text = ax2.text(j, i, int(confusion[i, j]),
                               ha="center", va="center", color="white" if confusion[i, j] > np.max(confusion)/2 else "black")
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax2)
        cbar.set_label('Count')
        
        # Add overall accuracy
        fig.suptitle(f'Scoring Model Accuracy: {metrics_to_use["accuracy_score"]*100:.1f}%', fontsize=16)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accuracy_evaluation_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(filename)
        logger.info(f"Visualization saved to {filename}")
        
        # If running in a notebook, display the plot
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
        except (ImportError, NameError):
            pass
    
    def generate_report(self):
        """Generate a comprehensive report of the evaluation."""
        # Calculate metrics
        before_metrics = self.calculate_metrics("before")
        after_metrics = self.calculate_metrics("after")
        
        # Prepare report
        report = {
            "milestone": self.milestone,
            "timestamp": datetime.now().isoformat(),
            "total_test_cases": len(TEST_CASES),
            "before": {
                "cases_tested": len(self.results["before"]),
                "metrics": before_metrics
            },
            "after": {
                "cases_tested": len(self.results["after"]),
                "metrics": after_metrics
            },
            "comparison": {
                "accuracy_score_change": after_metrics["accuracy_score"] - before_metrics["accuracy_score"] if self.results["before"] and self.results["after"] else None,
                "accuracy_label_change": after_metrics["accuracy_label"] - before_metrics["accuracy_label"] if self.results["before"] and self.results["after"] else None
            },
            "test_cases": [
                {
                    "response": case[0],
                    "expected_score": case[1],
                    "expected_label": case[2],
                    "description": case[3]
                }
                for case in TEST_CASES
            ],
            "detailed_results": {
                "before": self.results["before"],
                "after": self.results["after"]
            }
        }
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accuracy_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filename}")
        
        # Print summary
        logger.info("=== Accuracy Report Summary ===")
        active_metrics = after_metrics if self.results["after"] else before_metrics
        active_label = "Enhanced Model" if self.results["after"] else "Base Model"
        
        logger.info(f"Milestone: {self.milestone}")
        logger.info(f"Model: {active_label}")
        logger.info(f"Total test cases: {len(TEST_CASES)}")
        logger.info(f"Cases tested: {active_metrics['total_cases']}")
        logger.info(f"Score accuracy: {active_metrics['accuracy_score']*100:.1f}%")
        logger.info(f"Label accuracy: {active_metrics['accuracy_label']*100:.1f}%")
        
        # Print comparison if both before and after are available
        if self.results["before"] and self.results["after"]:
            score_change = report["comparison"]["accuracy_score_change"] * 100
            label_change = report["comparison"]["accuracy_label_change"] * 100
            
            score_change_str = f"+{score_change:.1f}%" if score_change > 0 else f"{score_change:.1f}%"
            label_change_str = f"+{label_change:.1f}%" if label_change > 0 else f"{label_change:.1f}%"
            
            logger.info("=== Before vs After Comparison ===")
            logger.info(f"Score accuracy change: {score_change_str}")
            logger.info(f"Label accuracy change: {label_change_str}")
        
        return report

def install_dependencies():
    """Install required dependencies if not already present."""
    try:
        import matplotlib
        import numpy
        logger.info("Dependencies already installed")
        return True
    except ImportError:
        logger.info("Installing required dependencies...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy", "requests"])
            logger.info("Dependencies successfully installed")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to install dependencies")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of the ASD Assessment API scoring model")
    parser.add_argument("--api-url", default="http://localhost:8002", help="API URL (default: http://localhost:8002)")
    parser.add_argument("--milestone", default=DEFAULT_MILESTONE, help=f"Milestone to test (default: {DEFAULT_MILESTONE})")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of results")
    parser.add_argument("--report-only", action="store_true", help="Generate report without running tests")
    args = parser.parse_args()
    
    # Ensure dependencies are installed
    if not install_dependencies():
        logger.error("Cannot continue without required dependencies")
        return 1
    
    # Initialize evaluator
    evaluator = AccuracyEvaluator(api_url=args.api_url, milestone=args.milestone)
    
    if not args.report_only:
        # Run evaluation
        if evaluator.run_evaluation():
            logger.info("Evaluation completed successfully")
        else:
            logger.error("Evaluation failed")
            return 1
    
    # Generate report
    evaluator.generate_report()
    
    # Visualize results if requested
    if args.visualize:
        try:
            evaluator.visualize_results()
        except Exception as e:
            logger.error(f"Failed to generate visualization: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 