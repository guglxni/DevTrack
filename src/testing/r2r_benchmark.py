#!/usr/bin/env python3
"""
R2R Benchmark Framework

This script provides a framework for benchmarking the R2R integration,
measuring accuracy, performance, and reliability metrics.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.scoring.base import Score, ScoringResult
from src.core.scoring.r2r_enhanced_scorer import R2REnhancedScorer
from src.core.retrieval.r2r_client import R2RClient
from src.testing.benchmark_framework import BenchmarkResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/r2r_benchmark.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("r2r_benchmark")

class R2RBenchmark:
    """Framework for benchmarking the R2R integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the benchmark with the given configuration."""
        self.config = config or self._default_config()
        self.results = []
        self._setup_output_dir()
        
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "output_dir": "test_results/r2r_benchmarks",
            "test_data_dir": "data/r2r_benchmark",
            "model_path": os.path.join("models", "mistral-7b-instruct-v0.2.Q3_K_S.gguf"),
            "data_dir": os.path.join("data", "documents"),
            "random_seed": 42,
            "num_runs": 3,
            "timeout": 60  # seconds
        }
        
    def _setup_output_dir(self) -> None:
        """Set up the output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config["output_dir"]) / f"benchmark_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Benchmark results will be saved to {self.output_dir}")
        
    def load_test_data(self, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load test data from file."""
        if filepath is None:
            # Find the most recent test data file
            test_data_dir = Path(self.config["test_data_dir"])
            test_data_files = list(test_data_dir.glob("r2r_test_data_*.json"))
            if not test_data_files:
                raise FileNotFoundError(f"No test data files found in {test_data_dir}")
            
            # Sort by modification time (most recent first)
            filepath = str(sorted(test_data_files, key=lambda f: f.stat().st_mtime, reverse=True)[0])
            logger.info(f"Using most recent test data file: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Extract test cases from the data structure
        test_cases = data.get("test_cases", [])
        logger.info(f"Loaded {len(test_cases)} test cases from {filepath}")
        return test_cases
        
    def initialize_scorer(self) -> R2REnhancedScorer:
        """Initialize the R2R Enhanced Scorer."""
        logger.info("Initializing R2R Enhanced Scorer...")
        config = {
            "model_path": self.config["model_path"],
            "data_dir": self.config["data_dir"]
        }
        return R2REnhancedScorer(config)
        
    def save_results(self, result: BenchmarkResult, name: str) -> None:
        """Save benchmark results to file."""
        result_file = self.output_dir / f"{name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved benchmark results to {result_file}")
        
    def generate_report(self, results: List[BenchmarkResult]) -> None:
        """Generate a comprehensive report from benchmark results."""
        report_path = self.output_dir / "benchmark_report.html"
        
        # Convert results to DataFrame for easier analysis
        results_data = []
        for result in results:
            data = {"name": result.name, "timestamp": result.timestamp}
            data.update(result.metrics)
            results_data.append(data)
        
        df = pd.DataFrame(results_data)
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>R2R Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .metric {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>R2R Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Benchmark Results</h2>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Timestamp</th>
                    {' '.join([f'<th>{col}</th>' for col in df.columns if col not in ['name', 'timestamp']])}
                </tr>
                {''.join([
                    f'<tr><td>{row["name"]}</td><td>{row["timestamp"]}</td>' + 
                    ''.join([f'<td>{row[col]}</td>' for col in df.columns if col not in ['name', 'timestamp']]) + 
                    '</tr>' for _, row in df.iterrows()
                ])}
            </table>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Generated benchmark report at {report_path}")
        
    def visualize_results(self, results: List[BenchmarkResult]) -> None:
        """Create visualizations of benchmark results."""
        # Convert results to DataFrame
        results_data = []
        for result in results:
            data = {"name": result.name}
            data.update(result.metrics)
            results_data.append(data)
        
        df = pd.DataFrame(results_data)
        
        # Create visualizations for each metric
        metrics = [col for col in df.columns if col != 'name']
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.bar(df['name'], df[metric])
            plt.title(f'{metric} by Benchmark')
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the figure
            output_path = self.output_dir / f"{metric}_visualization.png"
            plt.savefig(output_path)
            plt.close()
            
        logger.info(f"Saved benchmark visualizations to {self.output_dir}")

    def run_accuracy_benchmark(self, test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Run accuracy benchmark on the R2R Enhanced Scorer."""
        logger.info(f"Running accuracy benchmark with {len(test_data)} test cases...")
        
        # Initialize the scorer
        scorer = self.initialize_scorer()
        
        # Track results
        results = {
            "correct": 0,
            "total": 0,
            "by_score": {score.name: {"correct": 0, "total": 0} for score in list(Score) if score != Score.NOT_RATED},
            "by_domain": {domain: {"correct": 0, "total": 0} for domain in ["MOTOR", "COMMUNICATION", "SOCIAL", "COGNITIVE"]},
            "by_test_type": {"standard": {"correct": 0, "total": 0}},
            "confusion_matrix": {score.name: {other.name: 0 for other in list(Score) if other != Score.NOT_RATED} 
                               for score in list(Score) if score != Score.NOT_RATED},
            "confidence_sum": 0,
            "latencies": []
        }
        
        # Process each test case
        detailed_results = []
        for case in tqdm(test_data, desc="Scoring test cases"):
            response = case.get("response", "")
            milestone_context = case.get("milestone_context", {})
            expected_score_name = case.get("expected_score", "")
            domain = case.get("domain", "")
            test_type = case.get("test_type", "standard")
            
            # Skip cases with missing data
            if not response or not milestone_context or not expected_score_name:
                logger.warning(f"Skipping test case with missing data: {case}")
                continue
                
            # Convert expected score name to Score enum
            try:
                expected_score = getattr(Score, expected_score_name)
            except AttributeError:
                logger.warning(f"Invalid expected score: {expected_score_name}")
                continue
                
            # Track test type
            if test_type not in results["by_test_type"]:
                results["by_test_type"][test_type] = {"correct": 0, "total": 0}
                
            # Score the response
            start_time = time.time()
            result = scorer.score(response, milestone_context)
            latency = time.time() - start_time
            
            # Check if correct
            is_correct = result.score == expected_score
            
            # Update results
            results["total"] += 1
            results["by_score"][expected_score_name]["total"] += 1
            results["by_domain"][domain]["total"] += 1
            results["by_test_type"][test_type]["total"] += 1
            results["confidence_sum"] += result.confidence
            results["latencies"].append(latency)
            
            if is_correct:
                results["correct"] += 1
                results["by_score"][expected_score_name]["correct"] += 1
                results["by_domain"][domain]["correct"] += 1
                results["by_test_type"][test_type]["correct"] += 1
                
            # Update confusion matrix
            results["confusion_matrix"][expected_score_name][result.score.name] += 1
            
            # Store detailed result
            detailed_results.append({
                "response": response[:100] + "..." if len(response) > 100 else response,
                "expected_score": expected_score_name,
                "predicted_score": result.score.name,
                "confidence": result.confidence,
                "latency": latency,
                "correct": is_correct,
                "domain": domain,
                "test_type": test_type,
                "reasoning": result.reasoning[:200] + "..." if result.reasoning and len(result.reasoning) > 200 else result.reasoning
            })
            
        # Calculate metrics
        total = results["total"]
        if total > 0:
            accuracy = results["correct"] / total
            avg_confidence = results["confidence_sum"] / total
            avg_latency = sum(results["latencies"]) / total
            
            # Calculate per-category metrics
            for category in results["by_score"]:
                cat_total = results["by_score"][category]["total"]
                if cat_total > 0:
                    results["by_score"][category]["accuracy"] = results["by_score"][category]["correct"] / cat_total
                    
            # Calculate per-domain metrics
            for domain in results["by_domain"]:
                domain_total = results["by_domain"][domain]["total"]
                if domain_total > 0:
                    results["by_domain"][domain]["accuracy"] = results["by_domain"][domain]["correct"] / domain_total
                    
            # Calculate per-test-type metrics
            for test_type in results["by_test_type"]:
                type_total = results["by_test_type"][test_type]["total"]
                if type_total > 0:
                    results["by_test_type"][test_type]["accuracy"] = results["by_test_type"][test_type]["correct"] / type_total
        else:
            accuracy = 0
            avg_confidence = 0
            avg_latency = 0
            
        # Create metrics dictionary
        metrics = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_latency": avg_latency,
            "total_cases": total,
            "correct_cases": results["correct"]
        }
        
        # Add domain-specific accuracies
        for domain in results["by_domain"]:
            if results["by_domain"][domain]["total"] > 0:
                metrics[f"{domain.lower()}_accuracy"] = results["by_domain"][domain]["accuracy"]
                
        # Add test-type-specific accuracies
        for test_type in results["by_test_type"]:
            if results["by_test_type"][test_type]["total"] > 0:
                safe_name = test_type.replace("-", "_")
                metrics[f"{safe_name}_accuracy"] = results["by_test_type"][test_type]["accuracy"]
        
        # Create details dictionary
        details = {
            "by_score": results["by_score"],
            "by_domain": results["by_domain"],
            "by_test_type": results["by_test_type"],
            "confusion_matrix": results["confusion_matrix"],
            "case_results": detailed_results
        }
        
        return BenchmarkResult("r2r_accuracy_benchmark", metrics, details)
        
    def visualize_accuracy_results(self, result: BenchmarkResult) -> None:
        """Create visualizations for accuracy benchmark results."""
        metrics = result.metrics
        details = result.details
        
        # 1. Overall accuracy bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(["Accuracy"], [metrics["accuracy"]], color='blue')
        plt.title("Overall Accuracy")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='r', linestyle='--', label="50% Baseline")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_accuracy.png")
        plt.close()
        
        # 2. Accuracy by score category
        by_score = details["by_score"]
        categories = []
        accuracies = []
        
        for category, data in by_score.items():
            if data.get("total", 0) > 0:
                categories.append(category)
                accuracies.append(data.get("accuracy", 0))
                
        plt.figure(figsize=(12, 6))
        bars = plt.bar(categories, accuracies)
        plt.title("Accuracy by Score Category")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.axhline(y=metrics["accuracy"], color='r', linestyle='--', label="Overall Accuracy")
        
        # Add count labels on top of bars
        for i, bar in enumerate(bars):
            count = f"{by_score[categories[i]]['correct']}/{by_score[categories[i]]['total']}"
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    count, ha='center', va='bottom', rotation=0)
                    
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_by_category.png")
        plt.close()
        
        # 3. Accuracy by domain
        by_domain = details["by_domain"]
        domains = []
        domain_accuracies = []
        
        for domain, data in by_domain.items():
            if data.get("total", 0) > 0:
                domains.append(domain)
                domain_accuracies.append(data.get("accuracy", 0))
                
        plt.figure(figsize=(10, 6))
        bars = plt.bar(domains, domain_accuracies)
        plt.title("Accuracy by Domain")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.axhline(y=metrics["accuracy"], color='r', linestyle='--', label="Overall Accuracy")
        
        # Add count labels
        for i, bar in enumerate(bars):
            count = f"{by_domain[domains[i]]['correct']}/{by_domain[domains[i]]['total']}"
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    count, ha='center', va='bottom', rotation=0)
                    
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_by_domain.png")
        plt.close()
        
        # 4. Confusion matrix
        confusion_matrix = details["confusion_matrix"]
        categories = [cat for cat in confusion_matrix.keys() if cat != "NOT_RATED"]
        
        # Convert to numpy array
        cm = np.zeros((len(categories), len(categories)))
        for i, true_cat in enumerate(categories):
            for j, pred_cat in enumerate(categories):
                cm[i, j] = confusion_matrix[true_cat].get(pred_cat, 0)
                
        # Normalize by row (true categories)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=45)
        plt.yticks(tick_marks, categories)
        plt.ylabel("True Category")
        plt.xlabel("Predicted Category")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, f"{cm[i, j]:.0f}\n({cm_normalized[i, j]:.2f})",
                        ha="center", va="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")
                        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()
        
        # 5. Test type comparison
        by_test_type = details["by_test_type"]
        test_types = []
        test_type_accuracies = []
        
        for test_type, data in by_test_type.items():
            if data.get("total", 0) > 0:
                test_types.append(test_type)
                test_type_accuracies.append(data.get("accuracy", 0))
                
        plt.figure(figsize=(12, 6))
        bars = plt.bar(test_types, test_type_accuracies)
        plt.title("Accuracy by Test Type")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.axhline(y=metrics["accuracy"], color='r', linestyle='--', label="Overall Accuracy")
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, bar in enumerate(bars):
            count = f"{by_test_type[test_types[i]]['correct']}/{by_test_type[test_types[i]]['total']}"
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    count, ha='center', va='bottom', rotation=0)
                    
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_by_test_type.png")
        plt.close()
        
        logger.info(f"Saved accuracy visualizations to {self.output_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run R2R benchmarks")
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data file (default: most recent file in data/r2r_benchmark)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/r2r_benchmarks",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "mistral-7b-instruct-v0.2.Q3_K_S.gguf"),
        help="Path to the local model file"
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        choices=["accuracy", "performance", "all"],
        default="all",
        help="Type of benchmark to run"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize benchmark
    config = {
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "data_dir": os.path.join("data", "documents")  # Add default data_dir
    }
    benchmark = R2RBenchmark(config)
    
    # Load test data
    test_data = benchmark.load_test_data(args.test_data)
    
    # Track all results
    all_results = []
    
    # Run benchmarks based on type
    if args.benchmark_type in ["accuracy", "all"]:
        logger.info("Running accuracy benchmarks...")
        accuracy_result = benchmark.run_accuracy_benchmark(test_data)
        all_results.append(accuracy_result)
        
        # Save results
        benchmark.save_results(accuracy_result, "accuracy")
        
        # Create visualizations
        benchmark.visualize_accuracy_results(accuracy_result)
        
    if args.benchmark_type in ["performance", "all"]:
        logger.info("Running performance benchmarks...")
        # Performance benchmarks will be implemented in the next step
    
    # Generate comprehensive report if multiple benchmarks were run
    if len(all_results) > 1:
        benchmark.generate_report(all_results)
    
    logger.info("Benchmark completed successfully")

if __name__ == "__main__":
    main() 