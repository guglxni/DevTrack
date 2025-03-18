#!/usr/bin/env python3
"""
Benchmark Domain-Specific Prompts

This script benchmarks the performance of the LLM scorer with and without
domain-specific prompts to measure the impact on accuracy and confidence.
"""

import sys
import os
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import random
from datetime import datetime

# Add the project root to the Python path
sys.path.append('.')

from src.core.scoring.llm_scorer import LLMBasedScorer
from src.core.scoring.base import Score, ScoringResult
from src.testing.gold_standard_manager import GoldStandardManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("domain_prompt_benchmark")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark domain-specific prompts for the LLM scorer"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="1.0.0",
        help="Gold standard dataset version to use"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples to use for benchmarking"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/domain_prompts",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def setup_output_directory(output_dir: str) -> Path:
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"benchmark_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def load_test_data(version: str, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    """Load test data from gold standard dataset."""
    logger.info(f"Loading gold standard dataset version: {version}")
    
    # Initialize gold standard manager
    manager = GoldStandardManager()
    
    # Load dataset
    dataset = manager.load_dataset(version)
    logger.info(f"Loaded {len(dataset)} samples from dataset")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample data if needed
    if sample_size and sample_size < len(dataset):
        # Ensure we have samples from each domain
        domains = set(item.get("milestone_context", {}).get("domain", "unknown") 
                     for item in dataset)
        
        # Filter out unknown domains
        domains = [d for d in domains if d and d.lower() != "unknown"]
        
        # Calculate samples per domain
        samples_per_domain = max(1, sample_size // len(domains))
        
        # Group by domain
        domain_groups = {}
        for item in dataset:
            domain = item.get("milestone_context", {}).get("domain", "").lower()
            if domain in domains:
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(item)
        
        # Sample from each domain
        sampled_data = []
        for domain, items in domain_groups.items():
            domain_samples = random.sample(
                items, 
                min(samples_per_domain, len(items))
            )
            sampled_data.extend(domain_samples)
            logger.info(f"Selected {len(domain_samples)} samples from {domain} domain")
        
        # If we need more samples to reach sample_size, randomly select from remaining
        if len(sampled_data) < sample_size:
            remaining = [item for item in dataset if item not in sampled_data]
            additional = random.sample(
                remaining,
                min(sample_size - len(sampled_data), len(remaining))
            )
            sampled_data.extend(additional)
            logger.info(f"Added {len(additional)} additional samples to reach target size")
        
        # If we have too many samples, trim to exact size
        if len(sampled_data) > sample_size:
            sampled_data = sampled_data[:sample_size]
        
        dataset = sampled_data
    
    logger.info(f"Using {len(dataset)} test samples")
    return dataset

def run_benchmark_with_config(
    test_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    config_name: str
) -> Tuple[List[Dict[str, Any]], float]:
    """Run benchmark with a specific configuration."""
    logger.info(f"Running benchmark with {config_name} configuration")
    
    # Initialize scorer with config
    scorer = LLMBasedScorer(config)
    
    results = []
    start_time = time.time()
    
    # Score each sample
    for i, sample in enumerate(test_data):
        response = sample.get("response", "")
        milestone_context = sample.get("milestone_context", {})
        expected_score = sample.get("expected_score")
        
        if isinstance(expected_score, str):
            expected_score = Score[expected_score]
        
        logger.info(f"Scoring sample {i+1}/{len(test_data)}")
        
        try:
            # Score the response
            result = scorer.score(response, milestone_context)
            
            # Record result
            results.append({
                "sample_id": sample.get("id", f"sample_{i}"),
                "response": response,
                "milestone_context": milestone_context,
                "expected_score": expected_score.name if expected_score else None,
                "predicted_score": result.score.name,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "correct": result.score == expected_score if expected_score else None
            })
            
            # Log result
            logger.info(f"  Score: {result.score.name}, Confidence: {result.confidence:.2f}")
            if expected_score:
                logger.info(f"  Expected: {expected_score.name}, Correct: {result.score == expected_score}")
            
        except Exception as e:
            logger.error(f"Error scoring sample {i+1}: {e}")
            results.append({
                "sample_id": sample.get("id", f"sample_{i}"),
                "response": response,
                "milestone_context": milestone_context,
                "expected_score": expected_score.name if expected_score else None,
                "error": str(e)
            })
    
    # Calculate total time
    total_time = time.time() - start_time
    avg_time = total_time / len(test_data) if test_data else 0
    
    logger.info(f"Benchmark completed in {total_time:.2f} seconds")
    logger.info(f"Average time per sample: {avg_time:.2f} seconds")
    
    return results, avg_time

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy and other metrics from results."""
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    scored = total - errors
    
    # Calculate accuracy
    correct = sum(1 for r in results if r.get("correct") is True)
    accuracy = correct / scored if scored > 0 else 0
    
    # Calculate average confidence
    confidences = [r.get("confidence", 0) for r in results if "confidence" in r]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Calculate domain-specific metrics
    domain_metrics = {}
    for result in results:
        if "error" in result:
            continue
            
        domain = result.get("milestone_context", {}).get("domain", "unknown").lower()
        if domain not in domain_metrics:
            domain_metrics[domain] = {
                "total": 0,
                "correct": 0,
                "confidence": 0
            }
        
        domain_metrics[domain]["total"] += 1
        if result.get("correct"):
            domain_metrics[domain]["correct"] += 1
        domain_metrics[domain]["confidence"] += result.get("confidence", 0)
    
    # Calculate domain accuracies and average confidences
    for domain, metrics in domain_metrics.items():
        metrics["accuracy"] = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
        metrics["avg_confidence"] = metrics["confidence"] / metrics["total"] if metrics["total"] > 0 else 0
    
    return {
        "total_samples": total,
        "errors": errors,
        "scored_samples": scored,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "domain_metrics": domain_metrics
    }

def save_results(
    output_dir: Path,
    standard_results: List[Dict[str, Any]],
    domain_results: List[Dict[str, Any]],
    standard_metrics: Dict[str, Any],
    domain_metrics: Dict[str, Any],
    standard_time: float,
    domain_time: float,
    config: Dict[str, Any]
):
    """Save benchmark results to files."""
    # Save standard results
    with open(output_dir / "standard_results.json", "w") as f:
        json.dump(standard_results, f, indent=2)
    
    # Save domain-specific results
    with open(output_dir / "domain_results.json", "w") as f:
        json.dump(domain_results, f, indent=2)
    
    # Save metrics
    metrics = {
        "standard_prompt": {
            "metrics": standard_metrics,
            "avg_time_per_sample": standard_time
        },
        "domain_prompt": {
            "metrics": domain_metrics,
            "avg_time_per_sample": domain_time
        },
        "improvement": {
            "accuracy": domain_metrics["accuracy"] - standard_metrics["accuracy"],
            "confidence": domain_metrics["avg_confidence"] - standard_metrics["avg_confidence"],
            "time_diff": domain_time - standard_time
        }
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

def generate_report(output_dir: Path, standard_metrics: Dict[str, Any], domain_metrics: Dict[str, Any]):
    """Generate a human-readable report of the benchmark results."""
    report = [
        "# Domain-Specific Prompts Benchmark Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Metric | Standard Prompt | Domain-Specific Prompt | Improvement |",
        "|--------|----------------|------------------------|-------------|",
        f"| Accuracy | {standard_metrics['accuracy']:.2f} | {domain_metrics['accuracy']:.2f} | {domain_metrics['accuracy'] - standard_metrics['accuracy']:.2f} |",
        f"| Confidence | {standard_metrics['avg_confidence']:.2f} | {domain_metrics['avg_confidence']:.2f} | {domain_metrics['avg_confidence'] - standard_metrics['avg_confidence']:.2f} |",
        "",
        "## Domain-Specific Results",
        ""
    ]
    
    # Add domain-specific results
    report.append("### Standard Prompt")
    report.append("")
    report.append("| Domain | Accuracy | Avg. Confidence |")
    report.append("|--------|----------|----------------|")
    
    for domain, metrics in standard_metrics.get("domain_metrics", {}).items():
        report.append(f"| {domain} | {metrics['accuracy']:.2f} | {metrics['avg_confidence']:.2f} |")
    
    report.append("")
    report.append("### Domain-Specific Prompt")
    report.append("")
    report.append("| Domain | Accuracy | Avg. Confidence |")
    report.append("|--------|----------|----------------|")
    
    for domain, metrics in domain_metrics.get("domain_metrics", {}).items():
        report.append(f"| {domain} | {metrics['accuracy']:.2f} | {metrics['avg_confidence']:.2f} |")
    
    # Write report to file
    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))
    
    logger.info(f"Report generated at {output_dir / 'report.md'}")

def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    logger.info(f"Benchmark results will be saved to: {output_dir}")
    
    # Load test data
    test_data = load_test_data(args.dataset, args.sample_size, args.seed)
    
    # Define configurations
    standard_config = {
        "use_domain_specific_prompts": False,
        "n_ctx": 2048,
        "n_batch": 512,
        "n_gpu_layers": 16,
        "n_threads": 4,
        "f16_kv": True
    }
    
    domain_config = {
        "use_domain_specific_prompts": True,
        "custom_templates_dir": "config/prompt_templates",
        "n_ctx": 2048,
        "n_batch": 512,
        "n_gpu_layers": 16,
        "n_threads": 4,
        "f16_kv": True
    }
    
    # Run benchmarks
    standard_results, standard_time = run_benchmark_with_config(
        test_data, standard_config, "standard"
    )
    
    domain_results, domain_time = run_benchmark_with_config(
        test_data, domain_config, "domain-specific"
    )
    
    # Calculate metrics
    standard_metrics = calculate_metrics(standard_results)
    domain_metrics = calculate_metrics(domain_results)
    
    # Log summary
    logger.info("\nBenchmark Summary:")
    logger.info(f"  Standard Prompt Accuracy: {standard_metrics['accuracy']:.2f}")
    logger.info(f"  Domain-Specific Prompt Accuracy: {domain_metrics['accuracy']:.2f}")
    logger.info(f"  Accuracy Improvement: {domain_metrics['accuracy'] - standard_metrics['accuracy']:.2f}")
    logger.info(f"  Standard Prompt Avg. Confidence: {standard_metrics['avg_confidence']:.2f}")
    logger.info(f"  Domain-Specific Prompt Avg. Confidence: {domain_metrics['avg_confidence']:.2f}")
    logger.info(f"  Confidence Improvement: {domain_metrics['avg_confidence'] - standard_metrics['avg_confidence']:.2f}")
    
    # Save results
    save_results(
        output_dir,
        standard_results,
        domain_results,
        standard_metrics,
        domain_metrics,
        standard_time,
        domain_time,
        {
            "standard_config": standard_config,
            "domain_config": domain_config,
            "dataset_version": args.dataset,
            "sample_size": args.sample_size,
            "seed": args.seed
        }
    )
    
    # Generate report
    generate_report(output_dir, standard_metrics, domain_metrics)
    
    logger.info("Benchmark completed successfully")

if __name__ == "__main__":
    main() 