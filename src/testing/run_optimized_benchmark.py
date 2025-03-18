#!/usr/bin/env python3
"""
Optimized Benchmark Runner for Developmental Milestone Scoring System

This script runs a targeted benchmark to assess the tiered scoring approach
for speed and accuracy balance.
"""

import sys
import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append('.')

from src.testing.benchmark_framework import AccuracyBenchmark, BenchmarkResult
from src.testing.gold_standard_manager import GoldStandardManager
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
from src.core.scoring.base import Score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/optimized_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optimized_benchmark")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run optimized benchmark for the developmental milestone scoring system"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="latest",
        help="Gold standard dataset version to use (default: latest)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/optimized_benchmark.json",
        help="Path to optimized engine configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/benchmarks/optimized",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of samples to use (0 for all)"
    )
    parser.add_argument(
        "--focus-category",
        type=str,
        choices=["WITH_SUPPORT", "EMERGING", "INDEPENDENT", "LOST_SKILL", "CANNOT_DO", "ALL"],
        default="ALL",
        help="Focus on specific score category (default: ALL)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def load_engine_config(config_path: str) -> Dict[str, Any]:
    """Load engine configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return {}

def setup_output_directory(output_dir: str) -> Path:
    """Set up output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark_dir = output_path / f"benchmark_{timestamp}"
    benchmark_dir.mkdir(exist_ok=True)
    return benchmark_dir

def filter_dataset(data: List[Dict[str, Any]], category: str, sample_size: int) -> List[Dict[str, Any]]:
    """Filter dataset to focus on specific category."""
    if category != "ALL":
        filtered = [item for item in data if item.get("expected_score") == category]
        logger.info(f"Filtered to {len(filtered)} samples for category {category}")
    else:
        filtered = data
    
    if sample_size > 0 and sample_size < len(filtered):
        import random
        random.shuffle(filtered)
        filtered = filtered[:sample_size]
        logger.info(f"Using {len(filtered)} random samples")
    
    return filtered

def run_accuracy_benchmark(
    engine: ImprovedDevelopmentalScoringEngine,
    test_data: List[Dict[str, Any]],
    output_dir: Path
) -> BenchmarkResult:
    """Run accuracy benchmark with performance timing."""
    logger.info(f"Running accuracy benchmark with {len(test_data)} samples...")
    
    benchmark = AccuracyBenchmark({"output_dir": str(output_dir)})
    
    # Run the benchmark
    start_time = time.time()
    result = benchmark.run_accuracy_benchmark(engine, test_data)
    total_time = time.time() - start_time
    
    # Add timing info to metrics
    result.metrics["total_time_seconds"] = total_time
    result.metrics["avg_time_per_sample"] = total_time / len(test_data) if test_data else 0
    
    # Save detailed results
    with open(output_dir / "accuracy_results.json", 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    # Generate visualization
    visualize_results(result, test_data, output_dir)
    
    return result

def run_confusion_matrix(
    engine: ImprovedDevelopmentalScoringEngine,
    test_data: List[Dict[str, Any]],
    output_dir: Path
) -> BenchmarkResult:
    """Generate confusion matrix."""
    logger.info(f"Generating confusion matrix with {len(test_data)} samples...")
    
    benchmark = AccuracyBenchmark({"output_dir": str(output_dir)})
    result = benchmark.run_confusion_matrix(engine, test_data)
    
    # Save detailed results
    with open(output_dir / "confusion_matrix.json", 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    return result

def visualize_results(
    result: BenchmarkResult,
    test_data: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """Create visualizations of the benchmark results."""
    metrics = result.metrics
    
    # Plot 1: Overall Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(['Accuracy'], [metrics['accuracy']], color='blue')
    plt.ylim(0, 1)
    plt.title('Overall Accuracy')
    plt.savefig(output_dir / 'accuracy.png')
    
    # Plot 2: Per-category Accuracy
    categories = [cat for cat in Score.__members__ if cat != 'NOT_RATED']
    category_accuracies = [metrics.get(f"{cat}_accuracy", 0) for cat in categories]
    category_samples = [metrics.get(f"{cat}_samples", 0) for cat in categories]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, category_accuracies, color='green')
    
    # Add sample counts as text
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.02, 
                 f"n={category_samples[i]}", 
                 ha='center')
    
    plt.ylim(0, 1)
    plt.title('Accuracy by Category')
    plt.savefig(output_dir / 'category_accuracy.png')
    
    # Plot 3: Average Time per Sample
    plt.figure(figsize=(10, 6))
    plt.bar(['Avg Time per Sample'], [metrics['avg_time_per_sample']])
    plt.title('Average Processing Time per Sample (seconds)')
    plt.savefig(output_dir / 'avg_time.png')

def main():
    """Main function."""
    args = parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Set up output directory
    output_dir = setup_output_directory(args.output_dir)
    logger.info(f"Benchmark results will be saved to: {output_dir}")
    
    # Load gold standard dataset
    gold_manager = GoldStandardManager()
    
    if args.dataset == "latest":
        logger.info("Loading latest gold standard dataset")
        test_data = gold_manager.load_dataset()
    else:
        logger.info(f"Loading gold standard dataset version: {args.dataset}")
        test_data = gold_manager.load_dataset(args.dataset)
    
    # Filter dataset if needed
    test_data = filter_dataset(test_data, args.focus_category, args.sample_size)
    logger.info(f"Using {len(test_data)} test samples")
    
    # Save test data copy
    with open(output_dir / "test_data.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Load engine configuration
    config_path = args.config
    engine_config = load_engine_config(config_path)
    
    # Save config copy
    with open(output_dir / "engine_config.json", 'w') as f:
        json.dump(engine_config, f, indent=2)
    
    # Initialize scoring engine
    logger.info("Initializing scoring engine with optimized configuration")
    engine = ImprovedDevelopmentalScoringEngine(engine_config)
    
    # Run accuracy benchmark
    logger.info("Running accuracy benchmark")
    accuracy_result = run_accuracy_benchmark(engine, test_data, output_dir)
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info(f"  Accuracy: {accuracy_result.metrics['accuracy']:.2f}")
    logger.info(f"  Samples: {accuracy_result.metrics['samples']}")
    logger.info(f"  Avg. Score Distance: {accuracy_result.metrics['avg_score_distance']:.2f}")
    logger.info(f"  Avg. Time per Sample: {accuracy_result.metrics['avg_time_per_sample']:.2f} seconds")
    logger.info(f"  Total Time: {accuracy_result.metrics['total_time_seconds']:.2f} seconds")
    
    # Per-category breakdown
    logger.info("\nCategory Breakdown:")
    for cat in Score.__members__:
        if cat != "NOT_RATED" and f"{cat}_accuracy" in accuracy_result.metrics:
            samples = accuracy_result.metrics.get(f"{cat}_samples", 0)
            if samples > 0:
                acc = accuracy_result.metrics.get(f"{cat}_accuracy", 0)
                logger.info(f"  {cat}: {acc:.2f} ({samples} samples)")
    
    # Generate confusion matrix
    logger.info("Generating confusion matrix")
    confusion_result = run_confusion_matrix(engine, test_data, output_dir)
    
    logger.info(f"Benchmark completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 