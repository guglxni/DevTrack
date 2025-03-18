#!/usr/bin/env python3
"""
Benchmark Runner for Developmental Milestone Scoring System

This script runs benchmarks to evaluate the scoring system against gold standard datasets.
It integrates with the GoldStandardManager to track performance over time.
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
sys.path.append('.')

from src.testing.benchmark_framework import (
    AccuracyBenchmark, 
    PerformanceBenchmark, 
    ConfigurationBenchmark,
    BenchmarkResult
)
from src.testing.gold_standard_manager import GoldStandardManager
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/benchmark_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark_runner")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for the developmental milestone scoring system"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="latest",
        help="Gold standard dataset version to use (default: latest)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/benchmarks",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        choices=["accuracy", "performance", "config", "all"],
        default="all",
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--engine-config",
        type=str,
        help="Path to custom engine configuration file"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for performance benchmarks"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of benchmark results"
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        help="Previous benchmark result to compare with"
    )
    return parser.parse_args()

def load_engine_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load engine configuration from file or return default config."""
    default_config = {
        "enable_keyword_scorer": True,
        "enable_embedding_scorer": True,
        "enable_transformer_scorer": True,
        "enable_llm_scorer": False,
        "enable_continuous_learning": True,
        "score_weights": {
            "keyword": 0.4,
            "embedding": 0.3,
            "transformer": 0.3
        }
    }
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                return {**default_config, **custom_config}
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
            logger.info("Using default configuration.")
    
    return default_config

def setup_output_directory(output_dir: str) -> Path:
    """Set up the output directory for benchmark results."""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = output_path / f"benchmark_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir

def run_accuracy_benchmarks(
    engine: ImprovedDevelopmentalScoringEngine,
    test_data: List[Dict[str, Any]],
    output_dir: Path,
    visualize: bool = False
) -> Dict[str, BenchmarkResult]:
    """Run accuracy benchmarks."""
    logger.info("Running accuracy benchmarks...")
    
    benchmark = AccuracyBenchmark()
    results = {}
    
    # Run overall accuracy benchmark
    accuracy_result = benchmark.run_accuracy_benchmark(engine, test_data)
    results["accuracy"] = accuracy_result
    
    # Run confusion matrix benchmark
    confusion_result = benchmark.run_confusion_matrix(engine, test_data)
    results["confusion_matrix"] = confusion_result
    
    # Run component comparison benchmark
    component_result = benchmark.run_component_comparison(test_data)
    results["component_comparison"] = component_result
    
    # Save results
    for name, result in results.items():
        result_file = output_dir / f"{name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if visualize:
            viz_file = output_dir / f"{name}_visualization.png"
            benchmark.visualize_results(str(viz_file))
    
    # Generate HTML report
    report_file = output_dir / "accuracy_report.html"
    benchmark.generate_report(str(report_file))
    
    return results

def run_performance_benchmarks(
    engine: ImprovedDevelopmentalScoringEngine,
    test_data: List[Dict[str, Any]],
    output_dir: Path,
    threads: int = 4,
    visualize: bool = False
) -> Dict[str, BenchmarkResult]:
    """Run performance benchmarks."""
    logger.info("Running performance benchmarks...")
    
    benchmark = PerformanceBenchmark()
    results = {}
    
    # Run latency benchmark
    latency_result = benchmark.run_latency_benchmark(engine, test_data)
    results["latency"] = latency_result
    
    # Run throughput benchmark
    throughput_result = benchmark.run_throughput_benchmark(engine, test_data, threads)
    results["throughput"] = throughput_result
    
    # Run memory benchmark
    def engine_factory():
        return ImprovedDevelopmentalScoringEngine(load_engine_config())
    
    memory_result = benchmark.run_memory_benchmark(engine_factory, test_data)
    results["memory"] = memory_result
    
    # Run component benchmarks
    component_results = benchmark.run_component_benchmarks(test_data)
    for i, result in enumerate(component_results):
        results[f"component_{i}"] = result
    
    # Save results
    for name, result in results.items():
        result_file = output_dir / f"{name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if visualize:
            viz_file = output_dir / f"{name}_visualization.png"
            benchmark.visualize_results(str(viz_file))
    
    # Generate HTML report
    report_file = output_dir / "performance_report.html"
    benchmark.generate_report(str(report_file))
    
    return results

def run_configuration_benchmarks(
    test_data: List[Dict[str, Any]],
    output_dir: Path,
    visualize: bool = False
) -> Dict[str, BenchmarkResult]:
    """Run configuration optimization benchmarks."""
    logger.info("Running configuration optimization benchmarks...")
    
    benchmark = ConfigurationBenchmark()
    results = {}
    
    # Define parameter grid
    param_grid = {
        "enable_keyword_scorer": [True, False],
        "enable_embedding_scorer": [True, False],
        "enable_transformer_scorer": [True, False],
        "score_weights": [
            {"keyword": 0.6, "embedding": 0.2, "transformer": 0.2},
            {"keyword": 0.4, "embedding": 0.4, "transformer": 0.2},
            {"keyword": 0.2, "embedding": 0.2, "transformer": 0.6}
        ],
        "keyword_scorer": [
            {"confidence_threshold": 0.6},
            {"confidence_threshold": 0.8}
        ]
    }
    
    # Run configuration optimization
    config_result = benchmark.run_config_optimization(test_data, param_grid)
    results["config_optimization"] = config_result
    
    # Run LLM prompt optimization if available
    try:
        llm_result = benchmark.run_llm_prompt_optimization(test_data)
        results["llm_prompt_optimization"] = llm_result
    except Exception as e:
        logger.warning(f"LLM prompt optimization failed: {e}")
    
    # Save results
    for name, result in results.items():
        result_file = output_dir / f"{name}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        if visualize:
            viz_file = output_dir / f"{name}_visualization.png"
            benchmark.visualize_results(str(viz_file))
    
    # Generate HTML report
    report_file = output_dir / "config_report.html"
    benchmark.generate_report(str(report_file))
    
    return results

def compare_with_previous(
    current_results: Dict[str, BenchmarkResult],
    previous_result_path: str,
    output_dir: Path
) -> Dict[str, Any]:
    """Compare current benchmark results with previous results."""
    logger.info(f"Comparing with previous benchmark: {previous_result_path}")
    
    try:
        with open(previous_result_path, 'r') as f:
            previous_results = json.load(f)
        
        comparison = {}
        
        # Compare accuracy metrics
        if "accuracy" in current_results and "accuracy" in previous_results:
            current_acc = current_results["accuracy"].to_dict()["metrics"]["overall_accuracy"]
            previous_acc = previous_results["accuracy"]["metrics"]["overall_accuracy"]
            
            comparison["accuracy_change"] = current_acc - previous_acc
            comparison["accuracy_percent_change"] = (comparison["accuracy_change"] / previous_acc) * 100
        
        # Compare performance metrics
        if "latency" in current_results and "latency" in previous_results:
            current_latency = current_results["latency"].to_dict()["metrics"]["avg_latency_ms"]
            previous_latency = previous_results["latency"]["metrics"]["avg_latency_ms"]
            
            comparison["latency_change"] = current_latency - previous_latency
            comparison["latency_percent_change"] = (comparison["latency_change"] / previous_latency) * 100
        
        # Save comparison results
        comparison_file = output_dir / "benchmark_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    except Exception as e:
        logger.error(f"Failed to compare with previous benchmark: {e}")
        return {}

def main():
    """Main function to run benchmarks."""
    args = parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
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
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Load engine configuration
    engine_config = load_engine_config(args.engine_config)
    
    # Initialize scoring engine
    engine = ImprovedDevelopmentalScoringEngine(engine_config)
    
    # Run benchmarks
    all_results = {}
    
    if args.benchmark_type in ["accuracy", "all"]:
        accuracy_results = run_accuracy_benchmarks(
            engine, test_data, output_dir, args.visualize
        )
        all_results.update(accuracy_results)
    
    if args.benchmark_type in ["performance", "all"]:
        performance_results = run_performance_benchmarks(
            engine, test_data, output_dir, args.threads, args.visualize
        )
        all_results.update(performance_results)
    
    if args.benchmark_type in ["config", "all"]:
        config_results = run_configuration_benchmarks(
            test_data, output_dir, args.visualize
        )
        all_results.update(config_results)
    
    # Save combined results
    combined_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_version": args.dataset,
            "engine_config": engine_config
        },
        "results": {name: result.to_dict() for name, result in all_results.items()}
    }
    
    combined_file = output_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Compare with previous benchmark if specified
    if args.compare_with:
        comparison = compare_with_previous(all_results, args.compare_with, output_dir)
        
        if comparison:
            logger.info("Benchmark comparison:")
            for key, value in comparison.items():
                logger.info(f"  {key}: {value}")
    
    logger.info(f"Benchmark completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 