#!/usr/bin/env python3
"""
Developmental Milestone Scoring System - Benchmarking Framework

This module provides comprehensive benchmarking tools for the milestone scoring system,
allowing systematic evaluation of performance, accuracy, and reliability across
different scoring components and configurations.

Features:
- Performance benchmarking (throughput, latency)
- Accuracy benchmarking against gold standard datasets
- Component-specific benchmarking
- Configuration optimization
- Visualization of benchmark results
"""

import os
import sys
import json
import time
import statistics
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging

# Import scoring system
from src.core.scoring.base import Score, BaseScorer, ScoringResult
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
from src.core.scoring.keyword_scorer import KeywordBasedScorer
from src.core.scoring.embedding_scorer import SemanticEmbeddingScorer
from src.core.scoring.transformer_scorer import TransformerBasedScorer
from src.core.scoring.llm_scorer import LLMBasedScorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark")

class BenchmarkResult:
    """Data class to store benchmark results"""
    def __init__(self, 
                 name: str, 
                 metrics: Dict[str, Any],
                 details: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metrics = metrics
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary representation"""
        result = cls(data['name'], data['metrics'], data['details'])
        result.timestamp = datetime.fromisoformat(data['timestamp'])
        return result


class ScoringBenchmark:
    """Base class for scoring benchmarks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the benchmark with the given configuration."""
        self.config = config or self._default_config()
        self.results: List[BenchmarkResult] = []
        self._setup_output_dir()
        
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "output_dir": "test_results/benchmarks",
            "random_seed": 42,
            "num_samples": 100,
            "batch_size": 10,
            "num_runs": 3,
            "timeout": 30  # seconds
        }
    
    def _setup_output_dir(self) -> None:
        """Set up the output directory."""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self, filepath: str) -> List[Dict[str, Any]]:
        """Load test data from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to file."""
        output_path = Path(self.config["output_dir"]) / filename
        results_data = [r.to_dict() for r in self.results]
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Saved benchmark results to {output_path}")
    
    def generate_report(self, filename: str) -> None:
        """Generate a report from benchmark results."""
        report_path = Path(self.config["output_dir"]) / filename
        
        # Convert results to DataFrame for easier analysis
        results_data = []
        for result in self.results:
            data = {"name": result.name, "timestamp": result.timestamp}
            data.update(result.metrics)
            results_data.append(data)
        
        df = pd.DataFrame(results_data)
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>Scoring Benchmark Report</title>
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
            <h1>Scoring Benchmark Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Benchmark Results</h2>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Timestamp</th>
                    {' '.join([f'<th>{col}</th>' for col in df.columns if col not in ['name', 'timestamp']])}
                </tr>
                {''.join([
                    f'<tr><td>{row["name"]}</td><td>{row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}</td>' + 
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
    
    def visualize_results(self, filename: str) -> None:
        """Create visualizations of benchmark results."""
        output_path = Path(self.config["output_dir"]) / filename
        
        # Convert results to DataFrame
        results_data = []
        for result in self.results:
            data = {"name": result.name}
            data.update(result.metrics)
            results_data.append(data)
        
        df = pd.DataFrame(results_data)
        
        # Create a set of visualizations
        metrics = [col for col in df.columns if col != 'name']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            df.plot(x='name', y=metric, kind='bar', ax=axes[i], title=f'{metric} by Benchmark')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Saved benchmark visualizations to {output_path}")


class PerformanceBenchmark(ScoringBenchmark):
    """Benchmark for performance metrics (throughput, latency)"""
    
    def run_latency_benchmark(self, engine: ImprovedDevelopmentalScoringEngine, 
                              test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Run latency benchmark on the scoring engine."""
        logger.info(f"Running latency benchmark with {len(test_data)} samples...")
        
        latencies = []
        for sample in tqdm(test_data):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            
            start_time = time.time()
            engine.score_response(response, milestone_context)
            latency = time.time() - start_time
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p50_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        metrics = {
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        details = {
            "all_latencies": latencies,
            "sample_count": len(test_data)
        }
        
        return BenchmarkResult("latency_benchmark", metrics, details)
    
    def run_throughput_benchmark(self, engine: ImprovedDevelopmentalScoringEngine, 
                                test_data: List[Dict[str, Any]],
                                num_threads: int = 4) -> BenchmarkResult:
        """Run throughput benchmark on the scoring engine."""
        logger.info(f"Running throughput benchmark with {len(test_data)} samples and {num_threads} threads...")
        
        # Define worker function
        def score_sample(sample):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            start_time = time.time()
            engine.score_response(response, milestone_context)
            return time.time() - start_time
        
        throughputs = []
        batch_sizes = [1, 5, 10, 20, 50]  # Different batch sizes to test
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Prepare batches
            batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
            batch_times = []
            
            for batch in tqdm(batches):
                batch_start = time.time()
                
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    executor.map(score_sample, batch)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
            
            avg_batch_time = statistics.mean(batch_times)
            throughput = batch_size / avg_batch_time
            throughputs.append((batch_size, throughput))
        
        metrics = {
            f"throughput_batch_{batch_size}": throughput
            for batch_size, throughput in throughputs
        }
        
        metrics["max_throughput"] = max(t for _, t in throughputs)
        metrics["optimal_batch_size"] = next(bs for bs, t in throughputs if t == metrics["max_throughput"])
        
        details = {
            "throughputs_by_batch": dict(throughputs),
            "num_threads": num_threads,
            "sample_count": len(test_data)
        }
        
        return BenchmarkResult("throughput_benchmark", metrics, details)
    
    def run_memory_benchmark(self, engine_factory: Callable[[], ImprovedDevelopmentalScoringEngine], 
                            test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Run memory usage benchmark."""
        try:
            import psutil
            import gc
        except ImportError:
            logger.error("psutil is required for memory benchmarking.")
            return BenchmarkResult("memory_benchmark", {"error": "psutil not available"})
        
        logger.info("Running memory usage benchmark...")
        
        # Force garbage collection before starting
        gc.collect()
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a new engine instance
        engine = engine_factory()
        
        # Get memory after engine initialization
        init_memory = process.memory_info().rss / 1024 / 1024  # MB
        engine_memory = init_memory - baseline_memory
        
        # Run scoring on test data
        for sample in tqdm(test_data):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            engine.score_response(response, milestone_context)
        
        # Get memory after processing
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        processing_memory = peak_memory - init_memory
        
        metrics = {
            "baseline_memory_mb": baseline_memory,
            "engine_memory_mb": engine_memory,
            "processing_memory_mb": processing_memory,
            "peak_memory_mb": peak_memory,
            "total_memory_increase_mb": peak_memory - baseline_memory
        }
        
        return BenchmarkResult("memory_benchmark", metrics)

    def run_component_benchmarks(self, test_data: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """Run benchmarks on individual scoring components."""
        results = []
        
        # Benchmark KeywordBasedScorer
        logger.info("Benchmarking KeywordBasedScorer...")
        keyword_scorer = KeywordBasedScorer()
        keyword_times = []
        
        for sample in tqdm(test_data):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            
            start_time = time.time()
            keyword_scorer.score(response, milestone_context)
            keyword_times.append(time.time() - start_time)
        
        results.append(BenchmarkResult(
            "keyword_scorer_benchmark",
            {
                "avg_latency": statistics.mean(keyword_times),
                "p95_latency": np.percentile(keyword_times, 95),
                "max_latency": max(keyword_times)
            }
        ))
        
        # Benchmark SemanticEmbeddingScorer if available
        try:
            logger.info("Benchmarking SemanticEmbeddingScorer...")
            embedding_scorer = SemanticEmbeddingScorer()
            embedding_times = []
            
            for sample in tqdm(test_data):
                response = sample.get("response", "")
                milestone_context = sample.get("milestone_context", {})
                
                start_time = time.time()
                embedding_scorer.score(response, milestone_context)
                embedding_times.append(time.time() - start_time)
            
            results.append(BenchmarkResult(
                "embedding_scorer_benchmark",
                {
                    "avg_latency": statistics.mean(embedding_times),
                    "p95_latency": np.percentile(embedding_times, 95),
                    "max_latency": max(embedding_times)
                }
            ))
        except Exception as e:
            logger.warning(f"Could not benchmark SemanticEmbeddingScorer: {str(e)}")
        
        # Benchmark TransformerBasedScorer if available
        try:
            logger.info("Benchmarking TransformerBasedScorer...")
            transformer_scorer = TransformerBasedScorer()
            transformer_times = []
            
            for sample in tqdm(test_data[:10]):  # Use fewer samples due to potential slowness
                response = sample.get("response", "")
                milestone_context = sample.get("milestone_context", {})
                
                start_time = time.time()
                transformer_scorer.score(response, milestone_context)
                transformer_times.append(time.time() - start_time)
            
            results.append(BenchmarkResult(
                "transformer_scorer_benchmark",
                {
                    "avg_latency": statistics.mean(transformer_times),
                    "p95_latency": np.percentile(transformer_times, 95),
                    "max_latency": max(transformer_times)
                }
            ))
        except Exception as e:
            logger.warning(f"Could not benchmark TransformerBasedScorer: {str(e)}")
            
        # Benchmark LLMBasedScorer if available
        try:
            logger.info("Benchmarking LLMBasedScorer...")
            llm_scorer = LLMBasedScorer()
            
            if llm_scorer.model is not None:
                llm_times = []
                
                # Use a smaller subset due to LLM being slower
                llm_samples = test_data[:5]
                
                for sample in tqdm(llm_samples):
                    response = sample.get("response", "")
                    milestone_context = sample.get("milestone_context", {})
                    
                    start_time = time.time()
                    llm_scorer.score(response, milestone_context)
                    llm_times.append(time.time() - start_time)
                
                results.append(BenchmarkResult(
                    "llm_scorer_benchmark",
                    {
                        "avg_latency": statistics.mean(llm_times),
                        "p95_latency": np.percentile(llm_times, 95) if len(llm_times) > 1 else max(llm_times),
                        "max_latency": max(llm_times),
                        "samples": len(llm_samples)
                    }
                ))
            else:
                logger.warning("LLMBasedScorer model could not be initialized")
        except Exception as e:
            logger.warning(f"Could not benchmark LLMBasedScorer: {str(e)}")
        
        return results


class AccuracyBenchmark(ScoringBenchmark):
    """Benchmark for accuracy and reliability metrics"""
    
    def run_accuracy_benchmark(self, engine: ImprovedDevelopmentalScoringEngine, 
                              test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Run accuracy benchmark against gold standard data."""
        logger.info(f"Running accuracy benchmark with {len(test_data)} samples...")
        
        correct = 0
        confidence_sum = 0
        score_distances = []
        
        # Score categories for calculating distance
        score_values = {
            Score.CANNOT_DO.value: 0,
            Score.LOST_SKILL.value: 1,
            Score.EMERGING.value: 2,
            Score.WITH_SUPPORT.value: 3,
            Score.INDEPENDENT.value: 4
        }
        
        # Track performance by category
        category_metrics = {score: {"correct": 0, "total": 0} for score in Score.__members__}
        
        for sample in tqdm(test_data):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            expected_score = sample.get("expected_score")
            
            if expected_score is None:
                logger.warning(f"Skipping sample without expected_score: {sample}")
                continue
            
            # Get expected score value
            if isinstance(expected_score, str):
                expected_score_value = next((s.value for s in Score if s.name == expected_score), None)
                if expected_score_value is None:
                    logger.warning(f"Unknown expected score: {expected_score}, skipping")
                    continue
            else:
                expected_score_value = expected_score
            
            # Run scoring
            result = engine.score_response(response, milestone_context, detailed=True)
            
            # Get the predicted score
            predicted_score_value = result.score.value if hasattr(result, 'score') else result.get('score_value')
            confidence = result.confidence if hasattr(result, 'confidence') else result.get('confidence', 0)
            
            # Track metrics
            is_correct = predicted_score_value == expected_score_value
            if is_correct:
                correct += 1
            
            confidence_sum += confidence
            
            # Calculate score distance (how far off the prediction was)
            if predicted_score_value in score_values and expected_score_value in score_values:
                distance = abs(score_values[predicted_score_value] - score_values[expected_score_value])
                score_distances.append(distance)
            
            # Update category metrics
            expected_score_name = next((s.name for s in Score if s.value == expected_score_value), "UNKNOWN")
            if expected_score_name in category_metrics:
                category_metrics[expected_score_name]["total"] += 1
                if is_correct:
                    category_metrics[expected_score_name]["correct"] += 1
        
        # Calculate overall metrics
        accuracy = correct / len(test_data) if len(test_data) > 0 else 0
        avg_confidence = confidence_sum / len(test_data) if len(test_data) > 0 else 0
        avg_score_distance = statistics.mean(score_distances) if score_distances else 0
        
        # Calculate per-category accuracy
        category_accuracy = {}
        for category, data in category_metrics.items():
            if data["total"] > 0:
                category_accuracy[f"{category}_accuracy"] = data["correct"] / data["total"]
                category_accuracy[f"{category}_samples"] = data["total"]
        
        metrics = {
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_score_distance": avg_score_distance,
            "samples": len(test_data),
            **category_accuracy
        }
        
        return BenchmarkResult("accuracy_benchmark", metrics)
    
    def run_confusion_matrix(self, engine: ImprovedDevelopmentalScoringEngine, 
                           test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Generate confusion matrix for scoring results."""
        logger.info(f"Generating confusion matrix with {len(test_data)} samples...")
        
        # Initialize confusion matrix
        score_names = [s.name for s in Score if s.name != 'NOT_RATED']
        matrix = {expected: {predicted: 0 for predicted in score_names} for expected in score_names}
        
        for sample in tqdm(test_data):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            expected_score = sample.get("expected_score")
            
            if expected_score is None:
                continue
            
            # Get expected score name
            if isinstance(expected_score, int):
                expected_score_name = next((s.name for s in Score if s.value == expected_score), None)
            else:
                expected_score_name = expected_score
            
            if expected_score_name not in score_names:
                continue
            
            # Run scoring
            result = engine.score_response(response, milestone_context, detailed=True)
            
            # Get the predicted score
            if hasattr(result, 'score'):
                predicted_score_name = result.score.name
            else:
                predicted_score_value = result.get('score_value')
                predicted_score_name = next((s.name for s in Score if s.value == predicted_score_value), None)
            
            if predicted_score_name not in score_names:
                continue
            
            # Update confusion matrix
            matrix[expected_score_name][predicted_score_name] += 1
        
        metrics = {
            "confusion_matrix": matrix,
            "samples": len(test_data)
        }
        
        # Calculate precision, recall, F1 for each category
        precision_recall = {}
        for category in score_names:
            # Calculate true positives, false positives, and false negatives
            true_positives = matrix[category][category]
            false_positives = sum(matrix[other][category] for other in score_names if other != category)
            false_negatives = sum(matrix[category][other] for other in score_names if other != category)
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_recall[f"{category}_precision"] = precision
            precision_recall[f"{category}_recall"] = recall
            precision_recall[f"{category}_f1"] = f1
        
        metrics.update(precision_recall)
        
        return BenchmarkResult("confusion_matrix", metrics)
    
    def run_component_comparison(self, test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Compare performance of individual scoring components."""
        logger.info(f"Running component comparison with {len(test_data)} samples...")
        
        # Initialize scorers
        scorers = {}
        try:
            scorers["keyword"] = KeywordBasedScorer()
            scorers["embedding"] = SemanticEmbeddingScorer()
            scorers["transformer"] = TransformerBasedScorer()
            
            # Add LLM-based scorer if available
            llm_scorer = LLMBasedScorer()
            if llm_scorer.model is not None:
                scorers["llm"] = llm_scorer
                logger.info("LLM-based scorer included in component comparison")
            else:
                logger.warning("LLM-based scorer not available for component comparison")
                
        except Exception as e:
            logger.warning(f"Could not initialize all scorers: {str(e)}")
        
        # Track performance for each component
        component_metrics = {name: {"correct": 0, "incorrect": 0} for name in scorers.keys()}
        
        for sample in tqdm(test_data):
            response = sample.get("response", "")
            milestone_context = sample.get("milestone_context", {})
            expected_score = sample.get("expected_score")
            
            if expected_score is None:
                continue
            
            # Get expected score value
            if isinstance(expected_score, str):
                expected_score_value = next((s.value for s in Score if s.name == expected_score), None)
            else:
                expected_score_value = expected_score
            
            if expected_score_value is None:
                continue
            
            # Score with each component
            for name, scorer in scorers.items():
                try:
                    # For LLM scorer, we'll use a smaller subset of data
                    if name == "llm" and scorers[name].model is not None:
                        # Only process every 10th sample for LLM to save time
                        if test_data.index(sample) % 10 != 0:
                            continue
                    
                    result = scorer.score(response, milestone_context)
                    is_correct = result.score.value == expected_score_value
                    
                    if is_correct:
                        component_metrics[name]["correct"] += 1
                    else:
                        component_metrics[name]["incorrect"] += 1
                except Exception as e:
                    logger.warning(f"Error scoring with {name}: {str(e)}")
        
        # Calculate metrics for each component
        metrics = {}
        for name, data in component_metrics.items():
            total = data["correct"] + data["incorrect"]
            if total > 0:
                metrics[f"{name}_accuracy"] = data["correct"] / total
                metrics[f"{name}_samples"] = total
        
        return BenchmarkResult("component_comparison", metrics)


class ConfigurationBenchmark(ScoringBenchmark):
    """Benchmark for testing different configurations"""
    
    def run_config_optimization(self, 
                              test_data: List[Dict[str, Any]], 
                              param_grid: Dict[str, List[Any]]) -> BenchmarkResult:
        """Run benchmark with different configurations to find optimal settings."""
        logger.info(f"Running configuration optimization with {len(param_grid)} parameters...")
        
        # Generate all combinations of parameters
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        best_accuracy = 0
        best_config = None
        
        for combination in tqdm(combinations):
            # Create configuration
            config = {name: value for name, value in zip(param_names, combination)}
            logger.info(f"Testing configuration: {config}")
            
            # Create engine with this configuration
            engine = ImprovedDevelopmentalScoringEngine(config)
            
            # Benchmark with this configuration
            accuracy_benchmark = AccuracyBenchmark()
            accuracy_result = accuracy_benchmark.run_accuracy_benchmark(engine, test_data)
            accuracy = accuracy_result.metrics.get("accuracy", 0)
            
            results.append({
                "config": config,
                "accuracy": accuracy,
                **accuracy_result.metrics
            })
            
            # Track best configuration
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
        
        metrics = {
            "best_accuracy": best_accuracy,
            "configurations_tested": len(combinations)
        }
        
        details = {
            "best_config": best_config,
            "all_results": results
        }
        
        return BenchmarkResult("config_optimization", metrics, details)
        
    def run_llm_prompt_optimization(self, test_data: List[Dict[str, Any]]) -> BenchmarkResult:
        """Benchmark different prompt templates for the LLM-based scorer."""
        logger.info("Running LLM prompt optimization...")
        
        try:
            # Check if LLM is available
            base_scorer = LLMBasedScorer()
            if base_scorer.model is None:
                logger.error("LLM model not available for prompt optimization")
                return BenchmarkResult(
                    "llm_prompt_optimization",
                    {"error": "LLM model not available"}
                )
                
            # Define prompt variations to test
            prompt_templates = {
                "basic": """<s>[INST] Analyze this developmental milestone response and determine the appropriate score:
Milestone: {behavior}
Response: "{response}"
Score as: CANNOT_DO, LOST_SKILL, EMERGING, WITH_SUPPORT, or INDEPENDENT
[/INST]""",
                
                "detailed": base_scorer._get_default_prompt_template(),
                
                "structured": """<s>[INST] You are assessing a child's developmental milestone.
MILESTONE: {behavior}
CRITERIA: {criteria}
AGE RANGE: {age_range}
RESPONSE: "{response}"

Score categories:
- CANNOT_DO: Child cannot do this at all
- LOST_SKILL: Child could do this before but lost the ability
- EMERGING: Child shows beginning signs of this skill
- WITH_SUPPORT: Child can do this with help
- INDEPENDENT: Child can do this without assistance

Provide your assessment in this format:
SCORE: [category]
CONFIDENCE: [0-1]
REASONING: [explanation]
[/INST]"""
            }
            
            # Test a small subset of the data
            test_subset = test_data[:10]
            
            prompt_results = {}
            
            for prompt_name, prompt_template in prompt_templates.items():
                logger.info(f"Testing prompt template: {prompt_name}")
                
                # Configure scorer with this prompt
                prompt_config = base_scorer.config.copy()
                prompt_config["prompt_template"] = prompt_template
                scorer = LLMBasedScorer(prompt_config)
                
                if scorer.model is None:
                    logger.error("Failed to initialize LLM model")
                    continue
                
                correct = 0
                total = 0
                generation_times = []
                
                for sample in tqdm(test_subset):
                    response = sample.get("response", "")
                    milestone_context = sample.get("milestone_context", {})
                    expected_score = sample.get("expected_score")
                    
                    if expected_score is None:
                        continue
                        
                    # Get expected score value
                    if isinstance(expected_score, str):
                        expected_score_value = next((s.value for s in Score if s.name == expected_score), None)
                    else:
                        expected_score_value = expected_score
                    
                    if expected_score_value is None:
                        continue
                    
                    # Score with this prompt
                    start_time = time.time()
                    result = scorer.score(response, milestone_context)
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                    
                    # Check correctness
                    is_correct = result.score.value == expected_score_value
                    if is_correct:
                        correct += 1
                    total += 1
                
                # Calculate metrics for this prompt
                if total > 0:
                    accuracy = correct / total
                    avg_generation_time = statistics.mean(generation_times)
                    prompt_results[prompt_name] = {
                        "accuracy": accuracy,
                        "avg_generation_time": avg_generation_time,
                        "samples": total
                    }
            
            # Determine best prompt
            best_prompt = None
            best_accuracy = 0
            
            for prompt_name, metrics in prompt_results.items():
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_prompt = prompt_name
            
            return BenchmarkResult(
                "llm_prompt_optimization",
                {
                    "best_prompt": best_prompt,
                    "best_accuracy": best_accuracy
                },
                {
                    "prompt_results": prompt_results
                }
            )
                
        except Exception as e:
            logger.error(f"Error in LLM prompt optimization: {str(e)}")
            return BenchmarkResult(
                "llm_prompt_optimization",
                {"error": str(e)}
            )


def main():
    """Main entry point for benchmark command line tool."""
    parser = argparse.ArgumentParser(description="Developmental Milestone Scoring Benchmark Tool")
    subparsers = parser.add_subparsers(dest="command", help="Benchmark command")
    
    # Performance benchmark command
    perf_parser = subparsers.add_parser("performance", help="Run performance benchmark")
    perf_parser.add_argument("--data", required=True, help="Path to test data file")
    perf_parser.add_argument("--output", default="performance_benchmark", help="Output filename prefix")
    perf_parser.add_argument("--threads", type=int, default=4, help="Number of threads for throughput testing")
    perf_parser.add_argument("--memory", action="store_true", help="Run memory usage benchmark")
    perf_parser.add_argument("--include-llm", action="store_true", help="Include LLM-based scorer in benchmarks")
    
    # Accuracy benchmark command
    acc_parser = subparsers.add_parser("accuracy", help="Run accuracy benchmark")
    acc_parser.add_argument("--data", required=True, help="Path to gold standard test data file")
    acc_parser.add_argument("--output", default="accuracy_benchmark", help="Output filename prefix")
    acc_parser.add_argument("--confusion", action="store_true", help="Generate confusion matrix")
    acc_parser.add_argument("--components", action="store_true", help="Compare scoring components")
    acc_parser.add_argument("--include-llm", action="store_true", help="Include LLM-based scorer in benchmarks")
    
    # Configuration benchmark command
    config_parser = subparsers.add_parser("config", help="Run configuration benchmark")
    config_parser.add_argument("--data", required=True, help="Path to test data file")
    config_parser.add_argument("--params", required=True, help="Path to parameter grid JSON file")
    config_parser.add_argument("--output", default="config_benchmark", help="Output filename prefix")
    
    # LLM prompt optimization command
    llm_parser = subparsers.add_parser("llm-prompts", help="Optimize LLM prompts")
    llm_parser.add_argument("--data", required=True, help="Path to test data file")
    llm_parser.add_argument("--output", default="llm_prompt_benchmark", help="Output filename prefix")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "performance":
        # Run performance benchmark
        benchmark = PerformanceBenchmark()
        test_data = benchmark.load_test_data(args.data)
        engine = ImprovedDevelopmentalScoringEngine()
        
        # Run latency benchmark
        latency_result = benchmark.run_latency_benchmark(engine, test_data)
        benchmark.results.append(latency_result)
        
        # Run throughput benchmark
        throughput_result = benchmark.run_throughput_benchmark(engine, test_data, args.threads)
        benchmark.results.append(throughput_result)
        
        # Run memory benchmark if requested
        if args.memory:
            memory_result = benchmark.run_memory_benchmark(
                lambda: ImprovedDevelopmentalScoringEngine(), test_data)
            benchmark.results.append(memory_result)
        
        # Run component benchmarks
        component_results = benchmark.run_component_benchmarks(test_data)
        benchmark.results.extend(component_results)
        
        # Save results and generate report
        benchmark.save_results(f"{args.output}.json")
        benchmark.generate_report(f"{args.output}.html")
        benchmark.visualize_results(f"{args.output}.png")
        
    elif args.command == "accuracy":
        # Run accuracy benchmark
        benchmark = AccuracyBenchmark()
        test_data = benchmark.load_test_data(args.data)
        engine = ImprovedDevelopmentalScoringEngine()
        
        # Run accuracy benchmark
        accuracy_result = benchmark.run_accuracy_benchmark(engine, test_data)
        benchmark.results.append(accuracy_result)
        
        # Run confusion matrix if requested
        if args.confusion:
            confusion_result = benchmark.run_confusion_matrix(engine, test_data)
            benchmark.results.append(confusion_result)
        
        # Run component comparison if requested
        if args.components:
            component_result = benchmark.run_component_comparison(test_data)
            benchmark.results.append(component_result)
        
        # Save results and generate report
        benchmark.save_results(f"{args.output}.json")
        benchmark.generate_report(f"{args.output}.html")
        benchmark.visualize_results(f"{args.output}.png")
        
    elif args.command == "config":
        # Run configuration benchmark
        benchmark = ConfigurationBenchmark()
        test_data = benchmark.load_test_data(args.data)
        
        # Load parameter grid
        with open(args.params, 'r') as f:
            param_grid = json.load(f)
        
        # Run config optimization
        config_result = benchmark.run_config_optimization(test_data, param_grid)
        benchmark.results.append(config_result)
        
        # Save results and generate report
        benchmark.save_results(f"{args.output}.json")
        benchmark.generate_report(f"{args.output}.html")
        
    elif args.command == "llm-prompts":
        # Run LLM prompt optimization
        benchmark = ConfigurationBenchmark()
        test_data = benchmark.load_test_data(args.data)
        
        # Run prompt optimization
        prompt_result = benchmark.run_llm_prompt_optimization(test_data)
        benchmark.results.append(prompt_result)
        
        # Save results and generate report
        benchmark.save_results(f"{args.output}.json")
        benchmark.generate_report(f"{args.output}.html")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 