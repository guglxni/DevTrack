#!/usr/bin/env python3
"""
R2R Performance Benchmark

This script measures the performance characteristics of the R2R integration, including:
- Latency (response time)
- Throughput (requests per second)
- Resource usage (CPU, memory)

Usage:
    python3 r2r_performance_benchmark.py [options]

Options:
    --test-data PATH       Path to test data file
    --output-dir PATH      Directory for benchmark results (default: data/r2r_benchmark)
    --num-runs INT         Number of benchmark runs (default: 3)
    --warmup-runs INT      Number of warmup runs (default: 1)
"""

import os
import sys
import json
import time
import psutil
import logging
import argparse
import threading
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.scoring.r2r_enhanced_scorer import R2REnhancedScorer
from src.core.retrieval.r2r_client import R2RClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/r2r_performance_benchmark.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("r2r_performance_benchmark")

class R2RPerformanceBenchmark:
    """Framework for benchmarking the performance of the R2R integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the benchmark with the given configuration."""
        self.config = config or self._default_config()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "latencies": [],
            "cpu_usage": [],
            "memory_usage": [],
            "throughput": 0.0,
            "average_latency": 0.0,
            "latency_p50": 0.0,
            "latency_p90": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0,
            "peak_memory_mb": 0.0,
            "average_cpu_percent": 0.0,
            "total_cases": 0,
            "errors": 0
        }
        self._setup_output_dir()
        
    def _default_config(self) -> Dict[str, Any]:
        """Return the default configuration for the benchmark."""
        return {
            "output_dir": "data/r2r_benchmark",
            "num_runs": 3,
            "warmup_runs": 1,
            "concurrent_requests": 1,
            "timeout_seconds": 60,
            "monitor_interval_seconds": 0.5
        }
    
    def _setup_output_dir(self) -> None:
        """Ensure the output directory exists."""
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self, filepath: str) -> List[Dict[str, Any]]:
        """Load test data from the specified file."""
        logger.info(f"Loading test data from {filepath}")
        with open(filepath, 'r') as f:
            test_data = json.load(f)
        
        logger.info(f"Loaded {len(test_data)} test cases")
        return test_data
    
    def initialize_scorer(self) -> R2REnhancedScorer:
        """Initialize the R2R Enhanced Scorer."""
        logger.info("Initializing R2R Enhanced Scorer...")
        
        # Create R2R client
        client = R2RClient(
            model_name="local",
            temperature=0.0,
            max_tokens=1024
        )
        
        # Create scorer
        scorer = R2REnhancedScorer(client)
        
        logger.info("R2R Enhanced Scorer initialized successfully")
        return scorer
    
    def _monitor_resources(self, stop_event: threading.Event) -> Dict[str, List[float]]:
        """Monitor system resources during the benchmark."""
        process = psutil.Process(os.getpid())
        cpu_usage = []
        memory_usage = []
        
        while not stop_event.is_set():
            # Get CPU usage (percent)
            cpu_percent = process.cpu_percent(interval=0.1)
            cpu_usage.append(cpu_percent)
            
            # Get memory usage (MB)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_usage.append(memory_mb)
            
            # Sleep for the specified interval
            time.sleep(self.config["monitor_interval_seconds"])
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }
    
    def _process_test_case(self, scorer: R2REnhancedScorer, test_case: Dict[str, Any]) -> Tuple[float, Any, Optional[str]]:
        """Process a single test case and return the latency and result."""
        start_time = time.time()
        
        try:
            # Extract test case data
            domain = test_case["domain"]
            milestone = test_case["milestone"]
            caregiver_response = test_case["caregiver_response"]
            
            # Score the response
            result = scorer.score_milestone(
                domain=domain,
                milestone=milestone,
                caregiver_response=caregiver_response
            )
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return latency, result, None
        except Exception as e:
            # Calculate latency even for errors
            latency = (time.time() - start_time) * 1000
            return latency, None, str(e)
    
    def run_performance_test(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the performance benchmark using the specified test data."""
        logger.info("Starting performance benchmark...")
        
        # Initialize the scorer
        scorer = self.initialize_scorer()
        
        # Perform warmup runs
        if self.config["warmup_runs"] > 0:
            logger.info(f"Performing {self.config['warmup_runs']} warmup runs...")
            for i in range(min(self.config["warmup_runs"], len(test_data))):
                self._process_test_case(scorer, test_data[i])
        
        # Start resource monitoring
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(
            target=lambda: self._monitor_resources(stop_monitoring),
            daemon=True
        )
        monitor_thread.start()
        
        # Run the benchmark
        all_latencies = []
        errors = 0
        
        logger.info(f"Running performance benchmark with {len(test_data)} test cases...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config["concurrent_requests"]) as executor:
            futures = [executor.submit(self._process_test_case, scorer, test_case) for test_case in test_data]
            
            for future in futures:
                try:
                    latency, result, error = future.result(timeout=self.config["timeout_seconds"])
                    all_latencies.append(latency)
                    
                    if error:
                        errors += 1
                        logger.warning(f"Error processing test case: {error}")
                except Exception as e:
                    errors += 1
                    logger.error(f"Exception during benchmark: {str(e)}")
        
        # Calculate total execution time and throughput
        total_time = time.time() - start_time
        throughput = len(test_data) / total_time if total_time > 0 else 0
        
        # Stop resource monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Get resource usage data
        resource_data = self._monitor_resources(stop_monitoring)
        
        # Calculate statistics
        if all_latencies:
            average_latency = np.mean(all_latencies)
            latency_p50 = np.percentile(all_latencies, 50)
            latency_p90 = np.percentile(all_latencies, 90)
            latency_p95 = np.percentile(all_latencies, 95)
            latency_p99 = np.percentile(all_latencies, 99)
        else:
            average_latency = latency_p50 = latency_p90 = latency_p95 = latency_p99 = 0
        
        peak_memory_mb = max(resource_data["memory_usage"]) if resource_data["memory_usage"] else 0
        average_cpu_percent = np.mean(resource_data["cpu_usage"]) if resource_data["cpu_usage"] else 0
        
        # Compile results
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "latencies": all_latencies,
            "cpu_usage": resource_data["cpu_usage"],
            "memory_usage": resource_data["memory_usage"],
            "throughput": throughput,
            "average_latency": average_latency,
            "latency_p50": latency_p50,
            "latency_p90": latency_p90,
            "latency_p95": latency_p95,
            "latency_p99": latency_p99,
            "peak_memory_mb": peak_memory_mb,
            "average_cpu_percent": average_cpu_percent,
            "total_cases": len(test_data),
            "errors": errors
        }
        
        logger.info(f"Performance benchmark completed. Processed {len(test_data)} test cases with {errors} errors.")
        logger.info(f"Average latency: {average_latency:.2f} ms, Throughput: {throughput:.2f} requests/second")
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save benchmark results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.config["output_dir"], f"performance_benchmark_{timestamp}.json")
        
        # Create a copy of results without large arrays for JSON serialization
        serializable_results = results.copy()
        serializable_results["latencies"] = results["latencies"][:10]  # Save only first 10 latencies
        serializable_results["cpu_usage"] = results["cpu_usage"][:10]  # Save only first 10 CPU measurements
        serializable_results["memory_usage"] = results["memory_usage"][:10]  # Save only first 10 memory measurements
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Performance benchmark results saved to {output_path}")
        return output_path
    
    def visualize_results(self, results: Dict[str, Any]) -> None:
        """Visualize benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create latency histogram
        plt.figure(figsize=(10, 6))
        plt.hist(results["latencies"], bins=20, alpha=0.7, color='blue')
        plt.axvline(results["average_latency"], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {results["average_latency"]:.2f} ms')
        plt.axvline(results["latency_p95"], color='green', linestyle='dashed', linewidth=2, label=f'95th Percentile: {results["latency_p95"]:.2f} ms')
        plt.title('Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config["output_dir"], f"latency_histogram_{timestamp}.png"))
        
        # Create resource usage over time chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # CPU usage
        time_points = np.arange(len(results["cpu_usage"])) * self.config["monitor_interval_seconds"]
        ax1.plot(time_points, results["cpu_usage"], 'b-')
        ax1.set_title('CPU Usage Over Time')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage
        ax2.plot(time_points, results["memory_usage"], 'g-')
        ax2.set_title('Memory Usage Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["output_dir"], f"resource_usage_{timestamp}.png"))
        
        # Create latency over time chart (for report)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(results["latencies"])), results["latencies"], 'b-')
        plt.axhline(results["average_latency"], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {results["average_latency"]:.2f} ms')
        plt.title('Latency Over Time')
        plt.xlabel('Request Number')
        plt.ylabel('Latency (ms)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.config["output_dir"], "performance_latency_chart.png"))
        
        logger.info(f"Performance visualizations saved to {self.config['output_dir']}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="R2R Performance Benchmark")
    parser.add_argument("--test-data", required=True, help="Path to test data file")
    parser.add_argument("--output-dir", default="data/r2r_benchmark", help="Directory for benchmark results")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--concurrent-requests", type=int, default=1, help="Number of concurrent requests")
    return parser.parse_args()

def main():
    """Main function to run the performance benchmark."""
    args = parse_args()
    
    # Create benchmark configuration
    config = {
        "output_dir": args.output_dir,
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "concurrent_requests": args.concurrent_requests,
        "timeout_seconds": 60,
        "monitor_interval_seconds": 0.5
    }
    
    # Create benchmark instance
    benchmark = R2RPerformanceBenchmark(config)
    
    # Load test data
    test_data = benchmark.load_test_data(args.test_data)
    
    # Run performance benchmark
    results = benchmark.run_performance_test(test_data)
    
    # Save results
    benchmark.save_results(results)
    
    # Visualize results
    benchmark.visualize_results(results)
    
    logger.info("Performance benchmark completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 