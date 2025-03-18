#!/usr/bin/env python3
"""
R2R Benchmark Runner

This script orchestrates the complete R2R benchmarking process, including:
1. Test data generation
2. Accuracy benchmarks
3. Performance benchmarks
4. Report generation

Usage:
    python3 run_r2r_benchmarks.py [options]

Options:
    --test-data PATH       Path to existing test data (skips generation)
    --skip-test-gen        Skip test data generation
    --skip-accuracy        Skip accuracy benchmarks
    --skip-performance     Skip performance benchmarks
    --num-samples INT      Number of test samples to generate (default: 50)
    --output-dir PATH      Directory for benchmark results (default: data/r2r_benchmark)
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.testing.r2r_benchmark import R2RBenchmark
from src.testing.r2r_test_data_generator import R2RTestDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/r2r_benchmark_runner.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("r2r_benchmark_runner")

def ensure_directory_exists(directory_path: str) -> None:
    """Ensure the specified directory exists."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def run_command(command: List[str]) -> int:
    """Run a command and return its exit code."""
    logger.info(f"Running command: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def generate_test_data(num_samples: int, output_dir: str) -> str:
    """Generate test data for benchmarking."""
    logger.info(f"Generating {num_samples} test samples...")
    
    # Create generator instance
    generator = R2RTestDataGenerator(num_samples=num_samples)
    
    # Generate test data
    test_data = generator.generate_test_data()
    
    # Create timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"r2r_test_data_{timestamp}.json")
    
    # Save test data to file
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Test data saved to {output_path}")
    return output_path

def run_accuracy_benchmark(test_data_path: str, output_dir: str) -> str:
    """Run accuracy benchmark using the specified test data."""
    logger.info(f"Running accuracy benchmark with test data: {test_data_path}")
    
    # Create benchmark instance
    benchmark = R2RBenchmark({
        "output_dir": output_dir,
        "data_dir": os.path.dirname(test_data_path)
    })
    
    # Load test data
    test_data = benchmark.load_test_data(test_data_path)
    
    # Run accuracy benchmark
    result = benchmark.run_accuracy_benchmark(test_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"accuracy_benchmark_{timestamp}"
    benchmark.save_results(result, result_name)
    
    # Visualize results
    benchmark.visualize_accuracy_results(result)
    
    logger.info(f"Accuracy benchmark completed. Results saved to {output_dir}/{result_name}.json")
    return f"{output_dir}/{result_name}.json"

def run_performance_benchmark(test_data_path: str, output_dir: str) -> str:
    """Run performance benchmark using the specified test data."""
    logger.info(f"Running performance benchmark with test data: {test_data_path}")
    
    # Run the performance benchmark script
    command = [
        "python3", 
        "src/testing/r2r_performance_benchmark.py",
        "--test-data", test_data_path,
        "--output-dir", output_dir
    ]
    
    exit_code = run_command(command)
    
    if exit_code != 0:
        logger.error(f"Performance benchmark failed with exit code {exit_code}")
        return None
    
    # Determine the output file path (this assumes the performance benchmark script follows a similar naming convention)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"{output_dir}/performance_benchmark_{timestamp}.json"
    
    logger.info(f"Performance benchmark completed. Results saved to {result_path}")
    return result_path

def generate_report(accuracy_result_path: str, performance_result_path: str, output_dir: str) -> str:
    """Generate a comprehensive HTML report from benchmark results."""
    logger.info("Generating comprehensive benchmark report...")
    
    # Load benchmark results
    accuracy_results = None
    performance_results = None
    
    if accuracy_result_path and os.path.exists(accuracy_result_path):
        with open(accuracy_result_path, 'r') as f:
            accuracy_results = json.load(f)
    
    if performance_result_path and os.path.exists(performance_result_path):
        with open(performance_result_path, 'r') as f:
            performance_results = json.load(f)
    
    # Create timestamp for the report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"r2r_benchmark_report_{timestamp}.html")
    
    # Create a simple HTML report
    with open(report_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>R2R Benchmark Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin-bottom: 30px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; }}
        .chart {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>R2R Benchmark Report</h1>
    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="section summary">
        <h2>Summary</h2>
        <p>This report contains the results of benchmarking the R2R integration.</p>
    </div>
""")

        # Add accuracy results if available
        if accuracy_results:
            f.write(f"""
    <div class="section">
        <h2>Accuracy Benchmark Results</h2>
        <p>Overall accuracy: {accuracy_results.get('overall_accuracy', 'N/A'):.2f}%</p>
        <p>Total test cases: {accuracy_results.get('total_cases', 'N/A')}</p>
        <p>Correct predictions: {accuracy_results.get('correct_predictions', 'N/A')}</p>
        
        <h3>Accuracy by Domain</h3>
        <table>
            <tr>
                <th>Domain</th>
                <th>Accuracy</th>
                <th>Test Cases</th>
            </tr>
""")
            
            # Add domain-specific results
            for domain, stats in accuracy_results.get('domain_accuracy', {}).items():
                f.write(f"""
            <tr>
                <td>{domain}</td>
                <td>{stats.get('accuracy', 0):.2f}%</td>
                <td>{stats.get('total', 0)}</td>
            </tr>""")
            
            f.write("""
        </table>
        
        <h3>Confusion Matrix</h3>
        <p>The confusion matrix shows the distribution of predicted scores versus expected scores.</p>
        <img src="accuracy_confusion_matrix.png" alt="Confusion Matrix" class="chart">
    </div>
""")

        # Add performance results if available
        if performance_results:
            f.write(f"""
    <div class="section">
        <h2>Performance Benchmark Results</h2>
        <p>Average latency: {performance_results.get('average_latency', 'N/A'):.2f} ms</p>
        <p>Throughput: {performance_results.get('throughput', 'N/A'):.2f} requests/second</p>
        <p>Total test cases: {performance_results.get('total_cases', 'N/A')}</p>
        
        <h3>Latency Distribution</h3>
        <table>
            <tr>
                <th>Percentile</th>
                <th>Latency (ms)</th>
            </tr>
            <tr>
                <td>50th (median)</td>
                <td>{performance_results.get('latency_p50', 'N/A'):.2f}</td>
            </tr>
            <tr>
                <td>90th</td>
                <td>{performance_results.get('latency_p90', 'N/A'):.2f}</td>
            </tr>
            <tr>
                <td>95th</td>
                <td>{performance_results.get('latency_p95', 'N/A'):.2f}</td>
            </tr>
            <tr>
                <td>99th</td>
                <td>{performance_results.get('latency_p99', 'N/A'):.2f}</td>
            </tr>
        </table>
        
        <h3>Resource Usage</h3>
        <p>Peak memory usage: {performance_results.get('peak_memory_mb', 'N/A')} MB</p>
        <p>Average CPU usage: {performance_results.get('average_cpu_percent', 'N/A')}%</p>
        
        <h3>Latency Over Time</h3>
        <img src="performance_latency_chart.png" alt="Latency Over Time" class="chart">
    </div>
""")

        # Close the HTML document
        f.write("""
</body>
</html>
""")
    
    logger.info(f"Benchmark report generated: {report_path}")
    return report_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="R2R Benchmark Runner")
    parser.add_argument("--test-data", help="Path to existing test data (skips generation)")
    parser.add_argument("--skip-test-gen", action="store_true", help="Skip test data generation")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy benchmarks")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance benchmarks")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of test samples to generate")
    parser.add_argument("--output-dir", default="data/r2r_benchmark", help="Directory for benchmark results")
    return parser.parse_args()

def main():
    """Main function to run the complete benchmark process."""
    args = parse_args()
    
    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)
    ensure_directory_exists("logs")
    
    # Initialize variables for result paths
    test_data_path = args.test_data
    accuracy_result_path = None
    performance_result_path = None
    
    # Step 1: Generate test data if needed
    if not args.skip_test_gen and not test_data_path:
        test_data_path = generate_test_data(args.num_samples, args.output_dir)
    
    if not test_data_path or not os.path.exists(test_data_path):
        logger.error("No test data available. Please provide a valid test data path or enable test data generation.")
        return 1
    
    # Step 2: Run accuracy benchmark
    if not args.skip_accuracy:
        accuracy_result_path = run_accuracy_benchmark(test_data_path, args.output_dir)
    
    # Step 3: Run performance benchmark
    if not args.skip_performance:
        performance_result_path = run_performance_benchmark(test_data_path, args.output_dir)
    
    # Step 4: Generate comprehensive report
    if accuracy_result_path or performance_result_path:
        report_path = generate_report(accuracy_result_path, performance_result_path, args.output_dir)
        logger.info(f"Benchmark process completed. Report available at: {report_path}")
    else:
        logger.warning("No benchmark results available. Report generation skipped.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 