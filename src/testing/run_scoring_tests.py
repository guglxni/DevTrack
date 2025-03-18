#!/usr/bin/env python3
"""
Developmental Milestone Scoring System - Test Runner

This script provides a unified interface for running various tests and benchmarks
for the scoring system. It can run unit tests, integration tests, generate test data,
and run performance benchmarks.

Usage:
  python run_scoring_tests.py [command] [options]
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests and benchmarks for the scoring system")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Unit tests command
    unit_parser = subparsers.add_parser("unit", help="Run unit tests")
    unit_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    unit_parser.add_argument("--filter", "-k", help="Filter tests by keyword")
    unit_parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    
    # Integration tests command
    integration_parser = subparsers.add_parser("integration", help="Run integration tests")
    integration_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Generate test data command
    generate_parser = subparsers.add_parser("generate", help="Generate test data")
    generate_parser.add_argument("--output", default="test_data/scoring/benchmark_data.json",
                               help="Output file path")
    generate_parser.add_argument("--count", type=int, default=100, help="Number of test cases to generate")
    generate_parser.add_argument("--domain", default="all", help="Domain to generate data for")
    generate_parser.add_argument("--include-edge-cases", action="store_true", help="Include edge cases")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--type", choices=["performance", "accuracy", "config", "llm-prompts"], 
                                required=True, help="Type of benchmark to run")
    benchmark_parser.add_argument("--data", help="Path to test data file")
    benchmark_parser.add_argument("--output", help="Output filename prefix")
    benchmark_parser.add_argument("--threads", type=int, default=4, help="Number of threads for testing")
    benchmark_parser.add_argument("--memory", action="store_true", help="Include memory usage benchmarks")
    benchmark_parser.add_argument("--confusion", action="store_true", help="Generate confusion matrix")
    benchmark_parser.add_argument("--components", action="store_true", help="Compare scoring components")
    benchmark_parser.add_argument("--params", help="Path to parameter grid JSON file")
    benchmark_parser.add_argument("--include-llm", action="store_true", help="Include LLM-based scorer in benchmarks")
    
    # LLM test command
    llm_parser = subparsers.add_parser("llm", help="Test the LLM-based scorer")
    llm_parser.add_argument("--response", help="Text response to score")
    llm_parser.add_argument("--milestone", help="Milestone behavior description")
    llm_parser.add_argument("--criteria", help="Criteria for the milestone")
    llm_parser.add_argument("--age-range", help="Age range for the milestone")
    llm_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    # All tests command
    all_parser = subparsers.add_parser("all", help="Run all tests and benchmarks")
    all_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    all_parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmarks")
    
    return parser.parse_args()


def run_unit_tests(verbose: bool = False, keyword: Optional[str] = None, 
                   skip_slow: bool = False) -> bool:
    """Run unit tests and return True if all passed."""
    print("Running unit tests...")
    
    # Build pytest command
    cmd = ["pytest", "-xvs" if verbose else "-xqs", "src/testing/test_scoring_framework.py"]
    
    if keyword:
        cmd.extend(["-k", keyword])
    
    if skip_slow:
        cmd.append("-m not slow")
    
    # Execute tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=not verbose)
    duration = time.time() - start_time
    
    # Report results
    if result.returncode == 0:
        print(f"✅ All unit tests passed in {duration:.2f} seconds")
        return True
    else:
        print(f"❌ Unit tests failed in {duration:.2f} seconds")
        if not verbose:
            print("Standard output:")
            print(result.stdout.decode())
            print("Standard error:")
            print(result.stderr.decode())
        return False


def run_integration_tests(verbose: bool = False) -> bool:
    """Run integration tests and return True if all passed."""
    print("Running integration tests...")
    
    # Build pytest command
    cmd = ["pytest", "-xvs" if verbose else "-xqs", "tests/test_integration.py"]
    
    # Execute tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=not verbose)
    duration = time.time() - start_time
    
    # Report results
    if result.returncode == 0:
        print(f"✅ All integration tests passed in {duration:.2f} seconds")
        return True
    else:
        print(f"❌ Integration tests failed in {duration:.2f} seconds")
        if not verbose:
            print("Standard output:")
            print(result.stdout.decode())
            print("Standard error:")
            print(result.stderr.decode())
        return False


def generate_test_data(output: str, count: int, domain: str = "all", 
                     include_edge_cases: bool = False) -> bool:
    """Generate test data and return True if successful."""
    print(f"Generating {count} test data samples...")
    
    # Build command
    cmd = [
        "python", "src/testing/generate_test_data.py",
        "--output", output,
        "--count", str(count),
        "--domain", domain
    ]
    
    if include_edge_cases:
        cmd.append("--include-edge-cases")
    
    # Execute command
    result = subprocess.run(cmd)
    
    # Report results
    if result.returncode == 0:
        print(f"✅ Test data generation completed successfully")
        return True
    else:
        print(f"❌ Test data generation failed")
        return False


def run_benchmark(benchmark_type: str, data_file: Optional[str] = None, 
                output: Optional[str] = None, threads: int = 4, memory: bool = False,
                confusion: bool = False, components: bool = False,
                params_file: Optional[str] = None, include_llm: bool = False) -> bool:
    """Run benchmarks and return True if successful."""
    print(f"Running {benchmark_type} benchmark...")
    
    # Ensure we have test data
    if not data_file:
        default_data_file = "test_data/scoring/benchmark_data.json"
        if not os.path.exists(default_data_file):
            print(f"No test data file provided and default file {default_data_file} not found.")
            print("Generating test data first...")
            if not generate_test_data(default_data_file, 100, include_edge_cases=True):
                return False
        data_file = default_data_file
    
    # Build command
    cmd = [
        "python", "src/testing/benchmark_framework.py",
        benchmark_type,
        "--data", data_file
    ]
    
    if output:
        cmd.extend(["--output", output])
    
    if benchmark_type == "performance":
        cmd.extend(["--threads", str(threads)])
        if memory:
            cmd.append("--memory")
        if include_llm:
            cmd.append("--include-llm")
    
    elif benchmark_type == "accuracy":
        if confusion:
            cmd.append("--confusion")
        if components:
            cmd.append("--components")
        if include_llm:
            cmd.append("--include-llm")
    
    elif benchmark_type == "config":
        if not params_file:
            print("Error: Parameter grid file is required for configuration benchmarks")
            return False
        cmd.extend(["--params", params_file])
        
    elif benchmark_type == "llm-prompts":
        # No additional arguments needed for LLM prompt optimization
        pass
    
    # Execute command
    start_time = time.time()
    result = subprocess.run(cmd)
    duration = time.time() - start_time
    
    # Report results
    if result.returncode == 0:
        print(f"✅ {benchmark_type.capitalize()} benchmark completed in {duration:.2f} seconds")
        return True
    else:
        print(f"❌ {benchmark_type.capitalize()} benchmark failed")
        return False


def run_llm_test(response: str, milestone: str, criteria: Optional[str] = None, 
               age_range: Optional[str] = None, verbose: bool = False) -> bool:
    """Run a test of the LLM-based scorer with specific inputs."""
    print("Testing LLM-based scorer...")
    
    try:
        # Need to import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.core.scoring.llm_scorer import LLMBasedScorer
        
        # Initialize the scorer
        scorer = LLMBasedScorer()
        
        if scorer.model is None:
            print("❌ LLM model could not be initialized. Check logs for details.")
            return False
        
        # Create milestone context
        milestone_context = {
            "behavior": milestone,
            "criteria": criteria or f"Child can {milestone.lower()}",
            "age_range": age_range or "18-36 months"
        }
        
        # Score the response
        print(f"Scoring response: \"{response}\"")
        print(f"Milestone: {milestone_context['behavior']}")
        
        start_time = time.time()
        result = scorer.score(response, milestone_context)
        duration = time.time() - start_time
        
        # Print results
        print("\nResults:")
        print(f"Score: {result.score.name} ({result.score.value})")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.method}")
        print(f"Time taken: {duration:.2f} seconds")
        
        if verbose:
            print("\nReasoning:")
            print(result.reasoning)
            
            if result.details:
                print("\nDetails:")
                for key, value in result.details.items():
                    if key == "full_response" and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LLM-based scorer: {str(e)}")
        return False


def run_all_tests(verbose: bool = False, skip_benchmarks: bool = False) -> bool:
    """Run all tests and benchmarks, return True if all passed."""
    success = True
    
    # Run unit tests
    if not run_unit_tests(verbose):
        success = False
    
    # Run integration tests
    if not run_integration_tests(verbose):
        success = False
    
    # Run benchmarks if not skipped
    if not skip_benchmarks:
        # Generate test data
        test_data_file = "test_data/scoring/benchmark_data.json"
        if not generate_test_data(test_data_file, 100, include_edge_cases=True):
            success = False
        
        # Performance benchmark
        if not run_benchmark("performance", data_file=test_data_file):
            success = False
        
        # Accuracy benchmark
        if not run_benchmark("accuracy", data_file=test_data_file, confusion=True, components=True):
            success = False
    
    return success


def create_default_param_grid():
    """Create a default parameter grid for configuration benchmarks."""
    import json
    
    param_grid = {
        "enable_keyword_scorer": [True, False],
        "enable_embedding_scorer": [True, False],
        "score_weights": [
            {"keyword": 0.7, "embedding": 0.3},
            {"keyword": 0.5, "embedding": 0.5},
            {"keyword": 0.3, "embedding": 0.7}
        ],
        "keyword_scorer": [
            {"confidence_threshold": 0.6},
            {"confidence_threshold": 0.7},
            {"confidence_threshold": 0.8}
        ]
    }
    
    params_file = "test_data/scoring/param_grid.json"
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    
    with open(params_file, 'w') as f:
        json.dump(param_grid, f, indent=2)
    
    return params_file


def main():
    """Main function to run tests based on command line arguments."""
    args = parse_args()
    
    if args.command == "unit":
        success = run_unit_tests(args.verbose, args.filter, args.skip_slow)
    
    elif args.command == "integration":
        success = run_integration_tests(args.verbose)
    
    elif args.command == "generate":
        success = generate_test_data(args.output, args.count, args.domain, args.include_edge_cases)
    
    elif args.command == "benchmark":
        if args.type == "config" and not args.params:
            print("Creating default parameter grid file...")
            args.params = create_default_param_grid()
        
        success = run_benchmark(
            args.type, args.data, args.output, args.threads, args.memory,
            args.confusion, args.components, args.params, args.include_llm
        )
    
    elif args.command == "llm":
        if not args.response or not args.milestone:
            print("Error: --response and --milestone arguments are required")
            return 1
            
        success = run_llm_test(
            args.response, args.milestone, 
            args.criteria, args.age_range, args.verbose
        )
        
    elif args.command == "all":
        success = run_all_tests(args.verbose, args.skip_benchmarks)
    
    else:
        print("No command specified. Use --help for usage information.")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 