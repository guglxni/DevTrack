#!/usr/bin/env python3
"""
Benchmark script for LLM performance with Metal GPU acceleration.
This script sends multiple requests to the LLM scoring endpoint and measures response times.
"""

import requests
import json
import time
import statistics
import argparse

def run_benchmark(num_tests=5, show_results=False):
    """Run benchmark tests against the LLM scoring endpoint."""
    
    url = "http://localhost:8003/llm-scoring/direct-test"
    headers = {"Content-Type": "application/json"}
    
    # Define a set of test cases with varying complexity
    test_cases = [
        {
            "question": "Does your child recognize familiar people?",
            "milestone": "Recognizes familiar people",
            "response": "Yes, she recognizes all family members easily"
        },
        {
            "question": "Does your child walk independently?",
            "milestone": "Walks independently",
            "response": "She's just starting to take a few steps on her own but still needs support sometimes."
        },
        {
            "question": "Does your child use words to communicate?",
            "milestone": "Uses words to communicate",
            "response": "No, he doesn't say any words yet, just makes sounds."
        },
        {
            "question": "Can your child stack blocks?",
            "milestone": "Stacks blocks",
            "response": "He used to be able to stack 3-4 blocks but lately he hasn't been interested in doing it."
        },
        {
            "question": "Does your child point to ask for things?",
            "milestone": "Points to ask for something",
            "response": "Yes, but only when I prompt her and show her how to point first."
        }
    ]
    
    print(f"Running {num_tests} benchmark tests for each of {len(test_cases)} test cases...")
    print("-" * 60)
    
    all_times = []
    results_by_case = {}
    
    for i, test_case in enumerate(test_cases):
        print(f"Test Case #{i+1}: {test_case['milestone']}")
        times = []
        
        for j in range(num_tests):
            # Send request and measure time
            start_time = time.time()
            response = requests.post(url, headers=headers, json=test_case)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Record timing
            times.append(elapsed_time)
            all_times.append(elapsed_time)
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                print(f"  Test {j+1}: {elapsed_time:.2f}s - Score: {result['score_label']} ({result['score']})")
                if show_results:
                    print(f"     Reasoning: {result['reasoning'][:100]}...")
            else:
                print(f"  Test {j+1}: {elapsed_time:.2f}s - Error: {response.status_code}")
                
        # Calculate stats for this test case
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        results_by_case[i] = {
            "milestone": test_case['milestone'],
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time
        }
        
        print(f"  Average: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s")
        print("-" * 60)
    
    # Calculate overall stats
    overall_avg = statistics.mean(all_times)
    overall_min = min(all_times)
    overall_max = max(all_times)
    
    print("Summary of results:")
    print(f"Overall average response time: {overall_avg:.2f}s")
    print(f"Overall min response time: {overall_min:.2f}s")
    print(f"Overall max response time: {overall_max:.2f}s")
    
    # Return stats for possible comparison
    return {
        "overall": {
            "avg": overall_avg,
            "min": overall_min,
            "max": overall_max
        },
        "by_case": results_by_case
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM performance")
    parser.add_argument("-n", "--num-tests", type=int, default=3,
                        help="Number of tests to run for each case (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed results including reasoning")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        # Check if LLM is available
        health_response = requests.get("http://localhost:8003/llm-scoring/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"LLM Status: {health_data.get('status', 'Unknown')}")
            print(f"Mode: {health_data.get('mode', 'Unknown')}")
            print(f"Model: {health_data.get('model', 'Unknown')}")
            print("-" * 60)
            
            # Run benchmark
            run_benchmark(num_tests=args.num_tests, show_results=args.verbose)
        else:
            print(f"Error: LLM is not available. Status code: {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")
    
    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60) 