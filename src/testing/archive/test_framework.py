#!/usr/bin/env python3
"""
ASD Developmental Milestone Assessment API - Testing Framework

This script provides a comprehensive framework for testing the ASD Assessment API.
It can generate configurable test data, run automated tests, and provide detailed reports.

Features:
- Automated API endpoint testing
- Random test data generation with configurable parameters
- Edge case testing
- Performance testing
- Detailed test reports
"""

import os
import sys
import json
import asyncio
import random
import time
import statistics
import argparse
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to import the EnhancedAssessmentEngine for direct data generation
try:
    from enhanced_assessment_engine import EnhancedAssessmentEngine, Score, DevelopmentalMilestone
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Warning: EnhancedAssessmentEngine not available for import. Some features will be limited.")

# API configuration
API_BASE_URL = "http://localhost:8002"  # Default, can be overridden with command line args

# Milestone domains for organization and reporting
DOMAINS = ["GM", "FM", "ADL", "RL", "EL", "COG", "SOC", "EMO"]

class APITestResult:
    """Data class to store API test results"""
    def __init__(self, endpoint: str, passed: bool, response_time: float, details: str = ""):
        self.endpoint = endpoint
        self.passed = passed
        self.response_time = response_time
        self.details = details
        self.timestamp = datetime.now()
    
    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"{self.endpoint}: {status} ({self.response_time:.3f}s) - {self.details}"

class TestFramework:
    """Comprehensive testing framework for the ASD Assessment API"""
    
    def __init__(self, api_url: str = API_BASE_URL, use_engine: bool = ENGINE_AVAILABLE):
        """
        Initialize the testing framework
        
        Args:
            api_url: Base URL for the API
            use_engine: Whether to use the EnhancedAssessmentEngine directly for data generation
        """
        self.api_url = api_url
        self.use_engine = use_engine
        self.test_results = []
        self.synthetic_data = []
        
        # Initialize engine if available
        self.engine = None
        if self.use_engine:
            try:
                self.engine = EnhancedAssessmentEngine(use_embeddings=False)
                print("Direct engine access enabled for enhanced testing")
            except Exception as e:
                print(f"Error initializing engine: {e}")
                self.use_engine = False
        
        # Test configuration with defaults
        self.config = {
            "num_tests": 50,              # Number of test cases to generate
            "age": 24,                    # Child age in months
            "gender": "random",           # Child gender (male/female/they/random)
            "response_length": "medium",  # short/medium/long/random
            "include_edge_cases": True,   # Include edge cases in testing
            "domains_to_test": DOMAINS,   # Which developmental domains to test
            "concurrent_requests": 5,     # Max concurrent API requests
            "verbose": True               # Show detailed progress
        }
    
    def configure(self, **kwargs):
        """Update the test configuration with provided parameters"""
        self.config.update(kwargs)
        return self
    
    def _make_api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Tuple[Dict, float, bool]:
        """
        Make an API request and time the response
        
        Returns:
            Tuple of (response_data, response_time, success)
        """
        url = f"{self.api_url}{endpoint}"
        start_time = time.time()
        success = False
        response_data = {}
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                return {}, 0, False
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                success = True
                response_data = response.json()
            else:
                response_data = {
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            response_time = time.time() - start_time
            response_data = {"error": str(e)}
        
        return response_data, response_time, success
    
    def test_api_health(self) -> APITestResult:
        """Test if the API is running and responding"""
        try:
            response_data, response_time, success = self._make_api_request("GET", "/next-milestone")
            
            if success:
                return APITestResult("API Health Check", True, response_time, "API is responding")
            else:
                return APITestResult("API Health Check", False, response_time, 
                                    f"API not responding: {response_data.get('error', 'Unknown error')}")
        except Exception as e:
            return APITestResult("API Health Check", False, 0, f"Error checking API health: {str(e)}")
    
    def test_set_child_age(self, age: int) -> APITestResult:
        """Test setting the child's age"""
        data = {"age": age}
        response_data, response_time, success = self._make_api_request("POST", "/set-child-age", data)
        
        if success and response_data.get("total_milestones", 0) > 0:
            details = f"Successfully set age to {age}, {response_data.get('total_milestones')} milestones available"
            return APITestResult("Set Child Age", True, response_time, details)
        else:
            details = f"Failed to set age to {age}: {response_data}"
            return APITestResult("Set Child Age", False, response_time, details)
    
    def test_next_milestone(self) -> APITestResult:
        """Test retrieving the next milestone"""
        response_data, response_time, success = self._make_api_request("GET", "/next-milestone")
        
        if success:
            if response_data.get("complete", False):
                details = "No more milestones to assess"
            else:
                details = f"Retrieved milestone: {response_data.get('behavior')}"
            return APITestResult("Get Next Milestone", True, response_time, details)
        else:
            details = f"Failed to get next milestone: {response_data}"
            return APITestResult("Get Next Milestone", False, response_time, details)
    
    def test_score_response(self, milestone: str, response: str) -> APITestResult:
        """Test scoring a response for a specific milestone"""
        data = {
            "milestone_behavior": milestone,
            "response": response
        }
        response_data, response_time, success = self._make_api_request("POST", "/score-response", data)
        
        if success:
            details = f"Scored milestone '{milestone}' as {response_data.get('score_label')} ({response_data.get('score')})"
            return APITestResult("Score Response", True, response_time, details)
        else:
            details = f"Failed to score response: {response_data}"
            return APITestResult("Score Response", False, response_time, details)
    
    def test_batch_score(self, batch_data: List[Dict]) -> APITestResult:
        """Test batch scoring multiple responses"""
        data = {"responses": batch_data}
        response_data, response_time, success = self._make_api_request("POST", "/batch-score", data)
        
        if success:
            details = f"Successfully scored {len(response_data)} responses"
            return APITestResult("Batch Score", True, response_time, details)
        else:
            details = f"Failed to batch score responses: {response_data}"
            return APITestResult("Batch Score", False, response_time, details)
    
    def test_generate_report(self) -> APITestResult:
        """Test generating a comprehensive report"""
        response_data, response_time, success = self._make_api_request("GET", "/generate-report")
        
        if success:
            details = f"Generated report with {len(response_data.get('scores', []))} scored milestones"
            return APITestResult("Generate Report", True, response_time, details)
        else:
            details = f"Failed to generate report: {response_data}"
            return APITestResult("Generate Report", False, response_time, details)
    
    def _generate_random_response(self, milestone: Dict, score_level: Optional[int] = None) -> str:
        """
        Generate a random response for testing with specified characteristics
        
        Args:
            milestone: Dictionary with milestone information
            score_level: Optional score level to target (0-4)
        
        Returns:
            A synthetic response string
        """
        # Define templates based on score levels
        templates = {
            0: [  # CANNOT_DO
                "No, my child cannot {behavior_lower} at all.",
                "My child hasn't shown any ability to {behavior_lower} yet.",
                "We've tried to get our child to {behavior_lower}, but they don't show any signs of this skill."
            ],
            1: [  # LOST_SKILL
                "My child used to be able to {behavior_lower}, but has regressed.",
                "Around {regression_age} months, my child could {behavior_lower}, but now they don't anymore.",
                "This skill has disappeared. They previously could {behavior_lower} but now they can't."
            ],
            2: [  # EMERGING
                "My child is just beginning to {behavior_lower}, but it's inconsistent.",
                "Sometimes my child will try to {behavior_lower}, but they're still learning.",
                "I've seen early signs that my child is starting to {behavior_lower}."
            ],
            3: [  # WITH_SUPPORT
                "My child can {behavior_lower} with my help or guidance.",
                "With support, my child is able to {behavior_lower} successfully.",
                "They need assistance, but they can {behavior_lower} when I help them."
            ],
            4: [  # INDEPENDENT
                "My child does this completely independently.",
                "They have mastered this skill and can {behavior_lower} without any help.",
                "My child consistently and independently can {behavior_lower}."
            ]
        }
        
        # If no score level specified, choose randomly
        if score_level is None:
            score_level = random.randint(0, 4)
        
        # Select a template for the specified score level
        template = random.choice(templates[score_level])
        
        # Extract behavior
        behavior = milestone.get("behavior", "").lower()
        
        # Random age for regression (for LOST_SKILL responses)
        regression_age = random.randint(6, 18)
        
        # Generate the response
        response = template.format(
            behavior_lower=behavior,
            regression_age=regression_age
        )
        
        # Adjust response length based on configuration
        length_modifier = self.config.get("response_length", "medium")
        
        if length_modifier == "random":
            length_modifier = random.choice(["short", "medium", "long"])
        
        if length_modifier == "short":
            # Keep it as is
            pass
        elif length_modifier == "medium":
            # Add a detail sentence
            details = [
                f"I've noticed this happening especially when we're {random.choice(['playing', 'eating', 'out with others', 'reading books'])}.",
                f"This is most evident when {random.choice(['they are tired', 'they are excited', 'we have visitors', 'we are in a new environment'])}.",
                f"They seem to {random.choice(['enjoy', 'struggle with', 'be motivated by', 'be interested in'])} this particular skill."
            ]
            response += " " + random.choice(details)
        elif length_modifier == "long":
            # Add multiple detailed sentences
            details1 = [
                f"I've observed that {random.choice(['in the morning', 'after naps', 'when well-rested', 'when hungry'])} is when they perform best.",
                f"They seem to do better with this when {random.choice(['we provide encouragement', 'they have an audience', 'they are in a familiar setting', 'there are no distractions'])}.",
                f"This skill appears to be {random.choice(['developing rapidly', 'advancing slowly', 'consistent', 'variable day to day'])}."
            ]
            details2 = [
                f"Compared to other children, my child seems {random.choice(['ahead', 'behind', 'on track', 'unique in their approach'])}.",
                f"I {random.choice(['wonder if', 'hope that', 'think that', 'doubt that'])} this is developing typically.",
                f"Their {random.choice(['pediatrician', 'teacher', 'therapist', 'daycare provider'])} has also noticed this pattern."
            ]
            response += " " + random.choice(details1) + " " + random.choice(details2)
        
        return response
    
    def generate_test_data(self) -> List[Dict]:
        """
        Generate synthetic test data based on configuration
        
        Returns:
            List of dictionaries with milestone and response data
        """
        age = self.config.get("age", 24)
        num_tests = self.config.get("num_tests", 50)
        include_edge_cases = self.config.get("include_edge_cases", True)
        verbose = self.config.get("verbose", True)
        
        # Set child age
        result = self.test_set_child_age(age)
        if not result.passed:
            print(f"Error setting child age: {result.details}")
            return []
        
        # Get milestones
        milestones = []
        complete = False
        
        if verbose:
            print(f"Collecting milestones for age {age} months...")
        
        while not complete and len(milestones) < 100:  # Safety limit
            response_data, _, success = self._make_api_request("GET", "/next-milestone")
            
            if not success:
                print("Error retrieving milestones")
                break
            
            if response_data.get("complete", False):
                complete = True
            else:
                milestones.append(response_data)
        
        if verbose:
            print(f"Collected {len(milestones)} milestones")
        
        # Reset the engine for clean test data generation
        self._make_api_request("POST", "/reset")
        self.test_set_child_age(age)
        
        # Filter milestones by domain if specified
        domains_to_test = self.config.get("domains_to_test", DOMAINS)
        if domains_to_test != DOMAINS:
            milestones = [m for m in milestones if m.get("domain") in domains_to_test]
            if verbose:
                print(f"Filtered to {len(milestones)} milestones in domains {domains_to_test}")
        
        # Generate test data
        test_data = []
        
        # Add edge cases if requested
        if include_edge_cases:
            edge_cases = self._generate_edge_cases(milestones)
            test_data.extend(edge_cases)
            
            if verbose:
                print(f"Generated {len(edge_cases)} edge cases")
        
        # Calculate how many regular tests to generate
        remaining_tests = max(0, num_tests - len(test_data))
        
        # Generate regular test cases
        if remaining_tests > 0:
            # Sample milestones (with replacement if we need more tests than available milestones)
            sampled_milestones = random.choices(milestones, k=remaining_tests)
            
            # Generate responses for each sampled milestone
            for milestone in sampled_milestones:
                # Randomly choose a score level
                score_level = random.randint(0, 4)
                
                response = self._generate_random_response(milestone, score_level)
                
                test_data.append({
                    "milestone_behavior": milestone["behavior"],
                    "response": response,
                    "expected_score": score_level,
                    "domain": milestone["domain"],
                    "is_edge_case": False
                })
        
        if verbose:
            print(f"Generated {len(test_data)} total test cases")
        
        self.synthetic_data = test_data
        return test_data
    
    def _generate_edge_cases(self, milestones: List[Dict]) -> List[Dict]:
        """Generate edge cases for testing"""
        edge_cases = []
        
        if not milestones:
            return edge_cases
        
        # Sample a subset of milestones for edge cases
        sampled_milestones = random.sample(milestones, min(10, len(milestones)))
        
        # 1. Empty responses
        edge_cases.append({
            "milestone_behavior": sampled_milestones[0]["behavior"],
            "response": "",
            "expected_score": -1,  # NOT_RATED
            "domain": sampled_milestones[0]["domain"],
            "is_edge_case": True,
            "edge_case_type": "empty_response"
        })
        
        # 2. Very long responses
        long_response = "My child " + " ".join(["very " * 20 + "much"] * 5)
        edge_cases.append({
            "milestone_behavior": sampled_milestones[1]["behavior"],
            "response": long_response,
            "expected_score": None,  # Unpredictable
            "domain": sampled_milestones[1]["domain"],
            "is_edge_case": True,
            "edge_case_type": "very_long_response"
        })
        
        # 3. Mixed signals (contradictory statements)
        mixed_response = "My child can do this independently. Actually, they can't do it at all."
        edge_cases.append({
            "milestone_behavior": sampled_milestones[2]["behavior"],
            "response": mixed_response,
            "expected_score": None,  # Unpredictable
            "domain": sampled_milestones[2]["domain"],
            "is_edge_case": True,
            "edge_case_type": "contradictory_response"
        })
        
        # 4. Non-existent milestone
        edge_cases.append({
            "milestone_behavior": "This milestone does not exist",
            "response": "My child does this well",
            "expected_score": None,  # Should raise an error
            "domain": "Unknown",
            "is_edge_case": True,
            "edge_case_type": "invalid_milestone"
        })
        
        # 5. Response in different language
        foreign_response = "Mi hijo puede hacer esto muy bien. Ha dominado esta habilidad."
        edge_cases.append({
            "milestone_behavior": sampled_milestones[3]["behavior"],
            "response": foreign_response,
            "expected_score": None,  # Unpredictable
            "domain": sampled_milestones[3]["domain"],
            "is_edge_case": True,
            "edge_case_type": "foreign_language"
        })
        
        return edge_cases
    
    def run_single_tests(self) -> List[APITestResult]:
        """Run tests for individual API endpoints"""
        results = []
        verbose = self.config.get("verbose", True)
        
        # 1. Test API health
        results.append(self.test_api_health())
        if verbose:
            print(results[-1])
        
        # If API is not responding, don't run further tests
        if not results[-1].passed:
            return results
        
        # 2. Test setting child age
        age = self.config.get("age", 24)
        results.append(self.test_set_child_age(age))
        if verbose:
            print(results[-1])
        
        # 3. Test getting next milestone
        results.append(self.test_next_milestone())
        if verbose:
            print(results[-1])
        
        # Get a milestone for response testing
        response_data, _, success = self._make_api_request("GET", "/next-milestone")
        if success and not response_data.get("complete", False):
            milestone = response_data.get("behavior")
            
            # 4. Test scoring a response
            if milestone:
                response = self._generate_random_response(response_data)
                results.append(self.test_score_response(milestone, response))
                if verbose:
                    print(results[-1])
        
        # 5. Test generating a report
        results.append(self.test_generate_report())
        if verbose:
            print(results[-1])
        
        return results
    
    def run_batch_tests(self) -> List[APITestResult]:
        """Run batch processing tests with generated data"""
        if not self.synthetic_data:
            self.generate_test_data()
        
        results = []
        verbose = self.config.get("verbose", True)
        
        # Prepare data for batch scoring
        batch_data = [
            {"milestone_behavior": item["milestone_behavior"], "response": item["response"]} 
            for item in self.synthetic_data if not item.get("edge_case_type") == "invalid_milestone"
        ]
        
        # Split into smaller batches if many items
        batch_size = 25
        batches = [batch_data[i:i+batch_size] for i in range(0, len(batch_data), batch_size)]
        
        if verbose:
            print(f"Running {len(batches)} batch tests with {len(batch_data)} total items...")
        
        # Process each batch
        for i, batch in enumerate(batches):
            result = self.test_batch_score(batch)
            if verbose:
                print(f"Batch {i+1}/{len(batches)}: {result}")
            results.append(result)
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run a comprehensive set of tests and return detailed results
        
        Returns:
            Dictionary with test results and statistics
        """
        start_time = time.time()
        verbose = self.config.get("verbose", True)
        
        if verbose:
            print(f"Starting comprehensive API test against {self.api_url}")
            print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Generate test data if not already generated
        if not self.synthetic_data:
            self.generate_test_data()
        
        # Run basic endpoint tests
        if verbose:
            print("\n=== Testing API Endpoints ===")
        endpoint_results = self.run_single_tests()
        
        # If basic tests fail, don't continue
        basic_checks_passed = all(result.passed for result in endpoint_results[:2])  # Health and age setting
        
        if not basic_checks_passed:
            if verbose:
                print("\n❌ Basic API checks failed. Aborting further tests.")
            return {
                "success": False,
                "endpoint_results": endpoint_results,
                "duration": time.time() - start_time,
                "error": "Basic API checks failed"
            }
        
        # Run individual test cases
        if verbose:
            print("\n=== Testing Individual Responses ===")
        
        individual_results = []
        concurrent_requests = min(self.config.get("concurrent_requests", 5), len(self.synthetic_data))
        
        # Process test cases with progress bar
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            test_cases = [(item["milestone_behavior"], item["response"]) for item in self.synthetic_data]
            
            if verbose:
                test_iterator = tqdm(test_cases, desc="Testing responses")
            else:
                test_iterator = test_cases
            
            for milestone, response in test_iterator:
                future = executor.submit(self.test_score_response, milestone, response)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                individual_results.append(result)
        
        # Run batch tests
        if verbose:
            print("\n=== Testing Batch Processing ===")
        batch_results = self.run_batch_tests()
        
        # Generate report
        if verbose:
            print("\n=== Generating Final Report ===")
        report_result = self.test_generate_report()
        
        # Calculate statistics
        all_results = endpoint_results + individual_results + batch_results + [report_result]
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = sum(1 for r in all_results if not r.passed)
        
        if verbose:
            print(f"\n=== Test Summary ===")
            print(f"Total tests run: {len(all_results)}")
            print(f"Tests passed: {passed_tests} ({passed_tests/len(all_results)*100:.1f}%)")
            print(f"Tests failed: {failed_tests} ({failed_tests/len(all_results)*100:.1f}%)")
            print(f"Total test duration: {time.time() - start_time:.2f} seconds")
        
        # Collect response times
        response_times = [r.response_time for r in all_results]
        
        statistics_data = {
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)]
        }
        
        if verbose:
            print(f"\n=== Performance Statistics ===")
            print(f"Minimum response time: {statistics_data['min_response_time']:.3f}s")
            print(f"Maximum response time: {statistics_data['max_response_time']:.3f}s")
            print(f"Average response time: {statistics_data['avg_response_time']:.3f}s")
            print(f"Median response time: {statistics_data['median_response_time']:.3f}s")
            print(f"95th percentile response time: {statistics_data['p95_response_time']:.3f}s")
        
        # Collect results by endpoint type
        results_by_endpoint = {}
        for result in all_results:
            if result.endpoint not in results_by_endpoint:
                results_by_endpoint[result.endpoint] = []
            results_by_endpoint[result.endpoint].append(result)
        
        # Collect failures
        failures = [r for r in all_results if not r.passed]
        
        return {
            "success": True,
            "endpoint_results": endpoint_results,
            "individual_results": individual_results,
            "batch_results": batch_results,
            "report_result": report_result,
            "total_tests": len(all_results),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": passed_tests / len(all_results) if all_results else 0,
            "statistics": statistics_data,
            "results_by_endpoint": results_by_endpoint,
            "failures": failures,
            "duration": time.time() - start_time
        }
    
    def generate_test_report(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Generate a detailed test report in HTML format"""
        if not results:
            print("No test results to report")
            return
        
        # Create a report timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ASD Assessment API Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .summary {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .endpoint {{ font-weight: bold; }}
                .details {{ color: #666; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>ASD Developmental Milestone Assessment API Test Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p>Total tests run: <strong>{results['total_tests']}</strong></p>
                <p>Tests passed: <strong class="pass">{results['passed_tests']} ({results['pass_rate']*100:.1f}%)</strong></p>
                <p>Tests failed: <strong class="fail">{results['failed_tests']} ({(1-results['pass_rate'])*100:.1f}%)</strong></p>
                <p>Total test duration: <strong>{results['duration']:.2f} seconds</strong></p>
            </div>
            
            <h2>Performance Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Minimum response time</td>
                    <td>{results['statistics']['min_response_time']:.3f}s</td>
                </tr>
                <tr>
                    <td>Maximum response time</td>
                    <td>{results['statistics']['max_response_time']:.3f}s</td>
                </tr>
                <tr>
                    <td>Average response time</td>
                    <td>{results['statistics']['avg_response_time']:.3f}s</td>
                </tr>
                <tr>
                    <td>Median response time</td>
                    <td>{results['statistics']['median_response_time']:.3f}s</td>
                </tr>
                <tr>
                    <td>95th percentile response time</td>
                    <td>{results['statistics']['p95_response_time']:.3f}s</td>
                </tr>
            </table>
            
            <h2>API Endpoint Tests</h2>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Response Time</th>
                    <th>Details</th>
                </tr>
        """
        
        # Add endpoint results
        for result in results['endpoint_results']:
            status = "PASS" if result.passed else "FAIL"
            status_class = "pass" if result.passed else "fail"
            html += f"""
                <tr>
                    <td class="endpoint">{result.endpoint}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{result.response_time:.3f}s</td>
                    <td class="details">{result.details}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Failed Tests</h2>
        """
        
        if results['failures']:
            html += """
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Response Time</th>
                    <th>Details</th>
                </tr>
            """
            
            for failure in results['failures']:
                html += f"""
                    <tr>
                        <td class="endpoint">{failure.endpoint}</td>
                        <td>{failure.response_time:.3f}s</td>
                        <td class="details">{failure.details}</td>
                    </tr>
                """
            
            html += "</table>"
        else:
            html += "<p>No failed tests! 🎉</p>"
        
        html += """
        </body>
        </html>
        """
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"Test report saved to {output_file}")
        
        return html

def parse_arguments():
    parser = argparse.ArgumentParser(description="ASD Developmental Milestone Assessment API Testing Framework")
    
    # API configuration
    parser.add_argument("--api-url", default=API_BASE_URL, help="Base URL for the API")
    
    # Test configuration
    parser.add_argument("--num-tests", type=int, default=50, help="Number of test cases to generate")
    parser.add_argument("--age", type=int, default=24, help="Child age in months for testing")
    parser.add_argument("--gender", choices=["male", "female", "they", "random"], default="random", help="Child gender for test data")
    parser.add_argument("--response-length", choices=["short", "medium", "long", "random"], default="medium", help="Length of generated responses")
    parser.add_argument("--no-edge-cases", action="store_true", help="Disable edge case testing")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, help="Specific domains to test (default: all)")
    parser.add_argument("--concurrent", type=int, default=5, help="Maximum concurrent API requests")
    
    # Report configuration
    parser.add_argument("--output", help="Output file for test report (HTML format)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Configure the test framework
    config = {
        "num_tests": args.num_tests,
        "age": args.age,
        "gender": args.gender,
        "response_length": args.response_length,
        "include_edge_cases": not args.no_edge_cases,
        "concurrent_requests": args.concurrent,
        "verbose": args.verbose and not args.quiet
    }
    
    if args.domains:
        config["domains_to_test"] = args.domains
    
    # Initialize and run the framework
    framework = TestFramework(api_url=args.api_url)
    framework.configure(**config)
    
    try:
        results = framework.run_comprehensive_test()
        
        if args.output:
            framework.generate_test_report(results, args.output)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        raise

if __name__ == "__main__":
    main() 