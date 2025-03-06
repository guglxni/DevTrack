#!/usr/bin/env python3
"""
Comprehensive API Testing Framework for ASD Assessment API

This script provides a complete testing framework for the ASD Assessment API.
It can test all endpoints, including the recently added endpoints for questions,
keywords, and manual scoring. The framework supports detailed reporting,
performance metrics, and batch testing.
"""

import os
import sys
import json
import time
import requests
import pytest
import pandas as pd
import statistics
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default API URL
DEFAULT_API_URL = "http://localhost:8003"

# Terminal colors for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestResult:
    """Class to store results from API tests"""
    
    def __init__(self, endpoint: str, success: bool, response_time: float, 
                 details: Dict = None, status_code: int = None,
                 expected: Any = None, actual: Any = None):
        """
        Initialize a test result object
        
        Args:
            endpoint: The API endpoint tested
            success: Whether the test passed
            response_time: Time taken for the API response in seconds
            details: Additional details about the test
            status_code: HTTP status code returned
            expected: Expected value or result
            actual: Actual value or result returned by the API
        """
        self.endpoint = endpoint
        self.success = success
        self.response_time = response_time
        self.details = details or {}
        self.status_code = status_code
        self.expected = expected
        self.actual = actual
        self.timestamp = datetime.now().isoformat()
    
    def __str__(self):
        """String representation of the test result"""
        return f"Test [{self.endpoint}] - {'✅ PASS' if self.success else '❌ FAIL'} - {self.response_time:.4f}s - Status: {self.status_code}"
    
    def to_dict(self):
        """Convert the test result to a dictionary"""
        return {
            "endpoint": self.endpoint,
            "success": self.success,
            "response_time": self.response_time,
            "details": self.details,
            "status_code": self.status_code,
            "expected": self.expected,
            "actual": self.actual,
            "timestamp": self.timestamp
        }


class ComprehensiveAPITester:
    """A comprehensive tester for the ASD Assessment API endpoints"""
    
    def __init__(self, api_url: str = DEFAULT_API_URL, verbose: bool = False):
        """
        Initialize the API tester
        
        Args:
            api_url: The base URL for the API
            verbose: Whether to print verbose output during testing
        """
        self.api_url = api_url
        self.verbose = verbose
        self.test_results = []
        self.response_times = {}
        self.success_rates = {}
        
        # Create results directory if it doesn't exist
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info(f"Initialized API tester with URL: {api_url}")
    
    def set_verbose(self, verbose: bool):
        """Set the verbose flag"""
        self.verbose = verbose
    
    def log(self, message: str):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            logger.info(message)
    
    def make_request(self, method: str, endpoint: str, data: Dict = None, 
                     params: Dict = None, expected_status: int = 200) -> TestResult:
        """
        Make an HTTP request to the API and return the result
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/health")
            data: Request body for POST/PUT requests
            params: Query parameters for GET requests
            expected_status: Expected HTTP status code
        
        Returns:
            TestResult object containing the results of the test
        """
        url = f"{self.api_url}{endpoint}"
        self.log(f"Making {method} request to {url}")
        
        if data:
            self.log(f"Request data: {json.dumps(data, indent=2)}")
        
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Try to parse response as JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = response.text
            
            # Check if the status code matches expected
            success = response.status_code == expected_status
            
            # Store response times by endpoint
            endpoint_key = endpoint.split('?')[0]  # Remove query params
            if endpoint_key not in self.response_times:
                self.response_times[endpoint_key] = []
            self.response_times[endpoint_key].append(response_time)
            
            # Store success rates by endpoint
            if endpoint_key not in self.success_rates:
                self.success_rates[endpoint_key] = {"success": 0, "total": 0}
            self.success_rates[endpoint_key]["total"] += 1
            if success:
                self.success_rates[endpoint_key]["success"] += 1
            
            # Create and return the test result
            result = TestResult(
                endpoint=endpoint,
                success=success,
                response_time=response_time,
                details={"response": response_data},
                status_code=response.status_code,
                expected=expected_status,
                actual=response.status_code
            )
            
            self.test_results.append(result)
            self.log(str(result))
            
            if not success:
                logger.warning(f"Test failed: {result}")
            
            return result
        
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error making request to {url}: {str(e)}")
            
            # Create a failed test result
            result = TestResult(
                endpoint=endpoint,
                success=False,
                response_time=response_time,
                details={"error": str(e)},
                status_code=None,
                expected=expected_status,
                actual=None
            )
            
            self.test_results.append(result)
            self.log(str(result))
            
            return result

    # Basic Endpoints Tests
    
    def test_health(self) -> TestResult:
        """Test the health check endpoint"""
        return self.make_request("GET", "/health")
    
    def test_set_child_age(self, age: int, name: str = None) -> TestResult:
        """
        Test setting the child's age
        
        Args:
            age: Child's age in months (0-36)
            name: Child's name (optional)
        """
        data = {"age": age}
        if name:
            data["name"] = name
        
        return self.make_request("POST", "/set-child-age", data=data)
    
    def test_next_milestone(self) -> TestResult:
        """Test getting the next milestone"""
        return self.make_request("GET", "/next-milestone")
    
    def test_score_response(self, milestone: str, response_text: str) -> TestResult:
        """
        Test scoring a response for a specific milestone
        
        Args:
            milestone: The milestone behavior to score
            response_text: The caregiver's response text
        """
        data = {
            "milestone_behavior": milestone,
            "response": response_text
        }
        
        return self.make_request("POST", "/score-response", data=data)
    
    def test_batch_score(self, batch_data: List[Dict]) -> List[TestResult]:
        """
        Test batch scoring multiple responses
        
        Args:
            batch_data: List of response data objects
                Each object should have:
                - milestone_behavior: The milestone behavior
                - response: The caregiver's response text
        """
        # Format the request data
        data = {
            "responses": batch_data
        }
        
        # Make the request
        result = self.make_request("POST", "/batch-score", data=data)
        
        # Parse individual results
        individual_results = []
        
        if result.success and "response" in result.details:
            batch_response = result.details["response"]
            
            if isinstance(batch_response, list):
                # Create individual test results for each response
                for i, response_item in enumerate(batch_response):
                    individual_result = TestResult(
                        endpoint=f"/batch-score/item-{i}",
                        success=True,
                        response_time=result.response_time / len(batch_response),
                        details={"response": response_item},
                        status_code=result.status_code,
                        expected=batch_data[i],
                        actual=response_item
                    )
                    individual_results.append(individual_result)
        
        return [result] + individual_results
    
    def test_generate_report(self) -> TestResult:
        """Test generating a comprehensive assessment report"""
        return self.make_request("GET", "/generate-report")
    
    def test_reset(self) -> TestResult:
        """Test resetting the assessment engine"""
        return self.make_request("POST", "/reset")
    
    def test_all_milestones(self) -> TestResult:
        """Test retrieving all available milestones"""
        return self.make_request("GET", "/all-milestones")
    
    # New Endpoint Tests
    
    def test_question(self, question_text: str, milestone_id: str = None) -> TestResult:
        """
        Test submitting a question about a child's behavior
        
        Args:
            question_text: The question text
            milestone_id: Associated milestone ID (optional)
        """
        data = {"text": question_text}
        if milestone_id:
            data["milestone_id"] = milestone_id
            
        return self.make_request("POST", "/question", data=data)
    
    def test_update_keywords(self, category: str, keywords: List[str]) -> TestResult:
        """
        Test updating keywords for a scoring category
        
        Args:
            category: Scoring category (e.g., "CANNOT_DO")
            keywords: List of keywords for this category
        """
        data = {
            "category": category,
            "keywords": keywords
        }
        
        return self.make_request("POST", "/keywords", data=data)
    
    def test_send_score(self, milestone_id: str, score: int, score_label: str) -> TestResult:
        """
        Test manually setting a score for a specific milestone
        
        Args:
            milestone_id: The milestone ID (behavior)
            score: The numeric score value (0-4)
            score_label: The score label (e.g., "INDEPENDENT")
        """
        data = {
            "milestone_id": milestone_id,
            "score": score,
            "score_label": score_label
        }
        
        return self.make_request("POST", "/send-score", data=data)
    
    # Complex test scenarios
    
    def run_complete_assessment(self, age: int = 24, responses: Dict[str, str] = None) -> List[TestResult]:
        """
        Run a complete assessment flow
        
        Args:
            age: Child's age in months
            responses: Dictionary mapping milestone behaviors to response texts
                If not provided, random responses will be generated
        
        Returns:
            List of test results from each step
        """
        results = []
        
        # Step 1: Reset the assessment
        results.append(self.test_reset())
        
        # Step 2: Set the child's age
        results.append(self.test_set_child_age(age))
        
        # Step 3: Get all milestones to have a reference
        all_milestones_result = self.test_all_milestones()
        results.append(all_milestones_result)
        
        # Extract milestone behaviors if we don't have predefined responses
        milestones = []
        if all_milestones_result.success and "response" in all_milestones_result.details:
            response_data = all_milestones_result.details["response"]
            if isinstance(response_data, dict) and "milestones" in response_data:
                milestones = [m["behavior"] for m in response_data["milestones"]]
        
        # Create random responses if none provided
        if not responses:
            responses = {}
            response_templates = [
                "Yes, my child can do that independently.",
                "No, my child cannot do that yet.",
                "Sometimes, it depends on the situation.",
                "My child used to do this but has lost this skill.",
                "My child can do this with help and support."
            ]
            
            for milestone in milestones:
                responses[milestone] = random.choice(response_templates)
        
        # Step 4: Score each milestone
        for milestone, response in responses.items():
            results.append(self.test_score_response(milestone, response))
        
        # Step 5: Generate the final report
        results.append(self.test_generate_report())
        
        return results
    
    def test_keywords_workflow(self) -> List[TestResult]:
        """
        Test the complete keywords update workflow
        
        This tests updating keywords for each score category and then
        verifying they work by scoring responses with those keywords.
        """
        results = []
        
        # Define test keywords for each category
        keyword_tests = {
            "CANNOT_DO": [
                "no", "not", "never", "doesn't", "does not", 
                "cannot", "can't", "unable", "hasn't", "has not", 
                "not able", "not at all", "not yet started", "not capable"
            ],
            "LOST_SKILL": [
                "used to", "previously", "before", "no longer", 
                "stopped", "regressed", "lost", "forgotten"
            ],
            "EMERGING": [
                "sometimes", "occasionally", "starting to", "beginning to",
                "trying to", "inconsistent", "not consistent"
            ],
            "WITH_SUPPORT": [
                "with help", "when assisted", "with support", "with guidance",
                "needs help", "when prompted", "specific situations"
            ],
            "INDEPENDENT": [
                "always", "independently", "by themselves", "on their own",
                "without help", "consistently", "mastered", "yes"
            ]
        }
        
        # Step 1: Reset the assessment
        results.append(self.test_reset())
        
        # Step 2: Set a child age to work with
        results.append(self.test_set_child_age(24))
        
        # Step 3: Update keywords for each category
        for category, keywords in keyword_tests.items():
            results.append(self.test_update_keywords(category, keywords))
        
        # Step 4: Test scoring with different keywords
        # First get a milestone to test with
        next_milestone_result = self.test_next_milestone()
        results.append(next_milestone_result)
        
        if next_milestone_result.success and "response" in next_milestone_result.details:
            response_data = next_milestone_result.details["response"]
            if isinstance(response_data, dict) and "behavior" in response_data:
                milestone = response_data["behavior"]
                
                # Test each category with appropriate responses
                response_tests = {
                    "CANNOT_DO": f"No, my child cannot do this at all.",
                    "LOST_SKILL": f"My child used to do this but has regressed and no longer can.",
                    "EMERGING": f"My child is sometimes trying to do this but is inconsistent.",
                    "WITH_SUPPORT": f"My child can do this with help and support from me.",
                    "INDEPENDENT": f"Yes, my child always does this independently and consistently."
                }
                
                for expected_category, response_text in response_tests.items():
                    score_result = self.test_score_response(milestone, response_text)
                    results.append(score_result)
                    
                    # Verify the score matches the expected category
                    if score_result.success and "response" in score_result.details:
                        score_data = score_result.details["response"]
                        if isinstance(score_data, dict) and "score_label" in score_data:
                            actual_category = score_data["score_label"]
                            if actual_category == expected_category:
                                self.log(f"✅ Response correctly scored as {expected_category}")
                            else:
                                self.log(f"❌ Response incorrectly scored as {actual_category}, expected {expected_category}")
        
        return results
    
    def run_load_test(self, endpoint: str, method: str, data: Dict = None, 
                     params: Dict = None, requests_count: int = 10,
                     concurrent: bool = True) -> List[TestResult]:
        """
        Run a load test for a specific endpoint
        
        Args:
            endpoint: The endpoint to test
            method: HTTP method to use
            data: Request data (for POST requests)
            params: Query parameters (for GET requests)
            requests_count: Number of requests to make
            concurrent: Whether to run requests concurrently
        
        Returns:
            List of test results
        """
        self.log(f"Running load test for {endpoint}: {requests_count} requests, concurrent={concurrent}")
        
        results = []
        
        if concurrent:
            # Run requests concurrently using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, requests_count)) as executor:
                futures = []
                for i in range(requests_count):
                    futures.append(executor.submit(
                        self.make_request, method, endpoint, data, params
                    ))
                
                # Collect results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Requests"):
                    results.append(future.result())
        else:
            # Run requests sequentially
            for i in tqdm(range(requests_count), desc="Requests"):
                results.append(self.make_request(method, endpoint, data, params))
        
        return results
    
    # Reporting methods
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of test results"""
        if not self.test_results:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0,
                "average_response_time": 0,
                "endpoints": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # Count total tests, passed tests, and failed tests
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate overall success rate
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Calculate average response time
        average_response_time = sum(result.response_time for result in self.test_results) / total_tests
        
        # Generate endpoint-specific stats
        endpoints = {}
        for endpoint, times in self.response_times.items():
            success_rate = (self.success_rates[endpoint]["success"] / self.success_rates[endpoint]["total"]) * 100
            
            endpoints[endpoint] = {
                "total_tests": self.success_rates[endpoint]["total"],
                "success_rate": success_rate,
                "min_response_time": min(times),
                "max_response_time": max(times),
                "avg_response_time": statistics.mean(times),
                "median_response_time": statistics.median(times)
            }
            
            # Calculate standard deviation if we have enough samples
            if len(times) > 1:
                endpoints[endpoint]["std_dev_response_time"] = statistics.stdev(times)
            else:
                endpoints[endpoint]["std_dev_response_time"] = 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "average_response_time": average_response_time,
            "endpoints": endpoints,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results_to_json(self, filename: str = "test_results/api_test_results.json"):
        """Save test results to a JSON file"""
        data = {
            "summary": self._generate_summary(),
            "results": [result.to_dict() for result in self.test_results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved test results to {filename}")
        return filename
    
    def generate_charts(self, output_dir: str = "test_results"):
        """Generate performance charts from test results"""
        summary = self._generate_summary()
        
        if not summary["endpoints"]:
            logger.warning("No endpoints data to generate charts")
            return
        
        # Create response time chart
        plt.figure(figsize=(12, 6))
        
        endpoints = list(summary["endpoints"].keys())
        avg_times = [summary["endpoints"][e]["avg_response_time"] for e in endpoints]
        min_times = [summary["endpoints"][e]["min_response_time"] for e in endpoints]
        max_times = [summary["endpoints"][e]["max_response_time"] for e in endpoints]
        
        x = range(len(endpoints))
        plt.bar(x, avg_times, width=0.5, label='Average')
        plt.plot(x, min_times, 'g^', label='Min')
        plt.plot(x, max_times, 'rv', label='Max')
        
        plt.xlabel('Endpoint')
        plt.ylabel('Response Time (seconds)')
        plt.title('API Response Times by Endpoint')
        plt.xticks(x, [self._shorten_endpoint(e) for e in endpoints], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        response_time_chart = f"{output_dir}/response_time_chart.png"
        plt.savefig(response_time_chart)
        plt.close()
        
        # Create success rate chart
        plt.figure(figsize=(12, 6))
        
        success_rates = [summary["endpoints"][e]["success_rate"] for e in endpoints]
        
        plt.bar(x, success_rates, width=0.5)
        plt.axhline(y=100, color='g', linestyle='--', alpha=0.7)
        plt.axhline(y=90, color='y', linestyle='--', alpha=0.7)
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.7)
        
        plt.xlabel('Endpoint')
        plt.ylabel('Success Rate (%)')
        plt.title('API Success Rates by Endpoint')
        plt.xticks(x, [self._shorten_endpoint(e) for e in endpoints], rotation=45, ha='right')
        plt.ylim(0, 105)
        plt.tight_layout()
        
        success_rate_chart = f"{output_dir}/success_rate_chart.png"
        plt.savefig(success_rate_chart)
        plt.close()
        
        logger.info(f"Generated performance charts in {output_dir}")
        return response_time_chart, success_rate_chart
    
    def _shorten_endpoint(self, endpoint: str) -> str:
        """Shorten endpoint names for display in charts"""
        # Remove leading slash
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        
        # Truncate long endpoint names
        if len(endpoint) > 15:
            endpoint = endpoint[:12] + "..."
        
        return endpoint
    
    def generate_html_report(self, output_file: str = "test_results/api_test_report.html"):
        """Generate an HTML report with test results and charts"""
        # Generate summary and charts
        summary = self._generate_summary()
        self.generate_charts()
        
        # Generate HTML report
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ASD Assessment API Test Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #0066cc;
                }
                .summary {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .success-rate {
                    font-size: 24px;
                    font-weight: bold;
                }
                .success-rate.high {
                    color: #28a745;
                }
                .success-rate.medium {
                    color: #ffc107;
                }
                .success-rate.low {
                    color: #dc3545;
                }
                .charts {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .charts img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .endpoint-name {
                    font-weight: bold;
                }
                .passed {
                    color: #28a745;
                }
                .failed {
                    color: #dc3545;
                }
                .detail-toggle {
                    cursor: pointer;
                    color: #0066cc;
                    text-decoration: underline;
                }
                .details {
                    display: none;
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 10px;
                    white-space: pre-wrap;
                    font-family: monospace;
                    max-height: 300px;
                    overflow-y: auto;
                }
            </style>
            <script>
                function toggleDetails(id) {
                    var details = document.getElementById(id);
                    if (details.style.display === 'block') {
                        details.style.display = 'none';
                    } else {
                        details.style.display = 'block';
                    }
                }
            </script>
        </head>
        <body>
            <h1>ASD Assessment API Test Report</h1>
            <p>Generated on: {{ timestamp }}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>
                    <strong>Total Tests:</strong> {{ total_tests }}<br>
                    <strong>Passed:</strong> {{ passed_tests }}<br>
                    <strong>Failed:</strong> {{ failed_tests }}<br>
                    <strong>Success Rate:</strong> 
                    <span class="success-rate {{ success_rate_class }}">{{ success_rate }}%</span><br>
                    <strong>Average Response Time:</strong> {{ average_response_time }} seconds
                </p>
            </div>
            
            <div class="charts">
                <div>
                    <h3>Response Times by Endpoint</h3>
                    <img src="response_time_chart.png" alt="Response Time Chart">
                </div>
                <div>
                    <h3>Success Rates by Endpoint</h3>
                    <img src="success_rate_chart.png" alt="Success Rate Chart">
                </div>
            </div>
            
            <h2>Endpoint Performance</h2>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Tests</th>
                    <th>Success Rate</th>
                    <th>Avg Time (s)</th>
                    <th>Min Time (s)</th>
                    <th>Max Time (s)</th>
                </tr>
                {% for endpoint, stats in endpoints.items() %}
                <tr>
                    <td>{{ endpoint }}</td>
                    <td>{{ stats.total_tests }}</td>
                    <td>{{ stats.success_rate }}%</td>
                    <td>{{ stats.avg_response_time }}</td>
                    <td>{{ stats.min_response_time }}</td>
                    <td>{{ stats.max_response_time }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Status</th>
                    <th>Response Time (s)</th>
                    <th>Status Code</th>
                    <th>Details</th>
                </tr>
                {% for result in results %}
                <tr>
                    <td class="endpoint-name">{{ result.endpoint }}</td>
                    <td class="{{ 'passed' if result.success else 'failed' }}">
                        {{ "PASSED" if result.success else "FAILED" }}
                    </td>
                    <td>{{ result.response_time }}</td>
                    <td>{{ result.status_code }}</td>
                    <td>
                        <span class="detail-toggle" onclick="toggleDetails('details-{{ loop.index }}')">
                            Show/Hide Details
                        </span>
                        <div id="details-{{ loop.index }}" class="details">
                            <strong>Response:</strong>\n{{ result.details.response | tojson(indent=2) }}
                        </div>
                    </td>
                </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """
        
        from jinja2 import Template
        template = Template(html_template)
        
        # Determine success rate class
        success_rate_class = "high" if summary["success_rate"] >= 90 else "medium" if summary["success_rate"] >= 80 else "low"
        
        # Format numbers
        for endpoint in summary["endpoints"].values():
            endpoint["success_rate"] = f"{endpoint['success_rate']:.2f}"
            endpoint["avg_response_time"] = f"{endpoint['avg_response_time']:.4f}"
            endpoint["min_response_time"] = f"{endpoint['min_response_time']:.4f}"
            endpoint["max_response_time"] = f"{endpoint['max_response_time']:.4f}"
        
        # Render template
        html = template.render(
            timestamp=summary["timestamp"],
            total_tests=summary["total_tests"],
            passed_tests=summary["passed_tests"],
            failed_tests=summary["failed_tests"],
            success_rate=f"{summary['success_rate']:.2f}",
            success_rate_class=success_rate_class,
            average_response_time=f"{summary['average_response_time']:.4f}",
            endpoints=summary["endpoints"],
            results=self.test_results
        )
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated HTML report at {output_file}")
        return output_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive API Testing Framework for ASD Assessment API")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Base URL for the API")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Basic tests
    basic_parser = subparsers.add_parser("basic", help="Run basic API tests")
    basic_parser.add_argument("--age", type=int, default=24, help="Child's age in months for testing")
    
    # Complete assessment
    assessment_parser = subparsers.add_parser("assessment", help="Run a complete assessment flow")
    assessment_parser.add_argument("--age", type=int, default=24, help="Child's age in months")
    assessment_parser.add_argument("--responses", help="JSON file with milestone responses")
    
    # Keywords workflow
    keywords_parser = subparsers.add_parser("keywords", help="Test the keywords update workflow")
    
    # Load test
    load_parser = subparsers.add_parser("load", help="Run a load test")
    load_parser.add_argument("--endpoint", required=True, help="Endpoint to test")
    load_parser.add_argument("--method", default="GET", help="HTTP method to use")
    load_parser.add_argument("--data", help="JSON file with request data")
    load_parser.add_argument("--count", type=int, default=10, help="Number of requests to make")
    load_parser.add_argument("--concurrent", action="store_true", help="Run requests concurrently")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize tester
    tester = ComprehensiveAPITester(api_url=args.url, verbose=args.verbose)
    
    if args.command == "basic":
        # Run basic tests
        logger.info("Running basic API tests")
        
        # Health check
        tester.test_health()
        
        # Set child age
        tester.test_set_child_age(args.age)
        
        # Get all milestones
        tester.test_all_milestones()
        
        # Get next milestone
        next_milestone = tester.test_next_milestone()
        
        # Score a response if we have a milestone
        if next_milestone.success and "response" in next_milestone.details:
            response_data = next_milestone.details["response"]
            if isinstance(response_data, dict) and "behavior" in response_data:
                milestone = response_data["behavior"]
                tester.test_score_response(milestone, "Yes, my child can do that independently.")
        
        # Reset assessment
        tester.test_reset()
        
    elif args.command == "assessment":
        # Run a complete assessment
        logger.info("Running complete assessment flow")
        
        # Load responses from file if provided
        responses = None
        if args.responses:
            with open(args.responses, 'r') as f:
                responses = json.load(f)
        
        tester.run_complete_assessment(age=args.age, responses=responses)
        
    elif args.command == "keywords":
        # Test keywords workflow
        logger.info("Testing keywords update workflow")
        tester.test_keywords_workflow()
        
    elif args.command == "load":
        # Run load test
        logger.info(f"Running load test for {args.endpoint}")
        
        # Load data from file if provided
        data = None
        if args.data:
            with open(args.data, 'r') as f:
                data = json.load(f)
        
        tester.run_load_test(
            endpoint=args.endpoint,
            method=args.method,
            data=data,
            requests_count=args.count,
            concurrent=args.concurrent
        )
    else:
        # No command specified, run minimal health check
        logger.info("No command specified, running health check")
        tester.test_health()
    
    # Generate report
    tester.save_results_to_json()
    tester.generate_html_report()
    
    logger.info("Testing complete")


if __name__ == "__main__":
    main() 