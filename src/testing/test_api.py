#!/usr/bin/env python3
"""
API Testing Framework for ASD Developmental Milestone Assessment API

This script provides a comprehensive testing framework for the ASD Assessment API.
It can test various endpoints, generate performance metrics, and create detailed reports.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import requests
import pandas as pd
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Template
import matplotlib.pyplot as plt
from tqdm import tqdm

# Default API URL
DEFAULT_API_URL = "http://localhost:8002"

# Test result class
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
        self.timestamp = datetime.now()

    def __str__(self):
        """String representation of the test result"""
        status = "PASS" if self.success else "FAIL"
        return f"{status} - {self.endpoint} - {self.response_time:.3f}s - Code: {self.status_code}"

    def to_dict(self):
        """Convert test result to dictionary"""
        return {
            "endpoint": self.endpoint,
            "success": self.success,
            "response_time": self.response_time,
            "status_code": self.status_code,
            "details": self.details,
            "expected": str(self.expected) if self.expected is not None else None,
            "actual": str(self.actual) if self.actual is not None else None,
            "timestamp": self.timestamp.isoformat()
        }

class APITester:
    """Class to test the ASD Assessment API"""
    
    def __init__(self, api_url: str = DEFAULT_API_URL, verbose: bool = False):
        """
        Initialize the API tester
        
        Args:
            api_url: Base URL for the API
            verbose: Whether to print verbose output
        """
        self.api_url = api_url
        self.verbose = verbose
        self.results = []
        self.session = requests.Session()
        self.child_age = None  # Track current child age
        
        if self.verbose:
            print(f"Initialized API tester for {api_url}")

    def set_verbose(self, verbose: bool):
        """Set verbose mode"""
        self.verbose = verbose
    
    def log(self, message: str):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            print(message)

    def make_request(self, method: str, endpoint: str, data: Dict = None, 
                     params: Dict = None, expected_status: int = 200) -> TestResult:
        """
        Make an HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            data: JSON data to send
            params: URL parameters
            expected_status: Expected HTTP status code
            
        Returns:
            TestResult object with the test results
        """
        url = f"{self.api_url}{endpoint}"
        self.log(f"Making {method} request to {url}")
        
        if data and self.verbose:
            self.log(f"Request data: {json.dumps(data, indent=2)}")
        
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response_time = time.time() - start_time
            
            # Try to parse response as JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"text": response.text}
            
            # Check if status code matches expected
            success = response.status_code == expected_status
            
            # Create test result
            result = TestResult(
                endpoint=endpoint,
                success=success,
                response_time=response_time,
                details={"request": data, "response": response_data},
                status_code=response.status_code
            )
            
            # Log results
            if self.verbose:
                self.log(f"Response ({response.status_code}) in {response_time:.3f}s")
                self.log(f"Response data: {json.dumps(response_data, indent=2)}")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            error_result = TestResult(
                endpoint=endpoint,
                success=False,
                response_time=response_time,
                details={"error": str(e), "request": data},
                status_code=None
            )
            self.log(f"Error: {str(e)}")
            self.results.append(error_result)
            return error_result

    def test_health(self) -> TestResult:
        """Test API health by accessing next-milestone endpoint"""
        return self.make_request("GET", "/next-milestone")

    def test_set_child_age(self, age: int) -> TestResult:
        """Test setting the child's age"""
        self.child_age = age
        data = {"age": age}
        result = self.make_request("POST", "/set-child-age", data)
        
        # Check if response contains expected data
        success = result.success
        if success and "message" not in result.details["response"]:
            result.success = False
            result.details["reason"] = "Response missing message"
        
        return result

    def test_next_milestone(self) -> TestResult:
        """Test getting the next milestone"""
        if self.child_age is None:
            self.log("Warning: Child age not set, setting to 24 months")
            self.test_set_child_age(24)
            
        result = self.make_request("GET", "/next-milestone")
        
        # Check if response contains expected fields
        success = result.success
        if success:
            response = result.details["response"]
            if "behavior" not in response or not response["behavior"]:
                result.success = False
                result.details["reason"] = "Response missing behavior data"
            elif "domain" not in response or not response["domain"]:
                result.success = False
                result.details["reason"] = "Response missing domain data"
            elif "age_range" not in response or not response["age_range"]:
                result.success = False
                result.details["reason"] = "Response missing age_range data"
        
        return result

    def test_score_response(self, milestone: str, response_text: str) -> TestResult:
        """Test scoring a caregiver's response"""
        data = {
            "milestone_behavior": milestone,
            "response": response_text
        }
        
        result = self.make_request("POST", "/score-response", data)
        
        # Check if response contains expected fields
        success = result.success
        if success:
            response = result.details["response"]
            if "score" not in response:
                result.success = False
                result.details["reason"] = "Response missing score data"
        
        return result

    def test_batch_score(self, batch_data: List[Dict]) -> List[TestResult]:
        """Test batch scoring multiple responses"""
        # Convert the batch data to the format expected by the API
        formatted_batch = []
        for item in batch_data:
            formatted_item = {
                "milestone_behavior": item.get("milestone_name", item.get("milestone", "")),
                "response": item.get("response_text", item.get("response", item.get("caregiver_response", "")))
            }
            formatted_batch.append(formatted_item)
        
        data = {"responses": formatted_batch}
        
        result = self.make_request("POST", "/batch-score", data)
        
        # Check if response contains expected fields
        success = result.success
        if success:
            response = result.details["response"]
            if not isinstance(response, list):
                result.success = False
                result.details["reason"] = "Response is not a list"
            elif len(response) != len(batch_data):
                result.success = False
                result.details["reason"] = f"Expected {len(batch_data)} results, got {len(response)}"
        
        # For each individual result in the batch, create a separate TestResult
        individual_results = []
        
        if success and "results" in result.details["response"]:
            batch_results = result.details["response"]["results"]
            
            for i, (test_case, api_result) in enumerate(zip(batch_data, batch_results)):
                milestone = test_case.get("milestone_name", "")
                expected_score = test_case.get("expected_score")
                actual_score = api_result.get("score")
                
                individual_result = TestResult(
                    endpoint=f"/batch-score[{i}]",
                    success=actual_score is not None,
                    response_time=result.response_time / len(batch_data),
                    details={
                        "milestone": milestone,
                        "response": test_case.get("response_text", ""),
                        "api_result": api_result
                    },
                    status_code=result.status_code,
                    expected=expected_score,
                    actual=actual_score
                )
                individual_results.append(individual_result)
        
        # Add all individual results to the main results list
        self.results.extend(individual_results)
        
        return [result] + individual_results if individual_results else [result]

    def test_generate_report(self) -> TestResult:
        """Test generating a developmental report"""
        if self.child_age is None:
            self.log("Warning: Child age not set, setting to 24 months")
            self.test_set_child_age(24)
            
        result = self.make_request("GET", "/generate-report")
        
        # Check if response contains expected fields
        success = result.success
        if success:
            response = result.details["response"]
            if "domain_quotients" not in response:
                result.success = False
                result.details["reason"] = "Response missing domain_quotients data"
        
        return result

    def run_tests(self, test_data: List[Dict] = None, age: int = 24, domains: List[str] = None):
        """Run a comprehensive set of tests"""
        print(f"Running API tests against {self.api_url}...")
        
        # Test API health
        print("Running API health check...")
        health_result = self.test_health()
        if not health_result.success:
            print("API health check failed! Aborting tests.")
            return
        
        # Set child age
        print(f"Setting child age to {age} months...")
        age_result = self.test_set_child_age(age)
        if not age_result.success:
            print("Warning: Failed to set child age. Continuing with limited tests.")
        
        # Test next milestone
        print("Testing next milestone endpoint...")
        milestone_result = self.test_next_milestone()
        
        # If we have test data, run scoring tests
        if test_data:
            print(f"Using {len(test_data)} test cases...")
            
            # Test individual response scoring
            print("Testing individual response scoring...")
            print(f"Running {len(test_data)} scoring tests sequentially...")
            
            with tqdm(total=len(test_data), desc="Testing scoring") as pbar:
                for i, test_case in enumerate(test_data):
                    milestone = test_case.get("milestone", "")
                    response = test_case.get("caregiver_response", "")
                    
                    result = self.test_score_response(milestone, response)
                    pbar.update(1)
            
            # Test batch scoring
            print("Testing batch scoring...")
            batch_result = self.test_batch_score(test_data)
        
        # Test report generation
        print("Testing report generation...")
        report_result = self.test_generate_report()
        
        # Print summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        
        print("\nTest Summary:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests / total_tests * 100:.1f}%")
        
        # Calculate average response time
        avg_time = statistics.mean(r.response_time for r in self.results)
        print(f"Average Response Time: {avg_time:.3f}s")

    def _generate_summary(self) -> Dict:
        """Generate a summary of test results"""
        summary = {
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "average_response_time": 0,
            "endpoints": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Calculate average response time
        if summary["total_tests"] > 0:
            summary["average_response_time"] = statistics.mean(
                r.response_time for r in self.results
            )
        
        # Group by endpoint
        endpoint_results = {}
        for result in self.results:
            base_endpoint = result.endpoint.split('[')[0]  # Handle batch results
            if base_endpoint not in endpoint_results:
                endpoint_results[base_endpoint] = []
            endpoint_results[base_endpoint].append(result)
        
        # Calculate per-endpoint stats
        for endpoint, results in endpoint_results.items():
            summary["endpoints"][endpoint] = {
                "total": len(results),
                "passed": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "average_response_time": statistics.mean(r.response_time for r in results),
                "status_codes": {},
            }
            
            # Count status codes
            for result in results:
                if result.status_code:
                    status_code = str(result.status_code)
                    if status_code not in summary["endpoints"][endpoint]["status_codes"]:
                        summary["endpoints"][endpoint]["status_codes"][status_code] = 0
                    summary["endpoints"][endpoint]["status_codes"][status_code] += 1
        
        return summary

    def generate_html_report(self, summary: Dict, output_file: str = "api_test_report.html") -> str:
        """
        Generate an HTML report of test results
        
        Args:
            summary: Test summary dictionary
            output_file: Path to save the HTML report
            
        Returns:
            Path to the generated report file
        """
        # Convert results to DataFrames for better analysis
        results_data = []
        for result in self.results:
            results_data.append({
                "Endpoint": result.endpoint,
                "Success": result.success,
                "Response Time (s)": result.response_time,
                "Status Code": result.status_code or "Error",
                "Expected": result.expected,
                "Actual": result.actual,
                "Timestamp": result.timestamp
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Create charts for the report
        charts_data = {}
        
        # Create success rate chart
        if not results_df.empty:
            plt.figure(figsize=(10, 6))
            success_counts = results_df["Success"].value_counts()
            labels = ["Passed", "Failed"]
            values = [success_counts.get(True, 0), success_counts.get(False, 0)]
            colors = ["#28a745", "#dc3545"]  # green and red
            
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Test Success Rate')
            
            # Save the chart
            success_chart = "success_rate_chart.png"
            plt.savefig(success_chart)
            charts_data["success_chart"] = success_chart
            plt.close()
            
            # Create response time by endpoint chart
            plt.figure(figsize=(12, 6))
            endpoint_times = results_df.groupby("Endpoint")["Response Time (s)"].mean().sort_values(ascending=False)
            
            bars = plt.bar(endpoint_times.index, endpoint_times.values, color="#17a2b8")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.title('Average Response Time by Endpoint')
            plt.ylabel('Response Time (seconds)')
            
            # Add time values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                         f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
            
            # Save the chart
            time_chart = "response_time_chart.png"
            plt.savefig(time_chart)
            charts_data["time_chart"] = time_chart
            plt.close()
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Test Results</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .summary-box {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .endpoint-box {
                    background-color: #ffffff;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-left: 5px solid #17a2b8;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .charts {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin: 20px 0;
                }
                .chart-container {
                    flex: 1;
                    min-width: 300px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    padding: 15px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .stat {
                    display: inline-block;
                    margin-right: 20px;
                    margin-bottom: 10px;
                }
                .stat-label {
                    font-weight: bold;
                    margin-right: 5px;
                }
                .pass {
                    color: #28a745;
                }
                .fail {
                    color: #dc3545;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #e0e0e0;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .result-box {
                    padding: 5px 10px;
                    border-radius: 4px;
                    display: inline-block;
                }
                .pass-bg {
                    background-color: #d4edda;
                }
                .fail-bg {
                    background-color: #f8d7da;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ASD Assessment API Test Report</h1>
                <p>Report generated on {{ summary.timestamp }}</p>
                
                <div class="summary-box">
                    <h2>Summary</h2>
                    <div class="stat">
                        <span class="stat-label">Total Tests:</span>
                        <span>{{ summary.total_tests }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Passed:</span>
                        <span class="pass">{{ summary.passed }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Failed:</span>
                        <span class="fail">{{ summary.failed }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Success Rate:</span>
                        <span>{{ "%.1f"|format(summary.passed / summary.total_tests * 100 if summary.total_tests > 0 else 0) }}%</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Avg. Response Time:</span>
                        <span>{{ "%.3f"|format(summary.average_response_time) }}s</span>
                    </div>
                </div>
                
                {% if charts %}
                <div class="charts">
                    {% if charts.success_chart %}
                    <div class="chart-container">
                        <h3>Success Rate</h3>
                        <img src="{{ charts.success_chart }}" alt="Success rate chart" style="max-width:100%;">
                    </div>
                    {% endif %}
                    
                    {% if charts.time_chart %}
                    <div class="chart-container">
                        <h3>Response Times</h3>
                        <img src="{{ charts.time_chart }}" alt="Response time chart" style="max-width:100%;">
                    </div>
                    {% endif %}
                </div>
                {% endif %}
                
                <h2>Endpoint Results</h2>
                {% for endpoint, data in summary.endpoints.items() %}
                <div class="endpoint-box">
                    <h3>{{ endpoint }}</h3>
                    <div class="stat">
                        <span class="stat-label">Total:</span>
                        <span>{{ data.total }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Passed:</span>
                        <span class="pass">{{ data.passed }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Failed:</span>
                        <span class="fail">{{ data.failed }}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Success Rate:</span>
                        <span>{{ "%.1f"|format(data.passed / data.total * 100 if data.total > 0 else 0) }}%</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Avg. Response Time:</span>
                        <span>{{ "%.3f"|format(data.average_response_time) }}s</span>
                    </div>
                    
                    {% if data.status_codes %}
                    <div>
                        <span class="stat-label">Status Codes:</span>
                        {% for code, count in data.status_codes.items() %}
                        <span>{{ code }}: {{ count }}</span>{% if not loop.last %}, {% endif %}
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
                
                <h2>Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Endpoint</th>
                            <th>Status</th>
                            <th>Response Time</th>
                            <th>Status Code</th>
                            <th>Expected vs Actual</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in results %}
                        <tr>
                            <td>{{ result.Endpoint }}</td>
                            <td>
                                <div class="result-box {{ 'pass-bg' if result.Success else 'fail-bg' }}">
                                    {{ "PASS" if result.Success else "FAIL" }}
                                </div>
                            </td>
                            <td>{{ "%.3f"|format(result["Response Time (s)"]) }}s</td>
                            <td>{{ result["Status Code"] }}</td>
                            <td>
                                {% if result.Expected is not none and result.Actual is not none %}
                                {{ result.Expected }} vs {{ result.Actual }}
                                {% else %}
                                -
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Render the template
        template = Template(html_template)
        html_content = template.render(
            summary=summary,
            charts=charts_data,
            results=results_data
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_file

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ASD Assessment API Test Framework")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"API base URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--age", type=int, default=24, help="Child's age in months (default: 24)")
    parser.add_argument("--data", help="Path to test data JSON file")
    parser.add_argument("--tests", type=int, default=10, help="Number of test cases to run (default: 10)")
    parser.add_argument("--domains", help="Comma-separated list of domains to test")
    parser.add_argument("--report", default="api_test_report.html", help="Output file for HTML report (default: api_test_report.html)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--edge-cases", action="store_true", help="Include edge cases in tests")
    parser.add_argument("--concurrent", action="store_true", help="Run tests concurrently where possible")
    args = parser.parse_args()
    
    # Parse domains if provided
    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]
    
    # Initialize tester
    tester = APITester(api_url=args.url, verbose=args.verbose)
    
    # Load test data if provided
    test_data = []
    if args.data and os.path.exists(args.data):
        print(f"Loading test data from {args.data}...")
        try:
            with open(args.data, 'r') as f:
                data = json.load(f)
                if "test_cases" in data:
                    test_data = data["test_cases"]
                else:
                    test_data = data
        except Exception as e:
            print(f"Error loading test data: {str(e)}")
    
    # Run test suite
    print(f"Running API tests against {args.url}...")
    tester.run_tests(
        test_data=test_data,
        age=args.age,
        domains=domains
    )
    
    # Generate HTML report
    report_file = tester.generate_html_report(tester._generate_summary(), args.report)
    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    main() 