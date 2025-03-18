#!/usr/bin/env python3
"""
ASD Assessment System - CLI Testing Framework

This command-line tool provides a comprehensive framework for testing
and demonstrating the ASD assessment system. It allows users to:

1. Start and manage the API server
2. Test individual responses
3. Run batch tests
4. Generate reports
5. Run benchmarks
6. Start interactive assessments

Usage:
  python asd_test_cli.py [COMMAND] [OPTIONS]

Examples:
  python asd_test_cli.py start-api
  python asd_test_cli.py health
  python asd_test_cli.py set-age 24
  python asd_test_cli.py milestone
  python asd_test_cli.py test "walks independently" "yes, she can walk"
  python asd_test_cli.py batch-test
  python asd_test_cli.py demo
  python asd_test_cli.py report
  python asd_test_cli.py benchmark
  python asd_test_cli.py interactive
"""

import os
import sys
import time
import json
import requests
import pandas as pd
import random
import subprocess
import click
import signal
import matplotlib.pyplot as plt
from tabulate import tabulate
from typing import Dict, List, Any, Optional, Tuple
from termcolor import colored
import inquirer
from datetime import datetime

# API configuration
API_BASE_URL = "http://localhost:8002"
API_PORT = 8002

# Define constants
SCORE_LABELS = {
    0: "CANNOT_DO",
    1: "LOST_SKILL", 
    2: "EMERGING",
    3: "WITH_SUPPORT",
    4: "INDEPENDENT"
}

SCORE_DESCRIPTIONS = {
    0: "Skill not acquired",
    1: "Acquired but lost",
    2: "Emerging and inconsistent",
    3: "Acquired but consistent in specific situations only",
    4: "Acquired and present in all situations"
}

# Terminal colors
SCORE_COLORS = {
    0: "red",        # CANNOT_DO
    1: "yellow",     # LOST_SKILL
    2: "blue",       # EMERGING
    3: "cyan",       # WITH_SUPPORT
    4: "green"       # INDEPENDENT
}

# Example milestone domains
MILESTONE_DOMAINS = {
    "GM": "Gross Motor",
    "FM": "Fine Motor",
    "RL": "Receptive Language",
    "EL": "Expressive Language",
    "PS": "Problem Solving",
    "SE": "Social-Emotional"
}

# Response templates for generating test data
RESPONSE_TEMPLATES = {
    0: [  # CANNOT_DO
        "No, {pronoun} cannot {action}",
        "Not at all, {pronoun} has never {action}",
        "{pronoun} doesn't {action}",
        "No, not yet"
    ],
    1: [  # LOST_SKILL
        "{pronoun} used to {action} but has regressed",
        "{pronoun} previously could {action} but not anymore",
        "{pronoun} no longer {action}",
        "{pronoun} lost the ability to {action}"
    ],
    2: [  # EMERGING
        "Sometimes, but not consistently",
        "{pronoun} is just beginning to {action}",
        "{pronoun} occasionally {action}",
        "{pronoun} is trying to {action}"
    ],
    3: [  # WITH_SUPPORT
        "With help, {pronoun} can {action}",
        "{pronoun} needs support to {action}",
        "When prompted, {pronoun} will {action}",
        "{pronoun} can {action} with assistance"
    ],
    4: [  # INDEPENDENT
        "Yes, {pronoun} {action} independently",
        "{pronoun} consistently {action}",
        "Definitely, {pronoun} always {action}",
        "Yes, very well"
    ]
}

# API server process
api_process = None

# API Server Management
class APIServerManager:
    @staticmethod
    def start_server():
        """Start the API server as a background process"""
        if APIServerManager.check_running():
            click.echo(click.style("✓ API server is already running", fg="green"))
            return True
        
        try:
            # Start the server
            process = subprocess.Popen(
                ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(API_PORT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            global api_process
            api_process = process
            
            # Wait for API to start (with timeout)
            max_attempts = 10
            for i in range(max_attempts):
                if APIServerManager.check_running():
                    click.echo(click.style(f"✓ API server started on port {API_PORT}", fg="green"))
                    return True
                
                click.echo(f"Waiting for API server to start... ({i+1}/{max_attempts})")
                time.sleep(1)
            
            click.echo(click.style("✗ Failed to start API server (timeout)", fg="red"))
            return False
        except Exception as e:
            click.echo(click.style(f"✗ Failed to start API server: {str(e)}", fg="red"))
            return False
    
    @staticmethod
    def stop_server():
        """Stop the API server"""
        global api_process
        
        if api_process:
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
            
            api_process = None
            click.echo(click.style("✓ API server stopped", fg="green"))
            return True
        else:
            # Try to kill any running instances by port
            try:
                os.system("pkill -f 'uvicorn.*app:app.*8002'")
                click.echo(click.style("✓ API server stopped", fg="green"))
                return True
            except:
                click.echo(click.style("✗ No API server running", fg="yellow"))
                return False
    
    @staticmethod
    def check_running():
        """Check if the API is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

# API Request Manager
class APIRequestManager:
    @staticmethod
    def make_request(method, endpoint, data=None):
        """Make an API request with proper error handling"""
        if not APIServerManager.check_running():
            click.echo(click.style("✗ API server is not running. Use 'start-api' command first.", fg="red"))
            return None
        
        try:
            url = f"{API_BASE_URL}/{endpoint}"
            if method.lower() == "get":
                response = requests.get(url, timeout=10)
            elif method.lower() == "post":
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                click.echo(click.style(f"✗ Unsupported HTTP method: {method}", fg="red"))
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            click.echo(click.style(f"✗ API request error: {str(e)}", fg="red"))
            return None
    
    @staticmethod
    def set_child_age(age):
        """Set the child's age in the API"""
        result = APIRequestManager.make_request("post", "set-child-age", {"age": age})
        if result:
            click.echo(click.style(f"✓ Child age set to {age} months", fg="green"))
            return True
        return False
    
    @staticmethod
    def get_next_milestone():
        """Get the next milestone from the API"""
        return APIRequestManager.make_request("get", "next-milestone")
    
    @staticmethod
    def score_response(milestone, response):
        """Score a response for a milestone"""
        data = {
            "milestone_behavior": milestone,
            "response": response
        }
        return APIRequestManager.make_request("post", "score-response", data)
    
    @staticmethod
    def generate_report():
        """Generate a developmental report"""
        return APIRequestManager.make_request("get", "generate-report")

# Display utilities
class DisplayUtilities:
    @staticmethod
    def display_milestone(milestone):
        """Display milestone information in a formatted way"""
        if not milestone:
            click.echo(click.style("✗ No milestone available", fg="red"))
            return
        
        if milestone.get("complete", False):
            click.echo(click.style("✓ Assessment complete!", fg="green"))
            return
        
        click.echo()
        click.echo("═════════════════════════════════════")
        click.echo(click.style("MILESTONE", fg="cyan", bold=True))
        click.echo("═════════════════════════════════════")
        click.echo(f"Behavior:   {milestone.get('behavior', 'N/A')}")
        click.echo(f"Criteria:   {milestone.get('criteria', 'N/A')}")
        click.echo(f"Domain:     {milestone.get('domain', 'N/A')} ({MILESTONE_DOMAINS.get(milestone.get('domain', ''), 'Unknown')})")
        click.echo(f"Age Range:  {milestone.get('age_range', 'N/A')}")
        click.echo("═════════════════════════════════════")
        click.echo()
    
    @staticmethod
    def display_score(result):
        """Display score information in a formatted way"""
        if not result:
            click.echo(click.style("✗ No score available", fg="red"))
            return
        
        milestone = result.get("milestone", "N/A")
        domain = result.get("domain", "N/A")
        score = result.get("score", -1)
        score_label = result.get("score_label", "UNKNOWN")
        color = SCORE_COLORS.get(score, "white")
        
        click.echo()
        click.echo("═════════════════════════════════════")
        click.echo(click.style("SCORE RESULT", fg="cyan", bold=True))
        click.echo("═════════════════════════════════════")
        click.echo(f"Milestone:  {milestone}")
        click.echo(f"Domain:     {domain} ({MILESTONE_DOMAINS.get(domain, 'Unknown')})")
        click.echo(f"Score:      {colored(f'{score} - {score_label}', color, attrs=['bold'])}")
        
        # Display interpretation
        description = SCORE_DESCRIPTIONS.get(score, "Unknown score")
        click.echo(f"Meaning:    {description}")
        click.echo("═════════════════════════════════════")
        click.echo()
    
    @staticmethod
    def display_report(report):
        """Display a formatted developmental report"""
        if not report:
            click.echo(click.style("✗ No report available", fg="red"))
            return
        
        scores = report.get("scores", [])
        domain_quotients = report.get("domain_quotients", {})
        
        if not scores and not domain_quotients:
            click.echo(click.style("✗ Report contains no data", fg="yellow"))
            return
        
        click.echo()
        click.echo("══════════════════════════════════════════════════")
        click.echo(click.style("DEVELOPMENTAL ASSESSMENT REPORT", fg="cyan", bold=True))
        click.echo("══════════════════════════════════════════════════")
        
        if domain_quotients:
            click.echo()
            click.echo(click.style("DOMAIN QUOTIENTS", fg="cyan"))
            click.echo("──────────────────────────────────────────────────")
            
            # Format domain data for table
            domain_data = []
            for domain, quotient in domain_quotients.items():
                if quotient > 0:  # Only include domains with scores
                    domain_name = MILESTONE_DOMAINS.get(domain, domain)
                    
                    # Choose color based on quotient
                    if quotient >= 75:
                        color = "green"
                    elif quotient >= 50:
                        color = "cyan"
                    elif quotient >= 25:
                        color = "blue"
                    else:
                        color = "red"
                    
                    domain_data.append([
                        domain,
                        domain_name,
                        colored(f"{quotient:.1f}%", color)
                    ])
            
            if domain_data:
                # Display as table
                headers = ["Code", "Domain", "Quotient"]
                click.echo(tabulate(domain_data, headers=headers, tablefmt="simple"))
            else:
                click.echo(click.style("No domain quotients available", fg="yellow"))
        
        if scores:
            click.echo()
            click.echo(click.style("MILESTONE SCORES", fg="cyan"))
            click.echo("──────────────────────────────────────────────────")
            
            # Format score data for table
            score_data = []
            for score_item in scores:
                milestone = score_item.get("milestone", "")
                domain = score_item.get("domain", "")
                age_range = score_item.get("age_range", "")
                score = score_item.get("score", -1)
                score_label = score_item.get("score_label", "")
                color = SCORE_COLORS.get(score, "white")
                
                score_data.append([
                    milestone[:40] + "..." if len(milestone) > 40 else milestone,
                    domain,
                    age_range,
                    colored(f"{score} - {score_label}", color)
                ])
            
            if score_data:
                # Display as table
                headers = ["Milestone", "Domain", "Age Range", "Score"]
                click.echo(tabulate(score_data, headers=headers, tablefmt="simple"))
            else:
                click.echo(click.style("No milestone scores available", fg="yellow"))
        
        click.echo("══════════════════════════════════════════════════")
        click.echo()

# Test data generation
class TestDataGenerator:
    @staticmethod
    def generate_response(milestone, target_score):
        """Generate a response for the given milestone targeting a specific score"""
        templates = RESPONSE_TEMPLATES.get(target_score, [])
        if not templates:
            return "Yes"  # Default fallback
        
        template = random.choice(templates)
        
        # Extract action from milestone (simple heuristic)
        behavior = milestone.get("behavior", "")
        if isinstance(behavior, str):
            behavior = behavior.lower()
        else:
            behavior = ""
        
        # Try to extract the verb phrase
        action = behavior
        
        # Choose pronoun randomly
        pronoun = random.choice(["he", "she", "they"])
        
        # Format the template
        return template.format(pronoun=pronoun, action=action)

# CLI command group
@click.group()
def cli():
    """ASD Assessment System - Testing Framework
    
    This command-line tool provides a comprehensive framework for testing
    and demonstrating the ASD assessment system.
    """
    pass

# Command: start-api
@cli.command()
def start_api():
    """Start the API server."""
    APIServerManager.start_server()

# Command: stop-api
@cli.command()
def stop_api():
    """Stop the API server."""
    APIServerManager.stop_server()

# Command: health
@cli.command()
def health():
    """Check if the API server is running."""
    running = APIServerManager.check_running()
    if running:
        click.echo(click.style("✓ API server is running", fg="green"))
    else:
        click.echo(click.style("✗ API server is not running", fg="red"))

# Command: set-age
@cli.command()
@click.argument("age", type=int)
def set_age(age):
    """Set the child's age in months."""
    if age < 0 or age > 144:
        click.echo(click.style("✗ Age must be between 0 and 144 months", fg="red"))
        return
    
    APIRequestManager.set_child_age(age)

# Command: milestone
@cli.command()
def milestone():
    """Get the next milestone for assessment."""
    milestone = APIRequestManager.get_next_milestone()
    DisplayUtilities.display_milestone(milestone)

# Command: test
@cli.command()
@click.argument("milestone_behavior")
@click.argument("response")
def test(milestone_behavior, response):
    """Test a specific response for a milestone."""
    result = APIRequestManager.score_response(milestone_behavior, response)
    DisplayUtilities.display_score(result)

# Command: batch-test
@cli.command()
@click.option("--count", "-c", default=5, help="Number of tests to run")
@click.option("--age", "-a", default=24, help="Child age in months")
def batch_test(count, age):
    """Run batch tests with various responses."""
    # Set child age
    if not APIRequestManager.set_child_age(age):
        return
    
    click.echo(click.style(f"Running {count} batch tests...", fg="cyan"))
    
    # Get first milestone
    milestone = APIRequestManager.get_next_milestone()
    
    if not milestone:
        click.echo(click.style("✗ Failed to get milestone for batch testing", fg="red"))
        return
    
    # Track results
    results = []
    
    # Run tests
    for i in range(count):
        # Generate a random score target (0-4)
        score_target = random.randint(0, 4)
        
        # Generate response
        response = TestDataGenerator.generate_response(milestone, score_target)
        
        # Get milestone text
        milestone_text = milestone.get("behavior", "")
        
        # Score the response
        click.echo(f"Test {i+1}/{count}: Testing response: \"{response}\"")
        result = APIRequestManager.score_response(milestone_text, response)
        
        if result:
            score = result.get("score", -1)
            score_label = result.get("score_label", "UNKNOWN")
            color = SCORE_COLORS.get(score, "white")
            
            click.echo(f"  Score: {colored(f'{score} - {score_label}', color)}")
            
            results.append({
                "target_score": score_target,
                "actual_score": score,
                "response": response,
                "milestone": milestone_text
            })
        else:
            click.echo(click.style("  ✗ Failed to score response", fg="red"))
    
    # Show summary
    if results:
        click.echo()
        click.echo("═════════════════════════════════════")
        click.echo(click.style("BATCH TEST SUMMARY", fg="cyan", bold=True))
        click.echo("═════════════════════════════════════")
        
        # Count matches between target and actual
        matches = sum(1 for r in results if r["target_score"] == r["actual_score"])
        match_rate = (matches / len(results)) * 100 if results else 0
        
        click.echo(f"Tests run:       {len(results)}")
        click.echo(f"Target matches:  {matches}/{len(results)} ({match_rate:.1f}%)")
        
        # Count by score
        score_counts = {}
        for r in results:
            score = r["actual_score"]
            score_counts[score] = score_counts.get(score, 0) + 1
        
        click.echo()
        click.echo("Score distribution:")
        for score in range(5):
            count = score_counts.get(score, 0)
            percentage = (count / len(results)) * 100 if results else 0
            color = SCORE_COLORS.get(score, "white")
            click.echo(f"  {colored(f'{score} - {SCORE_LABELS.get(score, 'UNKNOWN')}', color)}: {count} ({percentage:.1f}%)")
        
        click.echo("═════════════════════════════════════")
        click.echo()

# Command: demo
@cli.command()
def demo():
    """Run a demonstration with various responses."""
    click.echo(click.style("ASD Assessment System - Demo Mode", fg="cyan", bold=True))
    click.echo("This demo will show how different responses are scored.")
    
    # Set age to 24 months
    if not APIRequestManager.set_child_age(24):
        return
    
    # Get a milestone
    milestone = APIRequestManager.get_next_milestone()
    if not milestone:
        click.echo(click.style("✗ Failed to get milestone for demo", fg="red"))
        return
    
    # Display milestone
    DisplayUtilities.display_milestone(milestone)
    
    milestone_text = milestone.get("behavior", "")
    
    # Test each score level
    click.echo(click.style("Testing responses for each score level:", fg="cyan"))
    
    for score in range(5):
        # Generate a response for this score level
        response = TestDataGenerator.generate_response(milestone, score)
        
        click.echo()
        click.echo(f"Testing {colored(SCORE_LABELS.get(score, 'UNKNOWN'), SCORE_COLORS.get(score, 'white'), attrs=['bold'])} response:")
        click.echo(f"Response: \"{response}\"")
        
        # Score the response
        result = APIRequestManager.score_response(milestone_text, response)
        
        if result:
            actual_score = result.get("score", -1)
            score_label = result.get("score_label", "UNKNOWN")
            color = SCORE_COLORS.get(actual_score, "white")
            
            click.echo(f"Score: {colored(f'{actual_score} - {score_label}', color)}")
            
            # Show if target was matched
            if actual_score == score:
                click.echo(click.style("Target score matched ✓", fg="green"))
            else:
                click.echo(click.style(f"Target score not matched ✗ (expected {score})", fg="yellow"))
        else:
            click.echo(click.style("✗ Failed to score response", fg="red"))
        
        click.echo("─────────────────────────────────────")
    
    click.echo()
    click.echo(click.style("Demo complete!", fg="green"))

# Command: report
@cli.command()
def report():
    """Generate a developmental report."""
    report_data = APIRequestManager.generate_report()
    DisplayUtilities.display_report(report_data)

# Command: benchmark
@cli.command()
@click.option("--iterations", "-i", default=50, help="Number of requests to make")
def benchmark(iterations):
    """Run a performance benchmark."""
    if not APIServerManager.check_running():
        click.echo(click.style("✗ API server is not running. Use 'start-api' command first.", fg="red"))
        return
    
    click.echo(click.style(f"Running benchmark with {iterations} iterations...", fg="cyan"))
    
    # Set age to 24 months
    APIRequestManager.set_child_age(24)
    
    # Get a milestone to use for testing
    milestone = APIRequestManager.get_next_milestone()
    if not milestone:
        click.echo(click.style("✗ Failed to get milestone for benchmark", fg="red"))
        return
    
    milestone_text = milestone.get("behavior", "")
    
    # Prepare responses for each score level
    test_responses = [
        TestDataGenerator.generate_response(milestone, score) for score in range(5)
    ]
    
    # Run benchmark
    results = []
    start_time = time.time()
    
    with click.progressbar(range(iterations), label="Running tests") as bar:
        for i in bar:
            # Choose a random response
            response = random.choice(test_responses)
            
            # Score the response and measure time
            request_start = time.time()
            result = APIRequestManager.score_response(milestone_text, response)
            request_end = time.time()
            
            if result:
                results.append({
                    "response": response,
                    "score": result.get("score", -1),
                    "response_time": (request_end - request_start) * 1000  # ms
                })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    success_count = len(results)
    success_rate = (success_count / iterations) * 100 if iterations > 0 else 0
    
    if success_count > 0:
        response_times = [r["response_time"] for r in results]
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
    else:
        response_times = []
        avg_response_time = min_response_time = max_response_time = 0
    
    rps = iterations / total_time if total_time > 0 else 0
    
    # Count by score level
    score_counts = {}
    for r in results:
        score = r["score"]
        score_counts[score] = score_counts.get(score, 0) + 1
    
    # Display results
    click.echo()
    click.echo("═════════════════════════════════════")
    click.echo(click.style("BENCHMARK RESULTS", fg="cyan", bold=True))
    click.echo("═════════════════════════════════════")
    click.echo(f"Iterations:         {iterations}")
    click.echo(f"Successful requests: {success_count}")
    click.echo(f"Success rate:       {success_rate:.1f}%")
    click.echo(f"Total time:         {total_time:.2f} seconds")
    click.echo(f"Requests per second: {rps:.2f}")
    click.echo(f"Avg response time:  {avg_response_time:.2f} ms")
    click.echo(f"Min response time:  {min_response_time:.2f} ms")
    click.echo(f"Max response time:  {max_response_time:.2f} ms")
    
    click.echo()
    click.echo("Score distribution:")
    for score in range(5):
        count = score_counts.get(score, 0)
        percentage = (count / success_count) * 100 if success_count > 0 else 0
        color = SCORE_COLORS.get(score, "white")
        click.echo(f"  {colored(f'{score} - {SCORE_LABELS.get(score, 'UNKNOWN')}', color)}: {count} ({percentage:.1f}%)")
    
    click.echo("═════════════════════════════════════")
    click.echo()

# Command: interactive
@cli.command()
def interactive():
    """Start an interactive assessment session."""
    click.echo(click.style("ASD Assessment System - Interactive Mode", fg="cyan", bold=True))
    
    # Prompt for child age
    questions = [
        inquirer.Text('age', message="Enter child's age in months (0-144)", validate=lambda _, x: 0 <= int(x) <= 144)
    ]
    answers = inquirer.prompt(questions)
    
    if not answers:
        return
    
    age = int(answers['age'])
    
    # Set child age
    if not APIRequestManager.set_child_age(age):
        return
    
    # Interactive assessment loop
    assessment_complete = False
    responses_given = 0
    
    while not assessment_complete:
        # Get next milestone
        milestone = APIRequestManager.get_next_milestone()
        
        if not milestone:
            click.echo(click.style("✗ Failed to get milestone", fg="red"))
            break
        
        # Check if assessment is complete
        if milestone.get("complete", False):
            assessment_complete = True
            click.echo(click.style("Assessment complete!", fg="green"))
            break
        
        # Display milestone
        DisplayUtilities.display_milestone(milestone)
        
        # Prompt for response
        milestone_text = milestone.get("behavior", "")
        
        response_options = [
            f"Cannot do (level 0): {TestDataGenerator.generate_response(milestone, 0)}",
            f"Lost skill (level 1): {TestDataGenerator.generate_response(milestone, 1)}",
            f"Emerging (level 2): {TestDataGenerator.generate_response(milestone, 2)}",
            f"With support (level 3): {TestDataGenerator.generate_response(milestone, 3)}",
            f"Independent (level 4): {TestDataGenerator.generate_response(milestone, 4)}",
            "Custom response"
        ]
        
        questions = [
            inquirer.List('response',
                          message=f"How would you respond to: \"{milestone_text}\"?",
                          choices=response_options)
        ]
        
        answers = inquirer.prompt(questions)
        
        if not answers:
            break
        
        selected_response = answers['response']
        
        # Handle custom response
        if selected_response == "Custom response":
            questions = [
                inquirer.Text('custom', message="Enter your custom response")
            ]
            custom_answers = inquirer.prompt(questions)
            
            if not custom_answers:
                continue
            
            response_text = custom_answers['custom']
        else:
            # Extract the example response from the selected option
            response_text = selected_response.split(": ", 1)[1]
        
        # Score the response
        click.echo(f"Scoring response: \"{response_text}\"")
        result = APIRequestManager.score_response(milestone_text, response_text)
        
        if result:
            DisplayUtilities.display_score(result)
            responses_given += 1
        else:
            click.echo(click.style("✗ Failed to score response", fg="red"))
    
    # Generate report if at least one response was given
    if responses_given > 0:
        click.echo("Generating developmental report...")
        report_data = APIRequestManager.generate_report()
        DisplayUtilities.display_report(report_data)
    
    click.echo(click.style("Interactive session complete!", fg="green"))

# Register signal handler for clean exit
def signal_handler(sig, frame):
    """Handle Ctrl+C to clean up resources"""
    click.echo("\nExiting...")
    if api_process:
        APIServerManager.stop_server()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    cli() 