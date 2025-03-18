#!/usr/bin/env python3
"""
ASD Assessment System - Comprehensive Testing Framework

This is a user-friendly testing framework for the ASD assessment system,
allowing clients to easily test and validate the system's functionality.
It provides a simple interface for testing various aspects of the system,
including individual milestone assessments, batch testing, and report generation.

Usage:
  python asd_test_framework.py [command] [options]

Commands:
  start-api        Start the API server
  health           Check the API health
  set-age          Set the child's age
  milestone        Get the next milestone
  test             Test a specific milestone response
  batch-test       Run batch tests with predefined responses
  demo             Run a demonstration with a variety of responses
  report           Generate a developmental report
  benchmark        Run a performance benchmark
  interactive      Start an interactive assessment session
"""

import os
import sys
import time
import json
import click
import requests
import pandas as pd
import random
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from contextlib import contextmanager
from termcolor import colored

# API configuration
API_BASE_URL = "http://localhost:8002"  # Default API URL
PORT = 8002  # Default port

# Milestone domains
DOMAINS = ["GM", "FM", "ADL", "RL", "EL", "Cog", "SOC", "Emo"]

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

SCORE_COLORS = {
    0: "red",
    1: "red",
    2: "yellow",
    3: "blue",
    4: "green"
}

# Context manager for API server
@contextmanager
def api_server():
    """Context manager to start and stop the API server"""
    process = None
    try:
        # Check if API is already running
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                click.echo(click.style("API server is already running", fg="green"))
                yield
                return
        except requests.RequestException:
            pass
        
        # Start API server
        click.echo(click.style("Starting API server...", fg="yellow"))
        process = subprocess.Popen(
            ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for API to start
        max_attempts = 10
        for i in range(max_attempts):
            try:
                response = requests.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    click.echo(click.style("API server started successfully", fg="green"))
                    break
            except requests.RequestException:
                if i < max_attempts - 1:
                    click.echo(click.style(f"Waiting for API to start (attempt {i+1}/{max_attempts})...", fg="yellow"))
                    time.sleep(2)
                else:
                    click.echo(click.style("Failed to start API server", fg="red"))
                    raise
        
        yield
    finally:
        if process:
            click.echo(click.style("Stopping API server...", fg="yellow"))
            process.terminate()
            process.wait(timeout=5)
            click.echo(click.style("API server stopped", fg="green"))

def make_api_request(method, endpoint, data=None):
    """Make an API request and handle errors consistently"""
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method.lower() == "get":
            response = requests.get(url)
        elif method.lower() == "post":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers)
        else:
            click.echo(click.style(f"Unsupported HTTP method: {method}", fg="red"))
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        click.echo(click.style(f"API request error: {str(e)}", fg="red"))
        return None

def display_milestone(milestone):
    """Display milestone information in a formatted way"""
    if not milestone:
        click.echo(click.style("No milestone available", fg="red"))
        return

    click.echo(click.style("\nMilestone Details:", fg="blue", bold=True))
    click.echo(click.style(f"Behavior: ", fg="green") + milestone.get("behavior", "N/A"))
    click.echo(click.style(f"Criteria: ", fg="green") + milestone.get("criteria", "N/A"))
    click.echo(click.style(f"Domain: ", fg="green") + milestone.get("domain", "N/A"))
    click.echo(click.style(f"Age Range: ", fg="green") + milestone.get("age_range", "N/A"))
    click.echo("")

def display_score(result):
    """Display score information in a formatted way"""
    if not result:
        click.echo(click.style("No score available", fg="red"))
        return

    score = result.get("score", -1)
    score_label = result.get("score_label", "UNKNOWN")
    
    click.echo(click.style("\nScore Result:", fg="blue", bold=True))
    click.echo(click.style(f"Milestone: ", fg="green") + result.get("milestone", "N/A"))
    click.echo(click.style(f"Domain: ", fg="green") + result.get("domain", "N/A"))
    
    color = SCORE_COLORS.get(score, "white")
    description = SCORE_DESCRIPTIONS.get(score, "Unknown")
    
    click.echo(click.style(f"Score: ", fg="green") + 
               click.style(f"{score} - {score_label}", fg=color) + 
               f" ({description})")
    click.echo("")

def display_report(report):
    """Display report information in a formatted way"""
    if not report:
        click.echo(click.style("No report available", fg="red"))
        return

    scores = report.get("scores", [])
    domain_quotients = report.get("domain_quotients", {})
    
    click.echo(click.style("\nDevelopmental Report:", fg="blue", bold=True))
    
    if scores:
        click.echo(click.style("\nMilestone Scores:", fg="green", bold=True))
        for score in scores:
            milestone = score.get("milestone", "N/A")
            domain = score.get("domain", "N/A")
            score_value = score.get("score", -1)
            score_label = score.get("score_label", "UNKNOWN")
            age_range = score.get("age_range", "N/A")
            
            color = SCORE_COLORS.get(score_value, "white")
            click.echo(f"  {milestone} ({domain}, {age_range}): " + 
                       click.style(f"{score_value} - {score_label}", fg=color))
    
    if domain_quotients:
        click.echo(click.style("\nDomain Quotients:", fg="green", bold=True))
        for domain, quotient in domain_quotients.items():
            # Use appropriate color based on quotient value
            if quotient >= 75:
                color = "green"
            elif quotient >= 50:
                color = "blue"
            elif quotient >= 25:
                color = "yellow"
            else:
                color = "red"
                
            click.echo(f"  {domain}: " + click.style(f"{quotient:.1f}", fg=color))
    
    click.echo("")

def generate_response(milestone, target_score):
    """Generate a response for the given milestone targeting a specific score"""
    templates = RESPONSE_TEMPLATES.get(target_score, [])
    if not templates:
        return "Yes"  # Default fallback
    
    template = random.choice(templates)
    
    # Extract action from milestone (simple heuristic)
    behavior = milestone.get("behavior", "").lower()
    
    # Try to extract the verb phrase
    action = behavior
    if " " in behavior:
        action = behavior
    
    # Choose pronoun randomly
    pronoun = random.choice(["he", "she", "they"])
    
    # Format the template
    return template.format(pronoun=pronoun, action=action)

# CLI commands
@click.group()
def cli():
    """ASD Assessment System Testing Framework"""
    pass

@cli.command()
def start_api():
    """Start the API server"""
    with api_server():
        click.echo(click.style("API server is running. Press CTRL+C to stop.", fg="green"))
        try:
            # Keep the server running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo(click.style("\nStopping API server...", fg="yellow"))

@cli.command()
def health():
    """Check the API health"""
    result = make_api_request("get", "health")
    if result:
        click.echo(click.style("API is healthy: ", fg="green") + json.dumps(result, indent=2))
    else:
        click.echo(click.style("API health check failed", fg="red"))

@cli.command()
@click.argument("age", type=int)
def set_age(age):
    """Set the child's age in months"""
    if age < 0 or age > 144:
        click.echo(click.style("Error: Age must be between 0 and 144 months", fg="red"))
        return
    
    result = make_api_request("post", "set-child-age", {"age": age})
    if result:
        click.echo(click.style("Child age set: ", fg="green") + json.dumps(result, indent=2))
    else:
        click.echo(click.style("Failed to set child age", fg="red"))

@cli.command()
def milestone():
    """Get the next milestone"""
    result = make_api_request("get", "next-milestone")
    if result:
        display_milestone(result)
    else:
        click.echo(click.style("Failed to get next milestone", fg="red"))

@cli.command()
@click.argument("milestone")
@click.argument("response")
def test(milestone, response):
    """Test a specific milestone response"""
    data = {
        "milestone_behavior": milestone,
        "response": response
    }
    result = make_api_request("post", "score-response", data)
    if result:
        display_score(result)
    else:
        click.echo(click.style("Failed to score response", fg="red"))

@cli.command()
def report():
    """Generate a developmental report"""
    result = make_api_request("get", "generate-report")
    if result:
        display_report(result)
    else:
        click.echo(click.style("Failed to generate report", fg="red"))

@cli.command()
@click.option("--count", default=10, help="Number of random tests to run")
def batch_test(count):
    """Run batch tests with random responses"""
    # First set a child age
    age = random.randint(12, 60)
    click.echo(click.style(f"Setting child age to {age} months", fg="blue"))
    make_api_request("post", "set-child-age", {"age": age})
    
    success_count = 0
    scores_by_level = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    with click.progressbar(length=count, label="Running tests") as bar:
        for i in range(count):
            # Get milestone
            milestone = make_api_request("get", "next-milestone")
            if not milestone:
                click.echo(click.style("Failed to get milestone, stopping batch test", fg="red"))
                break
            
            # Choose a random score to target
            target_score = random.randint(0, 4)
            
            # Generate appropriate response
            response = generate_response(milestone, target_score)
            
            # Score the response
            data = {
                "milestone_behavior": milestone.get("behavior", ""),
                "response": response
            }
            result = make_api_request("post", "score-response", data)
            
            if result:
                success_count += 1
                actual_score = result.get("score", -1)
                scores_by_level[actual_score] = scores_by_level.get(actual_score, 0) + 1
            
            bar.update(1)
    
    # Report results
    click.echo(click.style(f"\nBatch test completed: {success_count}/{count} successful", fg="green"))
    
    click.echo(click.style("\nScore distribution:", fg="blue"))
    for score, count in scores_by_level.items():
        color = SCORE_COLORS.get(score, "white")
        percentage = (count / success_count * 100) if success_count > 0 else 0
        label = SCORE_LABELS.get(score, "UNKNOWN")
        click.echo(f"  {score} - {label}: " + 
                  click.style(f"{count} ({percentage:.1f}%)", fg=color))
    
    # Generate final report
    click.echo(click.style("\nGenerating final report...", fg="blue"))
    report_result = make_api_request("get", "generate-report")
    if report_result:
        display_report(report_result)

@cli.command()
def demo():
    """Run a demonstration with a variety of responses"""
    # Set child age to 24 months
    click.echo(click.style("Setting child age to 24 months", fg="blue"))
    make_api_request("post", "set-child-age", {"age": 24})
    
    # Test each score level with appropriate responses
    click.echo(click.style("\nDemonstrating assessment with various response types:", fg="blue", bold=True))
    
    for score in range(5):
        # Get next milestone
        milestone = make_api_request("get", "next-milestone")
        if not milestone:
            click.echo(click.style("Failed to get milestone, stopping demo", fg="red"))
            break
        
        # Display milestone
        display_milestone(milestone)
        
        # Generate response for this score level
        response = generate_response(milestone, score)
        click.echo(click.style("Response: ", fg="yellow") + response)
        
        # Score the response
        data = {
            "milestone_behavior": milestone.get("behavior", ""),
            "response": response
        }
        result = make_api_request("post", "score-response", data)
        
        if result:
            display_score(result)
        else:
            click.echo(click.style("Failed to score response", fg="red"))
        
        click.echo("-" * 50)
    
    # Generate final report
    click.echo(click.style("\nGenerating final report...", fg="blue"))
    report_result = make_api_request("get", "generate-report")
    if report_result:
        display_report(report_result)

@cli.command()
def interactive():
    """Start an interactive assessment session"""
    # Set child age
    age = click.prompt("Enter child's age in months", type=int, default=24)
    click.echo(click.style(f"Setting child age to {age} months", fg="blue"))
    make_api_request("post", "set-child-age", {"age": age})
    
    responses = []
    
    while True:
        # Get next milestone
        milestone = make_api_request("get", "next-milestone")
        if not milestone or milestone.get("complete", False):
            click.echo(click.style("Assessment complete!", fg="green"))
            break
        
        # Display milestone
        display_milestone(milestone)
        
        # Get user response
        click.echo(click.style("Response options:", fg="yellow"))
        click.echo("0: Child cannot do this")
        click.echo("1: Child could do this before but lost the skill")
        click.echo("2: Child is starting to do this (emerging)")
        click.echo("3: Child can do this with help")
        click.echo("4: Child can do this independently")
        click.echo("q: Quit assessment")
        
        choice = click.prompt("\nChoose an option or enter a custom response", 
                             type=str, default="")
        
        if choice.lower() == 'q':
            click.echo(click.style("Assessment terminated by user", fg="yellow"))
            break
            
        if choice in ['0', '1', '2', '3', '4']:
            score = int(choice)
            response = generate_response(milestone, score)
            click.echo(click.style("Using response: ", fg="blue") + response)
        else:
            response = choice
        
        # Score the response
        data = {
            "milestone_behavior": milestone.get("behavior", ""),
            "response": response
        }
        result = make_api_request("post", "score-response", data)
        
        if result:
            display_score(result)
            responses.append({
                "milestone": milestone.get("behavior", ""),
                "response": response,
                "score": result.get("score", -1),
                "score_label": result.get("score_label", "UNKNOWN")
            })
        else:
            click.echo(click.style("Failed to score response", fg="red"))
        
        click.echo("-" * 50)
    
    # Generate final report
    if responses:
        click.echo(click.style("\nGenerating final report...", fg="blue"))
        report_result = make_api_request("get", "generate-report")
        if report_result:
            display_report(report_result)

@cli.command()
@click.option("--iterations", default=100, help="Number of iterations for the benchmark")
@click.option("--concurrency", default=1, help="Number of concurrent requests")
def benchmark(iterations, concurrency):
    """Run a performance benchmark"""
    import concurrent.futures
    
    # Prepare test data
    click.echo(click.style("Preparing benchmark data...", fg="blue"))
    
    # Set age to 24 months
    make_api_request("post", "set-child-age", {"age": 24})
    
    # Get a milestone to use for testing
    milestone = make_api_request("get", "next-milestone")
    if not milestone:
        click.echo(click.style("Failed to get milestone for benchmark", fg="red"))
        return
    
    milestone_text = milestone.get("behavior", "walks independently")
    
    # Prepare responses for each score level
    test_responses = [
        generate_response(milestone, score) for score in range(5)
    ]
    
    def score_single_response(response):
        data = {
            "milestone_behavior": milestone_text,
            "response": response
        }
        start_time = time.time()
        result = make_api_request("post", "score-response", data)
        end_time = time.time()
        return {
            "response": response,
            "result": result,
            "response_time": (end_time - start_time) * 1000  # Convert to ms
        }
    
    # Run benchmark
    click.echo(click.style(f"Running benchmark with {iterations} iterations and {concurrency} concurrent requests...", fg="blue"))
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Create a list of future to response mappings
        futures = []
        
        with click.progressbar(length=iterations, label="Running benchmark") as bar:
            for i in range(iterations):
                # Choose a random response from our test set
                response = random.choice(test_responses)
                future = executor.submit(score_single_response, response)
                futures.append(future)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                bar.update(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    success_count = sum(1 for r in results if r["result"] is not None)
    success_rate = (success_count / iterations) * 100 if iterations > 0 else 0
    
    response_times = [r["response_time"] for r in results if r["result"] is not None]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    rps = iterations / total_time if total_time > 0 else 0
    
    # Display results
    click.echo(click.style("\nBenchmark Results:", fg="blue", bold=True))
    click.echo(click.style(f"Total requests: ", fg="green") + str(iterations))
    click.echo(click.style(f"Successful requests: ", fg="green") + str(success_count))
    click.echo(click.style(f"Success rate: ", fg="green") + f"{success_rate:.2f}%")
    click.echo(click.style(f"Total time: ", fg="green") + f"{total_time:.2f} seconds")
    click.echo(click.style(f"Requests per second: ", fg="green") + f"{rps:.2f}")
    click.echo(click.style(f"Average response time: ", fg="green") + f"{avg_response_time:.2f} ms")
    
    if response_times:
        click.echo(click.style(f"Min response time: ", fg="green") + f"{min(response_times):.2f} ms")
        click.echo(click.style(f"Max response time: ", fg="green") + f"{max(response_times):.2f} ms")
    
    # Optionally, you could generate and save a graph of response times
    if response_times and plt:
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=20, alpha=0.7)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        output_file = "benchmark_results.png"
        plt.savefig(output_file)
        click.echo(click.style(f"\nResponse time distribution saved to: ", fg="green") + output_file)

if __name__ == "__main__":
    cli() 