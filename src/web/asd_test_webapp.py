#!/usr/bin/env python3
"""
ASD Assessment System - Interactive Testing Web Application

This Streamlit application provides a user-friendly interface for testing
and demonstrating the ASD assessment system. It allows users to:

1. Run interactive assessments
2. Test specific responses
3. View detailed reports
4. Benchmark the system
5. Explore example responses

Usage:
  streamlit run asd_test_webapp.py
"""

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import sys
import os
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="ASD Assessment Testing Framework",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8003"  # Updated API URL to use port 8003

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

SCORE_COLORS = {
    0: "#FF5252",  # red
    1: "#FF7043",  # orange-red
    2: "#FFD54F",  # yellow
    3: "#42A5F5",  # blue
    4: "#66BB6A"   # green
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

# Initialize session state variables
if "api_running" not in st.session_state:
    st.session_state.api_running = False
if "assessment_started" not in st.session_state:
    st.session_state.assessment_started = False
if "current_milestone" not in st.session_state:
    st.session_state.current_milestone = None
if "responses" not in st.session_state:
    st.session_state.responses = []
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "api_process" not in st.session_state:
    st.session_state.api_process = None
if "child_age" not in st.session_state:
    st.session_state.child_age = 24
if "test_history" not in st.session_state:
    st.session_state.test_history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None

# Helper functions
def check_api_running():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server as a background process"""
    if check_api_running():
        st.session_state.api_running = True
        return True
    
    try:
        # Start the server
        process = subprocess.Popen(
            ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        st.session_state.api_process = process
        
        # Wait for API to start (with timeout)
        max_attempts = 10
        for i in range(max_attempts):
            if check_api_running():
                st.session_state.api_running = True
                return True
            time.sleep(1)
        
        return False
    except Exception as e:
        st.error(f"Failed to start API server: {str(e)}")
        return False

def stop_api_server():
    """Stop the API server"""
    if st.session_state.api_process:
        st.session_state.api_process.terminate()
        try:
            st.session_state.api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            st.session_state.api_process.kill()
        
        st.session_state.api_process = None
        st.session_state.api_running = False
        return True
    else:
        # Try to kill any running instances by port
        try:
            os.system("pkill -f 'uvicorn.*8002'")
            st.session_state.api_running = False
            return True
        except:
            return False

def make_api_request(method, endpoint, data=None):
    """Make an API request with proper error handling"""
    if not check_api_running():
        st.error("API server is not running. Please start the server first.")
        return None
    
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        if method.lower() == "get":
            response = requests.get(url, timeout=10)
        elif method.lower() == "post":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"API request error: {str(e)}")
        return None

def set_child_age(age):
    """Set the child's age in the API"""
    result = make_api_request("post", "set-child-age", {"age": age})
    if result:
        st.session_state.child_age = age
        st.session_state.assessment_started = True
        return True
    return False

def get_next_milestone():
    """Get the next milestone from the API"""
    result = make_api_request("get", "next-milestone")
    if result:
        st.session_state.current_milestone = result
        return result
    return None

def score_response(milestone, response):
    """Score a response for a milestone"""
    data = {
        "milestone_behavior": milestone,
        "response": response
    }
    result = make_api_request("post", "score-response", data)
    if result:
        # Add to test history
        st.session_state.test_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "milestone": milestone,
            "response": response,
            "score": result.get("score", -1),
            "score_label": result.get("score_label", "UNKNOWN")
        })
        return result
    return None

def generate_report():
    """Generate a developmental report"""
    result = make_api_request("get", "generate-report")
    if result:
        st.session_state.last_report = result
        return result
    return None

def reset_assessment():
    """Reset the assessment"""
    # Just setting age again resets the assessment
    set_child_age(st.session_state.child_age)
    st.session_state.responses = []
    st.session_state.current_milestone = None
    # Also call the API reset endpoint
    try:
        response = requests.post(f"{API_BASE_URL}/reset")
        return response.status_code == 200
    except:
        return False

def get_all_milestones():
    """Get all available milestone behaviors"""
    try:
        response = requests.get(f"{API_BASE_URL}/all-milestones", timeout=5)
        if response.status_code == 200:
            return response.json().get("milestones", [])
        return []
    except:
        return []

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

def run_benchmark(iterations=50, concurrency=1):
    """Run a performance benchmark"""
    if not check_api_running():
        st.error("API server is not running. Please start the server first.")
        return None
    
    # Set age to 24 months
    set_child_age(24)
    
    # Get a milestone to use for testing
    milestone = get_next_milestone()
    if not milestone:
        st.error("Failed to get milestone for benchmark")
        return None
    
    milestone_text = milestone.get("behavior", "walks independently")
    
    # Prepare responses for each score level
    test_responses = [
        generate_response(milestone, score) for score in range(5)
    ]
    
    # Setup progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run benchmark
    results = []
    start_time = time.time()
    
    for i in range(iterations):
        # Update progress
        progress = (i + 1) / iterations
        progress_bar.progress(progress)
        status_text.text(f"Running benchmark: {i+1}/{iterations} requests completed")
        
        # Choose a random response
        response = random.choice(test_responses)
        
        # Score the response and measure time
        request_start = time.time()
        result = score_response(milestone_text, response)
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
    
    # Prepare benchmark results
    benchmark_results = {
        "iterations": iterations,
        "success_count": success_count,
        "success_rate": success_rate,
        "total_time": total_time,
        "avg_response_time": avg_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "requests_per_second": rps,
        "response_times": response_times,
        "score_counts": score_counts
    }
    
    # Store results in session state
    st.session_state.benchmark_results = benchmark_results
    
    # Clear progress display
    progress_bar.empty()
    status_text.empty()
    
    return benchmark_results

# Main application UI
def main():
    # Title and description
    st.title("ASD Assessment System - Testing Framework")
    st.markdown("""
    This interactive application allows you to test and demonstrate the ASD assessment system.
    You can run assessments, test specific responses, view reports, and benchmark the system.
    
    Start by checking if the API server is running, then explore the different testing options.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("API Server Controls")
        
        # Check API status
        api_status = check_api_running()
        st.session_state.api_running = api_status
        
        if api_status:
            st.success("✅ API Server is running")
            if st.button("Stop API Server"):
                if stop_api_server():
                    st.success("API Server stopped")
                    st.rerun()
                else:
                    st.error("Failed to stop API Server")
        else:
            st.error("❌ API Server is not running")
            if st.button("Start API Server"):
                with st.spinner("Starting API Server..."):
                    if start_api_server():
                        st.success("API Server started")
                        st.rerun()
                    else:
                        st.error("Failed to start API Server")
        
        st.divider()
        
        # Assessment controls
        st.header("Assessment Controls")
        child_age = st.number_input("Child's Age (months)", min_value=0, max_value=144, value=st.session_state.child_age)
        
        if st.button("Set Child Age"):
            if set_child_age(child_age):
                st.success(f"Child age set to {child_age} months")
                st.rerun()
        
        if st.button("Reset Assessment"):
            reset_assessment()
            st.success("Assessment reset")
            st.rerun()
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        page = st.radio("Select Page", ["Interactive Assessment", "Test Individual Responses", "Benchmark", "Test History", "Report"])
    
    # Main content
    if not st.session_state.api_running:
        st.warning("The API server is not running. Please start the server using the controls in the sidebar.")
        return
    
    # Display selected page
    if page == "Interactive Assessment":
        show_interactive_assessment()
    elif page == "Test Individual Responses":
        show_test_responses()
    elif page == "Benchmark":
        show_benchmark()
    elif page == "Test History":
        show_test_history()
    elif page == "Report":
        show_report()

def show_interactive_assessment():
    st.header("Interactive Assessment")
    st.markdown("""
    This page allows you to run an interactive assessment, stepping through milestones
    and providing responses. The system will score your responses and generate a report.
    """)
    
    # Start or continue assessment
    if not st.session_state.assessment_started:
        if st.button("Start Assessment"):
            if set_child_age(st.session_state.child_age):
                st.success(f"Assessment started with child age {st.session_state.child_age} months")
                st.rerun()
    else:
        # Get current milestone
        milestone = st.session_state.current_milestone
        if not milestone:
            milestone = get_next_milestone()
        
        if not milestone:
            st.error("Failed to get milestone")
            return
        
        if milestone.get("complete", False):
            st.success("Assessment complete!")
            report = generate_report()
            if report:
                display_report(report)
            else:
                st.error("Failed to generate report")
            return
        
        # Display milestone
        st.subheader("Current Milestone")
        st.markdown(f"**Behavior:** {milestone.get('behavior', 'N/A')}")
        st.markdown(f"**Criteria:** {milestone.get('criteria', 'N/A')}")
        st.markdown(f"**Domain:** {milestone.get('domain', 'N/A')}")
        st.markdown(f"**Age Range:** {milestone.get('age_range', 'N/A')}")
        
        # Response options
        st.subheader("Response")
        
        # Quick response buttons
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("Cannot Do", use_container_width=True):
                score_with_template(milestone, 0)
                st.rerun()
        
        with col2:
            if st.button("Lost Skill", use_container_width=True):
                score_with_template(milestone, 1)
                st.rerun()
        
        with col3:
            if st.button("Emerging", use_container_width=True):
                score_with_template(milestone, 2)
                st.rerun()
        
        with col4:
            if st.button("With Support", use_container_width=True):
                score_with_template(milestone, 3)
                st.rerun()
        
        with col5:
            if st.button("Independent", use_container_width=True):
                score_with_template(milestone, 4)
                st.rerun()
        
        # Custom response
        custom_response = st.text_input("Or enter a custom response:")
        if custom_response and st.button("Submit Custom Response"):
            result = score_response(milestone.get("behavior", ""), custom_response)
            if result:
                st.session_state.responses.append({
                    "milestone": milestone,
                    "response": custom_response,
                    "result": result
                })
                st.session_state.current_milestone = None  # Will get next milestone on rerun
                st.rerun()
            else:
                st.error("Failed to score response")
        
        # Display previous responses
        if st.session_state.responses:
            st.subheader("Previous Responses")
            for idx, response_data in enumerate(st.session_state.responses):
                with st.expander(f"Response {idx+1}: {response_data['milestone'].get('behavior', 'N/A')}"):
                    st.markdown(f"**Response:** {response_data['response']}")
                    result = response_data['result']
                    score = result.get("score", -1)
                    score_label = result.get("score_label", "UNKNOWN")
                    color = SCORE_COLORS.get(score, "#808080")
                    
                    st.markdown(f"**Score:** <span style='color:{color};font-weight:bold;'>{score} - {score_label}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Domain:** {result.get('domain', 'N/A')}")

def score_with_template(milestone, score_level):
    """Score a response using a template for the given score level"""
    response = generate_response(milestone, score_level)
    result = score_response(milestone.get("behavior", ""), response)
    if result:
        st.session_state.responses.append({
            "milestone": milestone,
            "response": response,
            "result": result
        })
        st.session_state.current_milestone = None  # Will get next milestone on rerun

def show_test_responses():
    st.header("Test Individual Responses")
    st.markdown("""
    This page allows you to test specific responses for any milestone.
    Select or enter a milestone and a response, and the system will score it.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get all available milestones
        milestones = get_all_milestones()
        milestone_behaviors = [m["behavior"] for m in milestones]
        
        # Allow user to select from dropdown or enter custom milestone
        use_dropdown = st.checkbox("Select from available milestones", value=True)
        
        if use_dropdown and milestone_behaviors:
            milestone = st.selectbox("Select Milestone Behavior", milestone_behaviors)
        else:
            milestone = st.text_input("Enter Milestone Behavior", "walks independently")
            
        response = st.text_area("Response", "Yes, he can do this well")
        
        if st.button("Score Response"):
            result = score_response(milestone, response)
            if result:
                display_score_result(result)
            else:
                st.error("Failed to score response")
    
    with col2:
        st.subheader("Milestone Behaviours")
        domains = {}
        
        # Group milestones by domain
        for m in milestones:
            domain = m.get("domain", "Unknown")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(m)
        
        # Create tabs for each domain
        if domains:
            domain_tabs = st.tabs(list(domains.keys()))
            
            for i, (domain, domain_milestones) in enumerate(domains.items()):
                with domain_tabs[i]:
                    # Group by age range within domain
                    age_ranges = {}
                    for m in domain_milestones:
                        age_range = m.get("age_range", "Unknown")
                        if age_range not in age_ranges:
                            age_ranges[age_range] = []
                        age_ranges[age_range].append(m)
                    
                    # Display milestones by age range
                    for age_range, age_milestones in sorted(age_ranges.items()):
                        st.markdown(f"**{age_range} months:**")
                        for m in age_milestones:
                            st.markdown(f"- {m['behavior']}")
        else:
            st.info("No milestone behaviors available. Please ensure the API server is running correctly.")

def display_score_result(result):
    score = result.get("score", -1)
    score_label = result.get("score_label", "UNKNOWN")
    color = SCORE_COLORS.get(score, "#808080")
    
    st.subheader("Score Result")
    st.markdown(f"**Milestone:** {result.get('milestone', 'N/A')}")
    st.markdown(f"**Domain:** {result.get('domain', 'N/A')}")
    st.markdown(f"**Score:** <span style='color:{color};font-weight:bold;'>{score} - {score_label}</span>", unsafe_allow_html=True)
    
    description = SCORE_DESCRIPTIONS.get(score, "Unknown score")
    st.markdown(f"**Interpretation:** {description}")

def show_benchmark():
    st.header("Benchmark Testing")
    st.markdown("""
    This page allows you to benchmark the performance of the assessment system.
    You can configure the number of requests and view detailed performance metrics.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        iterations = st.slider("Number of Requests", min_value=10, max_value=200, value=50, step=10)
        if st.button("Run Benchmark"):
            with st.spinner(f"Running benchmark with {iterations} requests..."):
                benchmark_results = run_benchmark(iterations)
                if benchmark_results:
                    st.success("Benchmark completed")
                    st.rerun()
                else:
                    st.error("Benchmark failed")
    
    # Display benchmark results if available
    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Rate", f"{results['success_rate']:.1f}%")
        with col2:
            st.metric("Avg Response Time", f"{results['avg_response_time']:.2f} ms")
        with col3:
            st.metric("Requests/Second", f"{results['requests_per_second']:.2f}")
        
        # Response time histogram
        if results['response_times']:
            fig = px.histogram(
                x=results['response_times'],
                nbins=20,
                labels={"x": "Response Time (ms)"},
                title="Response Time Distribution",
                color_discrete_sequence=["#42A5F5"]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        if results['score_counts']:
            score_data = []
            for score, count in results['score_counts'].items():
                if score in SCORE_LABELS:
                    score_data.append({
                        "Score": f"{score} - {SCORE_LABELS[score]}",
                        "Count": count,
                        "ScoreNum": score
                    })
            
            if score_data:
                df = pd.DataFrame(score_data)
                fig = px.bar(
                    df,
                    x="Score",
                    y="Count",
                    title="Score Distribution",
                    color="ScoreNum",
                    color_discrete_map={
                        0: SCORE_COLORS[0],
                        1: SCORE_COLORS[1],
                        2: SCORE_COLORS[2],
                        3: SCORE_COLORS[3],
                        4: SCORE_COLORS[4]
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        with st.expander("Detailed Metrics"):
            st.json({
                "iterations": results["iterations"],
                "success_count": results["success_count"],
                "success_rate": f"{results['success_rate']:.2f}%",
                "total_time": f"{results['total_time']:.2f} seconds",
                "avg_response_time": f"{results['avg_response_time']:.2f} ms",
                "min_response_time": f"{results['min_response_time']:.2f} ms",
                "max_response_time": f"{results['max_response_time']:.2f} ms",
                "requests_per_second": f"{results['requests_per_second']:.2f}"
            })

def show_test_history():
    st.header("Test History")
    st.markdown("""
    This page shows the history of tests you've run in this session.
    You can see all the responses you've tested and their scores.
    """)
    
    if not st.session_state.test_history:
        st.info("No test history available. Try scoring some responses first.")
        return
    
    # Create a dataframe from test history
    df = pd.DataFrame(st.session_state.test_history)
    
    # Add color coding for score
    def get_score_color(score):
        return SCORE_COLORS.get(score, "#808080")
    
    df["color"] = df["score"].apply(get_score_color)
    
    # Display as table
    st.dataframe(
        df[["timestamp", "milestone", "response", "score", "score_label"]],
        column_config={
            "timestamp": st.column_config.TextColumn("Time"),
            "milestone": st.column_config.TextColumn("Milestone"),
            "response": st.column_config.TextColumn("Response", width="large"),
            "score": st.column_config.NumberColumn(
                "Score",
                format="%d",
            ),
            "score_label": st.column_config.TextColumn("Label"),
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Show score distribution
    if len(df) > 0:
        st.subheader("Score Distribution")
        
        # Count scores
        score_counts = df["score"].value_counts().reset_index()
        score_counts.columns = ["Score", "Count"]
        
        # Add labels and colors
        score_counts["Label"] = score_counts["Score"].apply(lambda s: SCORE_LABELS.get(s, "UNKNOWN"))
        score_counts["ScoreWithLabel"] = score_counts.apply(lambda row: f"{row['Score']} - {row['Label']}", axis=1)
        
        # Create the chart
        fig = px.pie(
            score_counts,
            values="Count",
            names="ScoreWithLabel",
            title="Score Distribution",
            color="Score",
            color_discrete_map={
                0: SCORE_COLORS[0],
                1: SCORE_COLORS[1],
                2: SCORE_COLORS[2],
                3: SCORE_COLORS[3],
                4: SCORE_COLORS[4]
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Option to clear history
        if st.button("Clear Test History"):
            st.session_state.test_history = []
            st.success("Test history cleared")
            st.rerun()

def show_report():
    st.header("Assessment Report")
    st.markdown("""
    This page shows the developmental report based on the current assessment.
    You need to complete some milestone assessments before a report will be available.
    """)
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            report = generate_report()
            if report:
                st.session_state.last_report = report
                st.success("Report generated successfully")
                st.rerun()
            else:
                st.error("Failed to generate report. Please complete some assessments first.")
    
    # Display report if available
    if st.session_state.last_report:
        display_report(st.session_state.last_report)
    else:
        st.info("No report available. Complete some assessments or click 'Generate Report'.")

def display_report(report):
    """Display a formatted developmental report"""
    scores = report.get("scores", [])
    domain_quotients = report.get("domain_quotients", {})
    
    if not scores and not domain_quotients:
        st.warning("Report contains no data. Please complete some assessments first.")
        return
    
    # Display domain quotients as a chart
    if domain_quotients:
        st.subheader("Domain Quotients")
        
        # Create dataframe for chart
        df = pd.DataFrame([
            {"Domain": domain, "Quotient": quotient}
            for domain, quotient in domain_quotients.items()
            if quotient > 0  # Only include domains with scores
        ])
        
        if len(df) > 0:
            # Add colors based on quotient value
            def get_quotient_color(quotient):
                if quotient >= 75:
                    return SCORE_COLORS[4]  # green
                elif quotient >= 50:
                    return SCORE_COLORS[3]  # blue
                elif quotient >= 25:
                    return SCORE_COLORS[2]  # yellow
                else:
                    return SCORE_COLORS[0]  # red
            
            df["Color"] = df["Quotient"].apply(get_quotient_color)
            
            # Create bar chart
            fig = px.bar(
                df,
                x="Domain",
                y="Quotient",
                title="Domain Quotients",
                color="Color",
                color_discrete_map="identity"
            )
            
            # Customize layout
            fig.update_layout(
                yaxis=dict(
                    title="Quotient (%)",
                    range=[0, 100]
                ),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
            # Display quotients as metrics
            cols = st.columns(min(4, len(df)))
            for i, (_, row) in enumerate(df.iterrows()):
                if i < len(cols):
                    cols[i].metric(
                        row["Domain"],
                        f"{row['Quotient']:.1f}%"
                    )
    
    # Display individual milestone scores
    if scores:
        st.subheader("Milestone Scores")
        
        # Create dataframe for display
        df = pd.DataFrame([
            {
                "Milestone": score.get("milestone", ""),
                "Domain": score.get("domain", ""),
                "Age Range": score.get("age_range", ""),
                "Score": score.get("score", -1),
                "Label": score.get("score_label", "")
            }
            for score in scores
        ])
        
        # Display as table
        st.dataframe(
            df,
            column_config={
                "Milestone": st.column_config.TextColumn("Milestone", width="large"),
                "Domain": st.column_config.TextColumn("Domain"),
                "Age Range": st.column_config.TextColumn("Age Range"),
                "Score": st.column_config.NumberColumn("Score", format="%d"),
                "Label": st.column_config.TextColumn("Label"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Score distribution pie chart
        score_counts = df["Score"].value_counts().reset_index()
        score_counts.columns = ["Score", "Count"]
        
        # Add labels
        score_counts["Label"] = score_counts["Score"].apply(lambda s: SCORE_LABELS.get(s, "UNKNOWN"))
        score_counts["ScoreWithLabel"] = score_counts.apply(lambda row: f"{row['Score']} - {row['Label']}", axis=1)
        
        # Create pie chart
        fig = px.pie(
            score_counts,
            values="Count",
            names="ScoreWithLabel",
            title="Score Distribution",
            color="Score",
            color_discrete_map={
                0: SCORE_COLORS[0],
                1: SCORE_COLORS[1],
                2: SCORE_COLORS[2],
                3: SCORE_COLORS[3],
                4: SCORE_COLORS[4]
            }
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 