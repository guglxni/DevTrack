#!/usr/bin/env python3
"""
Test Report Generator for ASD Assessment API

This script consolidates test results from multiple test runs and generates a
comprehensive report with performance metrics for each endpoint.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import glob

def load_test_results(filepath):
    """Load test results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading test results: {e}")
        return None

def load_all_test_results(directory):
    """Load test results from all JSON files in the directory."""
    combined_results = {
        'summary': {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'success_rate': 0,
            'avg_response_time': 0,
            'endpoints': {}
        },
        'results': []
    }
    
    # Find the most recent test result file
    result_file = os.path.join(directory, 'api_test_results.json')
    if os.path.exists(result_file):
        results = load_test_results(result_file)
        if results:
            # Merge data from this file
            combined_results['results'].extend(results['results'])
            
            # Update overall summary
            combined_results['summary']['total_tests'] += results['summary']['total_tests']
            combined_results['summary']['passed_tests'] += results['summary']['passed_tests']
            combined_results['summary']['failed_tests'] += results['summary']['failed_tests']
            
            # Merge endpoint data
            for endpoint, data in results['summary']['endpoints'].items():
                if endpoint in combined_results['summary']['endpoints']:
                    # If this endpoint already exists, we need to merge the data
                    existing_data = combined_results['summary']['endpoints'][endpoint]
                    combined_results['summary']['endpoints'][endpoint] = {
                        'total_tests': existing_data['total_tests'] + data['total_tests'],
                        'passed_tests': existing_data['passed_tests'] + data['passed_tests'],
                        'failed_tests': existing_data['failed_tests'] + data['failed_tests'],
                        'success_rate': ((existing_data['passed_tests'] + data['passed_tests']) / 
                                        (existing_data['total_tests'] + data['total_tests'])) * 100,
                        'avg_response_time': (existing_data['avg_response_time'] * existing_data['total_tests'] + 
                                            data['avg_response_time'] * data['total_tests']) / 
                                            (existing_data['total_tests'] + data['total_tests']),
                        'min_response_time': min(existing_data['min_response_time'], data['min_response_time']),
                        'max_response_time': max(existing_data['max_response_time'], data['max_response_time']),
                        'median_response_time': (existing_data.get('median_response_time', 0) + 
                                                data.get('median_response_time', 0)) / 2,
                        'std_dev_response_time': (existing_data.get('std_dev_response_time', 0) + 
                                                data.get('std_dev_response_time', 0)) / 2
                    }
                else:
                    # Otherwise, just add this endpoint data
                    combined_results['summary']['endpoints'][endpoint] = data
    
    # Calculate the overall average response time based on all requests
    if combined_results['results']:
        combined_results['summary']['avg_response_time'] = sum(
            r['response_time'] for r in combined_results['results']
        ) / len(combined_results['results'])
        
        # Recalculate overall success rate
        if combined_results['summary']['total_tests'] > 0:
            combined_results['summary']['success_rate'] = (
                combined_results['summary']['passed_tests'] / combined_results['summary']['total_tests']
            ) * 100
    
    return combined_results

def create_endpoint_dataframe(results):
    """Create a DataFrame summarizing endpoint performance."""
    endpoint_data = []
    
    for endpoint, data in results['summary']['endpoints'].items():
        endpoint_data.append({
            'Endpoint': endpoint,
            'Tests': data['total_tests'],
            'Success Rate': f"{data['success_rate']:.2f}%",
            'Avg Time (s)': f"{data['avg_response_time']:.4f}",
            'Min Time (s)': f"{data['min_response_time']:.4f}",
            'Max Time (s)': f"{data['max_response_time']:.4f}",
            'Median Time (s)': f"{data.get('median_response_time', 0):.4f}",
            'StdDev (s)': f"{data.get('std_dev_response_time', 0):.4f}"
        })
    
    return pd.DataFrame(endpoint_data)

def create_request_dataframe(results):
    """Create a DataFrame with individual request data."""
    request_data = []
    
    for result in results['results']:
        request_data.append({
            'Endpoint': result['endpoint'],
            'Status': result['status_code'],
            'Response Time': result['response_time'],
            'Timestamp': result.get('timestamp', ''),
            'Success': result['success']
        })
    
    return pd.DataFrame(request_data)

def generate_charts(request_df, output_dir):
    """Generate performance charts from request data."""
    plt.figure(figsize=(10, 6))
    
    # Response time distribution
    plt.subplot(2, 1, 1)
    sns.histplot(request_df['Response Time'], kde=True)
    plt.title('Response Time Distribution')
    plt.xlabel('Response Time (s)')
    plt.ylabel('Frequency')
    
    # Response time by endpoint
    plt.subplot(2, 1, 2)
    if 'timestamp' in request_df.columns and not request_df['timestamp'].empty:
        sns.lineplot(x=range(len(request_df)), y='Response Time', 
                    hue='Endpoint', data=request_df)
    else:
        sns.lineplot(x=range(len(request_df)), y='Response Time', 
                    hue='Endpoint', data=request_df)
    plt.title('Response Time Trend')
    plt.xlabel('Request Number')
    plt.ylabel('Response Time (s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_charts.png'))
    plt.close()

def generate_html_report(endpoint_df, results, output_path):
    """Generate an HTML report with test results and performance metrics."""
    # Calculate overall statistics
    total_tests = results['summary']['total_tests']
    passed_tests = results['summary']['passed_tests']
    failed_tests = results['summary']['failed_tests']
    success_rate = results['summary']['success_rate']
    avg_response_time = results['summary']['avg_response_time']
    
    # Format the current date and time
    timestamp = datetime.now().isoformat()
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ASD Assessment API Test Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #444;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .summary {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }}
            .summary-item {{
                flex: 1;
                min-width: 200px;
                background-color: #f5f5f5;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .success {{
                color: green;
            }}
            .failure {{
                color: red;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            .chart-container {{
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>ASD Assessment API Test Report</h1>
        <p>Generated on: {timestamp}</p>
        
        <h2>Consolidated Test Summary</h2>
        <div class="summary">
            <div class="summary-item">
                <h3>Total Tests</h3>
                <p>{total_tests}</p>
            </div>
            <div class="summary-item">
                <h3>Passed</h3>
                <p class="success">{passed_tests}</p>
            </div>
            <div class="summary-item">
                <h3>Failed</h3>
                <p class="failure">{failed_tests}</p>
            </div>
            <div class="summary-item">
                <h3>Success Rate</h3>
                <p>{success_rate:.2f}%</p>
            </div>
            <div class="summary-item">
                <h3>Average Response Time</h3>
                <p>{avg_response_time:.4f} seconds</p>
            </div>
        </div>
        
        <h2>Response Time Distribution</h2>
        <div class="chart-container">
            <img src="performance_charts.png" alt="Response Time Distribution">
        </div>
        
        <h2>Endpoint Performance</h2>
        <table>
            <tr>
                {' '.join(f'<th>{col}</th>' for col in endpoint_df.columns)}
            </tr>
            {''.join(
                f'<tr>{"".join(f"<td>{cell}</td>" for cell in row)}</tr>' 
                for row in endpoint_df.values
            )}
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate consolidated test report')
    parser.add_argument('--output-dir', default='test_results', 
                        help='Directory to store the report')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all test results
    results = load_all_test_results(args.output_dir)
    
    if not results or not results['results']:
        print("No test results found.")
        return
    
    # Create DataFrames
    endpoint_df = create_endpoint_dataframe(results)
    request_df = create_request_dataframe(results)
    
    # Generate charts
    generate_charts(request_df, args.output_dir)
    
    # Generate HTML report
    html_path = os.path.join(args.output_dir, 'consolidated_report.html')
    generate_html_report(endpoint_df, results, html_path)

if __name__ == "__main__":
    main() 