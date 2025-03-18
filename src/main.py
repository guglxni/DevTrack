#!/usr/bin/env python3
"""
ASD Assessment System - Main Entry Point

This script serves as the main entry point for the ASD Assessment System.
It provides options to start the API server or the web application.

Usage:
  python main.py [--api | --web | --help]

Options:
  --api    Start the API server
  --web    Start the web application
  --help   Show this help message
"""

import os
import sys
import subprocess
import argparse

def start_api_server():
    """Start the API server"""
    print("Starting API server...")
    try:
        from src.api.app import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8003)
    except ImportError as e:
        print(f"Error importing API modules: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
        sys.exit(1)

def start_web_app():
    """Start the web application"""
    print("Starting web application...")
    try:
        # Use subprocess to run streamlit
        subprocess.run(["python3", "-m", "streamlit", "run", "src/web/asd_test_webapp.py"])
    except Exception as e:
        print(f"Error starting web application: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ASD Assessment System")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--api", action="store_true", help="Start the API server")
    group.add_argument("--web", action="store_true", help="Start the web application")
    
    args = parser.parse_args()
    
    if args.api:
        start_api_server()
    elif args.web:
        start_web_app()
    else:
        # If no arguments provided, show help
        parser.print_help()
        print("\nNo option specified. Use --api to start the API server or --web to start the web application.")

if __name__ == "__main__":
    main() 