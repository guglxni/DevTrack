#!/usr/bin/env python3
"""
Script to check the web interface.
"""

import requests
import sys
import webbrowser

def check_web_interface():
    """Check the web interface."""
    try:
        # Check if the web interface is accessible
        response = requests.get("http://localhost:8003/")
        if response.status_code == 200:
            print("Web interface is accessible.")
            
            # Open the web interface in the default browser
            webbrowser.open("http://localhost:8003/")
            
            return True
        else:
            print(f"Error: Web interface returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking web interface: {e}")
        return False

if __name__ == "__main__":
    success = check_web_interface()
    sys.exit(0 if success else 1)
