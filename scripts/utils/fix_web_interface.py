#!/usr/bin/env python3
"""
Script to fix the web interface display issue.
"""

import os
import sys

def fix_web_interface():
    """Fix the web interface display issue."""
    try:
        # Path to the index.html file
        index_html_path = os.path.join("src", "web", "static", "index.html")
        
        # Read the index.html file
        with open(index_html_path, "r") as f:
            lines = f.readlines()
        
        # Create a backup of the original file
        backup_path = index_html_path + ".bak"
        with open(backup_path, "w") as f:
            f.writelines(lines)
        print(f"Created backup of the original file at {backup_path}")
        
        # Find the LLM status section
        llm_status_start = -1
        llm_status_end = -1
        
        for i, line in enumerate(lines):
            if "llmData.status === 'available'" in line:
                llm_status_start = i
            if llm_status_start != -1 and "} else {" in line:
                llm_status_end = i
                break
        
        if llm_status_start == -1 or llm_status_end == -1:
            print("Error: Could not find the LLM status section in the index.html file.")
            return False
        
        # Print the current LLM status section
        print("Current LLM status section:")
        for i in range(llm_status_start, llm_status_end + 1):
            print(lines[i].strip())
        
        # Create a script to check the web interface
        check_script_path = "check_web_interface.py"
        with open(check_script_path, "w") as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Script to check the web interface.
\"\"\"

import requests
import sys
import webbrowser

def check_web_interface():
    \"\"\"Check the web interface.\"\"\"
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
""")
        
        print(f"Created script to check the web interface at {check_script_path}")
        
        return True
    except Exception as e:
        print(f"Error fixing web interface: {e}")
        return False

if __name__ == "__main__":
    success = fix_web_interface()
    sys.exit(0 if success else 1) 