#!/usr/bin/env python3
"""
Script to fix the domain mapping in the comprehensive assessment.
"""

import os
import sys
import re

def fix_domain_mapping():
    """Fix the domain mapping in the comprehensive assessment."""
    try:
        # Path to the app.py file
        app_py_path = os.path.join("src", "app.py")
        
        # Read the app.py file
        with open(app_py_path, "r") as f:
            content = f.read()
        
        # Create a backup of the original file
        backup_path = app_py_path + ".bak"
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Created backup of the original file at {backup_path}")
        
        # Find the domain mapping section
        domain_mapping_pattern = r"domain_mapping = \{[^}]*\}"
        domain_mapping_match = re.search(domain_mapping_pattern, content, re.DOTALL)
        
        if not domain_mapping_match:
            print("Error: Could not find the domain mapping section in the app.py file.")
            return False
        
        # Print the current domain mapping
        current_domain_mapping = domain_mapping_match.group(0)
        print("Current domain mapping:")
        print(current_domain_mapping)
        
        # Update the domain mapping
        updated_domain_mapping = """domain_mapping = {
        "uses gestures to communicate": "EL",
        "recognizes familiar people": "SOC",
        "walks independently": "GM",
        "stacks blocks": "FM",
        "follows simple directions": "RL",
        "points to ask for something": "EL",
        "imitates animal sounds": "EL",
        "uses words to communicate": "EL",
        "responds to name": "SOC",
        "makes eye contact": "SOC",
        "smiles responsively": "SOC",
        "lifts head when on tummy": "GM",
        "rolls from back to side": "GM",
        "sits with support": "GM",
        "clenches fist": "FM",
        "puts everything in mouth": "FM",
        "grasps objects": "FM",
        "coos and gurgles": "EL",
        "laughs": "EL",
        "makes consonant sounds": "EL"
    }"""
        
        # Replace the domain mapping
        updated_content = content.replace(current_domain_mapping, updated_domain_mapping)
        
        # Write the updated content back to the file
        with open(app_py_path, "w") as f:
            f.write(updated_content)
        
        print(f"Updated domain mapping in {app_py_path}")
        
        # Update the comprehensive assessment endpoint to use the domain mapping
        comprehensive_assessment_pattern = r"@app\.post\(\"/api/comprehensive-assessment\".*?def comprehensive_assessment\(.*?\):(.*?)return ComprehensiveResult"
        comprehensive_assessment_match = re.search(comprehensive_assessment_pattern, updated_content, re.DOTALL)
        
        if not comprehensive_assessment_match:
            print("Error: Could not find the comprehensive assessment endpoint in the app.py file.")
            return False
        
        # Print the current comprehensive assessment endpoint
        current_comprehensive_assessment = comprehensive_assessment_match.group(1)
        print("\nCurrent comprehensive assessment endpoint:")
        print(current_comprehensive_assessment)
        
        # Update the comprehensive assessment endpoint
        updated_comprehensive_assessment = current_comprehensive_assessment.replace(
            "domain=\"EL\"  # Default to Expressive Language",
            "domain=domain_mapping.get(assessment_data.milestone_behavior.lower(), \"EL\")  # Use domain mapping"
        )
        
        # Replace the comprehensive assessment endpoint
        updated_content = updated_content.replace(current_comprehensive_assessment, updated_comprehensive_assessment)
        
        # Write the updated content back to the file
        with open(app_py_path, "w") as f:
            f.write(updated_content)
        
        print(f"Updated comprehensive assessment endpoint in {app_py_path}")
        
        return True
    except Exception as e:
        print(f"Error fixing domain mapping: {e}")
        return False

if __name__ == "__main__":
    success = fix_domain_mapping()
    sys.exit(0 if success else 1) 