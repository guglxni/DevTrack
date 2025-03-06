#!/usr/bin/env python3
import sys
import os
import json
from enum import Enum

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the enhanced assessment engine
from src.core.enhanced_assessment_engine import EnhancedAssessmentEngine, Score, DevelopmentalMilestone

# Create a debug function to analyze keyword matching
def debug_keyword_matching(response, keywords):
    # Print response
    print(f"\nDebug: Analyzing response: '{response}'")
    
    # Check each keyword against the response
    response_lower = response.lower()
    for category, keyword_list in keywords.items():
        matches = []
        for keyword in keyword_list:
            if keyword.lower() in response_lower:
                matches.append(keyword)
        
        if matches:
            print(f"Debug: Found {category} keywords in response: {', '.join(matches)}")
        else:
            print(f"Debug: No {category} keywords found in response")

def main():
    # Initialize the assessment engine
    engine = EnhancedAssessmentEngine()
    print("Initialized assessment engine")
    
    # Load test data
    test_file = "test_data/comprehensive_keyword_test.json"
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded test data from {test_file}")
    
    # Find the milestone
    milestone_behavior = test_data["milestone_behavior"]
    milestone = engine.find_milestone_by_name(milestone_behavior)
    
    if not milestone:
        print(f"Error: Milestone '{milestone_behavior}' not found")
        return
    
    print(f"Found milestone: {milestone.behavior} ({milestone.domain}, {milestone.age_range})")
    
    # Get initial keyword map state
    milestone_key = engine._get_milestone_key(milestone)
    print(f"Milestone key: {milestone_key}")
    
    if milestone_key in engine._scoring_keywords_cache:
        print(f"Initial keyword map has {len(engine._scoring_keywords_cache[milestone_key])} entries")
    else:
        print("No initial keyword map found")
    
    # First, analyze the response without modified keywords
    parent_response = test_data["parent_response"]
    print(f"\n--- Test 1: Original response ---")
    print(f"Response text: '{parent_response}'")
    
    # Manual keyword check
    debug_keyword_matching(parent_response, test_data["keywords"])
    
    # Score with engine
    score = engine.score_response(milestone.behavior, parent_response)
    print(f"Score: {score.name} ({score.value})")
    
    # Now update the keyword map
    print(f"\n--- Test 2: Updating keywords ---")
    
    # Get the specific milestone key
    if milestone_key not in engine._scoring_keywords_cache:
        engine._scoring_keywords_cache[milestone_key] = {}
    
    keyword_map = engine._scoring_keywords_cache[milestone_key]
    
    # Print current keyword map
    print("Current keyword map:")
    categories = {}
    for keyword, score in keyword_map.items():
        if score.name not in categories:
            categories[score.name] = []
        categories[score.name].append(keyword)
    
    for category, keywords in categories.items():
        print(f"  {category}: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
    
    # Update keywords from test data
    for category, keywords in test_data["keywords"].items():
        # Find the score enum
        score_enum = None
        for score in Score:
            if score.name == category:
                score_enum = score
                break
        
        if not score_enum:
            print(f"Error: Could not find score enum for category: {category}")
            continue
        
        # Remove existing keywords for this category
        keys_to_remove = []
        for key, score in keyword_map.items():
            if score == score_enum:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del keyword_map[key]
        
        # Add new keywords
        for keyword in keywords:
            keyword_map[keyword.lower()] = score_enum
    
    print("\nUpdated keyword map:")
    categories = {}
    for keyword, score in keyword_map.items():
        if score.name not in categories:
            categories[score.name] = []
        categories[score.name].append(keyword)
    
    for category, keywords in categories.items():
        print(f"  {category}: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
    
    # Score again with updated keywords
    print(f"\n--- Test 3: Scoring with updated keywords ---")
    score = engine.score_response(milestone.behavior, parent_response)
    print(f"Score: {score.name} ({score.value})")
    
    # Modified keywords test - make "understands" and "responds" into EMERGING keywords
    print(f"\n--- Test 4: Testing with explicitly modified keywords ---")
    
    # Modify keyword map to move some keywords to EMERGING
    modified_map = keyword_map.copy()
    
    # Make sure the keywords "understands" and "responds" are in EMERGING
    for key in list(modified_map.keys()):
        if "understand" in key or "respond" in key:
            modified_map[key] = Score.EMERGING
    
    # Add these keywords explicitly
    modified_map["understands"] = Score.EMERGING
    modified_map["responds"] = Score.EMERGING
    
    # Replace the engine's cache with our modified map
    engine._scoring_keywords_cache[milestone_key] = modified_map
    
    # Score again with our modified keywords
    score = engine.score_response(milestone.behavior, parent_response)
    print(f"Score with 'understands' and 'responds' as EMERGING: {score.name} ({score.value})")
    
    # Test with WITH_SUPPORT keywords in the response
    print(f"\n--- Test 5: Testing WITH_SUPPORT keywords ---")
    
    # Modify the response to include WITH_SUPPORT keywords
    modified_response = "The child follows when repeated and needs visual cues to understand commands"
    print(f"Modified response: '{modified_response}'")
    
    # Manual keyword check
    debug_keyword_matching(modified_response, test_data["keywords"])
    
    # Score with engine
    score = engine.score_response(milestone.behavior, modified_response)
    print(f"Score: {score.name} ({score.value})")
    
    print("\nDebug complete!")

if __name__ == "__main__":
    main() 