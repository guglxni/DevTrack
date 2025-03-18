#!/usr/bin/env python3
"""
Example script demonstrating the use of the R2R Enhanced Scorer for
developmental assessment.

This script shows how to initialize the R2R Enhanced Scorer and use it to score
responses with retrieval-augmented capabilities using a local Mistral LLM model.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.scoring.base import Score
from src.core.scoring.r2r_enhanced_scorer import R2REnhancedScorer

def print_colored_score(score: Score, confidence: float):
    """Print a score with color based on the score value."""
    score_colors = {
        Score.CANNOT_DO: "\033[91m",  # Red
        Score.LOST_SKILL: "\033[38;5;208m",  # Orange
        Score.EMERGING: "\033[93m",  # Yellow
        Score.WITH_SUPPORT: "\033[94m",  # Blue
        Score.INDEPENDENT: "\033[92m",  # Green
        Score.NOT_RATED: "\033[90m",  # Gray
    }
    
    score_emoji = {
        Score.CANNOT_DO: "❌",
        Score.LOST_SKILL: "⬇️",
        Score.EMERGING: "🌱",
        Score.WITH_SUPPORT: "👋",
        Score.INDEPENDENT: "✅",
        Score.NOT_RATED: "❓",
    }
    
    reset = "\033[0m"
    color = score_colors.get(score, reset)
    emoji = score_emoji.get(score, "")
    
    # Print the score with color and emoji
    confidence_bars = "▓" * int(confidence * 10)
    confidence_spaces = "░" * (10 - int(confidence * 10))
    
    print(f"{color}Score: {score.name} ({score.value}) {emoji}{reset}")
    print(f"Confidence: {confidence:.2f} [{confidence_bars}{confidence_spaces}]")
    
def print_result_details(result):
    """Print detailed result information."""
    print("\n--- REASONING ---")
    print(result.reasoning or "No reasoning provided")
    
    if hasattr(result, 'details') and result.details:
        if 'sources' in result.details:
            print("\n--- SOURCES ---")
            for i, source in enumerate(result.details['sources'], 1):
                text = source.get('text', '')
                if text:
                    print(f"{i}. {text[:100]}...")

def main():
    # Set up example milestone contexts
    example_milestones = [
        {
            "id": "motor-1",
            "name": "Crawls on hands and knees",
            "domain": "MOTOR",
            "description": "Child moves forward on hands and knees for at least 3 feet",
            "age_range": "9-12 months"
        },
        {
            "id": "comm-1",
            "name": "Uses words to express needs",
            "domain": "COMMUNICATION",
            "description": "Child uses at least 5 different words to request objects or actions",
            "age_range": "18-24 months"
        },
        {
            "id": "social-1",
            "name": "Takes turns in games",
            "domain": "SOCIAL",
            "description": "Child waits for their turn and follows simple game rules",
            "age_range": "30-36 months"
        }
    ]
    
    # Set up example responses
    example_responses = [
        "She has just started trying to crawl. She can get up on her hands and knees and rock back and forth, but she hasn't figured out how to move forward yet. Sometimes she pushes backward instead.",
        "He can say 'mama', 'dada', 'ball', 'more', 'milk', and 'dog' when he wants something. He uses these words consistently and knows what they mean. He'll say 'more' when he wants more food and 'ball' when he wants to play.",
        "She loves playing simple board games now. She understands that she needs to wait until it's her turn, and she follows the basic rules. Sometimes she gets impatient if someone else's turn takes too long, but she's getting better at waiting."
    ]
    
    # Initialize the R2R Enhanced Scorer
    print("Initializing R2R Enhanced Scorer with local Mistral model...")
    try:
        # Configuration for local model
        config = {
            "model_path": os.path.join("models", "mistral-7b-instruct-v0.2.Q3_K_S.gguf"),
            "data_dir": os.path.join("data", "documents")
        }
        scorer = R2REnhancedScorer(config)
        print("Initialization successful!")
    except Exception as e:
        print(f"Error initializing scorer: {str(e)}")
        print("Note: Make sure the local model file exists in the models directory.")
        print("Continuing in demo mode with limited functionality.")
        # Initialize with a mock configuration for demonstration
        scorer = R2REnhancedScorer({})
    
    # Process each example
    for i, (milestone, response) in enumerate(zip(example_milestones, example_responses)):
        print("\n" + "="*80)
        print(f"EXAMPLE {i+1}: {milestone['domain']} - {milestone['name']}")
        print(f"Age Range: {milestone['age_range']}")
        print("-"*80)
        print(f"Response: {response}")
        print("-"*80)
        
        # Score the response
        result = scorer.score(response, milestone)
        
        # Print result
        print_colored_score(result.score, result.confidence)
        print(f"Method: {result.method}")
        print(f"Time: {getattr(result, 'duration', 0.0):.2f} seconds")
        
        # Print detailed result information
        print_result_details(result)
    
    print("\n" + "="*80)
    print("All examples processed successfully!")
    print("Note: The quality of results depends on the availability of the local model")
    print("and proper initialization of the R2R system.")

if __name__ == "__main__":
    main() 