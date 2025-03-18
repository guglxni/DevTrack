#!/usr/bin/env python3
"""
Improved Scoring Example

This script demonstrates how to use the improved developmental scoring engine
with its modular components and robust scoring features.
"""

import sys
import os
import json
from typing import Dict, Any

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
from src.core.scoring.base import Score


def print_score_result(result, milestone: Dict[str, Any]) -> None:
    """Print score result in a formatted way"""
    print("\n" + "=" * 70)
    print(f"Milestone: {milestone['behavior']}")
    if "criteria" in milestone:
        print(f"Criteria: {milestone['criteria']}")
    print("-" * 70)
    
    if hasattr(result, "to_dict"):
        result_dict = result.to_dict()
        
        print(f"Response scored as: {result_dict['score_label']}")
        print(f"Confidence: {result_dict['confidence']:.2f}")
        print(f"Method: {result_dict['method']}")
        
        if result_dict['reasoning']:
            print(f"Reasoning: {result_dict['reasoning']}")
            
    elif isinstance(result, dict):
        print(f"Response scored as: {result['score_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if result['reasoning']:
            print(f"Reasoning: {result['reasoning']}")
            
        if result.get('needs_review', False):
            print("NEEDS REVIEW: This response has been flagged for expert review")
            
        if 'component_results' in result:
            print("\nComponent Scores:")
            for comp in result['component_results']:
                print(f"  - {comp['method']}: {comp['score_label']} (confidence: {comp['confidence']:.2f})")
                if 'reasoning' in comp and comp['reasoning']:
                    print(f"    Reason: {comp['reasoning']}")
    print("=" * 70)


def main():
    # Example milestone contexts
    milestones = [
        {
            "id": "social_01",
            "domain": "social",
            "behavior": "Recognize familiar people",
            "criteria": "Child recognizes and shows preference for familiar caregivers",
            "age_range": "0-3 months"
        },
        {
            "id": "communication_03",
            "domain": "communication",
            "behavior": "Respond to their name",
            "criteria": "Child looks or turns when their name is called",
            "age_range": "6-9 months"
        },
        {
            "id": "motor_05",
            "domain": "motor",
            "behavior": "Stack blocks",
            "criteria": "Child can stack at least 3 blocks on top of each other",
            "age_range": "18-24 months"
        }
    ]
    
    # Example responses to score
    responses = [
        # Different responses for milestone 1 (recognize familiar people)
        {
            "milestone_id": "social_01",
            "text": "Yes, my baby always lights up and smiles when I enter the room. She definitely knows who I am.",
            "expected_score": Score.INDEPENDENT
        },
        {
            "milestone_id": "social_01",
            "text": "Sometimes she seems to recognize me, but not consistently. It depends on her mood.",
            "expected_score": Score.EMERGING
        },
        {
            "milestone_id": "social_01",
            "text": "No, she doesn't recognize me or anyone else yet.",
            "expected_score": Score.CANNOT_DO
        },
        
        # Different responses for milestone 2 (respond to name)
        {
            "milestone_id": "communication_03",
            "text": "He used to turn his head when we called his name, but now he doesn't respond to it anymore.",
            "expected_score": Score.LOST_SKILL
        },
        {
            "milestone_id": "communication_03",
            "text": "He only responds to his name when we get close to him and say it loudly.",
            "expected_score": Score.WITH_SUPPORT
        },
        
        # Different responses for milestone 3 (stack blocks)
        {
            "milestone_id": "motor_05",
            "text": "She's just starting to stack blocks, but usually only manages 2 before they fall.",
            "expected_score": Score.EMERGING
        },
        {
            "milestone_id": "motor_05",
            "text": "Yes, she can stack 5-6 blocks easily without help.",
            "expected_score": Score.INDEPENDENT
        }
    ]
    
    # Create milestone lookup by ID
    milestone_lookup = {m["id"]: m for m in milestones}
    
    # Initialize the improved scoring engine
    print("Initializing improved scoring engine...")
    engine = ImprovedDevelopmentalScoringEngine({
        "enable_keyword_scorer": True,
        "enable_embedding_scorer": True,
        "enable_transformer_scorer": False,  # Disabled for this example to run faster
        "enable_continuous_learning": True,
        "score_weights": {
            "keyword": 0.6,
            "embedding": 0.4
        }
    })
    
    # Score each response
    print("\nScoring responses...\n")
    
    for i, response_data in enumerate(responses):
        print(f"\nResponse {i+1}: \"{response_data['text']}\"")
        
        # Get the milestone context
        milestone_id = response_data["milestone_id"]
        milestone = milestone_lookup[milestone_id]
        
        # Score with basic result
        basic_result = engine.score_response(
            response=response_data["text"],
            milestone_context=milestone,
            detailed=False
        )
        
        # Print basic result
        print_score_result(basic_result, milestone)
        
        # Check if result matches expected score
        expected = response_data["expected_score"]
        actual = basic_result.score if hasattr(basic_result, "score") else basic_result["score"]
        
        if expected == actual:
            print("✓ Result matches expected score")
        else:
            print(f"✗ Result does not match expected score. Expected: {expected.name}")
            
            # Provide expert feedback for incorrect scoring
            print("   Providing expert feedback...")
            engine.with_expert_feedback(
                response=response_data["text"],
                milestone_context=milestone,
                correct_score=expected,
                notes="Correction from example script"
            )
    
    # Get performance metrics
    print("\n\nPerformance Metrics:")
    metrics = engine.get_performance_metrics()
    
    if "training_statistics" in metrics and metrics["training_statistics"]:
        training_stats = metrics["training_statistics"]
        print(f"Training examples: {training_stats.get('total_examples', 0)}")
        
        # Show number of examples per category
        by_score = training_stats.get("by_score", {})
        if by_score:
            print("Examples by category:")
            for score_name, count in by_score.items():
                print(f"  - {score_name}: {count}")
    
    # Check if there are pending reviews
    pending = engine.get_pending_reviews(limit=5)
    if pending:
        print(f"\nPending reviews: {len(pending)}")
        print("First pending review example:")
        print(f"  Response: \"{pending[0]['response']}\"")
        print(f"  Predicted score: {pending[0]['predicted_score']}")
        
        # In a real application, you would present these for expert review
        # and then call engine.with_expert_feedback() with the correct score


if __name__ == "__main__":
    main() 