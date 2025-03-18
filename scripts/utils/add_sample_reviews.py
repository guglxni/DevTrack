#!/usr/bin/env python3
"""
Script to add sample reviews with milestone context to the review queue for testing purposes.
"""

import json
import os
import uuid
from datetime import datetime

# Sample milestone data
SAMPLE_MILESTONES = [
    {
        "behavior": "Walks independently",
        "question": "Does your child walk independently?",
        "domain": "GM",
        "response": "Yes, he walks without any support now."
    },
    {
        "behavior": "Uses words to communicate",
        "question": "Does your child use words to communicate?",
        "domain": "EL",
        "response": "She says about 10 words clearly and tries to say more."
    },
    {
        "behavior": "Points to ask for things",
        "question": "Does your child point to ask for things?",
        "domain": "SOC",
        "response": "He points to things he wants but doesn't use words yet."
    },
    {
        "behavior": "Recognizes familiar people",
        "question": "Does your child recognize familiar people?",
        "domain": "SOC",
        "response": "Yes, she smiles when she sees family members."
    },
    {
        "behavior": "Plays with toys appropriately",
        "question": "Does your child play with toys appropriately?",
        "domain": "Cog",
        "response": "Sometimes, but often just throws them."
    }
]

def add_sample_reviews():
    """Add sample reviews to the review queue."""
    # Path to the review queue file
    review_queue_path = os.path.join("data", "continuous_learning", "review_queue.json")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(review_queue_path), exist_ok=True)
    
    # Load existing review queue or create a new one
    if os.path.exists(review_queue_path):
        with open(review_queue_path, "r") as f:
            try:
                review_queue = json.load(f)
            except json.JSONDecodeError:
                review_queue = []
    else:
        review_queue = []
    
    # Add sample reviews
    for i, milestone in enumerate(SAMPLE_MILESTONES):
        review_id = f"sample_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create review item with both milestone and milestone_context fields
        review_item = {
            "timestamp": datetime.now().isoformat(),
            "response": milestone["response"],
            "milestone": {
                "id": f"sample_milestone_{i}",
                "domain": milestone["domain"],
                "behavior": milestone["behavior"],
                "criteria": f"Child {milestone['behavior'].lower()} regularly and consistently",
                "age_range": "12-24 months"
            },
            "milestone_context": {
                "behavior": milestone["behavior"],
                "question": milestone["question"],
                "domain": milestone["domain"],
                "criteria": f"Child {milestone['behavior'].lower()} regularly and consistently",
                "age_range": "12-24 months"
            },
            "predicted_score": "NOT_RATED",
            "predicted_score_value": -1,
            "confidence": 0.9,  # High confidence for testing
            "reasoning": f"This is a sample review for testing the {milestone['domain']} domain.",
            "priority": 0.95,  # Very high priority for testing
            "status": "pending",
            "id": review_id
        }
        
        review_queue.append(review_item)
    
    # Save the updated review queue
    with open(review_queue_path, "w") as f:
        json.dump(review_queue, f, indent=2)
    
    print(f"Added {len(SAMPLE_MILESTONES)} sample reviews to the review queue at {review_queue_path}.")
    print("Restart the server to see the changes.")

if __name__ == "__main__":
    add_sample_reviews() 