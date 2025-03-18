#!/usr/bin/env python3
"""
Active Learning Demo

This script demonstrates the key features of the Active Learning Engine,
showing how it identifies valuable examples for expert review and
manages the continuous learning process.
"""

import os
import sys
import logging
import json
from datetime import datetime
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.scoring.active_learning import ActiveLearningEngine
from src.core.enhanced_assessment_engine import Score
from src.core.scoring.dynamic_ensemble import DynamicEnsembleScorer
from src.core.scoring.keyword_scorer import KeywordBasedScorer
from src.core.scoring.embedding_scorer import SemanticEmbeddingScorer
from src.core.scoring.transformer_scorer import TransformerBasedScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("active_learning_demo")

# Example milestone contexts for demonstration
EXAMPLE_MILESTONE_CONTEXTS = [
    {
        "behavior": "Points to show things to others",
        "domain": "SOCIAL",
        "age_range": "9-12 months",
        "criteria": "Child intentionally points to objects to direct another person's attention to something of interest"
    },
    {
        "behavior": "Says at least 2 words besides mama/dada",
        "domain": "COMMUNICATION",
        "age_range": "12-16 months",
        "criteria": "Child uses at least two distinct words meaningfully in appropriate contexts"
    },
    {
        "behavior": "Walks without holding on",
        "domain": "MOTOR",
        "age_range": "11-14 months",
        "criteria": "Child takes at least five steps independently without support"
    },
    {
        "behavior": "Copies simple actions",
        "domain": "COGNITIVE",
        "age_range": "12-18 months",
        "criteria": "Child imitates simple actions demonstrated by others"
    }
]

# Example responses with varying levels of ambiguity and information
EXAMPLE_RESPONSES = [
    # Clear CANNOT_DO responses
    "He doesn't do this at all. We've tried to encourage it but he shows no interest.",
    "No, she hasn't started pointing yet. She sometimes reaches for things but doesn't point.",
    
    # Clear INDEPENDENT responses
    "Yes, he does this all the time! He points to everything he wants us to see.",
    "She's very good at this. Points to things and looks at me to make sure I'm seeing what she's pointing at.",
    
    # Emerging responses
    "He's just starting to do this sometimes, but not consistently. Maybe once or twice a day.",
    "She's tried a few times but isn't quite there yet. She'll sometimes point when prompted.",
    
    # WITH_SUPPORT responses
    "He only does this when I ask him 'can you show me?' Otherwise he doesn't point on his own.",
    "She needs me to start pointing first, then she'll imitate and point too.",
    
    # Ambiguous responses
    "Sometimes yes, sometimes no. It depends on the day and his mood.",
    "Hard to say. I've seen her do it occasionally but I'm not sure if she's really pointing or just reaching.",
    
    # Contradictory responses
    "No, he doesn't point yet, but yesterday he did point at the dog and looked at me.",
    "She doesn't point to show things, but she will point when she wants something.",
    
    # Responses with qualifiers
    "He only points when he's really excited about something, otherwise no.",
    "Not usually, but there have been a few times when she definitely pointed to show me something."
]

def initialize_active_learning_engine() -> ActiveLearningEngine:
    """
    Initialize and configure the Active Learning Engine.
    
    Returns:
        ActiveLearningEngine: The configured engine
    """
    # Create a custom configuration
    config = {
        "data_dir": "demo_active_learning_data",
        "min_training_examples_per_category": 5,
        "retraining_threshold": 15,
        "active_learning_enabled": True,
        "use_enhanced_ambiguity_detection": True,
        "info_gain_weights": {
            "uncertainty": 0.4,
            "disagreement": 0.3,
            "linguistic_novelty": 0.2,
            "domain_coverage": 0.1
        }
    }
    
    # Initialize the engine with custom config
    engine = ActiveLearningEngine(config)
    
    # Create data directory if it doesn't exist
    os.makedirs(config["data_dir"], exist_ok=True)
    
    logger.info(f"Active Learning Engine initialized with data directory: {config['data_dir']}")
    
    return engine

def create_scoring_components():
    """
    Create sample scoring components to simulate disagreement.
    
    Returns:
        Dict: Dictionary of scoring components and their scores
    """
    # This simulates what would happen in the real system where
    # multiple scoring components might disagree
    return {
        "keyword_scorer": 0,  # CANNOT_DO
        "embedding_scorer": 2,  # EMERGING
        "transformer_scorer": 3,  # WITH_SUPPORT
        "llm_scorer": 4   # INDEPENDENT
    }

def demonstrate_valuable_example_identification(engine: ActiveLearningEngine):
    """
    Demonstrate how the engine identifies valuable examples.
    
    Args:
        engine: The Active Learning Engine
    """
    logger.info("\n===== DEMONSTRATING VALUABLE EXAMPLE IDENTIFICATION =====")
    
    # Create some example responses with metadata
    examples = []
    
    for i, (response, milestone_context) in enumerate(
        zip(EXAMPLE_RESPONSES[:8], EXAMPLE_MILESTONE_CONTEXTS * 2)
    ):
        # Create random component scores to simulate disagreement
        components = {}
        
        # For some examples, create high disagreement
        if i % 3 == 0:
            # High disagreement
            components = {
                "keyword_scorer": 0,  # CANNOT_DO
                "embedding_scorer": 2,  # EMERGING
                "transformer_scorer": 3,  # WITH_SUPPORT
                "llm_scorer": 4   # INDEPENDENT
            }
            confidence = 0.6  # Lower confidence
        elif i % 3 == 1:
            # Moderate disagreement
            components = {
                "keyword_scorer": 2,  # EMERGING
                "embedding_scorer": 2,  # EMERGING
                "transformer_scorer": 3,  # WITH_SUPPORT
                "llm_scorer": 3   # WITH_SUPPORT
            }
            confidence = 0.7  # Moderate confidence
        else:
            # Low disagreement
            components = {
                "keyword_scorer": 4,  # INDEPENDENT
                "embedding_scorer": 4,  # INDEPENDENT
                "transformer_scorer": 4,  # INDEPENDENT
                "llm_scorer": 4   # INDEPENDENT
            }
            confidence = 0.9  # High confidence
        
        # Select a predicted score (could be from any component or ensemble)
        predicted_score = random.choice(list(Score))
        if predicted_score == Score.NOT_RATED:
            predicted_score = Score.CANNOT_DO  # Avoid NOT_RATED for demo
        
        # Create example
        example = {
            "id": f"example_{i}",
            "response": response,
            "domain": milestone_context["domain"],
            "milestone": milestone_context["behavior"],
            "component_scores": components,
            "predicted_score": predicted_score.name,
            "confidence": confidence
        }
        
        examples.append(example)
    
    # Identify valuable examples
    valuable_examples = engine.identify_valuable_examples(examples)
    
    # Prioritize examples
    prioritized = engine.prioritize_expert_review(valuable_examples, max_count=3)
    
    logger.info(f"Found {len(valuable_examples)} valuable examples out of {len(examples)}")
    logger.info(f"Top 3 prioritized examples:")
    
    for i, example in enumerate(prioritized):
        logger.info(f"\nPriority Example #{i+1} (priority: {example['priority']:.2f}):")
        logger.info(f"Response: {example['response']}")
        logger.info(f"Domain: {example['domain']}, Milestone: {example['milestone']}")
        logger.info(f"Predicted Score: {example['predicted_score']}")
        logger.info(f"Confidence: {example['confidence']:.2f}")
        
        # Show breakdown of information gain metrics
        logger.info("Information Gain Metrics:")
        logger.info(f"- Uncertainty: {example['uncertainty_score']:.2f}")
        logger.info(f"- Disagreement: {example['disagreement_score']:.2f}")
        logger.info(f"- Linguistic Novelty: {example['novelty_score']:.2f}")
        logger.info(f"- Domain Coverage: {example['domain_coverage_score']:.2f}")
        
        logger.info("Component Scores:")
        for component, score in example['component_scores'].items():
            score_name = Score(score).name if isinstance(score, int) else score
            logger.info(f"- {component}: {score_name}")
    
    return prioritized

def demonstrate_queue_mechanism(engine: ActiveLearningEngine, valuable_examples):
    """
    Demonstrate how to queue examples for expert review.
    
    Args:
        engine: The Active Learning Engine
        valuable_examples: List of valuable examples identified
    """
    logger.info("\n===== DEMONSTRATING QUEUE MECHANISM =====")
    
    review_ids = []
    
    # Queue top examples for review
    for example in valuable_examples:
        milestone_context = {
            "behavior": example['milestone'],
            "domain": example['domain']
        }
        
        # Convert score name to Score enum if needed
        predicted_score = example['predicted_score']
        if isinstance(predicted_score, str):
            predicted_score = Score[predicted_score]
        
        # Queue with priority
        review_id = engine.queue_with_priority(
            response=example['response'],
            milestone_context=milestone_context,
            predicted_score=predicted_score,
            confidence=example['confidence'],
            component_scores=example['component_scores'],
            priority=example['priority'],
            expire_days=14  # Expire in 2 weeks
        )
        
        review_ids.append(review_id)
        logger.info(f"Queued example for review with ID: {review_id}")
    
    # Get pending reviews
    pending_reviews = engine._load_review_queue()
    
    logger.info(f"\nTotal pending reviews: {len(pending_reviews)}")
    
    return review_ids

def demonstrate_expert_feedback(engine: ActiveLearningEngine, review_ids):
    """
    Demonstrate providing expert feedback.
    
    Args:
        engine: The Active Learning Engine
        review_ids: List of review IDs
    """
    logger.info("\n===== DEMONSTRATING EXPERT FEEDBACK =====")
    
    # Select a review ID to provide feedback for
    review_id = review_ids[0]
    
    # Show review before feedback
    review_queue = engine._load_review_queue()
    review_item = next((r for r in review_queue if r.get("id") == review_id), None)
    
    if review_item:
        logger.info(f"Providing feedback for review: {review_id}")
        logger.info(f"Response: {review_item['response']}")
        logger.info(f"Predicted Score: {review_item['predicted_score']}")
        
        # Expert determines the correct score is different
        expert_score = Score.EMERGING
        logger.info(f"Expert determined score: {expert_score.name}")
        
        # Add expert feedback
        result = engine.add_expert_feedback(
            review_id=review_id,
            correct_score=expert_score,
            expert_notes="Child shows early signs of this behavior but needs more consistency."
        )
        
        if result:
            logger.info("Expert feedback added successfully!")
            logger.info("Review item moved to training examples.")
            
            # Get training examples
            training_examples = engine._load_training_examples()
            
            # Count examples by category
            examples_by_category = {}
            for key, examples in training_examples.items():
                examples_by_category[key] = len(examples)
            
            logger.info(f"Training examples by category: {examples_by_category}")
        else:
            logger.error("Failed to add expert feedback.")

def demonstrate_model_versioning(engine: ActiveLearningEngine):
    """
    Demonstrate model versioning capabilities.
    
    Args:
        engine: The Active Learning Engine
    """
    logger.info("\n===== DEMONSTRATING MODEL VERSIONING =====")
    
    # Get current version
    current_version = engine.get_current_version()
    logger.info(f"Current model version: {current_version['version']}")
    
    # Add some test metrics
    metrics = {
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.79,
        "f1_score": 0.81
    }
    
    # Increment minor version
    new_version = engine.increment_version(
        level="minor",
        description="Improved model with better handling of ambiguous cases",
        metrics=metrics
    )
    
    logger.info(f"Created new version: {new_version['version']}")
    logger.info(f"Description: {new_version['description']}")
    logger.info(f"Metrics: {json.dumps(new_version['metrics'], indent=2)}")
    
    # Show all versions
    all_versions = engine.model_versions
    logger.info(f"Total versions: {len(all_versions)}")
    
    for version in all_versions:
        logger.info(f"- {version['version']} ({version['timestamp']}): {version['description']}")

def demonstrate_system_statistics(engine: ActiveLearningEngine):
    """
    Demonstrate system statistics reporting.
    
    Args:
        engine: The Active Learning Engine
    """
    logger.info("\n===== DEMONSTRATING SYSTEM STATISTICS =====")
    
    # Get system statistics
    stats = engine.get_system_statistics()
    
    logger.info(f"Total examples: {stats['total_examples']}")
    logger.info(f"Examples by category: {stats['examples_by_category']}")
    logger.info(f"Pending reviews: {stats['pending_reviews']}")
    logger.info(f"Completed reviews: {stats['completed_reviews']}")
    logger.info(f"Current model version: {stats['current_model_version']}")
    logger.info(f"Total model versions: {stats['total_model_versions']}")
    
    # Get interface data
    interface_data = engine.export_feedback_interface_data()
    logger.info(f"Interface data contains {len(interface_data['pending_reviews'])} pending reviews")
    logger.info(f"Available score categories: {interface_data['categories']}")

def main():
    """Main entry point for the demo"""
    logger.info("Starting Active Learning Engine demo")
    
    # Initialize the engine
    engine = initialize_active_learning_engine()
    
    try:
        # Demonstrate key features
        valuable_examples = demonstrate_valuable_example_identification(engine)
        review_ids = demonstrate_queue_mechanism(engine, valuable_examples)
        demonstrate_expert_feedback(engine, review_ids)
        demonstrate_model_versioning(engine)
        demonstrate_system_statistics(engine)
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 