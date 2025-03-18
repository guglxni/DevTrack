#!/usr/bin/env python
"""
Dynamic Ensemble Example

This script demonstrates the dynamic ensemble weighting and component specialization
features from Phase 3 of the improved developmental scoring system.
"""

import sys
import os
import logging
import json
from typing import Dict, Any, List, Optional
import time

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.scoring.base import Score, ScoringResult
from src.core.scoring.dynamic_ensemble import DynamicEnsembleScorer
from src.core.scoring.component_specialization import (
    KeywordSpecializedScorer,
    EmbeddingSpecializedScorer,
    TransformerSpecializedScorer,
    LLMSpecializedScorer,
    analyze_response_features,
    specialize_ensemble_weights
)
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dynamic_ensemble_example")

# Example milestone contexts for different domains and age groups
EXAMPLE_MILESTONES = {
    "motor_infant": {
        "id": "m001",
        "domain": "MOTOR",
        "age_months": 8,
        "behavior": "Sits without support and can reach for toys without falling over",
        "criteria": "Child maintains sitting position for at least 30 seconds without support.",
        "age_range": "7-9 months"
    },
    "communication_toddler": {
        "id": "c002",
        "domain": "COMMUNICATION",
        "age_months": 18,
        "behavior": "Uses at least 10 words meaningfully",
        "criteria": "Child uses at least 10 different words appropriately in context.",
        "age_range": "16-20 months"
    },
    "social_preschool": {
        "id": "s003",
        "domain": "SOCIAL",
        "age_months": 48,
        "behavior": "Takes turns in games and activities with other children",
        "criteria": "Child waits for their turn and follows simple game rules with peers.",
        "age_range": "42-54 months"
    },
    "cognitive_preschool": {
        "id": "c004",
        "domain": "COGNITIVE",
        "age_months": 42,
        "behavior": "Sorts objects by color, shape, or size",
        "criteria": "Child can sort at least 10 objects into 3 different categories consistently.",
        "age_range": "36-48 months"
    }
}

# Example responses for each milestone with different qualities
EXAMPLE_RESPONSES = {
    "motor_infant": {
        "clear_independent": "Yes, Emma sits up on her own very well now. She's been doing it for about a month and can sit for several minutes without falling. She reaches for toys all around her and doesn't topple over.",
        "ambiguous": "Sometimes she can sit for a bit but other times she falls over. If I prop her with pillows she does better.",
        "clear_emerging": "She's just starting to sit up. She can stay up for maybe 5-10 seconds before she falls over. She's not quite steady enough to reach for toys yet."
    },
    "communication_toddler": {
        "clear_independent": "Liam has a lot of words now! He says mama, dada, ball, dog, cat, up, down, more, water, banana, and several others. He uses them correctly - like saying 'up' when he wants to be picked up or 'ball' when he sees one.",
        "ambiguous": "He has some words but it's hard to understand them all. He definitely says mama and dada, and maybe 3-4 other things that might be words but they're not very clear.",
        "clear_emerging": "He's mostly babbling still. He says mama and dada but I'm not sure he connects them to us specifically. He makes sounds that might be trying to be words but nothing consistent yet."
    }
}

def create_standard_engine() -> ImprovedDevelopmentalScoringEngine:
    """Create a standard scoring engine"""
    config = {
        "enable_keyword_scorer": True,
        "enable_embedding_scorer": True,
        "enable_transformer_scorer": False,
        "enable_llm_scorer": False,
        "enable_continuous_learning": False,
        "enable_audit_logging": True,
        "use_tiered_approach": False,
        "enable_component_specialization": False,
        "high_confidence_threshold": 0.8,
        "low_confidence_threshold": 0.5,
        "score_weights": {
            "keyword": 1.0,
            "embedding": 1.0
        }
    }
    
    return ImprovedDevelopmentalScoringEngine(config)

def create_dynamic_engine() -> ImprovedDevelopmentalScoringEngine:
    """Create a dynamic ensemble scoring engine"""
    config = {
        "enable_keyword_scorer": True,
        "enable_embedding_scorer": True,
        "enable_transformer_scorer": False,
        "enable_llm_scorer": False,
        "enable_continuous_learning": False,
        "enable_audit_logging": True,
        "use_tiered_approach": False,
        "enable_component_specialization": True,
        "high_confidence_threshold": 0.8,
        "low_confidence_threshold": 0.5,
        "score_weights": {
            "keyword": 1.0,
            "embedding": 1.0
        }
    }
    
    return ImprovedDevelopmentalScoringEngine(config)

def print_score_result(result):
    """Print the scoring result in a readable format"""
    if isinstance(result, ScoringResult):
        print(f"Score: {result.score.name}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.method}")
        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")
    elif isinstance(result, dict):
        print(f"Score: {result.get('score_label', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Method: {result.get('method', 'Unknown')}")
        if 'reasoning' in result:
            print(f"Reasoning: {result['reasoning']}")
        
        if 'component_results' in result:
            print("\nComponent Results:")
            for comp in result['component_results']:
                print(f"- {comp.get('method', 'Unknown')}: {comp.get('score_label', 'Unknown')} "
                      f"(confidence: {comp.get('confidence', 0.0):.2f})")
    else:
        print(f"Unknown result type: {type(result)}")

def compare_engines(domain: str, age_range: str, response_type: str):
    """Compare standard vs dynamic engine on the same response"""
    milestone = EXAMPLE_MILESTONES[f"{domain}_{age_range}"]
    response = EXAMPLE_RESPONSES[f"{domain}_{age_range}"][response_type]
    
    # Create engines
    standard_engine = create_standard_engine()
    dynamic_engine = create_dynamic_engine()
    
    # Print example information
    print("=" * 80)
    print(f"Example: {domain.upper()} domain, {age_range} ({milestone['age_months']} months)")
    print(f"Milestone: {milestone['behavior']}")
    print(f"Response Type: {response_type}")
    print("-" * 80)
    print(f"Response: \"{response}\"")
    print("-" * 80)
    
    # Analyze response features
    features = analyze_response_features(response)
    print("Detected features:")
    for feature in features:
        print(f"  - {feature.name}")
    print("-" * 80)
    
    # Score with standard engine
    print("STANDARD ENGINE RESULTS:")
    start_time = time.time()
    standard_result = standard_engine.score_response(response, milestone, detailed=True)
    standard_time = time.time() - start_time
    print_score_result(standard_result)
    print(f"Processing time: {standard_time:.4f} seconds")
    print("-" * 80)
    
    # Score with dynamic engine
    print("DYNAMIC ENGINE RESULTS:")
    start_time = time.time()
    dynamic_result = dynamic_engine.score_response(response, milestone, detailed=True)
    dynamic_time = time.time() - start_time
    print_score_result(dynamic_result)
    print(f"Processing time: {dynamic_time:.4f} seconds")
    print("=" * 80)
    print()

def main():
    """Run the dynamic ensemble example"""
    print("\n===== Dynamic Ensemble Scoring Example =====\n")
    
    # Create standard and dynamic engines
    standard_engine = create_standard_engine()
    dynamic_engine = create_dynamic_engine()
    
    # Example 1: Motor domain (infant)
    domain = "motor"
    age_months = 8
    response = "Yes, she can sit independently and reach for toys."
    
    print(f"\n----- Example 1: {domain.upper()} domain (infant {age_months} months) -----")
    print(f"Response: \"{response}\"")
    print("\nDetected features:")
    features = analyze_response_features(response)
    for feature in features:
        print(f"- {feature.name}")
    
    # Score with standard engine
    print("\nStandard Engine Results:")
    standard_result = standard_engine.score_response(response, {
        "domain": domain,
        "age_months": age_months,
        "behavior": "Sits independently"
    })
    print_score_result(standard_result)
    
    # Score with dynamic engine
    print("\nDynamic Engine Results:")
    dynamic_result = dynamic_engine.score_response(response, {
        "domain": domain,
        "age_months": age_months,
        "behavior": "Sits independently"
    })
    print_score_result(dynamic_result)
    
    # Example 2: Communication domain (toddler)
    domain = "communication"
    age_months = 18
    response = "He can say about 10 words meaningfully, including mama, dada, ball, dog, cat, up, more, milk, banana, and cookie."
    
    print(f"\n----- Example 2: {domain.upper()} domain (toddler {age_months} months) -----")
    print(f"Response: \"{response}\"")
    print("\nDetected features:")
    features = analyze_response_features(response)
    for feature in features:
        print(f"- {feature.name}")
    
    # Score with standard engine
    print("\nStandard Engine Results:")
    standard_result = standard_engine.score_response(response, {
        "domain": domain,
        "age_months": age_months,
        "behavior": "Uses meaningful words"
    })
    print_score_result(standard_result)
    
    # Score with dynamic engine
    print("\nDynamic Engine Results:")
    dynamic_result = dynamic_engine.score_response(response, {
        "domain": domain,
        "age_months": age_months,
        "behavior": "Uses meaningful words"
    })
    print_score_result(dynamic_result)
    
    # Component specialization analysis
    print("\n===== Component Specialization Analysis =====\n")
    
    # Create specialized scorers
    class SimpleKeywordScorer(KeywordSpecializedScorer):
        pass
        
    class SimpleEmbeddingScorer(EmbeddingSpecializedScorer):
        pass
        
    class SimpleTransformerScorer(TransformerSpecializedScorer):
        pass
        
    class SimpleLLMScorer(LLMSpecializedScorer):
        pass
    
    # Create a list of scorers
    scorers = [
        SimpleKeywordScorer(),
        SimpleEmbeddingScorer(),
        SimpleTransformerScorer(),
        SimpleLLMScorer()
    ]
    
    # Define domains and age groups to analyze
    domains = ["motor", "communication", "social", "cognitive"]
    age_groups = [6, 18, 36, 48]
    
    # Print header
    print(f"{'Domain':<15} {'Age':<8} {'Keyword':<10} {'Embedding':<10} {'Transformer':<12} {'LLM':<10} {'Best Component':<15}")
    print("-" * 80)
    
    # Analyze component strengths across domains and age groups
    for domain in domains:
        for age in age_groups:
            # Create a sample response
            if domain == "motor":
                response = "She can walk independently and climb stairs with support."
            elif domain == "communication":
                response = "He uses about 20 words and is starting to combine words."
            elif domain == "social":
                response = "She plays alongside other children and sometimes shares toys."
            else:  # cognitive
                response = "He can sort objects by color and shape when asked."
                
            # Get specialization scores for each component
            scores = {}
            response_length = len(response)
            
            for scorer in scorers:
                if isinstance(scorer, KeywordSpecializedScorer):
                    component = "Keyword"
                elif isinstance(scorer, EmbeddingSpecializedScorer):
                    component = "Embedding"
                elif isinstance(scorer, TransformerSpecializedScorer):
                    component = "Transformer"
                else:
                    component = "LLM"
                
                scores[component] = scorer.get_specialization_score(domain, age, response_length)
            
            # Find the best component
            best_component = max(scores.items(), key=lambda x: x[1])[0] if scores else "None"
            
            # Print the results
            print(f"{domain:<15} {age:<8} {scores.get('Keyword', 0):<10.2f} {scores.get('Embedding', 0):<10.2f} {scores.get('Transformer', 0):<12.2f} {scores.get('LLM', 0):<10.2f} {best_component:<15}")
    
    print("\nNote: Higher scores indicate better specialization for the domain/age combination.")
    print("      The dynamic ensemble will weight components based on these specialization scores.")

if __name__ == "__main__":
    main() 