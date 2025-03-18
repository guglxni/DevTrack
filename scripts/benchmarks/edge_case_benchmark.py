#!/usr/bin/env python3
"""
Edge Case Benchmark for Developmental Scoring System

This script tests the scoring system on challenging edge cases involving:
- Ambiguous responses
- Conflicting information
- Responses with qualifiers
- Context-dependent statements
"""

import json
import logging
import time
import sys
from typing import List, Dict, Any, Tuple, Optional

print("Starting edge case benchmark...")

try:
    import numpy as np
    from tqdm import tqdm

    from src.core.scoring.base import Score, ScoringResult
    from src.core.scoring.keyword_scorer import KeywordBasedScorer
    from src.core.scoring.embedding_scorer import SemanticEmbeddingScorer
    from src.core.scoring.transformer_scorer import TransformerBasedScorer
    from src.core.scoring.dynamic_ensemble import DynamicEnsembleScorer
    
    print("Successfully imported all modules")
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test data with challenging edge cases
EDGE_CASES = [
    {
        "response": "Yes, he can do it, but only sometimes. It really depends on the day.",
        "expected_score": Score.EMERGING,
        "category": "ambiguous_with_qualifier",
        "milestone_context": {
            "domain": "MOTOR",
            "behavior": "Walks independently",
            "age_months": 14
        }
    },
    {
        "response": "No, but also yes. It's complicated. She will do it when she feels like it.",
        "expected_score": Score.EMERGING,
        "category": "contradictory",
        "milestone_context": {
            "domain": "SOCIAL",
            "behavior": "Plays alongside other children",
            "age_months": 24
        }
    },
    {
        "response": "She never does it at home, but her teacher says she does it all the time at daycare.",
        "expected_score": Score.EMERGING,
        "category": "context_dependent",
        "milestone_context": {
            "domain": "COMMUNICATION",
            "behavior": "Uses 2-3 word sentences",
            "age_months": 30
        }
    },
    {
        "response": "He tries to, but can't quite manage it yet. He's getting closer though.",
        "expected_score": Score.EMERGING,
        "category": "attempting_not_achieving",
        "milestone_context": {
            "domain": "COGNITIVE",
            "behavior": "Completes simple puzzles",
            "age_months": 36
        }
    },
    {
        "response": "She used to do it all the time, but now she rarely does anymore.",
        "expected_score": Score.LOST_SKILL,
        "category": "regression",
        "milestone_context": {
            "domain": "COMMUNICATION",
            "behavior": "Responds to name",
            "age_months": 18
        }
    },
    {
        "response": "He does this with me but not with his dad. With me, he's a pro!",
        "expected_score": Score.WITH_SUPPORT,
        "category": "person_dependent",
        "milestone_context": {
            "domain": "SOCIAL",
            "behavior": "Makes eye contact during interactions",
            "age_months": 12
        }
    },
    {
        "response": "She'll do this if we start the activity, but never initiates it herself.",
        "expected_score": Score.WITH_SUPPORT,
        "category": "requires_initiation",
        "milestone_context": {
            "domain": "SOCIAL",
            "behavior": "Engages in pretend play",
            "age_months": 30
        }
    },
    {
        "response": "Yes and no. I wouldn't say he can't do it, but I also wouldn't say he can do it independently.",
        "expected_score": Score.EMERGING,
        "category": "double_negative",
        "milestone_context": {
            "domain": "MOTOR",
            "behavior": "Uses utensils to eat",
            "age_months": 24
        }
    },
    {
        "response": "It's 50/50. Some days perfect, other days not at all.",
        "expected_score": Score.EMERGING,
        "category": "inconsistent",
        "milestone_context": {
            "domain": "COGNITIVE",
            "behavior": "Follows two-step directions",
            "age_months": 30
        }
    },
    {
        "response": "I'm not sure I would say she can't do it, but it's definitely not something she does consistently.",
        "expected_score": Score.EMERGING,
        "category": "qualified_negative",
        "milestone_context": {
            "domain": "COMMUNICATION",
            "behavior": "Points to objects when named",
            "age_months": 18
        }
    }
]

print(f"Defined {len(EDGE_CASES)} edge cases for testing")

def create_standard_ensemble() -> DynamicEnsembleScorer:
    """Create a standard ensemble scorer without the enhancements"""
    print("Creating standard ensemble...")
    keyword_scorer = KeywordBasedScorer()
    embedding_scorer = SemanticEmbeddingScorer()
    transformer_scorer = TransformerBasedScorer()
    
    scorers = [keyword_scorer, embedding_scorer, transformer_scorer]
    weights = [1.0, 1.0, 1.0]
    
    # Disable our enhancements
    config = {
        "dynamic_weighting_enabled": True,
        "specialization_enabled": True,
        "track_performance": False,
        # Set a flag to disable our conflict and ambiguity detection enhancements
        "use_enhanced_ambiguity_detection": False
    }
    
    return DynamicEnsembleScorer(scorers, weights, config)

def create_enhanced_ensemble() -> DynamicEnsembleScorer:
    """Create an enhanced ensemble scorer with our improvements"""
    print("Creating enhanced ensemble...")
    keyword_scorer = KeywordBasedScorer()
    embedding_scorer = SemanticEmbeddingScorer()
    transformer_scorer = TransformerBasedScorer()
    
    scorers = [keyword_scorer, embedding_scorer, transformer_scorer]
    weights = [1.0, 1.0, 1.0]
    
    # Enable our enhancements
    config = {
        "dynamic_weighting_enabled": True,
        "specialization_enabled": True,
        "track_performance": True,
        # Set a flag to enable our conflict and ambiguity detection enhancements
        "use_enhanced_ambiguity_detection": True
    }
    
    return DynamicEnsembleScorer(scorers, weights, config)

def run_benchmark(test_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run benchmark on test data comparing standard and enhanced scoring
    
    Args:
        test_data: List of test cases
        
    Returns:
        Tuple of (standard_results, enhanced_results)
    """
    logger.info("Creating scorers...")
    try:
        standard_scorer = create_standard_ensemble()
        enhanced_scorer = create_enhanced_ensemble()
        print("Successfully created scorers")
    except Exception as e:
        print(f"Error creating scorers: {e}")
        raise
    
    standard_results = []
    enhanced_results = []
    
    logger.info("Running benchmark...")
    for i, case in enumerate(test_data):
        print(f"Processing case {i+1}/{len(test_data)}: {case['category']}")
        try:
            response = case["response"]
            milestone_context = case.get("milestone_context", {})
            expected_score = case["expected_score"]
            
            # Score with standard system
            print(f"  Scoring with standard system...")
            standard_result = standard_scorer.score(response, milestone_context)
            case_result = {
                "response": response,
                "expected_score": expected_score.name,
                "predicted_score": standard_result.score.name,
                "confidence": standard_result.confidence,
                "correct": standard_result.score == expected_score,
                "reasoning": standard_result.reasoning,
                "category": case.get("category", "unknown")
            }
            standard_results.append(case_result)
            
            # Score with enhanced system
            print(f"  Scoring with enhanced system...")
            enhanced_result = enhanced_scorer.score(response, milestone_context)
            case_result = {
                "response": response,
                "expected_score": expected_score.name,
                "predicted_score": enhanced_result.score.name,
                "confidence": enhanced_result.confidence,
                "correct": enhanced_result.score == expected_score,
                "reasoning": enhanced_result.reasoning,
                "category": case.get("category", "unknown")
            }
            enhanced_results.append(case_result)
            
            print(f"  Standard: {standard_result.score.name} ({standard_result.confidence:.2f}) - Enhanced: {enhanced_result.score.name} ({enhanced_result.confidence:.2f})")
        except Exception as e:
            print(f"Error processing case {i+1}: {e}")
            continue
    
    return standard_results, enhanced_results

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate metrics from benchmark results
    
    Args:
        results: List of case results
        
    Returns:
        Dictionary of metrics
    """
    print("Calculating metrics...")
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    avg_confidence = np.mean([r["confidence"] for r in results])
    
    # Calculate per-category metrics
    categories = {}
    for r in results:
        category = r["category"]
        if category not in categories:
            categories[category] = {"total": 0, "correct": 0, "confidence": []}
        
        categories[category]["total"] += 1
        if r["correct"]:
            categories[category]["correct"] += 1
        categories[category]["confidence"].append(r["confidence"])
    
    # Calculate accuracy and confidence for each category
    category_metrics = {}
    for category, data in categories.items():
        category_metrics[category] = {
            "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
            "confidence": np.mean(data["confidence"]),
            "total": data["total"],
            "correct": data["correct"]
        }
    
    return {
        "accuracy": accuracy,
        "confidence": avg_confidence,
        "total": total,
        "correct": correct,
        "by_category": category_metrics
    }

def print_results(standard_metrics: Dict[str, Any], enhanced_metrics: Dict[str, Any]):
    """
    Print comparison of benchmark results
    
    Args:
        standard_metrics: Metrics from standard scoring
        enhanced_metrics: Metrics from enhanced scoring
    """
    print("Printing results...")
    print("\n===== EDGE CASE BENCHMARK RESULTS =====\n")
    
    # Overall metrics
    print("Overall Performance:")
    print(f"  Standard:  Accuracy: {standard_metrics['accuracy']:.2f} ({standard_metrics['correct']}/{standard_metrics['total']}), Confidence: {standard_metrics['confidence']:.2f}")
    print(f"  Enhanced:  Accuracy: {enhanced_metrics['accuracy']:.2f} ({enhanced_metrics['correct']}/{enhanced_metrics['total']}), Confidence: {enhanced_metrics['confidence']:.2f}")
    
    acc_diff = (enhanced_metrics['accuracy'] - standard_metrics['accuracy']) * 100
    conf_diff = (enhanced_metrics['confidence'] - standard_metrics['confidence']) * 100
    print(f"  Difference: {acc_diff:.2f}% accuracy, {conf_diff:.2f}% confidence\n")
    
    # Category performance
    print("Category Performance:")
    all_categories = set(list(standard_metrics['by_category'].keys()) + list(enhanced_metrics['by_category'].keys()))
    
    for category in sorted(all_categories):
        std_cat = standard_metrics['by_category'].get(category, {"accuracy": 0, "total": 0, "correct": 0})
        enh_cat = enhanced_metrics['by_category'].get(category, {"accuracy": 0, "total": 0, "correct": 0})
        
        print(f"  {category}:")
        print(f"    Standard: {std_cat['accuracy']:.2f} ({std_cat['correct']}/{std_cat['total']}), Confidence: {std_cat.get('confidence', 0):.2f}")
        print(f"    Enhanced: {enh_cat['accuracy']:.2f} ({enh_cat['correct']}/{enh_cat['total']}), Confidence: {enh_cat.get('confidence', 0):.2f}")
        
        # Calculate difference
        acc_diff = (enh_cat['accuracy'] - std_cat['accuracy']) * 100
        conf_diff = (enh_cat.get('confidence', 0) - std_cat.get('confidence', 0)) * 100
        print(f"    Difference: {acc_diff:.2f}% accuracy, {conf_diff:.2f}% confidence\n")
    
    # Key improvements
    print("Key Improvements:")
    improved_categories = []
    for category in all_categories:
        std_cat = standard_metrics['by_category'].get(category, {"accuracy": 0})
        enh_cat = enhanced_metrics['by_category'].get(category, {"accuracy": 0})
        
        if enh_cat['accuracy'] > std_cat['accuracy']:
            improved_categories.append(category)
    
    if improved_categories:
        for category in improved_categories:
            std_cat = standard_metrics['by_category'].get(category, {"accuracy": 0})
            enh_cat = enhanced_metrics['by_category'].get(category, {"accuracy": 0})
            acc_diff = (enh_cat['accuracy'] - std_cat['accuracy']) * 100
            print(f"  - {category}: +{acc_diff:.2f}% accuracy")
    else:
        print("  No significant accuracy improvements in this benchmark.")
        
    # If confidence improved but accuracy stayed the same
    conf_improved = enhanced_metrics['confidence'] > standard_metrics['confidence']
    if conf_improved and abs(enhanced_metrics['accuracy'] - standard_metrics['accuracy']) < 0.01:
        print(f"\nConfidence improved by {conf_diff:.2f}% while maintaining the same accuracy.")

def main():
    """Main function"""
    print("Starting main function...")
    start_time = time.time()
    
    try:
        standard_results, enhanced_results = run_benchmark(EDGE_CASES)
        
        standard_metrics = calculate_metrics(standard_results)
        enhanced_metrics = calculate_metrics(enhanced_results)
        
        print_results(standard_metrics, enhanced_metrics)
        
        elapsed = time.time() - start_time
        print(f"\nBenchmark completed in {elapsed:.2f} seconds.")
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Script starting...")
    main() 