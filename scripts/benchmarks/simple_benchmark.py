#!/usr/bin/env python3
"""
Simple Benchmark for Developmental Scoring System

This script performs a simple benchmark on the scoring system with our test data.
"""

import json
import sys
import os
from typing import Dict, List, Any, Tuple, Optional

# Import scoring modules
from src.core.scoring.base import Score, ScoringResult
from src.core.scoring.keyword_scorer import KeywordBasedScorer
from src.core.scoring.embedding_scorer import SemanticEmbeddingScorer
from src.core.scoring.dynamic_ensemble import DynamicEnsembleScorer

# Test data path
TEST_DATA_PATH = './test_results/benchmark_test_data.json'

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Print stats about the loaded data
        score_counts = {}
        for case in data:
            expected_score = case.get("expected_score", "UNKNOWN")
            if expected_score not in score_counts:
                score_counts[expected_score] = 0
            score_counts[expected_score] += 1
            
        print(f"Loaded {len(data)} test cases with expected scores: {score_counts}")
        return data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return []

def run_benchmark(test_data: List[Dict[str, Any]], 
                 use_enhanced: bool = False) -> Tuple[float, float, Dict[str, float]]:
    """Run benchmark tests with the given configuration"""
    # Create scorers
    keyword_scorer = KeywordBasedScorer()
    embedding_scorer = SemanticEmbeddingScorer()
    
    # Create ensemble with different configurations
    if use_enhanced:
        ensemble = DynamicEnsembleScorer(
            scorers=[keyword_scorer, embedding_scorer],
            weights=[1.0, 1.0],
            config={
                "confidence_power": 2.5,  # Enhanced
                "minimum_weight": 0.15,   # Enhanced
                "handle_ambiguity": True  # Enhanced
            }
        )
        print("Using ENHANCED configuration")
    else:
        ensemble = DynamicEnsembleScorer(
            scorers=[keyword_scorer, embedding_scorer],
            weights=[1.0, 1.0],
            config={
                "confidence_power": 2.0,
                "minimum_weight": 0.1,
                "handle_ambiguity": False  # Standard
            }
        )
        print("Using STANDARD configuration")
    
    # Track metrics
    total_cases = len(test_data)
    correct = 0
    total_confidence = 0.0
    category_correct = {}
    category_total = {}
    
    # Process each test case
    for case in test_data:
        response = case.get("response", "")
        expected_score_str = case.get("expected_score", "NOT_RATED")
        domain = case.get("domain", "")
        age_months = case.get("age_months", 0)
        
        # Skip empty cases
        if not response:
            continue
        
        try:
            expected_score = Score[expected_score_str]
        except KeyError:
            print(f"Warning: Invalid expected score: {expected_score_str}")
            continue
            
        # Create milestone context if domain or age is provided
        milestone_context = {}
        if domain:
            milestone_context["domain"] = domain
        if age_months:
            milestone_context["age_months"] = age_months
        
        # Score the response
        result = ensemble.score(response, milestone_context)
        
        # Track overall metrics
        if result.score == expected_score:
            correct += 1
        
        total_confidence += result.confidence
        
        # Track category-specific metrics
        if expected_score_str not in category_total:
            category_total[expected_score_str] = 0
            category_correct[expected_score_str] = 0
        
        category_total[expected_score_str] += 1
        if result.score == expected_score:
            category_correct[expected_score_str] += 1
            
        # Print result for debugging
        print(f"Response: '{response[:30]}...'")
        print(f"  Expected: {expected_score_str}, Got: {result.score.name}, Confidence: {result.confidence:.2f}")
        print(f"  {'✓' if result.score == expected_score else '✗'} {result.reasoning[:100]}...")
        print()
    
    # Calculate final metrics
    accuracy = correct / total_cases if total_cases > 0 else 0
    avg_confidence = total_confidence / total_cases if total_cases > 0 else 0
    
    # Calculate category-specific metrics
    category_accuracy = {}
    for score, total in category_total.items():
        if total > 0:
            category_accuracy[score] = category_correct[score] / total
        else:
            category_accuracy[score] = 0
    
    return accuracy, avg_confidence, category_accuracy

def main():
    """Main function to run the benchmark"""
    # Load test data
    test_data = load_test_data(TEST_DATA_PATH)
    
    if not test_data:
        print("No test data found. Exiting.")
        return
    
    # Run standard benchmark
    print("\n=== STANDARD BENCHMARK ===\n")
    std_accuracy, std_confidence, std_category_acc = run_benchmark(test_data, use_enhanced=False)
    
    # Run enhanced benchmark
    print("\n=== ENHANCED BENCHMARK ===\n")
    enh_accuracy, enh_confidence, enh_category_acc = run_benchmark(test_data, use_enhanced=True)
    
    # Compare results
    print("\n=== BENCHMARK COMPARISON ===\n")
    print(f"Overall Accuracy: Standard: {std_accuracy:.2f}, Enhanced: {enh_accuracy:.2f}, " 
          f"Difference: {(enh_accuracy - std_accuracy) * 100:.2f}%")
    print(f"Average Confidence: Standard: {std_confidence:.2f}, Enhanced: {enh_confidence:.2f}, "
          f"Difference: {(enh_confidence - std_confidence) * 100:.2f}%")
    
    print("\nCategory Accuracy:")
    for category in set(list(std_category_acc.keys()) + list(enh_category_acc.keys())):
        std_acc = std_category_acc.get(category, 0)
        enh_acc = enh_category_acc.get(category, 0)
        diff = (enh_acc - std_acc) * 100
        
        print(f"  {category}: Standard: {std_acc:.2f}, Enhanced: {enh_acc:.2f}, "
              f"Difference: {diff:.2f}%")

if __name__ == "__main__":
    main() 