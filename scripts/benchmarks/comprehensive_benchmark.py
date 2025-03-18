import json
import sys
import re
from src.core.scoring.keyword_scorer import KeywordBasedScorer
from src.core.scoring.embedding_scorer import SemanticEmbeddingScorer
from src.core.scoring.dynamic_ensemble import DynamicEnsembleScorer
from src.core.scoring.base import Score

def run_benchmark(test_data, use_enhanced=False):
    """
    Run benchmark on test data
    
    Args:
        test_data: Test data to benchmark
        use_enhanced: Whether to use enhanced scorers
        
    Returns:
        Dictionary with benchmark results
    """
    # Initialize scorers
    keyword_scorer = KeywordBasedScorer()
    embedding_scorer = SemanticEmbeddingScorer()
    
    # Configure ensemble
    if use_enhanced:
        # Enhanced configuration with ambiguity handling
        ensemble_scorer = DynamicEnsembleScorer(
            scorers=[keyword_scorer, embedding_scorer],
            weights=[0.6, 0.4],
            config={
                "confidence_power": 2.0,
                "minimum_weight": 0.1,
                "specialization_enabled": True,
                "dynamic_weighting_enabled": True,
                "handle_ambiguity": True
            }
        )
    else:
        # Standard configuration
        ensemble_scorer = DynamicEnsembleScorer(
            scorers=[keyword_scorer, embedding_scorer],
            weights=[0.5, 0.5],
            config={
                "confidence_power": 1.5,
                "minimum_weight": 0.2,
                "specialization_enabled": False,
                "dynamic_weighting_enabled": True,
                "handle_ambiguity": False
            }
        )
    
    # Run benchmark
    results = {
        "correct": 0,
        "total": 0,
        "confidence_sum": 0,
        "by_category": {
            "CANNOT_DO": {"correct": 0, "total": 0},
            "LOST_SKILL": {"correct": 0, "total": 0},
            "EMERGING": {"correct": 0, "total": 0},
            "WITH_SUPPORT": {"correct": 0, "total": 0},
            "INDEPENDENT": {"correct": 0, "total": 0}
        },
        "confusion_matrix": {},
        "details": []
    }
    
    for case in test_data.get('test_cases', []):
        response = case.get('caregiver_response', '')
        expected_label = case.get('expected_label')
        
        # Map expected score to enum
        expected = None
        if expected_label == 'CANNOT_DO':
            expected = Score.CANNOT_DO
        elif expected_label == 'LOST_SKILL':
            expected = Score.LOST_SKILL
        elif expected_label == 'EMERGING':
            expected = Score.EMERGING
        elif expected_label == 'WITH_SUPPORT':
            expected = Score.WITH_SUPPORT
        elif expected_label == 'INDEPENDENT':
            expected = Score.INDEPENDENT
        
        if expected is None:
            continue
        
        # Create milestone context
        milestone_context = {
            'domain': case.get('domain'),
            'age_months': test_data.get('metadata', {}).get('age'),
            'behavior': case.get('milestone')
        }
        
        # Score with ensemble
        result = ensemble_scorer.score(response, milestone_context)
        
        # Check if correct
        is_correct = result.score == expected
        if is_correct:
            results["correct"] += 1
            results["by_category"][expected.name]["correct"] += 1
        
        results["total"] += 1
        results["by_category"][expected.name]["total"] += 1
        results["confidence_sum"] += result.confidence
        
        # Update confusion matrix
        predicted = result.score.name
        if expected.name not in results["confusion_matrix"]:
            results["confusion_matrix"][expected.name] = {}
        
        if predicted not in results["confusion_matrix"][expected.name]:
            results["confusion_matrix"][expected.name][predicted] = 0
        
        results["confusion_matrix"][expected.name][predicted] += 1
        
        # Store details
        results["details"].append({
            "response": response[:100] + "..." if len(response) > 100 else response,
            "expected": expected.name,
            "predicted": predicted,
            "confidence": result.confidence,
            "correct": is_correct,
            "reasoning": result.reasoning
        })
    
    # Calculate accuracy
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["avg_confidence"] = results["confidence_sum"] / results["total"] if results["total"] > 0 else 0
    
    # Calculate per-category accuracy
    for category in results["by_category"]:
        cat_data = results["by_category"][category]
        cat_data["accuracy"] = cat_data["correct"] / cat_data["total"] if cat_data["total"] > 0 else 0
    
    return results

def print_results(standard_results, enhanced_results):
    """Print benchmark results comparison"""
    print("\n===== BENCHMARK RESULTS =====\n")
    
    print("Overall Performance:")
    print(f"  Standard:  Accuracy: {standard_results['accuracy']:.2f} ({standard_results['correct']}/{standard_results['total']}), Confidence: {standard_results['avg_confidence']:.2f}")
    print(f"  Enhanced:  Accuracy: {enhanced_results['accuracy']:.2f} ({enhanced_results['correct']}/{enhanced_results['total']}), Confidence: {enhanced_results['avg_confidence']:.2f}")
    print(f"  Difference: {(enhanced_results['accuracy'] - standard_results['accuracy'])*100:.2f}% accuracy, {(enhanced_results['avg_confidence'] - standard_results['avg_confidence'])*100:.2f}% confidence")
    
    print("\nCategory Performance:")
    for category in ["CANNOT_DO", "LOST_SKILL", "EMERGING", "WITH_SUPPORT", "INDEPENDENT"]:
        std_cat = standard_results["by_category"][category]
        enh_cat = enhanced_results["by_category"][category]
        
        if std_cat["total"] == 0:
            continue
            
        print(f"  {category}:")
        print(f"    Standard: {std_cat['accuracy']:.2f} ({std_cat['correct']}/{std_cat['total']})")
        print(f"    Enhanced: {enh_cat['accuracy']:.2f} ({enh_cat['correct']}/{enh_cat['total']})")
        print(f"    Difference: {(enh_cat['accuracy'] - std_cat['accuracy'])*100:.2f}%")
    
    print("\nKey Improvements:")
    improved_cases = []
    for i, (std_case, enh_case) in enumerate(zip(standard_results["details"], enhanced_results["details"])):
        if not std_case["correct"] and enh_case["correct"]:
            improved_cases.append((i, std_case, enh_case))
    
    for i, std_case, enh_case in improved_cases[:5]:  # Show top 5 improvements
        print(f"  Case {i+1}:")
        print(f"    Response: {std_case['response'][:50]}...")
        print(f"    Expected: {std_case['expected']}")
        print(f"    Standard: {std_case['predicted']} (conf: {std_case['confidence']:.2f})")
        print(f"    Enhanced: {enh_case['predicted']} (conf: {enh_case['confidence']:.2f})")
        print(f"    Reasoning: {enh_case['reasoning'][:100]}...")
        print()

# Load test data
with open('./test_results/test_data.json', 'r') as f:
    data = json.load(f)

# Run benchmarks
print("Running standard benchmark...")
standard_results = run_benchmark(data, use_enhanced=False)

print("Running enhanced benchmark...")
enhanced_results = run_benchmark(data, use_enhanced=True)

# Print results
print_results(standard_results, enhanced_results)

# Save results
with open('benchmark_results_phase4.json', 'w') as f:
    json.dump({
        "standard": standard_results,
        "enhanced": enhanced_results
    }, f, indent=2) 