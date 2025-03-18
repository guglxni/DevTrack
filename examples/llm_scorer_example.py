#!/usr/bin/env python3
"""
LLM-based Scorer Example

This example demonstrates how to use the LLM-based scorer with the Mistral 7B model
to analyze parent/caregiver responses about developmental milestones.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.scoring.llm_scorer import LLMBasedScorer
from src.core.scoring.base import Score

# Sample data for testing
SAMPLE_RESPONSES = [
    {
        "response": "Yes, my child can definitely do this completely independently. She's been doing it for months now without any help.",
        "milestone_context": {
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": Score.INDEPENDENT
    },
    {
        "response": "He's starting to take a few steps while holding onto furniture, but still needs support.",
        "milestone_context": {
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": Score.WITH_SUPPORT
    },
    {
        "response": "She used to walk well on her own, but in the last few months she's regressed and now needs constant support.",
        "milestone_context": {
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": Score.LOST_SKILL
    },
    {
        "response": "Sometimes they can take 2-3 steps alone before falling, but it's not consistent yet.",
        "milestone_context": {
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": Score.EMERGING
    },
    {
        "response": "No, they can't walk at all yet, even with support.",
        "milestone_context": {
            "behavior": "Walks independently",
            "criteria": "Child walks without support for at least 10 steps",
            "age_range": "12-18 months"
        },
        "expected_score": Score.CANNOT_DO
    }
]

def print_with_color(text, label=None):
    """Print text with optional colored label."""
    if label:
        color_map = {
            "SUCCESS": "\033[92m",  # Green
            "ERROR": "\033[91m",    # Red
            "INFO": "\033[94m",     # Blue
            "WARNING": "\033[93m",  # Yellow
            "RESET": "\033[0m"      # Reset
        }
        color = color_map.get(label, color_map["INFO"])
        reset = color_map["RESET"]
        print(f"{color}[{label}]{reset} {text}")
    else:
        print(text)

def print_score_explanation(sample: dict, result: dict, response_time: float) -> None:
    """Print detailed explanation of scoring results."""
    print("\nSample response:", sample["response"])
    print("Milestone:", sample["milestone_context"]["behavior"])
    print("Criteria:", sample["milestone_context"]["criteria"])
    print("Age Range:", sample["milestone_context"]["age_range"])
    print("--------------------------------------------------")
    
    # Print the score comparison
    expected = sample["expected_score"].name
    actual = result.score.name
    
    if expected == actual:
        print_with_color(f"✓ Score matches expected: {actual}", "green")
    else:
        print_with_color(f"✗ Score mismatch - Expected: {expected}, Got: {actual}", "red")
    
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Response time: {response_time:.2f}s")
    
    if result.reasoning:
        print("\nReasoning:")
        print_with_color(result.reasoning, "cyan")
    
    print("\n" + "="*70)

def main():
    """Run the LLM scorer on sample data."""
    print_with_color("LLM-based Milestone Scorer Example (M4 Optimized)", "INFO")
    print("=" * 70)
    print("Using Mistral 7B to score developmental milestone responses")
    print("=" * 70)
    
    # Configure with options for demonstration
    models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
    model_path = os.path.join(models_dir, "mistral-7b-instruct-v0.2.Q3_K_S.gguf")
    
    # Check if running on Apple Silicon
    is_apple_silicon = False
    try:
        import platform
        is_apple_silicon = platform.system() == 'Darwin' and platform.processor() == 'arm'
        if is_apple_silicon:
            print_with_color("Apple Silicon (M4) detected - enabling maximum GPU utilization", "INFO")
    except ImportError:
        pass
    
    # Initialize the scorer with just the model path first
    print_with_color("Initializing LLM-based scorer...", "INFO")
    try:
        # Create a temporary instance to get the default config
        temp_scorer = LLMBasedScorer({"model_path": model_path})
        
        # Get the default config
        default_config = temp_scorer._default_config()
        
        # Merge with our custom settings optimized for M4 GPU
        config = default_config.copy()
        config.update({
            "model_path": model_path,          # Path to the model file
            "temperature": 0.1,                # Lower temperature for more consistent results
            "max_tokens": 512,                 # Maximum tokens to generate
            "n_ctx": 8192,                     # Context window size
            "top_p": 0.95,                     # Top-p sampling parameter
            
            # Maximum GPU utilization for M4
            "n_gpu_layers": 48 if is_apple_silicon else 0,  # Offload ALL layers to GPU for M4
            "n_batch": 2048 if is_apple_silicon else 512,   # Maximum batch size for GPU
            "n_threads": 8,                     # Utilize all CPU cores
            
            # Memory optimizations
            "use_mmap": True,                   # Use memory mapping
            "use_mlock": True,                  # Lock memory to prevent swapping
            "vocab_only": False,                # Load full model into GPU memory
            
            # Metal-specific optimizations for M4
            "metal_device": True if is_apple_silicon else False,  # Enable Metal backend
            "mmap": True if is_apple_silicon else False,          # Enable memory mapping
            "numa": True if is_apple_silicon else False,          # Enable NUMA optimization
            "tensor_split": None,               # Let the model decide tensor splitting
            
            # Advanced Metal optimizations
            "rope_scaling": {                   # Position embeddings scaling
                "type": "linear",
                "factor": 1.0
            },
            "mul_mat_q": True if is_apple_silicon else False,   # Matrix multiplication optimization
            
            # GPU memory optimizations
            "split_mode": 1 if is_apple_silicon else 0,         # Optimize memory splitting
            "main_gpu": 0,                      # Use primary GPU
            "tensor_parallel": True if is_apple_silicon else False,  # Enable tensor parallelism
            
            # Kernel optimizations
            "f16_kv": True if is_apple_silicon else False,      # Use FP16 for KV cache
            "logits_all": False,                # Only compute last logits
            "embedding_only": False,            # Compute full model
            
            "verbose": True if is_apple_silicon else False      # Show GPU utilization info
        })
        
        # Create the scorer with our optimized config
        scorer = LLMBasedScorer(config)
        
        if not scorer.model:
            print_with_color("Error: Failed to initialize the LLM model", "ERROR")
            sys.exit(1)
            
        print_with_color("Model initialized successfully with maximum M4 GPU utilization!", "SUCCESS")
        
        # Print model metadata if available
        if hasattr(scorer.model, "metadata") and scorer.model.metadata:
            print_with_color("Model Information:", "INFO")
            for key, value in scorer.model.metadata.items():
                if any(key.startswith(prefix) for prefix in ["llama.", "gpu.", "metal."]):
                    print(f"  - {key}: {value}")
            
            # Print GPU memory info if available
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"  - GPU Memory: {gpu_mem:.2f} GB")
            except ImportError:
                pass
    except Exception as e:
        print_with_color(f"Error initializing scorer: {str(e)}", "ERROR")
        return
        
    print("=" * 70)
    
    # Process each sample response
    print_with_color("Processing samples individually:", "INFO")
    correct = 0
    total = len(SAMPLE_RESPONSES)
    
    results = []
    individual_times = []
    for i, sample in enumerate(SAMPLE_RESPONSES, 1):
        print(f"\nSample {i}/{total}:")
        print(f"Response: \"{sample['response']}\"")
        print(f"Milestone: {sample['milestone_context']['behavior']}")
        print(f"Criteria: {sample['milestone_context']['criteria']}")
        print("-" * 50)
        
        # Time the scoring
        start_time = time.time()
        result = scorer.score(sample["response"], sample["milestone_context"])
        total_time = time.time() - start_time
        individual_times.append(total_time)
        
        # Print results
        print_score_explanation(sample, result, total_time)
        
        if result.score == sample["expected_score"]:
            correct += 1
        
        # Store results
        results.append({
            "response": sample["response"],
            "milestone": sample["milestone_context"],
            "expected_score": sample["expected_score"].name,
            "actual_score": result.score.name,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "response_time": total_time
        })
        
        print("-" * 50)
    
    # Now demonstrate batch processing
    print("\n" + "=" * 70)
    print_with_color("Demonstrating batch processing with M4 optimizations:", "INFO")
    print("=" * 70)
    
    # Prepare batch items
    batch_items = []
    for sample in SAMPLE_RESPONSES:
        batch_items.append({
            "response": sample["response"],
            "milestone_context": sample["milestone_context"]
        })
    
    # Time the batch scoring
    batch_start_time = time.time()
    batch_results = scorer.batch_score(batch_items)
    batch_total_time = time.time() - batch_start_time
    
    # Calculate batch accuracy
    batch_correct = 0
    for i, (result, sample) in enumerate(zip(batch_results, SAMPLE_RESPONSES)):
        if result.score == sample["expected_score"]:
            batch_correct += 1
    
    batch_accuracy = batch_correct / total
    
    # Print batch results summary
    print(f"\nBatch Processing Results:")
    print(f"Total batch processing time: {batch_total_time:.2f} seconds")
    print(f"Average time per response: {batch_total_time/total:.2f} seconds")
    print(f"Individual processing average: {sum(individual_times)/total:.2f} seconds")
    print(f"Speed improvement: {sum(individual_times)/batch_total_time:.2f}x")
    print(f"Batch accuracy: {batch_accuracy:.2f}")
    
    # Print overall accuracy for individual processing
    accuracy = correct / total
    
    # Save results to file if requested
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "llm_scorer_results.json")
    
    # Prepare summary data
    summary = {
        "individual_processing": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_time_per_response": sum(individual_times)/total
        },
        "batch_processing": {
            "accuracy": batch_accuracy,
            "correct": batch_correct,
            "total": total,
            "total_time": batch_total_time,
            "avg_time_per_response": batch_total_time/total,
            "speed_improvement": sum(individual_times)/batch_total_time if batch_total_time > 0 else 0
        },
        "model_info": {
            "is_apple_silicon": is_apple_silicon,
            "n_gpu_layers": config["n_gpu_layers"],
            "n_batch": config["n_batch"],
            "n_threads": config["n_threads"]
        },
        "sample_results": results
    }
    
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nOverall Results:")
    print(f"Individual Processing Accuracy: {accuracy:.2f} ({correct}/{total} correct)")
    print(f"Batch Processing Accuracy: {batch_accuracy:.2f} ({batch_correct}/{total} correct)")
    print(f"Individual avg time: {sum(individual_times)/total:.2f} seconds per response")
    print(f"Batch avg time: {batch_total_time/total:.2f} seconds per response")
    print(f"Speed improvement with batch processing: {sum(individual_times)/batch_total_time:.2f}x")
    print(f"\nResults saved to {results_file}")
    print("=" * 70)

if __name__ == "__main__":
    main() 