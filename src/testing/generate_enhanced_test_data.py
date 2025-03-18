#!/usr/bin/env python3
"""
Script to generate enhanced test data for the developmental milestone scoring system.
This script demonstrates the usage of the EnhancedTestDataGenerator.
"""

import sys
import argparse
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.append('.')
from src.testing.enhanced_test_generator import EnhancedTestDataGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced test data for milestone scoring system"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="enhanced_test_data.json",
        help="Output filename for the generated test data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of base test samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    return parser.parse_args()

def load_config(config_path: str = None) -> dict:
    """Load configuration from file or return default config."""
    default_config = {
        "output_dir": "test_data/scoring",
        "random_seed": 42,
        "num_samples_per_category": 20,
        "response_length_min": 5,
        "response_length_max": 100,
        "include_edge_cases": True,
        "include_multilingual": True,
        "include_progression": True
    }
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                return {**default_config, **custom_config}
        except Exception as e:
            print(f"Warning: Failed to load config file: {e}")
            print("Using default configuration.")
    
    return default_config

def main():
    """Main function to generate enhanced test data."""
    args = parse_args()
    config = load_config(args.config)
    
    # Update config with command line arguments
    config["random_seed"] = args.seed
    
    # Initialize generator
    generator = EnhancedTestDataGenerator(config)
    
    # Generate enhanced test data
    print(f"Generating {args.num_samples} base samples with enhancements...")
    test_data = generator.generate_enhanced_test_data(args.num_samples)
    
    # Save the generated data
    generator.save_enhanced_test_data(test_data, args.output)
    
    # Print statistics
    print("\nGeneration complete!")
    print(f"Total samples generated: {len(test_data)}")
    
    # Load the saved data to verify and display statistics
    output_path = Path(config["output_dir"]) / args.output
    with open(output_path, 'r') as f:
        saved_data = json.load(f)
    
    print("\nTest Data Statistics:")
    print("-" * 50)
    stats = saved_data["metadata"]["feature_statistics"]
    dist = saved_data["metadata"]["score_distribution"]
    
    print("\nScore Distribution:")
    for score, count in dist.items():
        print(f"  {score}: {count}")
    
    print("\nFeature Statistics:")
    print(f"  Complex Edge Cases: {stats['complex_edge_cases']}")
    print(f"  Multilingual Responses: {stats['multilingual_responses']}")
    print(f"  Progression Scenarios: {stats['progression_scenarios']}")
    print(f"  Average Response Length: {stats['average_response_length']:.1f} words")

if __name__ == "__main__":
    main() 