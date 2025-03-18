#!/usr/bin/env python3
"""
Gold Standard Dataset Management CLI

This script provides a command-line interface for managing gold standard datasets
used for evaluating the developmental milestone scoring system.
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append('.')

from src.testing.gold_standard_manager import GoldStandardManager
from src.testing.test_data_generator import TestDataGenerator, Score
from src.testing.enhanced_test_generator import EnhancedTestDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/gold_standard_cli.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gold_standard_cli")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage gold standard datasets for milestone scoring system"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available gold standard datasets")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new gold standard dataset")
    create_parser.add_argument(
        "--source",
        type=str,
        choices=["generate", "import", "expert"],
        default="generate",
        help="Source of the dataset (generate new, import from file, or expert review)"
    )
    create_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of samples to generate (if source is 'generate')"
    )
    create_parser.add_argument(
        "--input-file",
        type=str,
        help="Input file path (if source is 'import')"
    )
    create_parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced test data generator (if source is 'generate')"
    )
    create_parser.add_argument(
        "--version",
        type=str,
        help="Version for the new dataset (default: auto-increment)"
    )
    create_parser.add_argument(
        "--metadata",
        type=str,
        help="JSON file with metadata for the dataset"
    )
    
    # View command
    view_parser = subparsers.add_parser("view", help="View details of a gold standard dataset")
    view_parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Dataset version to view (default: latest)"
    )
    view_parser.add_argument(
        "--format",
        type=str,
        choices=["summary", "full", "stats"],
        default="summary",
        help="Output format (summary, full details, or statistics)"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a gold standard dataset")
    export_parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Dataset version to export (default: latest)"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv"],
        default="csv",
        help="Export format (json or csv)"
    )
    export_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: gold_standard_<version>.<format>)"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two gold standard datasets")
    compare_parser.add_argument(
        "--version1",
        type=str,
        required=True,
        help="First dataset version"
    )
    compare_parser.add_argument(
        "--version2",
        type=str,
        required=True,
        help="Second dataset version"
    )
    compare_parser.add_argument(
        "--output",
        type=str,
        help="Output file for comparison results"
    )
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split a gold standard dataset into train/test sets")
    split_parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Dataset version to split (default: latest)"
    )
    split_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (default: 0.8)"
    )
    split_parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify split by score categories"
    )
    split_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/split",
        help="Output directory for split datasets"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a gold standard dataset")
    validate_parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Dataset version to validate (default: latest)"
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Apply strict validation rules"
    )
    
    return parser.parse_args()

def list_datasets(manager: GoldStandardManager):
    """List all available gold standard datasets."""
    versions = manager._list_available_versions()
    
    if not versions:
        print("No gold standard datasets found.")
        return
    
    print(f"Found {len(versions)} gold standard datasets:")
    print("-" * 60)
    print(f"{'Version':<15} {'Date':<20} {'Samples':<10} {'Description'}")
    print("-" * 60)
    
    for version in versions:
        try:
            metadata = manager.get_metadata(version)
            date = metadata.get("creation_date", "Unknown")
            sample_count = metadata.get("sample_count", "Unknown")
            description = metadata.get("description", "")
            print(f"{version:<15} {date[:19]:<20} {sample_count:<10} {description[:30]}")
        except Exception as e:
            print(f"{version:<15} Error loading metadata: {str(e)}")
    
    print("-" * 60)
    latest = manager._get_latest_version()
    print(f"Latest version: {latest}")

def create_dataset(args, manager: GoldStandardManager):
    """Create a new gold standard dataset."""
    if args.source == "generate":
        logger.info(f"Generating {args.count} test samples...")
        
        if args.enhanced:
            generator = EnhancedTestDataGenerator()
            data = generator.generate_enhanced_test_data(args.count)
        else:
            generator = TestDataGenerator()
            data = generator.generate_test_data(args.count)
        
        logger.info(f"Generated {len(data)} test samples")
    
    elif args.source == "import":
        if not args.input_file:
            logger.error("Input file is required for import source")
            return
        
        logger.info(f"Importing data from {args.input_file}")
        
        try:
            with open(args.input_file, 'r') as f:
                if args.input_file.endswith('.json'):
                    data = json.load(f)
                elif args.input_file.endswith('.csv'):
                    df = pd.read_csv(args.input_file)
                    data = df.to_dict(orient='records')
                else:
                    logger.error("Unsupported file format. Use .json or .csv")
                    return
            
            logger.info(f"Imported {len(data)} test samples")
        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return
    
    elif args.source == "expert":
        logger.info("Starting expert review pipeline...")
        # This would typically launch an interactive review process
        # For now, we'll just show a message
        print("Expert review pipeline is not implemented in this CLI version.")
        print("Please use the web interface for expert review.")
        return
    
    # Load metadata if provided
    metadata = None
    if args.metadata:
        try:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata file: {e}")
            logger.info("Continuing with default metadata")
    
    # Add default metadata if not provided
    if not metadata:
        metadata = {
            "description": f"Gold standard dataset created via CLI ({args.source})",
            "creation_date": datetime.now().isoformat(),
            "source": args.source,
            "creator": os.getenv("USER", "unknown")
        }
    
    # Save the dataset
    version = manager.save_dataset(data, args.version, metadata)
    logger.info(f"Created gold standard dataset version {version}")
    
    # Validate the dataset
    valid, issues = manager.validate_dataset(data)
    if valid:
        logger.info("Dataset validation passed")
    else:
        logger.warning("Dataset validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")

def view_dataset(args, manager: GoldStandardManager):
    """View details of a gold standard dataset."""
    try:
        # Load the dataset
        data = manager.load_dataset(args.version)
        metadata = manager.get_metadata(args.version)
        
        # Display based on format
        if args.format == "summary":
            print(f"Gold Standard Dataset: {args.version}")
            print("-" * 60)
            print(f"Total samples: {len(data)}")
            print(f"Creation date: {metadata.get('creation_date', 'Unknown')}")
            print(f"Source: {metadata.get('source', 'Unknown')}")
            print(f"Creator: {metadata.get('creator', 'Unknown')}")
            print(f"Description: {metadata.get('description', '')}")
            
            # Score distribution
            scores = {}
            for item in data:
                score = item.get("expected_score", "Unknown")
                scores[score] = scores.get(score, 0) + 1
            
            print("\nScore Distribution:")
            for score, count in scores.items():
                percentage = (count / len(data)) * 100
                print(f"  {score}: {count} ({percentage:.1f}%)")
        
        elif args.format == "full":
            print(json.dumps(data[:10], indent=2))
            if len(data) > 10:
                print(f"\n... and {len(data) - 10} more items")
        
        elif args.format == "stats":
            # Generate and display statistics
            analysis = manager.analyze_dataset(data, args.version)
            print(json.dumps(analysis, indent=2))
    
    except Exception as e:
        logger.error(f"Failed to view dataset: {e}")

def export_dataset(args, manager: GoldStandardManager):
    """Export a gold standard dataset."""
    try:
        # Determine output filename
        if not args.output:
            version = args.version if args.version != "latest" else manager._get_latest_version()
            args.output = f"gold_standard_{version}.{args.format}"
        
        # Export the dataset
        if args.format == "csv":
            output_path = manager.export_as_csv(args.version, args.output)
        else:  # json
            data = manager.load_dataset(args.version)
            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
            output_path = args.output
        
        logger.info(f"Exported dataset to {output_path}")
        print(f"Dataset exported to {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")

def compare_datasets(args, manager: GoldStandardManager):
    """Compare two gold standard datasets."""
    try:
        # Compare the datasets
        comparison = manager.compare_versions(args.version1, args.version2)
        
        # Display comparison results
        print(f"Comparison of {args.version1} vs {args.version2}:")
        print("-" * 60)
        print(f"Dataset 1 samples: {comparison.get('dataset1_count', 'Unknown')}")
        print(f"Dataset 2 samples: {comparison.get('dataset2_count', 'Unknown')}")
        print(f"Common samples: {comparison.get('common_count', 'Unknown')}")
        print(f"Unique to dataset 1: {comparison.get('unique_to_dataset1', 'Unknown')}")
        print(f"Unique to dataset 2: {comparison.get('unique_to_dataset2', 'Unknown')}")
        
        # Score distribution changes
        print("\nScore Distribution Changes:")
        dist1 = comparison.get("dataset1_distribution", {})
        dist2 = comparison.get("dataset2_distribution", {})
        
        all_scores = set(list(dist1.keys()) + list(dist2.keys()))
        for score in all_scores:
            count1 = dist1.get(score, 0)
            count2 = dist2.get(score, 0)
            change = count2 - count1
            print(f"  {score}: {count1} -> {count2} ({'+' if change >= 0 else ''}{change})")
        
        # Save comparison results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\nComparison results saved to {args.output}")
    
    except Exception as e:
        logger.error(f"Failed to compare datasets: {e}")

def split_dataset(args, manager: GoldStandardManager):
    """Split a gold standard dataset into training and testing sets."""
    try:
        # Split the dataset
        train_data, test_data = manager.split_dataset(
            args.version, 
            args.train_ratio,
            args.stratify
        )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine version for filenames
        version = args.version if args.version != "latest" else manager._get_latest_version()
        
        # Save the split datasets
        train_file = output_dir / f"train_{version}.json"
        test_file = output_dir / f"test_{version}.json"
        
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Display results
        print(f"Dataset split complete:")
        print(f"  Training set: {len(train_data)} samples -> {train_file}")
        print(f"  Testing set: {len(test_data)} samples -> {test_file}")
        
        # Generate a simple visualization of the split
        if train_data and test_data:
            # Count score distribution in each set
            train_scores = {}
            for item in train_data:
                score = item.get("expected_score", "Unknown")
                train_scores[score] = train_scores.get(score, 0) + 1
            
            test_scores = {}
            for item in test_data:
                score = item.get("expected_score", "Unknown")
                test_scores[score] = test_scores.get(score, 0) + 1
            
            # Create visualization
            all_scores = sorted(set(list(train_scores.keys()) + list(test_scores.keys())))
            
            plt.figure(figsize=(10, 6))
            x = range(len(all_scores))
            width = 0.35
            
            train_values = [train_scores.get(score, 0) for score in all_scores]
            test_values = [test_scores.get(score, 0) for score in all_scores]
            
            plt.bar([i - width/2 for i in x], train_values, width, label='Train')
            plt.bar([i + width/2 for i in x], test_values, width, label='Test')
            
            plt.xlabel('Score Categories')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Scores in Train/Test Split')
            plt.xticks(x, all_scores)
            plt.legend()
            
            viz_file = output_dir / f"split_distribution_{version}.png"
            plt.savefig(viz_file)
            print(f"  Distribution visualization saved to {viz_file}")
    
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")

def validate_dataset(args, manager: GoldStandardManager):
    """Validate a gold standard dataset."""
    try:
        # Load the dataset
        data = manager.load_dataset(args.version)
        
        # Validate the dataset
        valid, issues = manager.validate_dataset(data)
        
        # Apply additional strict validation if requested
        if args.strict and valid:
            # Additional validation rules for strict mode
            strict_issues = []
            
            # Check for balanced score distribution
            scores = {}
            for item in data:
                score = item.get("expected_score", "Unknown")
                scores[score] = scores.get(score, 0) + 1
            
            # Check if any score category has less than 10% of samples
            total = len(data)
            for score, count in scores.items():
                percentage = (count / total) * 100
                if percentage < 10:
                    strict_issues.append(
                        f"Score category '{score}' has only {percentage:.1f}% of samples (below 10% threshold)"
                    )
            
            # Check for duplicate responses
            responses = {}
            for i, item in enumerate(data):
                response = item.get("response", "")
                if response in responses:
                    strict_issues.append(
                        f"Duplicate response found at indices {responses[response]} and {i}"
                    )
                responses[response] = i
            
            # Update validation result
            if strict_issues:
                valid = False
                issues.extend(strict_issues)
        
        # Display validation results
        if valid:
            print("Dataset validation passed successfully!")
        else:
            print("Dataset validation failed with the following issues:")
            for issue in issues:
                print(f"  - {issue}")
    
    except Exception as e:
        logger.error(f"Failed to validate dataset: {e}")

def main():
    """Main function to run the CLI."""
    args = parse_args()
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Initialize the gold standard manager
    manager = GoldStandardManager()
    
    # Execute the requested command
    if args.command == "list":
        list_datasets(manager)
    elif args.command == "create":
        create_dataset(args, manager)
    elif args.command == "view":
        view_dataset(args, manager)
    elif args.command == "export":
        export_dataset(args, manager)
    elif args.command == "compare":
        compare_datasets(args, manager)
    elif args.command == "split":
        split_dataset(args, manager)
    elif args.command == "validate":
        validate_dataset(args, manager)
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main() 