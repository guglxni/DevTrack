"""
Gold Standard Dataset Manager for Developmental Milestone Scoring System.

This module provides tools for creating, validating, storing, and managing gold standard
datasets used for training and evaluating the milestone scoring system.
"""

import os
import json
import shutil
import hashlib
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.core.scoring.base import Score


class GoldStandardManager:
    """
    Manager for gold standard datasets with versioning and validation.
    
    This class provides functionality to:
    1. Create and store versioned gold standard datasets
    2. Validate datasets for quality and coverage
    3. Analyze dataset characteristics
    4. Split datasets for training and evaluation
    5. Track dataset performance metrics
    """
    
    def __init__(self, base_dir: str = "data/gold_standard"):
        """
        Initialize the gold standard dataset manager.
        
        Args:
            base_dir: Base directory for storing gold standard datasets
        """
        self.base_dir = base_dir
        self.ensure_directories()
        self.current_version = self._get_latest_version()
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "analysis"), exist_ok=True)
    
    def _get_latest_version(self) -> str:
        """Get the latest dataset version."""
        versions = self._list_available_versions()
        if not versions:
            return "0.0.0"
        return max(versions)
    
    def _list_available_versions(self) -> List[str]:
        """List all available dataset versions."""
        versions = []
        for item in os.listdir(self.base_dir):
            if os.path.isfile(os.path.join(self.base_dir, item)) and item.endswith(".json"):
                # Extract version from filename (format: gold_standard_vX.Y.Z.json)
                if item.startswith("gold_standard_v") and item.endswith(".json"):
                    version = item[len("gold_standard_v"):-len(".json")]
                    if self._is_valid_version(version):
                        versions.append(version)
        return versions
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if a version string is valid (format: X.Y.Z)."""
        try:
            parts = version.split(".")
            if len(parts) != 3:
                return False
            for part in parts:
                int(part)  # Check if all parts are integers
            return True
        except (ValueError, IndexError):
            return False
    
    def _increment_version(self, version: str, level: str = "patch") -> str:
        """
        Increment the version number.
        
        Args:
            version: Current version (format: X.Y.Z)
            level: Which level to increment ('major', 'minor', or 'patch')
            
        Returns:
            New version string
        """
        major, minor, patch = map(int, version.split("."))
        
        if level == "major":
            return f"{major + 1}.0.0"
        elif level == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
    
    def _get_dataset_path(self, version: str) -> str:
        """Get the file path for a specific dataset version."""
        return os.path.join(self.base_dir, f"gold_standard_v{version}.json")
    
    def _get_metadata_path(self, version: str) -> str:
        """Get the file path for dataset metadata."""
        return os.path.join(self.base_dir, "metadata", f"metadata_v{version}.json")
    
    def save_dataset(self, 
                   data: List[Dict[str, Any]], 
                   version: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   increment_level: str = "patch") -> str:
        """
        Save a gold standard dataset with versioning.
        
        Args:
            data: List of dataset entries
            version: Specific version to use (if None, increment current version)
            metadata: Additional metadata to store with the dataset
            increment_level: Which version level to increment ('major', 'minor', or 'patch')
            
        Returns:
            Version string of the saved dataset
        """
        # Determine version
        if version is None:
            version = self._increment_version(self.current_version, increment_level)
        
        # Validate the dataset before saving
        self.validate_dataset(data)
        
        # Save dataset
        dataset_path = self._get_dataset_path(version)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Generate and save metadata
        if metadata is None:
            metadata = {}
        
        # Add standard metadata
        metadata.update({
            "version": version,
            "created_at": datetime.datetime.now().isoformat(),
            "entry_count": len(data),
            "file_hash": self._compute_file_hash(dataset_path),
            "distribution": self._get_distribution(data)
        })
        
        metadata_path = self._get_metadata_path(version)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Update current version
        self.current_version = version
        
        # Generate analysis report
        self.analyze_dataset(data, version)
        
        return version
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of scores in the dataset."""
        scores = []
        for item in data:
            if "expected_score" in item:
                scores.append(item["expected_score"])
        
        return dict(Counter(scores))
    
    def validate_dataset(self, data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate dataset for quality and completeness.
        
        Args:
            data: Dataset to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data is empty
        if not data:
            errors.append("Dataset is empty")
            return False, errors
        
        # Check required fields
        required_fields = ["response", "milestone_context", "expected_score"]
        
        for idx, item in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in item:
                    errors.append(f"Item {idx} is missing required field '{field}'")
            
            # Check milestone context fields
            if "milestone_context" in item:
                milestone_context = item["milestone_context"]
                milestone_required = ["id", "behavior"]
                for field in milestone_required:
                    if field not in milestone_context:
                        errors.append(f"Item {idx} milestone_context is missing required field '{field}'")
            
            # Validate expected score
            if "expected_score" in item:
                expected_score = item["expected_score"]
                if not isinstance(expected_score, str) or expected_score not in [s.name for s in Score]:
                    errors.append(f"Item {idx} has invalid expected_score '{expected_score}'")
        
        # Check distribution across score categories
        score_distribution = self._get_distribution(data)
        for score in [s.name for s in Score if s.name != "NOT_RATED"]:
            if score not in score_distribution or score_distribution[score] < 5:
                errors.append(f"Insufficient examples for score category '{score}' (minimum 5 required)")
        
        return len(errors) == 0, errors
    
    def load_dataset(self, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load a gold standard dataset.
        
        Args:
            version: Version to load (if None, load latest version)
            
        Returns:
            Dataset as a list of dictionaries
        """
        if version is None:
            version = self.current_version
        
        file_path = self._get_dataset_path(version)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset version {version} not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_metadata(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a dataset version.
        
        Args:
            version: Version to get metadata for (if None, use latest version)
            
        Returns:
            Metadata dictionary
        """
        if version is None:
            version = self.current_version
        
        metadata_path = self._get_metadata_path(version)
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata for version {version} not found")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_dataset(self, data: List[Dict[str, Any]], version: str) -> Dict[str, Any]:
        """
        Analyze dataset characteristics and generate reports.
        
        Args:
            data: Dataset to analyze
            version: Version of the dataset
            
        Returns:
            Analysis results dictionary
        """
        analysis = {}
        
        # Basic statistics
        analysis["count"] = len(data)
        
        # Score distribution
        score_distribution = self._get_distribution(data)
        analysis["score_distribution"] = score_distribution
        
        # Domain distribution
        domains = []
        for item in data:
            if "milestone_context" in item and "domain" in item["milestone_context"]:
                domains.append(item["milestone_context"]["domain"])
        
        domain_distribution = dict(Counter(domains))
        analysis["domain_distribution"] = domain_distribution
        
        # Response length statistics
        response_lengths = [len(item["response"]) for item in data if "response" in item]
        if response_lengths:
            analysis["response_length"] = {
                "min": min(response_lengths),
                "max": max(response_lengths),
                "avg": sum(response_lengths) / len(response_lengths)
            }
        
        # Save analysis report
        analysis_path = os.path.join(self.base_dir, "analysis", f"analysis_v{version}.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Generate visualizations
        self._generate_visualizations(analysis, version)
        
        return analysis
    
    def _generate_visualizations(self, analysis: Dict[str, Any], version: str) -> None:
        """Generate visualizations for dataset analysis."""
        # Create directory for visualizations
        viz_dir = os.path.join(self.base_dir, "analysis", f"visualizations_v{version}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Score distribution chart
        if "score_distribution" in analysis:
            plt.figure(figsize=(10, 6))
            score_dist = analysis["score_distribution"]
            plt.bar(score_dist.keys(), score_dist.values())
            plt.title(f"Score Distribution - v{version}")
            plt.xlabel("Score Categories")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "score_distribution.png"))
            plt.close()
        
        # Domain distribution chart
        if "domain_distribution" in analysis:
            plt.figure(figsize=(10, 6))
            domain_dist = analysis["domain_distribution"]
            plt.bar(domain_dist.keys(), domain_dist.values())
            plt.title(f"Domain Distribution - v{version}")
            plt.xlabel("Developmental Domains")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "domain_distribution.png"))
            plt.close()
    
    def split_dataset(self, 
                    version: Optional[str] = None, 
                    train_ratio: float = 0.8,
                    stratify_by_score: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into training and evaluation sets.
        
        Args:
            version: Dataset version to split (if None, use latest)
            train_ratio: Proportion of data to use for training
            stratify_by_score: Whether to ensure proportional representation of score categories
            
        Returns:
            Tuple of (training_data, evaluation_data)
        """
        from sklearn.model_selection import train_test_split
        
        # Load dataset
        data = self.load_dataset(version)
        
        if stratify_by_score:
            # Extract scores for stratification
            scores = [item["expected_score"] for item in data]
            train_data, eval_data = train_test_split(
                data, 
                train_size=train_ratio, 
                stratify=scores,
                random_state=42
            )
        else:
            train_data, eval_data = train_test_split(
                data, 
                train_size=train_ratio,
                random_state=42
            )
        
        return train_data, eval_data
    
    def export_as_csv(self, version: Optional[str] = None, filepath: Optional[str] = None) -> str:
        """
        Export dataset as CSV file.
        
        Args:
            version: Dataset version to export (if None, use latest)
            filepath: Target filepath (if None, generate based on version)
            
        Returns:
            Path to exported CSV file
        """
        if version is None:
            version = self.current_version
        
        if filepath is None:
            filepath = os.path.join(self.base_dir, f"gold_standard_v{version}.csv")
        
        # Load dataset
        data = self.load_dataset(version)
        
        # Flatten the nested structure for CSV
        flattened_data = []
        for item in data:
            flat_item = {
                "response": item["response"],
                "expected_score": item["expected_score"]
            }
            
            # Add milestone context fields
            if "milestone_context" in item:
                for key, value in item["milestone_context"].items():
                    flat_item[f"milestone_{key}"] = value
            
            flattened_data.append(flat_item)
        
        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(flattened_data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two dataset versions and report differences.
        
        Args:
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary of differences
        """
        # Load datasets
        data1 = self.load_dataset(version1)
        data2 = self.load_dataset(version2)
        
        # Get metadata
        meta1 = self.get_metadata(version1)
        meta2 = self.get_metadata(version2)
        
        # Compare basic stats
        comparison = {
            "version1": version1,
            "version2": version2,
            "count_diff": len(data2) - len(data1),
            "metadata1": meta1,
            "metadata2": meta2
        }
        
        # Compare score distributions
        dist1 = meta1.get("distribution", {})
        dist2 = meta2.get("distribution", {})
        
        diff_distribution = {}
        all_scores = set(list(dist1.keys()) + list(dist2.keys()))
        
        for score in all_scores:
            count1 = dist1.get(score, 0)
            count2 = dist2.get(score, 0)
            diff_distribution[score] = count2 - count1
        
        comparison["distribution_diff"] = diff_distribution
        
        return comparison


def create_gold_standard_from_test_data(
    test_data: List[Dict[str, Any]],
    output_version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a gold standard dataset from test data with expert review.
    
    Args:
        test_data: Source test data
        output_version: Version to assign (if None, auto-increment)
        metadata: Additional metadata to store
        
    Returns:
        Version of the created gold standard
    """
    manager = GoldStandardManager()
    
    # Here we'd typically implement an expert review process
    # For now, we'll just use the data as-is with a metadata note
    if metadata is None:
        metadata = {}
    
    metadata["source"] = "test_data_direct"
    metadata["notes"] = "Created from test data without expert review. Expert review recommended."
    
    # Save as gold standard
    version = manager.save_dataset(test_data, version=output_version, metadata=metadata)
    
    return version


def expert_review_pipeline(test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process for expert review of test data.
    
    In a real implementation, this would include:
    1. UI for experts to review and correct classifications
    2. Tracking of disagreements
    3. Resolution process for conflicting assessments
    
    Args:
        test_data: Data to review
        
    Returns:
        Reviewed data with expert corrections
    """
    # This is a placeholder for the actual expert review process
    # In a real implementation, this would involve human experts
    reviewed_data = test_data.copy()
    
    # Add a "reviewed" flag to each item
    for item in reviewed_data:
        item["expert_reviewed"] = True
        item["review_date"] = datetime.datetime.now().isoformat()
        
        # In a real implementation, the expert might change the expected_score
        # Here we're just simulating that by adding a review note
        item["expert_notes"] = "This item was reviewed by an expert and the classification was confirmed."
    
    return reviewed_data


def save_gold_standard(data: List[Dict[str, Any]], version: Optional[str] = None) -> str:
    """
    Save a reviewed dataset as a gold standard.
    
    Args:
        data: Reviewed dataset
        version: Version to assign (if None, auto-increment)
        
    Returns:
        Version of the saved gold standard
    """
    manager = GoldStandardManager()
    
    metadata = {
        "source": "expert_reviewed",
        "reviewed_by": "developmental_experts",
        "review_date": datetime.datetime.now().isoformat()
    }
    
    version = manager.save_dataset(data, version=version, metadata=metadata)
    
    return version 