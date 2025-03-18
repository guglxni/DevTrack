#!/usr/bin/env python3
"""
Benchmark for Age-Specific Knowledge Integration

This script benchmarks the performance of scoring with and without
age-specific knowledge integration.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import random

# Add project root to path
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.core.scoring.base import Score, ScoringResult, BaseScorer
from src.core.knowledge import adjust_category_for_age, get_age_bracket
from src.testing.benchmark_framework import BenchmarkResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("age_specific_benchmark")

# Add a mock LLM scorer class
class MockLLMScorer(BaseScorer):
    """Mock LLM scorer for benchmarking without an actual LLM"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration"""
        super().__init__(config or {})
        self.use_domain_prompts = self.config.get("use_domain_specific_prompts", True)
        self.use_age_prompts = self.config.get("use_age_specific_prompts", True)
        
    def _format_prompt(self, response: str, milestone_context: Dict[str, Any]) -> str:
        """Format a prompt for scoring"""
        prompt = f"Developmental Assessment\n\n"
        
        # Add domain-specific content if enabled
        if self.use_domain_prompts and "domain" in milestone_context:
            prompt += f"Domain: {milestone_context['domain']}\n"
        
        # Add age-specific content if enabled
        if self.use_age_prompts and "age_months" in milestone_context:
            age_months = milestone_context["age_months"]
            age_bracket = get_age_bracket(age_months)
            prompt += f"Age: {age_months} months ({age_bracket})\n"
        
        # Add milestone context
        if "milestone" in milestone_context:
            prompt += f"Milestone: {milestone_context['milestone']}\n"
        
        # Add response
        prompt += f"Response: {response}\n"
        
        return prompt
    
    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Simulate scoring a response"""
        if not milestone_context:
            milestone_context = {}
            
        # Format the prompt to simulate LLM processing
        prompt = self._format_prompt(response, milestone_context)
        
        # Generate a mock score based on the response and context
        mock_scores = {
            "CANNOT_DO": ["cannot", "unable", "doesn't", "not yet", "never"],
            "LOST_SKILL": ["used to", "previously", "before", "regressed", "lost", "stopped"],
            "EMERGING": ["sometimes", "occasionally", "trying", "beginning", "starting"],
            "WITH_SUPPORT": ["help", "assist", "guidance", "support", "when I", "with me"],
            "INDEPENDENT": ["always", "consistently", "by himself", "by herself", "independently", "without help"]
        }
        
        # Determine the most likely category based on keyword matching
        response_lower = response.lower()
        scores = {}
        for category, keywords in mock_scores.items():
            score_value = 0
            for keyword in keywords:
                if keyword in response_lower:
                    score_value += 1
            scores[category] = score_value
        
        # Get the category with the highest score
        if any(scores.values()):
            category = max(scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to EMERGING if no keywords match
            category = "EMERGING"
            
        # Simulate a confidence score between 0.65 and 0.95
        confidence = 0.7 + (random.random() * 0.2)
        
        # Apply age adjustment if the config has it enabled
        if self.use_age_prompts and "age_months" in milestone_context:
            confidence = min(0.95, confidence + 0.05)  # Small boost for age-aware prompts
        
        # Create a Score enum from the category string
        score_enum = getattr(Score, category)
        
        return ScoringResult(
            score=score_enum,
            confidence=confidence,
            method="mock_llm",
            reasoning=f"Mock reasoning: {category} detected with confidence {confidence:.2f}",
            details={"prompt_length": len(prompt), "age_aware": self.use_age_prompts}
        )

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark"""
    data_path: str
    output_dir: str
    domains: List[str] = field(default_factory=lambda: ["MOTOR", "COMMUNICATION", "SOCIAL", "COGNITIVE"])
    age_groups: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 12), (13, 36), (37, 60)])
    run_llm: bool = False
    save_figures: bool = True
    sample_size: Optional[int] = None

@dataclass
class ScoringConfig:
    """Configuration for different scoring setups"""
    name: str
    description: str
    config: Dict[str, Any]
    color: str  # For visualization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "color": self.color
        }

def parse_args() -> BenchmarkConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Age-Specific Knowledge Integration")
    parser.add_argument("--data", required=True, help="Path to test data JSON file")
    parser.add_argument("--output", required=True, help="Directory to save benchmark results")
    parser.add_argument("--domains", nargs="+", default=["MOTOR", "COMMUNICATION", "SOCIAL", "COGNITIVE"],
                        help="Domains to benchmark")
    parser.add_argument("--age-groups", nargs="+", default=["0-12", "13-36", "37-60"],
                        help="Age groups to analyze (format: min-max)")
    parser.add_argument("--run-llm", action="store_true", help="Actually run LLM (slow, costs tokens)")
    parser.add_argument("--no-figures", action="store_true", help="Skip generating figures")
    parser.add_argument("--sample", type=int, help="Sample a subset of test cases")
    
    args = parser.parse_args()
    
    # Parse age groups
    age_groups = []
    for group in args.age_groups:
        min_age, max_age = map(int, group.split("-"))
        age_groups.append((min_age, max_age))
    
    return BenchmarkConfig(
        data_path=args.data,
        output_dir=args.output,
        domains=args.domains,
        age_groups=age_groups,
        run_llm=args.run_llm,
        save_figures=not args.no_figures,
        sample_size=args.sample
    )

def setup_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_test_data(config: BenchmarkConfig) -> List[Dict[str, Any]]:
    """Load and filter test data"""
    logger.info(f"Loading test data from {config.data_path}")
    
    try:
        with open(config.data_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return []
    
    # Filter by domains if specified
    if config.domains:
        logger.info(f"Filtering to domains: {config.domains}")
        original_count = len(data)
        data = [
            case for case in data
            if case.get("milestone_context", {}).get("domain") in config.domains
        ]
        logger.info(f"Filtered from {original_count} to {len(data)} cases")
    
    # Ensure age information is present and add some if needed
    ages = [case.get("milestone_context", {}).get("age_months") for case in data]
    ages = [age for age in ages if age is not None]
    
    if not ages:
        logger.warning("No age information found in test data, adding example ages")
        # Add age information spanning across age groups
        additional_cases = []
        for domain in config.domains:
            for age_group in config.age_groups:
                min_age, max_age = age_group
                # Add examples at start, middle, and end of range
                for age in [min_age, (min_age + max_age) // 2, max_age]:
                    for original_case in data[:5]:  # Use first 5 cases as templates
                        case_copy = dict(original_case)
                        if "milestone_context" not in case_copy:
                            case_copy["milestone_context"] = {}
                        case_copy["milestone_context"]["age_months"] = age
                        case_copy["milestone_context"]["domain"] = domain
                        additional_cases.append(case_copy)
        
        logger.info(f"Added {len(additional_cases)} cases with example ages")
        data.extend(additional_cases)
    
    # Sample if requested
    if config.sample_size and config.sample_size < len(data):
        logger.info(f"Sampling {config.sample_size} cases from {len(data)} total")
        data = random.sample(data, config.sample_size)
    
    return data

def run_scoring_benchmark(
    config: BenchmarkConfig,
    test_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run benchmark with different configurations"""
    
    # Define configurations to test
    configs = [
        ScoringConfig(
            name="standard",
            description="Standard scoring without age-specific knowledge",
            config={
                "use_domain_specific_prompts": True,
                "use_age_specific_prompts": False
            },
            color="blue"
        ),
        ScoringConfig(
            name="age_prompts",
            description="Scoring with age-specific prompt templates",
            config={
                "use_domain_specific_prompts": True,
                "use_age_specific_prompts": True,
                "custom_templates_dir": "config/prompt_templates"
            },
            color="green"
        ),
        ScoringConfig(
            name="age_adjusted",
            description="Scoring with full age adjustment (prompts + confidence)",
            config={
                "use_domain_specific_prompts": True,
                "use_age_specific_prompts": True,
                "custom_templates_dir": "config/prompt_templates"
            },
            color="red"
        )
    ]
    
    # Results storage
    results = {
        "configs": [c.to_dict() for c in configs],  # Convert configs to dictionaries
        "metrics": {},
        "raw_results": {},
        "runtime": {},
        "by_age_group": {}
    }
    
    # Initialize age group results
    for min_age, max_age in config.age_groups:
        age_key = f"{min_age}-{max_age}"
        results["by_age_group"][age_key] = {}
    
    # Run benchmark for each configuration
    for cfg in configs:
        logger.info(f"\nBenchmarking configuration: {cfg.name}")
        logger.info(f"Description: {cfg.description}")
        
        # Initialize scorer
        try:
            # Use MockLLMScorer instead of LLMBasedScorer
            scorer = MockLLMScorer(cfg.config)
            
            # Storage for this configuration's results
            config_results = []
            age_adjusted_results = []
            start_time = time.time()
            
            # Process each test case
            for i, case in enumerate(test_data):
                response = case.get("response", "")
                milestone_context = case.get("milestone_context", {})
                expected_score = case.get("expected_score", "NOT_RATED")
                
                # Track which age group this belongs to
                age_months = milestone_context.get("age_months", 0)
                domain = milestone_context.get("domain", "unknown")
                age_group = next(
                    (f"{min_age}-{max_age}" for min_age, max_age in config.age_groups 
                     if min_age <= age_months <= max_age), 
                    "unknown"
                )
                
                logger.info(f"Processing case {i+1}/{len(test_data)} - Age: {age_months}m, Domain: {domain}")
                
                # Score the response
                if config.run_llm:
                    # Run actual scoring
                    result = scorer.score(response, milestone_context)
                    
                    # Store original result
                    config_results.append({
                        "case_id": i,
                        "age_months": age_months,
                        "domain": domain,
                        "age_group": age_group,
                        "expected_score": expected_score,
                        "score": result.score.name,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning
                    })
                    
                    # If this is the age_adjusted config, also apply manual adjustment
                    if cfg.name == "age_adjusted":
                        adjusted_score, adjusted_confidence = adjust_category_for_age(
                            result.score.name,
                            result.confidence,
                            age_months,
                            domain.lower() if domain else None
                        )
                        
                        age_adjusted_results.append({
                            "case_id": i,
                            "age_months": age_months,
                            "domain": domain,
                            "age_group": age_group,
                            "expected_score": expected_score,
                            "original_score": result.score.name,
                            "original_confidence": result.confidence,
                            "adjusted_score": adjusted_score,
                            "adjusted_confidence": adjusted_confidence
                        })
                else:
                    # Just format the prompt and simulate scoring
                    prompt = scorer._format_prompt(response, milestone_context)
                    
                    # Add a simulated result
                    result = scorer.score(response, milestone_context)
                    
                    config_results.append({
                        "case_id": i,
                        "age_months": age_months,
                        "domain": domain,
                        "age_group": age_group,
                        "expected_score": expected_score,
                        "score": result.score.name,
                        "confidence": result.confidence,
                        "prompt_length": len(prompt),
                        "has_age_specific": "age_months" in prompt.lower(),
                        "has_domain_specific": domain.lower() in prompt.lower() if domain else False
                    })
                    
                    # If this is the age_adjusted config, also apply manual adjustment
                    if cfg.name == "age_adjusted":
                        adjusted_score, adjusted_confidence = adjust_category_for_age(
                            result.score.name,
                            result.confidence,
                            age_months,
                            domain.lower() if domain else None
                        )
                        
                        age_adjusted_results.append({
                            "case_id": i,
                            "age_months": age_months,
                            "domain": domain,
                            "age_group": age_group,
                            "expected_score": expected_score,
                            "original_score": result.score.name,
                            "original_confidence": result.confidence,
                            "adjusted_score": adjusted_score,
                            "adjusted_confidence": adjusted_confidence
                        })
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Store results
            results["raw_results"][cfg.name] = config_results
            results["runtime"][cfg.name] = runtime
            
            if cfg.name == "age_adjusted" and age_adjusted_results:
                results["raw_results"]["age_adjusted_manual"] = age_adjusted_results
            
            # Calculate metrics
            metrics = calculate_metrics(config_results, expected_available=True)
            results["metrics"][cfg.name] = metrics
            
            # Calculate metrics by age group
            for age_group in results["by_age_group"]:
                age_group_cases = [
                    case for case in config_results 
                    if case["age_group"] == age_group
                ]
                
                if age_group_cases:
                    age_group_metrics = calculate_metrics(age_group_cases, expected_available=True)
                    results["by_age_group"][age_group][cfg.name] = age_group_metrics
            
            # If we have age_adjusted results, calculate those metrics too
            if cfg.name == "age_adjusted" and age_adjusted_results:
                adj_metrics = calculate_metrics_for_adjusted(age_adjusted_results)
                results["metrics"]["age_adjusted_manual"] = adj_metrics
                
                # By age group
                for age_group in results["by_age_group"]:
                    age_group_cases = [
                        case for case in age_adjusted_results 
                        if case["age_group"] == age_group
                    ]
                    
                    if age_group_cases:
                        age_group_metrics = calculate_metrics_for_adjusted(age_group_cases)
                        results["by_age_group"][age_group]["age_adjusted_manual"] = age_group_metrics
            
            logger.info(f"Completed {cfg.name} in {runtime:.2f}s")
            
        except Exception as e:
            logger.error(f"Error benchmarking {cfg.name}: {e}")
            results["raw_results"][cfg.name] = []
            results["runtime"][cfg.name] = 0
            results["metrics"][cfg.name] = {"error": str(e)}
    
    return results

def calculate_metrics(cases: List[Dict[str, Any]], expected_available: bool = False) -> Dict[str, Any]:
    """Calculate metrics for a set of cases"""
    if not cases:
        return {"error": "No cases provided"}
    
    # Initialize metrics
    metrics = {
        "count": len(cases),
        "avg_confidence": sum(case.get("confidence", 0) for case in cases) / len(cases),
        "categories": {}
    }
    
    # Calculate category distribution
    for case in cases:
        category = case.get("score", "UNKNOWN")
        if category not in metrics["categories"]:
            metrics["categories"][category] = 0
        metrics["categories"][category] += 1
    
    # Normalize category distribution
    for category in metrics["categories"]:
        metrics["categories"][category] /= len(cases)
    
    # Calculate agreement with expected score if available
    if expected_available:
        agreement_count = sum(
            1 for case in cases 
            if case.get("expected_score") and case.get("score") == case.get("expected_score")
        )
        metrics["agreement"] = agreement_count / len(cases) if len(cases) > 0 else 0
    
    return metrics

def calculate_metrics_for_adjusted(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate metrics specifically for adjusted scores"""
    if not cases:
        return {"error": "No cases provided"}
    
    # Initialize metrics
    metrics = {
        "count": len(cases),
        "avg_confidence_original": sum(case.get("original_confidence", 0) for case in cases) / len(cases),
        "avg_confidence_adjusted": sum(case.get("adjusted_confidence", 0) for case in cases) / len(cases),
        "categories_original": {},
        "categories_adjusted": {},
        "category_changes": {},
        "improved_cases": 0,
        "worsened_cases": 0,
        "unchanged_cases": 0
    }
    
    # Calculate category distributions
    for case in cases:
        original_category = case.get("original_score", "UNKNOWN")
        adjusted_category = case.get("adjusted_score", "UNKNOWN")
        expected_score = case.get("expected_score")
        
        if original_category not in metrics["categories_original"]:
            metrics["categories_original"][original_category] = 0
        metrics["categories_original"][original_category] += 1
        
        if adjusted_category not in metrics["categories_adjusted"]:
            metrics["categories_adjusted"][adjusted_category] = 0
        metrics["categories_adjusted"][adjusted_category] += 1
        
        # Track category changes
        if original_category != adjusted_category:
            change_key = f"{original_category}->{adjusted_category}"
            if change_key not in metrics["category_changes"]:
                metrics["category_changes"][change_key] = 0
            metrics["category_changes"][change_key] += 1
            
            # Track impact on agreement
            if expected_score:
                if original_category == expected_score and adjusted_category != expected_score:
                    # Agreement got worse
                    metrics["worsened_cases"] += 1
                elif original_category != expected_score and adjusted_category == expected_score:
                    # Agreement improved
                    metrics["improved_cases"] += 1
                else:
                    # No change in agreement
                    metrics["unchanged_cases"] += 1
        elif expected_score:
            # No category change, but still count for agreement stats
            metrics["unchanged_cases"] += 1
    
    # Normalize category distributions
    for category in metrics["categories_original"]:
        metrics["categories_original"][category] /= len(cases)
    
    for category in metrics["categories_adjusted"]:
        metrics["categories_adjusted"][category] /= len(cases)
    
    # Calculate agreement with expected score if available
    agreement_original = sum(
        1 for case in cases 
        if case.get("expected_score") and case.get("original_score") == case.get("expected_score")
    )
    agreement_adjusted = sum(
        1 for case in cases 
        if case.get("expected_score") and case.get("adjusted_score") == case.get("expected_score")
    )
    
    metrics["agreement_original"] = agreement_original / len(cases) if len(cases) > 0 else 0
    metrics["agreement_adjusted"] = agreement_adjusted / len(cases) if len(cases) > 0 else 0
    
    # Calculate how often the score changed
    changes = sum(1 for case in cases if case.get("original_score") != case.get("adjusted_score"))
    metrics["change_rate"] = changes / len(cases) if len(cases) > 0 else 0
    
    # Calculate percentages for agreement impact
    if len(cases) > 0:
        metrics["improved_rate"] = metrics["improved_cases"] / len(cases)
        metrics["worsened_rate"] = metrics["worsened_cases"] / len(cases)
        metrics["unchanged_rate"] = metrics["unchanged_cases"] / len(cases)
    
    return metrics

def generate_visualizations(results: Dict[str, Any], output_dir: str) -> None:
    """Generate visualizations from benchmark results"""
    if not results or not results.get("raw_results"):
        logger.error("No results to visualize")
        return
    
    logger.info("Generating visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Average confidence by configuration
    plt.figure(figsize=(10, 6))
    
    config_names = []
    confidences = []
    
    for config in results["configs"]:
        name = config["name"]
        if name in results["metrics"]:
            config_names.append(name)
            
            # Handle error case where average confidence might be missing
            if "avg_confidence" in results["metrics"][name]:
                confidences.append(results["metrics"][name]["avg_confidence"])
            elif "error" in results["metrics"][name]:
                logger.warning(f"Skipping visualization for {name}: {results['metrics'][name]['error']}")
                confidences.append(0)
            else:
                confidences.append(0)
            
            # Add adjusted version if available
            if name == "age_adjusted" and "age_adjusted_manual" in results["metrics"]:
                config_names.append("age_adjusted_manual")
                if "avg_confidence_adjusted" in results["metrics"]["age_adjusted_manual"]:
                    confidences.append(results["metrics"]["age_adjusted_manual"]["avg_confidence_adjusted"])
                else:
                    confidences.append(0)
    
    if config_names and confidences:
        plt.bar(config_names, confidences, color=[config["color"] for config in results["configs"] if config["name"] in config_names] + ["purple"])
        plt.title("Average Confidence by Configuration")
        plt.xlabel("Configuration")
        plt.ylabel("Average Confidence")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confidence_by_config.png"))
    else:
        logger.warning("No confidence data to visualize")
    
    # Agreement with expected score if available
    agreements = []
    for name in config_names:
        if name == "age_adjusted_manual":
            if "agreement_adjusted" in results["metrics"].get("age_adjusted_manual", {}):
                agreements.append(results["metrics"]["age_adjusted_manual"]["agreement_adjusted"])
            else:
                agreements.append(0)
        elif "agreement" in results["metrics"].get(name, {}):
            agreements.append(results["metrics"][name]["agreement"])
        else:
            agreements.append(0)
    
    if any(agreements):
        plt.figure(figsize=(10, 6))
        plt.bar(config_names, agreements, color=[config["color"] for config in results["configs"] if config["name"] in config_names] + ["purple"])
        plt.title("Agreement with Expected Score by Configuration")
        plt.xlabel("Configuration")
        plt.ylabel("Agreement Rate")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "agreement_by_config.png"))
    else:
        logger.warning("No agreement data to visualize")
    
    # Category distribution by configuration
    for config in results["configs"]:
        name = config["name"]
        if name in results["metrics"] and "categories" in results["metrics"][name]:
            plt.figure(figsize=(10, 6))
            
            categories = list(results["metrics"][name]["categories"].keys())
            counts = []
            
            for category in categories:
                count = results["metrics"][name]["categories"].get(category, 0)
                counts.append(count)
            
            if categories and counts:
                plt.bar(categories, counts, color=config["color"])
                plt.title(f"Category Distribution: {name}")
                plt.xlabel("Category")
                plt.ylabel("Proportion")
                plt.ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"categories_{name}.png"))
            else:
                logger.warning(f"No category data to visualize for {name}")
    
    # Visualize category changes if available
    if "age_adjusted_manual" in results["metrics"] and "category_changes" in results["metrics"]["age_adjusted_manual"]:
        category_changes = results["metrics"]["age_adjusted_manual"]["category_changes"]
        if category_changes:
            plt.figure(figsize=(12, 8))
            
            changes = list(category_changes.keys())
            counts = list(category_changes.values())
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]  # Descending order
            sorted_changes = [changes[i] for i in sorted_indices]
            sorted_counts = [counts[i] for i in sorted_indices]
            
            plt.bar(sorted_changes, sorted_counts, color="purple")
            plt.title("Category Changes from Age-Specific Adjustments")
            plt.xlabel("Category Change")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "category_changes.png"))
        
        # Visualize impact on agreement
        if all(k in results["metrics"]["age_adjusted_manual"] for k in ["improved_rate", "worsened_rate", "unchanged_rate"]):
            plt.figure(figsize=(10, 6))
            
            impact_labels = ["Improved", "Worsened", "Unchanged"]
            impact_values = [
                results["metrics"]["age_adjusted_manual"]["improved_rate"],
                results["metrics"]["age_adjusted_manual"]["worsened_rate"],
                results["metrics"]["age_adjusted_manual"]["unchanged_rate"]
            ]
            
            plt.bar(impact_labels, impact_values, color=["green", "red", "gray"])
            plt.title("Impact of Age-Specific Adjustments on Agreement")
            plt.xlabel("Impact Type")
            plt.ylabel("Proportion of Cases")
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "agreement_impact.png"))
    
    # By age group if available
    if "by_age_group" in results and results["by_age_group"]:
        age_groups = list(results["by_age_group"].keys())
        if age_groups:
            plt.figure(figsize=(14, 8))
            
            x = np.arange(len(age_groups))
            width = 0.2
            
            # Check if we have any valid data to plot
            has_valid_data = False
            
            for i, config in enumerate(results["configs"]):
                name = config["name"]
                confidences = []
                
                for age_group in age_groups:
                    if (name in results["by_age_group"][age_group] and 
                        "avg_confidence" in results["by_age_group"][age_group][name]):
                        confidences.append(results["by_age_group"][age_group][name]["avg_confidence"])
                        has_valid_data = True
                    else:
                        confidences.append(0)
                
                if confidences:
                    plt.bar(x + i*width, confidences, width, label=name, color=config["color"])
            
            if has_valid_data:
                plt.title("Confidence by Age Group and Configuration")
                plt.xlabel("Age Group (months)")
                plt.ylabel("Average Confidence")
                plt.xticks(x + width, age_groups)
                plt.legend()
                plt.ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "confidence_by_age.png"))
            else:
                logger.warning("No confidence by age group data to visualize")
        
            # Agreement by age group
            agreements_available = False
            for age_group in age_groups:
                for config in results["configs"]:
                    if (config["name"] in results["by_age_group"][age_group] and 
                        "agreement" in results["by_age_group"][age_group][config["name"]]):
                        agreements_available = True
                        break
            
            if agreements_available:
                plt.figure(figsize=(14, 8))
                
                # Check if we have any valid data to plot
                has_valid_data = False
                
                for i, config in enumerate(results["configs"]):
                    name = config["name"]
                    agreements = []
                    
                    for age_group in age_groups:
                        if (name in results["by_age_group"][age_group] and 
                            "agreement" in results["by_age_group"][age_group][name]):
                            agreements.append(results["by_age_group"][age_group][name]["agreement"])
                            has_valid_data = True
                        else:
                            agreements.append(0)
                    
                    if agreements:
                        plt.bar(x + i*width, agreements, width, label=name, color=config["color"])
                
                if has_valid_data:
                    plt.title("Agreement with Expected Score by Age Group")
                    plt.xlabel("Age Group (months)")
                    plt.ylabel("Agreement Rate")
                    plt.xticks(x + width, age_groups)
                    plt.legend()
                    plt.ylim(0, 1)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "agreement_by_age.png"))
                else:
                    logger.warning("No agreement by age group data to visualize")

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save results to output files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary metrics
    summary = {
        "sample_size": len(next(iter(results["raw_results"].values()), [])),
        "runtime": {name: time for name, time in results["runtime"].items()},
        "configurations": [c["name"] for c in results["configs"]]
    }
    
    for config_name, metrics in results["metrics"].items():
        summary[config_name] = {
            k: v for k, v in metrics.items() 
            if k not in ["categories", "categories_original", "categories_adjusted"]
        }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    details = {
        "configs": [{"name": c["name"], "description": c["description"]} for c in results["configs"]],
        "raw_results": results["raw_results"],
        "by_age_group": results["by_age_group"] if "by_age_group" in results else {}
    }
    
    with open(os.path.join(output_dir, "details.json"), 'w') as f:
        json.dump(details, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

def main():
    """Main benchmark function"""
    logger.info("Starting Age-Specific Knowledge Benchmark")
    
    # Parse arguments
    config = parse_args()
    logger.info(f"Using test data: {config.data_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Domains: {config.domains}")
    logger.info(f"Age groups: {config.age_groups}")
    logger.info(f"Run LLM: {config.run_llm}")
    logger.info(f"Sample size: {config.sample_size}")
    
    # Load test data
    test_data = load_test_data(config)
    logger.info(f"Loaded {len(test_data)} test cases")
    
    if not test_data:
        logger.error("No test data available, exiting")
        return
    
    # Run benchmark
    results = run_scoring_benchmark(config, test_data)
    
    # Generate visualizations
    if config.save_figures:
        generate_visualizations(results, config.output_dir)
    
    # Save results
    save_results(results, config.output_dir)
    
    logger.info("Benchmark completed")

if __name__ == "__main__":
    main() 