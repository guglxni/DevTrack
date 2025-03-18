"""
Active Learning Engine for Continuous Learning System

This module provides a specialized ContinuousLearningEngine that implements
active learning techniques to identify valuable examples for expert review
and model improvement. It provides methods for continuous learning through
expert feedback.
"""

import os
import json
import uuid
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Import from ContinuousLearningEngine
from src.core.scoring.continuous_learning import ContinuousLearningEngine
from src.core.enhanced_assessment_engine import Score

# Configure logging
logger = logging.getLogger(__name__)

class ActiveLearningEngine(ContinuousLearningEngine):
    """
    Active Learning Engine for enhancing the continuous learning system.
    
    This engine extends the continuous learning system by implementing active learning
    techniques to identify valuable examples for expert review. It focuses on examples
    that are likely to improve model performance, including examples with:
    
    1. High disagreement between different scoring components
    2. Borderline confidence scores
    3. Examples with unusual linguistic patterns
    4. Examples that expose limitations in the current model
    
    It also implements versioning and tracking for model improvements.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ActiveLearningEngine.
        
        Args:
            config: Configuration dictionary for the ActiveLearningEngine
        """
        # Initialize with default config if none provided
        if config is None:
            config = self._default_config()
            
        # Call parent class init
        super().__init__(config)
        
        # Ensure data_dir is set up properly
        self.data_dir = self.config.get("data_directory", "data/continuous_learning")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load model version history if it exists
        try:
            self.model_versions = self._load_model_versions()
        except Exception as e:
            logger.error(f"Error loading model versions: {e}")
            # Initialize with a default version
            self.model_versions = [{
                "version": "0.1.0",
                "timestamp": datetime.now().isoformat(),
                "description": "Initial version",
                "metrics": {},
                "training_examples_count": 0
            }]
        
        # Track information gain and model improvement metrics
        self.information_gain_metrics = {}
        self.model_improvement_metrics = {}
        
        # Performance metrics tracking
        self.performance_history = []
        
        logger.info("Active Learning Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values for the ActiveLearningEngine.
        
        Returns:
            Dict: Default configuration values
        """
        # Get the base configuration from the parent class
        config = super()._default_config()
        
        # Add active learning specific configuration
        active_learning_config = {
            "uncertainty_threshold": 0.65,  # Threshold for considering examples with uncertain predictions
            "disagreement_threshold": 0.25,  # Threshold for considering examples with high disagreement
            "max_prioritized_examples": 50,  # Maximum number of examples to prioritize for review
            "info_gain_weights": {
                "uncertainty": 0.4,
                "disagreement": 0.3,
                "linguistic_novelty": 0.2,
                "domain_coverage": 0.1
            },
            "use_enhanced_ambiguity_detection": True,
            "active_learning_enabled": True,
            "version_history_file": "model_versions.json",
            "data_dir": "active_learning_data",
            "min_training_examples_per_category": 20  # Minimum examples per category for balanced training
        }
        
        # Update the config with active learning settings
        config.update(active_learning_config)
        
        return config
    
    def _load_model_versions(self) -> List[Dict[str, Any]]:
        """
        Load model version history from a JSON file.
        
        Returns:
            List[Dict]: List of model version entries
        """
        # Make sure data_dir is set
        if not hasattr(self, 'data_dir') or not self.data_dir:
            self.data_dir = self.config.get("data_directory", "data/continuous_learning")
            os.makedirs(self.data_dir, exist_ok=True)
            
        version_file = os.path.join(self.data_dir, self.config.get("version_history_file", "model_versions.json"))
        
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        # Convert to list if it's not already
                        return [data]
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading model versions: {e}")
                return []
        else:
            # Initialize with a default version
            initial_version = [{
                "version": "0.1.0",
                "timestamp": datetime.now().isoformat(),
                "description": "Initial version",
                "metrics": {},
                "training_examples_count": 0
            }]
            
            # Save the initial version
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(version_file), exist_ok=True)
                with open(version_file, 'w') as f:
                    json.dump(initial_version, f, indent=2)
                logger.info(f"Created initial model version file at {version_file}")
            except IOError as e:
                logger.error(f"Error creating model version file: {e}")
            
            return initial_version
    
    def _save_model_versions(self, versions: List[Dict[str, Any]]) -> None:
        """
        Save model version history to a JSON file.
        
        Args:
            versions: List of model version entries
        """
        version_file = os.path.join(self.data_dir, self.config.get("version_history_file", "model_versions.json"))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(version_file), exist_ok=True)
        
        try:
            with open(version_file, 'w') as f:
                json.dump(versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model versions: {e}")
    
    def identify_valuable_examples(self, responses: List[Dict]) -> List[Dict]:
        """
        Identify examples that would be valuable for expert review based on various criteria.
        
        Args:
            responses: List of response dictionaries with text, predicted scores, etc.
            
        Returns:
            List of responses with added information gain scores and priority
        """
        # Skip if active learning is disabled
        if not self.config.get("active_learning_enabled", True):
            return responses
        
        valuable_examples = []
        
        for response in responses:
            # Skip if already in review queue
            if self._is_in_review_queue(response.get("id", "")):
                continue
                
            # 1. Check for component disagreement
            disagreement_score = self._calculate_component_disagreement(response)
            
            # 2. Check for borderline confidence
            uncertainty_score = self._calculate_uncertainty_score(response)
            
            # 3. Check for unusual patterns in text
            novelty_score = self._calculate_linguistic_novelty(response)
            
            # 4. Check domain coverage (are we lacking examples in this domain?)
            domain_coverage_score = self._calculate_domain_coverage_score(response)
            
            # Calculate overall information gain
            info_gain = self._calculate_information_gain(
                uncertainty_score, 
                disagreement_score,
                novelty_score,
                domain_coverage_score
            )
            
            # Add scores to the response
            enriched_response = response.copy()
            enriched_response.update({
                "info_gain": info_gain,
                "uncertainty_score": uncertainty_score,
                "disagreement_score": disagreement_score,
                "novelty_score": novelty_score,
                "domain_coverage_score": domain_coverage_score,
                "priority": info_gain  # Use info gain as priority value
            })
            
            valuable_examples.append(enriched_response)
        
        return valuable_examples
    
    def _is_in_review_queue(self, response_id: str) -> bool:
        """
        Check if a response is already in the review queue.
        
        Args:
            response_id: The ID of the response
            
        Returns:
            bool: True if in review queue, False otherwise
        """
        review_queue = self._load_review_queue()
        return any(item.get("id") == response_id for item in review_queue)
    
    def _calculate_component_disagreement(self, response: Dict) -> float:
        """
        Calculate the level of disagreement between scoring components.
        
        Args:
            response: Response dictionary with scoring components
            
        Returns:
            float: Disagreement score between 0 and 1
        """
        # Get component scores if available
        component_scores = response.get("component_scores", {})
        
        if not component_scores or len(component_scores) < 2:
            return 0.0
        
        # Calculate standard deviation of component scores
        scores = list(component_scores.values())
        score_std = np.std(scores)
        
        # Normalize to a 0-1 range (assuming max std is around 2.0 for 5 score categories)
        disagreement = min(score_std / 2.0, 1.0)
        
        return disagreement
    
    def _calculate_uncertainty_score(self, response: Dict) -> float:
        """
        Calculate uncertainty score based on confidence.
        
        Args:
            response: Response dictionary with confidence score
            
        Returns:
            float: Uncertainty score between 0 and 1
        """
        confidence = response.get("confidence", 0.5)
        
        # Responses with confidence around 0.5-0.75 are most uncertain
        if confidence < 0.5:
            # Scale 0-0.5 to 0-1
            uncertainty = confidence * 2
        else:
            # Scale 0.5-1.0 to 1-0
            uncertainty = 2 - (confidence * 2)
        
        return uncertainty
    
    def _calculate_linguistic_novelty(self, response: Dict) -> float:
        """
        Calculate how novel or unusual the linguistic patterns in the text are.
        
        Args:
            response: Response dictionary with text
            
        Returns:
            float: Novelty score between 0 and 1
        """
        # This could use more sophisticated NLP techniques in a real implementation
        # Here we'll use a simple heuristic based on word count and length
        
        text = response.get("response", "")
        if not text:
            return 0.0
        
        # Simple heuristics:
        # 1. Very short or very long responses may be unusual
        words = text.split()
        word_count = len(words)
        
        if word_count < 3 or word_count > 50:
            return 0.7
        
        # 2. Sentences with unusual punctuation patterns
        unusual_punct_count = text.count('!') + text.count('?') + text.count('...')
        if unusual_punct_count > 3:
            return 0.6
        
        # 3. Sentences with mixed positive and negative indicators
        positive_words = ['can', 'does', 'able', 'yes', 'good', 'well']
        negative_words = ['cannot', 'not', 'doesn\'t', 'unable', 'no', 'never']
        
        has_positive = any(word in text.lower() for word in positive_words)
        has_negative = any(word in text.lower() for word in negative_words)
        
        if has_positive and has_negative:
            return 0.8
            
        return 0.3  # Default modest novelty score
    
    def _calculate_domain_coverage_score(self, response: Dict) -> float:
        """
        Calculate how much this example would improve domain coverage.
        
        Args:
            response: Response dictionary with domain information
            
        Returns:
            float: Domain coverage score between 0 and 1
        """
        # Get training examples
        training_examples = self._load_training_examples()
        
        # Get domain and milestone
        domain = response.get("domain", "unknown")
        milestone = response.get("milestone", "unknown")
        predicted_score = response.get("predicted_score", "CANNOT_DO")
        
        # Count examples for this domain and score category
        domain_examples = sum(1 for ex in training_examples if ex.get("domain") == domain)
        milestone_examples = sum(1 for ex in training_examples if ex.get("milestone") == milestone)
        category_examples = sum(1 for ex in training_examples if ex.get("score") == predicted_score)
        
        # Calculate coverage scores (inverse of example counts, normalized)
        min_examples = self.config.get("min_training_examples_per_category", 20)
        
        domain_score = max(0, 1 - (domain_examples / min_examples)) if domain_examples < min_examples else 0
        milestone_score = max(0, 1 - (milestone_examples / min_examples)) if milestone_examples < min_examples else 0
        category_score = max(0, 1 - (category_examples / min_examples)) if category_examples < min_examples else 0
        
        # Combined score (weighted average)
        coverage_score = (0.4 * domain_score) + (0.4 * milestone_score) + (0.2 * category_score)
        
        return coverage_score
    
    def _calculate_information_gain(self, 
                                   uncertainty: float, 
                                   disagreement: float,
                                   novelty: float,
                                   domain_coverage: float) -> float:
        """
        Calculate the potential information gain from expert feedback.
        
        Args:
            uncertainty: Uncertainty score (0-1)
            disagreement: Disagreement score (0-1)
            novelty: Linguistic novelty score (0-1)
            domain_coverage: Domain coverage score (0-1)
            
        Returns:
            float: Information gain score between 0 and 1
        """
        # Get weights from config
        weights = self.config.get("info_gain_weights", {
            "uncertainty": 0.4,
            "disagreement": 0.3,
            "linguistic_novelty": 0.2,
            "domain_coverage": 0.1
        })
        
        # Calculate weighted sum
        info_gain = (
            (weights.get("uncertainty", 0.4) * uncertainty) +
            (weights.get("disagreement", 0.3) * disagreement) +
            (weights.get("linguistic_novelty", 0.2) * novelty) +
            (weights.get("domain_coverage", 0.1) * domain_coverage)
        )
        
        return info_gain
    
    def prioritize_expert_review(self, responses: List[Dict], max_count: int = None) -> List[Dict]:
        """
        Prioritize examples for expert review based on information gain.
        
        Args:
            responses: List of response dictionaries with info_gain scores
            max_count: Maximum number of examples to return
            
        Returns:
            List of responses, sorted by priority
        """
        if not max_count:
            max_count = self.config.get("max_prioritized_examples", 50)
        
        # Sort by information gain (priority)
        sorted_responses = sorted(
            responses, 
            key=lambda x: x.get("priority", 0), 
            reverse=True
        )
        
        # Return top N examples
        return sorted_responses[:max_count]
    
    def queue_with_priority(self, 
                           response: str, 
                           milestone_context: Dict, 
                           predicted_score: Score, 
                           confidence: float,
                           component_scores: Dict = None,
                           reasoning: str = None,
                           priority: float = None,
                           expire_days: int = 30) -> str:
        """
        Queue a response for expert review with priority information.
        
        Args:
            response: The response text to review
            milestone_context: Context about the milestone being assessed
            predicted_score: The predicted score from the model
            confidence: Confidence in the prediction
            component_scores: Optional dict of component scores
            reasoning: Optional reasoning for the prediction
            priority: Optional priority value (0-1)
            expire_days: Days until this review expires
            
        Returns:
            str: The ID of the queued review
        """
        # Generate a unique ID
        review_id = str(uuid.uuid4())
        
        # Calculate priority if not provided
        if priority is None:
            # Create a temporary response dict to calculate info gain
            temp_response = {
                "response": response,
                "confidence": confidence,
                "predicted_score": predicted_score.name if isinstance(predicted_score, Score) else predicted_score,
                "domain": milestone_context.get("domain", "unknown"),
                "milestone": milestone_context.get("behavior", "unknown"),
                "component_scores": component_scores or {}
            }
            
            # Identify valuable examples and get priority
            valuable_examples = self.identify_valuable_examples([temp_response])
            if valuable_examples:
                priority = valuable_examples[0].get("priority", 0.5)
            else:
                priority = 0.5
        
        # Calculate expiration date
        expiration_date = (datetime.now() + timedelta(days=expire_days)).isoformat()
        
        # Create the review item
        review_item = {
            "id": review_id,
            "response": response,
            "milestone_context": milestone_context,
            "predicted_score": predicted_score.name if isinstance(predicted_score, Score) else predicted_score,
            "predicted_score_value": predicted_score.value if isinstance(predicted_score, Score) else 0,
            "confidence": confidence,
            "component_scores": component_scores,
            "reasoning": reasoning,
            "priority": priority,
            "status": "PENDING",
            "created_at": datetime.now().isoformat(),
            "expires_at": expiration_date
        }
        
        # Add to the review queue
        review_queue = self._load_review_queue()
        review_queue.append(review_item)
        self._save_review_queue(review_queue)
        
        logger.info(f"Queued response for expert review with ID {review_id} and priority {priority:.2f}")
        
        return review_id
    
    def add_model_version(self, 
                         version: str, 
                         description: str, 
                         metrics: Dict = None,
                         training_examples_count: int = None) -> Dict:
        """
        Add a new model version to the version history.
        
        Args:
            version: Version string (e.g., "1.2.0")
            description: Description of changes in this version
            metrics: Performance metrics for this version
            training_examples_count: Number of training examples used
            
        Returns:
            Dict: The created version entry
        """
        if training_examples_count is None:
            # Count the current training examples
            training_examples = self._load_training_examples()
            training_examples_count = len(training_examples)
        
        # Create the version entry
        version_entry = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "metrics": metrics or {},
            "training_examples_count": training_examples_count
        }
        
        # Add to version history
        self.model_versions.append(version_entry)
        self._save_model_versions(self.model_versions)
        
        logger.info(f"Added new model version {version}: {description}")
        
        return version_entry
    
    def get_current_version(self) -> Dict:
        """
        Get the current model version information.
        
        Returns:
            Dict: Current version information
        """
        if not self.model_versions:
            return {
                "version": "0.1.0",
                "timestamp": datetime.now().isoformat(),
                "description": "Initial model version",
                "metrics": {},
                "training_examples_count": 0
            }
        
        # Find the most recent version by timestamp
        return sorted(
            self.model_versions,
            key=lambda v: v.get("timestamp", ""),
            reverse=True
        )[0]  # Return the first (most recent) item
    
    def increment_version(self, 
                         level: str = "patch", 
                         description: str = None, 
                         metrics: Dict = None) -> Dict:
        """
        Increment the version number and create a new version entry.
        
        Args:
            level: Version level to increment ("major", "minor", or "patch")
            description: Description of the version changes
            metrics: Performance metrics for this version
            
        Returns:
            Dict: The new version entry
        """
        current = self.get_current_version()
        current_version = current.get("version", "0.1.0")
        
        # Parse version components
        try:
            major, minor, patch = map(int, current_version.split("."))
        except ValueError:
            major, minor, patch = 0, 1, 0
        
        # Increment the appropriate level
        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        else:  # patch is default
            patch += 1
        
        # Create new version string
        new_version = f"{major}.{minor}.{patch}"
        
        # Default description if none provided
        if not description:
            description = f"Automatic version bump to {new_version}"
        
        # Add the new version
        return self.add_model_version(new_version, description, metrics)
    
    def record_performance_metrics(self, metrics: Dict) -> None:
        """
        Record performance metrics for tracking improvement over time.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        # Add timestamp
        metrics_with_time = metrics.copy()
        metrics_with_time["timestamp"] = datetime.now().isoformat()
        
        # Add to history
        self.performance_history.append(metrics_with_time)
        
        # Save to file
        metrics_file = os.path.join(self.data_dir, "performance_history.json")
        
        try:
            # Load existing history if file exists
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add new metrics
            history.append(metrics_with_time)
            
            # Save updated history
            with open(metrics_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
    
    def calculate_model_improvement(self, 
                                   baseline_metrics: Dict,
                                   current_metrics: Dict) -> Dict:
        """
        Calculate improvement between baseline and current metrics.
        
        Args:
            baseline_metrics: Baseline performance metrics
            current_metrics: Current performance metrics
            
        Returns:
            Dict: Improvement metrics
        """
        improvement = {}
        
        # Calculate improvement for each metric
        for key in current_metrics:
            if key in baseline_metrics and isinstance(current_metrics[key], (int, float)):
                baseline_value = baseline_metrics[key]
                current_value = current_metrics[key]
                
                # Calculate absolute difference
                abs_diff = current_value - baseline_value
                
                # Calculate percentage improvement if possible
                if baseline_value != 0:
                    pct_diff = (abs_diff / baseline_value) * 100
                    improvement[f"{key}_pct"] = pct_diff
                
                # Add absolute difference
                improvement[f"{key}_diff"] = abs_diff
        
        return improvement
    
    def export_feedback_interface_data(self) -> Dict:
        """
        Export data needed for the feedback interface.
        
        Returns:
            Dict: Data for the feedback interface
        """
        # Get current model version
        current_version = self.get_current_version()
        
        # Get pending reviews
        review_queue = self._load_review_queue()
        pending_reviews = [r for r in review_queue if r.get("status") == "PENDING"]
        
        # Sort by priority
        pending_reviews = sorted(
            pending_reviews,
            key=lambda r: r.get("priority", 0),
            reverse=True
        )
        
        # Get statistics
        statistics = self.get_system_statistics()
        
        # List of score categories
        score_categories = [s.name for s in Score]
        
        return {
            "model_version": current_version.get("version", "0.1.0"),
            "pending_reviews": pending_reviews,
            "statistics": statistics,
            "categories": score_categories
        }
    
    def get_system_statistics(self) -> Dict:
        """
        Get statistics about the active learning system.
        
        Returns:
            Dict: System statistics
        """
        try:
            # Get all data
            review_queue = self._load_review_queue()
            training_examples = self._load_training_examples()
            
            # Ensure review_queue and training_examples are lists
            if not isinstance(review_queue, list):
                review_queue = []
            if not isinstance(training_examples, list):
                training_examples = []
            
            # Calculate statistics
            pending_reviews = 0
            completed_reviews = 0
            
            for r in review_queue:
                # Check if it's a dictionary (expected case)
                if isinstance(r, dict):
                    if r.get("status") == "PENDING":
                        pending_reviews += 1
                    elif r.get("status") == "REVIEWED":
                        completed_reviews += 1
                # Handle string case (unexpected, but handle gracefully)
                elif isinstance(r, str):
                    # Log and continue - this is a problematic item
                    logger.warning(f"Found string item in review queue instead of dictionary: {r}")
                    continue
                else:
                    # Handle other types
                    logger.warning(f"Found unexpected type in review queue: {type(r)}")
                    continue
            
            # Count examples by category
            examples_by_category = defaultdict(int)
            for example in training_examples:
                if isinstance(example, dict):
                    category = example.get("score", "UNKNOWN")
                    examples_by_category[category] += 1
                elif isinstance(example, str):
                    logger.warning(f"Found string item in training examples instead of dictionary: {example}")
                    continue
                else:
                    logger.warning(f"Found unexpected type in training examples: {type(example)}")
                    continue
            
            # Get current version info
            current_version = self.get_current_version()
            
            return {
                "total_examples": len(training_examples),
                "examples_by_category": dict(examples_by_category),
                "pending_reviews": pending_reviews,
                "completed_reviews": completed_reviews,
                "current_model_version": current_version.get("version", "0.1.0") if isinstance(current_version, dict) else "0.1.0",
                "total_model_versions": len(self._load_model_versions())
            }
        except Exception as e:
            logger.error(f"Error getting system statistics: {str(e)}")
            return {
                "total_examples": 0,
                "examples_by_category": {},
                "pending_reviews": 0,
                "completed_reviews": 0,
                "current_model_version": "0.1.0",
                "total_model_versions": 0
            }
    
    def get_prioritized_reviews(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get pending reviews ordered by priority.
        
        Args:
            limit: Maximum number of reviews to return
            
        Returns:
            List of review items ordered by priority
        """
        try:
            # Get pending reviews from the parent class
            pending = self.get_pending_reviews(limit=100)  # Get more than we need to prioritize
            
            # Log the total number of pending reviews
            logger.info(f"Got {len(pending)} pending reviews")
            
            # Log the first few reviews to see what's in them
            for i, item in enumerate(pending[:5]):
                logger.info(f"Review {i}: id={item.get('id', 'unknown')}, status={item.get('status', 'unknown')}")
            
            # Check for sample reviews
            sample_reviews = [item for item in pending if item.get('id', '').startswith('sample_')]
            logger.info(f"Found {len(sample_reviews)} sample reviews")
            
            # If we have sample reviews, make sure they're included in the result
            if sample_reviews:
                # Log the first few sample reviews
                for i, item in enumerate(sample_reviews[:3]):
                    logger.info(f"Sample review {i}: id={item.get('id', 'unknown')}, status={item.get('status', 'unknown')}")
            
            if not pending:
                return []
                
            # Calculate priority for each review
            for item in pending:
                # Default priority if not set
                if "priority" not in item:
                    # Calculate priority based on information gain factors
                    uncertainty = item.get("confidence", 0.5)
                    uncertainty_score = 1.0 - uncertainty  # Lower confidence = higher uncertainty
                    
                    # Simple disagreement score (can be enhanced)
                    disagreement = 0.0
                    if "component_scores" in item:
                        scores = [s.get("score_value", 0) for s in item.get("component_scores", [])]
                        if scores and len(scores) > 1:
                            disagreement = np.std(scores) / 4.0  # Normalize by max possible std
                    
                    # Calculate priority
                    priority = (
                        uncertainty_score * 0.6 +  # Weight uncertainty more
                        disagreement * 0.4
                    )
                    
                    item["priority"] = min(max(priority, 0.0), 1.0)  # Ensure between 0 and 1
                    
                # Ensure sample reviews have high priority
                if item.get('id', '').startswith('sample_') and item.get('priority', 0.0) < 0.9:
                    item['priority'] = 0.95  # Give sample reviews very high priority
            
            # Sort by priority (highest first)
            sorted_reviews = sorted(pending, key=lambda x: x.get("priority", 0.0), reverse=True)
            
            # Ensure sample reviews are included in the result
            sample_ids = {item.get('id') for item in sample_reviews}
            prioritized_reviews = []
            
            # First, include all sample reviews
            for item in sorted_reviews:
                if item.get('id') in sample_ids:
                    prioritized_reviews.append(item)
            
            # Then add other high-priority reviews up to the limit
            for item in sorted_reviews:
                if item.get('id') not in sample_ids and len(prioritized_reviews) < limit:
                    prioritized_reviews.append(item)
            
            # Limit to the requested number
            result = prioritized_reviews[:limit]
            
            # Log the number of sample reviews in the result
            sample_count = sum(1 for item in result if item.get('id', '').startswith('sample_'))
            logger.info(f"Returning {len(result)} prioritized reviews, including {sample_count} sample reviews")
            
            return result
        except Exception as e:
            logger.error(f"Error in get_prioritized_reviews: {e}")
            return [] 