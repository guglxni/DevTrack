"""
Continuous Learning Engine Module

This module implements mechanisms for continuous model improvement through
expert feedback and retraining.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Set, Union
import threading
import logging

from .base import Score, ScoringResult


class ContinuousLearningEngine:
    """
    Engine for continuous learning and model improvement
    
    This implementation:
    1. Collects examples that need expert review
    2. Incorporates expert feedback into the model
    3. Builds a training dataset for model improvement
    4. Manages periodic model retraining
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the continuous learning engine"""
        self.config = config or self._default_config()
        
        # Set up logging
        self.logger = logging.getLogger("continuous_learning")
        self.logger.setLevel(logging.INFO)
        
        # Set up file paths
        self._setup_data_paths()
        
        # Load existing data
        self.review_queue = self._load_review_queue()
        self.training_examples = self._load_training_examples()
        
        # Initialize lock for thread safety
        self.queue_lock = threading.Lock()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
            "data_directory": "data/continuous_learning",  # Directory for data files
            "review_queue_file": "review_queue.json",      # File for items needing review
            "training_data_file": "training_examples.json",# File for training examples
            "max_queue_size": 1000,                        # Maximum review queue size
            "min_confidence_for_auto": 0.9,                # Min confidence for automatic addition
            "retraining_threshold": 100,                   # Number of new examples before retraining
            "enable_auto_retraining": False,               # Whether to automatically retrain
        }
    
    def _setup_data_paths(self) -> None:
        """Set up data file paths"""
        data_dir = self.config.get("data_directory", "data/continuous_learning")
        os.makedirs(data_dir, exist_ok=True)
        
        self.review_queue_path = os.path.join(
            data_dir, 
            self.config.get("review_queue_file", "review_queue.json")
        )
        
        self.training_data_path = os.path.join(
            data_dir, 
            self.config.get("training_data_file", "training_examples.json")
        )
    
    def _load_review_queue(self) -> List[Dict[str, Any]]:
        """Load the existing review queue"""
        if not os.path.exists(self.review_queue_path):
            return []
            
        try:
            with open(self.review_queue_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading review queue: {e}")
            return []
    
    def _save_review_queue(self) -> None:
        """Save the current review queue"""
        try:
            with open(self.review_queue_path, 'w') as f:
                json.dump(self.review_queue, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving review queue: {e}")
    
    def _load_training_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load existing training examples"""
        if not os.path.exists(self.training_data_path):
            return {score.name: [] for score in Score if score != Score.NOT_RATED}
            
        try:
            with open(self.training_data_path, 'r') as f:
                data = json.load(f)
                
                # Ensure all score categories exist
                for score in Score:
                    if score != Score.NOT_RATED and score.name not in data:
                        data[score.name] = []
                        
                return data
        except Exception as e:
            self.logger.error(f"Error loading training examples: {e}")
            return {score.name: [] for score in Score if score != Score.NOT_RATED}
    
    def _save_training_examples(self) -> None:
        """Save the current training examples"""
        try:
            with open(self.training_data_path, 'w') as f:
                json.dump(self.training_examples, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving training examples: {e}")
    
    def queue_for_review(self, 
                        response: str, 
                        milestone_context: Dict[str, Any], 
                        predicted_score: Union[Score, ScoringResult],
                        confidence: Optional[float] = None) -> None:
        """
        Queue a response for expert review
        
        Args:
            response: The text response
            milestone_context: Context about the milestone
            predicted_score: The predicted score
            confidence: Optional explicit confidence
        """
        # Extract score and confidence from ScoringResult if needed
        if isinstance(predicted_score, ScoringResult):
            confidence = predicted_score.confidence
            predicted_score = predicted_score.score
        
        # Create queue entry
        queue_entry = {
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "milestone": milestone_context,
            "predicted_score": predicted_score.name if predicted_score else "UNKNOWN",
            "predicted_score_value": predicted_score.value if predicted_score else -1,
            "confidence": confidence,
            "status": "pending",
            "id": f"review_{len(self.review_queue) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        # Add to queue with thread safety
        with self.queue_lock:
            # Enforce max queue size
            max_queue_size = self.config.get("max_queue_size", 1000)
            
            if len(self.review_queue) >= max_queue_size:
                # Remove oldest pending item
                pending_items = [i for i in range(len(self.review_queue)) 
                                if self.review_queue[i]["status"] == "pending"]
                
                if pending_items:
                    oldest_idx = min(pending_items)
                    self.review_queue.pop(oldest_idx)
            
            self.review_queue.append(queue_entry)
            self._save_review_queue()
            
        self.logger.info(f"Queued response for review: {queue_entry['id']}")
    
    def add_expert_feedback(self, 
                           review_id: str, 
                           correct_score: Score, 
                           expert_notes: Optional[str] = None) -> bool:
        """
        Add expert feedback for a queued review item
        
        Args:
            review_id: The ID of the review item
            correct_score: The correct score according to expert
            expert_notes: Optional notes from the expert
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Find the item in the queue
        with self.queue_lock:
            found = False
            
            for item in self.review_queue:
                if item["id"] == review_id:
                    found = True
                    
                    # Update item
                    item["status"] = "reviewed"
                    item["expert_score"] = correct_score.name
                    item["expert_score_value"] = correct_score.value
                    item["expert_notes"] = expert_notes
                    item["review_timestamp"] = datetime.now().isoformat()
                    
                    # Add to training examples
                    self._add_to_training_examples(item, correct_score)
                    break
            
            if found:
                self._save_review_queue()
                return True
                
        return False
    
    def _add_to_training_examples(self, 
                                review_item: Dict[str, Any], 
                                correct_score: Score) -> None:
        """
        Add a reviewed item to training examples
        
        Args:
            review_item: The reviewed item
            correct_score: The correct score
        """
        # Create training example
        training_example = {
            "response": review_item["response"],
            "milestone": review_item["milestone"],
            "score": correct_score.name,
            "score_value": correct_score.value,
            "timestamp": datetime.now().isoformat(),
            "predicted_score": review_item.get("predicted_score"),
            "predicted_score_value": review_item.get("predicted_score_value"),
            "confidence": review_item.get("confidence"),
            "expert_notes": review_item.get("expert_notes")
        }
        
        # Add to appropriate category
        self.training_examples[correct_score.name].append(training_example)
        
        # Save training examples
        self._save_training_examples()
        
        # Check if retraining is needed
        self._check_retraining_need()
    
    def _check_retraining_need(self) -> None:
        """Check if model retraining is needed based on new examples"""
        # Count total training examples
        total_examples = sum(len(examples) for examples in self.training_examples.values())
        
        # Check retraining threshold
        threshold = self.config.get("retraining_threshold", 100)
        
        if total_examples > 0 and total_examples % threshold == 0:
            if self.config.get("enable_auto_retraining", False):
                self.logger.info(f"Retraining threshold reached ({total_examples} examples). "
                                f"Initiating auto-retraining.")
                
                # Trigger retraining
                self._retrain_models()
            else:
                self.logger.info(f"Retraining threshold reached ({total_examples} examples). "
                                f"Auto-retraining is disabled.")
    
    def _retrain_models(self) -> None:
        """Retrain models using collected training examples"""
        # This would integrate with model training code
        # For now, we just log the intent
        self.logger.info("Model retraining requested. This would trigger model retraining.")
        
        # In a real implementation, this would:
        # 1. Convert training examples to appropriate format
        # 2. Split into train/validation sets
        # 3. Train or fine-tune models
        # 4. Evaluate performance
        # 5. Update production models if performance improves
    
    def get_pending_reviews(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of items pending expert review
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List[Dict]: Pending review items
        """
        with self.queue_lock:
            # Log the total number of items in the review queue
            self.logger.info(f"Total items in review queue: {len(self.review_queue)}")
            
            # Log the number of items with status=pending
            pending_count = sum(1 for item in self.review_queue if item.get("status") == "pending")
            self.logger.info(f"Items with status=pending: {pending_count}")
            
            # Log the number of sample reviews
            sample_count = sum(1 for item in self.review_queue if item.get("id", "").startswith("sample_"))
            self.logger.info(f"Sample reviews: {sample_count}")
            
            # Log the number of sample reviews with status=pending
            sample_pending_count = sum(1 for item in self.review_queue if item.get("id", "").startswith("sample_") and item.get("status") == "pending")
            self.logger.info(f"Sample reviews with status=pending: {sample_pending_count}")
            
            # Include items with status=pending or items with id starting with "sample_"
            pending = [
                item for item in self.review_queue 
                if item.get("status") == "pending" or 
                   (item.get("id", "").startswith("sample_") and item.get("status") != "completed")
            ]
            
            # Log the number of pending reviews after filtering
            self.logger.info(f"Pending reviews after filtering: {len(pending)}")
            
            # Find all sample reviews
            sample_reviews = [item for item in pending if item.get("id", "").startswith("sample_")]
            
            # Log the first few sample reviews
            for i, item in enumerate(sample_reviews[:3]):
                self.logger.info(f"Sample review {i}: id={item.get('id', 'unknown')}, status={item.get('status', 'unknown')}")
            
            # Ensure status field is set for sample reviews
            for item in pending:
                if item.get("id", "").startswith("sample_") and "status" not in item:
                    item["status"] = "pending"
            
            # Ensure milestone_context is present in each item
            for item in pending:
                # If milestone_context is missing but milestone is present, create milestone_context from milestone
                if "milestone_context" not in item and "milestone" in item:
                    milestone = item["milestone"]
                    item["milestone_context"] = {
                        "behavior": milestone.get("behavior", "Unknown behavior"),
                        "question": milestone.get("question", f"Does your child {milestone.get('behavior', 'do this')}?"),
                        "domain": milestone.get("domain", "Unknown"),
                        "criteria": milestone.get("criteria", ""),
                        "age_range": milestone.get("age_range", "")
                    }
            
            # Get regular pending reviews up to the limit
            regular_pending = [item for item in pending if not item.get("id", "").startswith("sample_")]
            regular_pending = regular_pending[:limit]
            
            # Combine regular pending reviews with sample reviews
            result = regular_pending + sample_reviews
            
            # Log the final result
            self.logger.info(f"Returning {len(result)} pending reviews, including {len(sample_reviews)} sample reviews")
            
            return result
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected training examples
        
        Returns:
            Dict: Training data statistics
        """
        stats = {
            "total_examples": 0,
            "by_score": {},
            "latest_example": None,
            "prediction_metrics": {
                "correct_predictions": 0,
                "total_predictions": 0,
                "accuracy": 0.0
            }
        }
        
        # Collect stats by score category
        for score_name, examples in self.training_examples.items():
            count = len(examples)
            stats["by_score"][score_name] = count
            stats["total_examples"] += count
            
            # Track prediction metrics
            for example in examples:
                if "predicted_score" in example and "predicted_score_value" in example:
                    stats["prediction_metrics"]["total_predictions"] += 1
                    
                    if example["score"] == example["predicted_score"]:
                        stats["prediction_metrics"]["correct_predictions"] += 1
            
            # Track latest example
            if examples and (not stats["latest_example"] or 
                           examples[-1]["timestamp"] > stats["latest_example"]["timestamp"]):
                stats["latest_example"] = examples[-1]["timestamp"]
        
        # Calculate accuracy
        if stats["prediction_metrics"]["total_predictions"] > 0:
            stats["prediction_metrics"]["accuracy"] = (
                stats["prediction_metrics"]["correct_predictions"] / 
                stats["prediction_metrics"]["total_predictions"]
            )
        
        return stats
    
    def export_training_data(self, output_format: str = "json") -> Dict[str, Any]:
        """
        Export training data for external use
        
        Args:
            output_format: Format for export (json, csv, etc.)
            
        Returns:
            Dict: Export information
        """
        # For now, only support JSON export
        if output_format.lower() != "json":
            self.logger.warning(f"Unsupported export format: {output_format}. Using JSON.")
        
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_examples": sum(len(examples) for examples in self.training_examples.values()),
                "version": "1.0"
            },
            "examples": self.training_examples
        }
        
        export_path = os.path.join(
            self.config.get("data_directory", "data/continuous_learning"),
            f"export_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        )
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            return {
                "success": True,
                "path": export_path,
                "count": export_data["metadata"]["total_examples"]
            }
        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
            
            return {
                "success": False,
                "error": str(e)
            } 