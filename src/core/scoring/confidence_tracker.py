"""
Confidence Tracking Module

This module implements techniques for tracking confidence and uncertainty.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
import numpy as np
import json
import os
from datetime import datetime

from .base import Score, ScoringResult


class ConfidenceTracker:
    """
    Tracks confidence in scoring predictions and maintains historical data
    
    This implementation:
    1. Aggregates confidence metrics from multiple scoring methods
    2. Maintains a history of confidence by category and method
    3. Provides uncertainty estimation for identifying low-confidence predictions
    4. Helps identify cases that need expert review
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the confidence tracker"""
        self.config = config or self._default_config()
        
        # Initialize history tracking
        self.history = {score.name: deque(maxlen=self.config["history_size"]) 
                        for score in Score}
        
        # Track method-specific performance
        self.method_performance = {}
        
        # Load existing history if available
        self._load_history()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
            "history_size": 1000,                # Max history entries to keep
            "confidence_threshold": 0.7,         # Threshold for confident predictions
            "uncertainty_threshold": 0.3,        # Threshold for uncertain predictions
            "method_weights": {                  # Default method weights
                "keyword": 0.3,
                "embedding": 0.3,
                "transformer": 0.4
            },
            "history_file": "data/confidence_history.json"  # File to store history
        }
    
    def _load_history(self) -> None:
        """Load confidence history from file if available"""
        history_file = self.config.get("history_file", "data/confidence_history.json")
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    
                    # Convert loaded data to our format
                    for score_name, entries in data.get("history", {}).items():
                        if score_name in self.history:
                            self.history[score_name] = deque(entries, maxlen=self.config["history_size"])
                    
                    # Load method performance data        
                    self.method_performance = data.get("method_performance", {})
        except Exception as e:
            print(f"Could not load confidence history: {e}")
    
    def save_history(self) -> None:
        """Save confidence history to file"""
        history_file = self.config.get("history_file", "data/confidence_history.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            # Convert history to serializable format
            serializable_history = {
                score_name: list(entries) for score_name, entries in self.history.items()
            }
            
            # Prepare data to save
            data_to_save = {
                "history": serializable_history,
                "method_performance": self.method_performance,
                "last_updated": datetime.now().isoformat()
            }
            
            # Save to file
            with open(history_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
        except Exception as e:
            print(f"Could not save confidence history: {e}")
    
    def calculate_confidence(self, 
                            score: Score, 
                            component_results: List[ScoringResult]) -> float:
        """
        Calculate confidence score based on component scores and history
        
        Args:
            score: The predicted score
            component_results: List of component scoring results
            
        Returns:
            float: Confidence score (0-1)
        """
        if not component_results:
            return 0.0
            
        # Extract method confidences and apply method weights
        method_confidences = {}
        for result in component_results:
            method = result.method
            if method not in method_confidences:
                method_confidences[method] = []
            
            # Only include results that match our predicted score
            if result.score == score:
                method_confidences[method].append(result.confidence)
        
        # If no results support our predicted score, confidence is low
        if not any(method_confidences.values()):
            return 0.0
            
        # Calculate weighted confidence from supporting methods
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for method, confidences in method_confidences.items():
            if not confidences:
                continue
                
            # Average confidence for this method
            avg_confidence = sum(confidences) / len(confidences)
            
            # Get method weight (default to 0.1 if not specified)
            method_weight = self.config.get("method_weights", {}).get(method, 0.1)
            
            weighted_confidence += avg_confidence * method_weight
            total_weight += method_weight
        
        # Normalize by total weight
        if total_weight > 0:
            confidence = weighted_confidence / total_weight
        else:
            confidence = 0.0
            
        # Apply historical adjustment
        if self.history and score.name in self.history and self.history[score.name]:
            # Calculate historical accuracy for this score category
            historical_accuracy = self._calculate_historical_accuracy(score)
            
            # Blend current confidence with historical accuracy
            confidence = 0.8 * confidence + 0.2 * historical_accuracy
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _calculate_historical_accuracy(self, score: Score) -> float:
        """Calculate historical accuracy for a score category"""
        if score.name not in self.history or not self.history[score.name]:
            return 0.5  # Default if no history
            
        # Get historical entries
        entries = self.history[score.name]
        
        # Count correct predictions (those with manual_correct = True)
        correct = sum(1 for entry in entries if entry.get("manual_correct", False))
        
        # Calculate accuracy
        return correct / len(entries) if entries else 0.5
    
    def update_with_expert_feedback(self, 
                                   result: ScoringResult, 
                                   correct: bool) -> None:
        """
        Update history with expert feedback for a prediction
        
        Args:
            result: The original scoring result
            correct: Whether the prediction was correct (expert judgment)
        """
        if not result.score or result.score == Score.NOT_RATED:
            return
            
        # Create history entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "confidence": result.confidence,
            "method": result.method,
            "manual_correct": correct
        }
        
        # Add to history
        self.history[result.score.name].append(entry)
        
        # Update method performance
        method = result.method
        if method not in self.method_performance:
            self.method_performance[method] = {"correct": 0, "total": 0}
            
        self.method_performance[method]["total"] += 1
        if correct:
            self.method_performance[method]["correct"] += 1
            
        # Save history periodically
        if sum(len(entries) for entries in self.history.values()) % 10 == 0:
            self.save_history()
    
    def should_request_review(self, result: ScoringResult) -> bool:
        """
        Determine if a prediction should be flagged for expert review
        
        Args:
            result: The scoring result to evaluate
            
        Returns:
            bool: True if the prediction should be reviewed
        """
        # Always review NOT_RATED results
        if result.score == Score.NOT_RATED:
            return True
            
        # Check confidence threshold
        if result.confidence < self.config.get("confidence_threshold", 0.7):
            return True
            
        # Check historical accuracy for this score category
        if result.score.name in self.history and self.history[result.score.name]:
            historical_accuracy = self._calculate_historical_accuracy(result.score)
            
            # If historical accuracy is low for this category, request review
            if historical_accuracy < 0.75:  # 75% accuracy threshold
                return True
        
        # If method has poor historical performance, request review
        method = result.method
        if method in self.method_performance and self.method_performance[method]["total"] > 10:
            method_accuracy = (self.method_performance[method]["correct"] / 
                              self.method_performance[method]["total"])
                              
            if method_accuracy < 0.7:  # 70% accuracy threshold for method
                return True
        
        return False
    
    def get_method_performance(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get performance metrics for each scoring method
        
        Returns:
            Dict: Method performance statistics
        """
        result = {}
        
        for method, stats in self.method_performance.items():
            total = stats["total"]
            correct = stats["correct"]
            
            # Calculate accuracy if we have data
            accuracy = correct / total if total > 0 else 0
            
            result[method] = {
                "total": total,
                "correct": correct,
                "accuracy": accuracy
            }
            
        return result 