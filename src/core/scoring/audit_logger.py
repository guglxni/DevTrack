"""
Audit Logging Module

This module implements audit logging for scoring decisions to enable
transparency, debugging, and continuous improvement.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging

from .base import Score, ScoringResult


class AuditLogger:
    """
    Logs scoring decisions for audit, analysis, and system improvement
    
    This implementation:
    1. Creates structured logs of all scoring decisions
    2. Provides searchable audit trail for debugging
    3. Enables analysis of system performance
    4. Supports continuous improvement through data collection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the audit logger"""
        self.config = config or self._default_config()
        
        # Set up logging
        self._setup_logging()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
            "log_directory": "logs/scoring",         # Directory for log files
            "log_level": "INFO",                     # Default log level
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "enable_json_logging": True,             # Enable structured JSON logs
            "json_log_file": "logs/scoring/scoring_audit.jsonl",
            "max_log_files": 10,                     # Maximum number of log files to keep
            "max_log_size_mb": 10,                   # Maximum size of each log file in MB
        }
    
    def _setup_logging(self) -> None:
        """Set up logging configuration"""
        # Create log directory if it doesn't exist
        log_dir = self.config.get("log_directory", "logs/scoring")
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up structured JSON logging
        if self.config.get("enable_json_logging", True):
            json_log_file = self.config.get("json_log_file", "logs/scoring/scoring_audit.jsonl")
            os.makedirs(os.path.dirname(json_log_file), exist_ok=True)
        
        # Set up Python logging
        log_level_name = self.config.get("log_level", "INFO")
        log_level = getattr(logging, log_level_name, logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger("scoring_audit")
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add file handler
        log_format = self.config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter(log_format)
        
        log_file = os.path.join(log_dir, "scoring.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler for debugging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
        self.logger.addHandler(console_handler)
    
    def _log_to_json(self, data: Dict[str, Any]) -> None:
        """Log structured data to JSON file"""
        if not self.config.get("enable_json_logging", True):
            return
            
        json_log_file = self.config.get("json_log_file", "logs/scoring/scoring_audit.jsonl")
        
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
        
        # Convert Score enum to string to avoid serialization issues
        serializable_data = self._make_serializable(data)
        
        try:
            with open(json_log_file, "a") as f:
                f.write(json.dumps(serializable_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to JSON log: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert Score enum values to strings in any data structure"""
        if isinstance(obj, Score):
            return obj.name
        elif isinstance(obj, dict):
            # Handle Score enum keys by converting them to strings
            return {
                (k.name if isinstance(k, Score) else k): self._make_serializable(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return self._make_serializable(obj.to_dict())
        else:
            return obj
    
    def record_scoring(self, 
                      response: str, 
                      milestone: Optional[Dict[str, Any]],
                      score: Union[Score, ScoringResult],
                      confidence: Optional[float] = None,
                      needs_review: bool = False,
                      expert_feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a scoring decision for audit
        
        Args:
            response: The text that was scored
            milestone: The milestone that was scored
            score: The score result
            confidence: Optional explicit confidence score
            needs_review: Whether this needs expert review
            expert_feedback: Optional feedback from an expert
        """
        # Extract the Score enum and confidence depending on the type
        if isinstance(score, ScoringResult):
            score_value = score.score
            confidence = score.confidence
            method = score.method
            reasoning = score.reasoning
            details = score.details
        else:
            score_value = score
            method = "unknown"
            reasoning = None
            details = None
        
        # Create structured log entry
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "milestone": milestone,
            "score": score_value.name if score_value else "UNKNOWN",
            "score_value": score_value.value if score_value else -1,
            "confidence": confidence,
            "method": method,
            "reasoning": reasoning,
            "needs_review": needs_review,
            "expert_feedback": expert_feedback
        }
        
        # Add details if available
        if details:
            log_data["details"] = details
        
        # Log to JSON file
        self._log_to_json(log_data)
        
        # Log to standard logger
        confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        log_message = f"Scored: {score_value.name if score_value else 'UNKNOWN'} (confidence: {confidence_str})"
        
        if needs_review:
            log_message += " - Needs review"
        
        if expert_feedback:
            log_message += f" - Expert feedback: {expert_feedback.get('score', 'N/A')}"
        
        self.logger.info(log_message)
    
    def record_error(self, 
                    error_message: str, 
                    context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error that occurred during scoring
        
        Args:
            error_message: The error message
            context: Additional context about the error
        """
        # Create structured log entry
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "context": context or {}
        }
        
        # Log to JSON file
        self._log_to_json(log_data)
        
        # Log to standard logger
        self.logger.error(f"Scoring error: {error_message}")
    
    def search_logs(self, 
                   query: Dict[str, Any], 
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search audit logs for matching entries
        
        Args:
            query: Query parameters to match
            limit: Maximum number of results to return
            
        Returns:
            List[Dict]: Matching log entries
        """
        if not self.config.get("enable_json_logging", True):
            return []
            
        json_log_file = self.config.get("json_log_file", "logs/scoring/scoring_audit.jsonl")
        
        if not os.path.exists(json_log_file):
            return []
            
        results = []
        try:
            with open(json_log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Check if entry matches all query parameters
                        matches = True
                        for k, v in query.items():
                            # Handle nested keys (e.g., "milestone.id")
                            if "." in k:
                                parts = k.split(".")
                                current = entry
                                for part in parts:
                                    if part not in current:
                                        matches = False
                                        break
                                    current = current[part]
                                
                                if matches and current != v:
                                    matches = False
                            elif k not in entry or entry[k] != v:
                                matches = False
                        
                        if matches:
                            results.append(entry)
                            
                            if len(results) >= limit:
                                break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self.logger.error(f"Error searching logs: {e}")
            
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged scoring decisions
        
        Returns:
            Dict: Statistics about logged decisions
        """
        if not self.config.get("enable_json_logging", True):
            return {}
            
        json_log_file = self.config.get("json_log_file", "logs/scoring/scoring_audit.jsonl")
        
        if not os.path.exists(json_log_file):
            return {}
            
        stats = {
            "total_scored": 0,
            "by_score": {},
            "by_method": {},
            "needs_review_count": 0,
            "expert_reviewed_count": 0,
            "average_confidence": 0.0
        }
        
        confidence_sum = 0
        confidence_count = 0
        
        try:
            with open(json_log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Skip error entries
                        if "error" in entry:
                            continue
                            
                        stats["total_scored"] += 1
                        
                        # Track by score
                        score = entry.get("score", "UNKNOWN")
                        if score not in stats["by_score"]:
                            stats["by_score"][score] = 0
                        stats["by_score"][score] += 1
                        
                        # Track by method
                        method = entry.get("method", "unknown")
                        if method not in stats["by_method"]:
                            stats["by_method"][method] = 0
                        stats["by_method"][method] += 1
                        
                        # Track review needs
                        if entry.get("needs_review", False):
                            stats["needs_review_count"] += 1
                            
                        # Track expert feedback
                        if entry.get("expert_feedback") is not None:
                            stats["expert_reviewed_count"] += 1
                            
                        # Track confidence
                        confidence = entry.get("confidence")
                        if confidence is not None:
                            confidence_sum += confidence
                            confidence_count += 1
                            
                    except json.JSONDecodeError:
                        continue
                        
            # Calculate average confidence
            if confidence_count > 0:
                stats["average_confidence"] = confidence_sum / confidence_count
                
            # Add timestamp
            stats["generated_at"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error generating statistics: {e}")
            
        return stats 