"""
Base Scorer Module

This module defines the base classes for all scoring components in the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Union


class Score(Enum):
    """Score categories for developmental milestones"""
    CANNOT_DO = 0
    LOST_SKILL = 1
    EMERGING = 2
    WITH_SUPPORT = 3
    INDEPENDENT = 4
    NOT_RATED = -1
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
        
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
        
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
        
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


@dataclass
class ScoringResult:
    """Structured result of a scoring operation"""
    score: Score
    confidence: float
    method: str
    reasoning: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "score": self.score.value,
            "score_label": self.score.name,
            "confidence": self.confidence,
            "method": self.method,
            "reasoning": self.reasoning,
            "details": self.details
        }


class BaseScorer(ABC):
    """Abstract base class for all scoring components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scorer
        
        Args:
            config: Configuration dictionary for the scorer
        """
        self.config = config or {}
    
    @abstractmethod
    def score(self, 
              response: str, 
              milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a response given optional milestone context
        
        Args:
            response: The text response to score
            milestone_context: Optional context about the milestone
            
        Returns:
            ScoringResult: Structured result with score and confidence
        """
        pass
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Provide default configuration
        
        Returns:
            Dict: Default configuration settings
        """
        return {}


class EnsembleScorer(BaseScorer):
    """Base class for scorers that combine multiple scoring methods"""
    
    def __init__(self, scorers: List[BaseScorer], weights: Optional[List[float]] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize with component scorers
        
        Args:
            scorers: List of component scorers
            weights: Optional weights for each scorer (must match scorers length)
            config: Configuration dictionary
        """
        super().__init__(config)
        self.scorers = scorers
        
        # Validate and normalize weights
        if weights is None:
            # Equal weights if not specified
            weights = [1.0] * len(scorers)
        elif len(weights) != len(scorers):
            raise ValueError("Number of weights must match number of scorers")
        
        # Normalize weights to sum to 1.0
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def score(self, 
              response: str,
              milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score using ensemble of component scorers
        
        Args:
            response: The text response to score
            milestone_context: Optional context about the milestone
            
        Returns:
            ScoringResult: Ensemble scoring result
        """
        # Get results from all component scorers
        results = []
        for scorer, weight in zip(self.scorers, self.weights):
            result = scorer.score(response, milestone_context)
            results.append((result, weight))
        
        return self._combine_results(results)
    
    def _combine_results(self, 
                         weighted_results: List[tuple[ScoringResult, float]]) -> ScoringResult:
        """
        Combine multiple weighted results
        
        Args:
            weighted_results: List of (result, weight) tuples
            
        Returns:
            ScoringResult: Combined result
        """
        # Filter valid results (skip NOT_RATED)
        valid_results = [(r, w) for r, w in weighted_results 
                          if r.score != Score.NOT_RATED]
        
        if not valid_results:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="ensemble",
                reasoning="No valid scores available",
                details={"component_methods": [r.method for r, _ in weighted_results]}
            )
        
        # Calculate weighted scores for each category
        category_scores = {s: 0.0 for s in Score if s != Score.NOT_RATED}
        for result, weight in valid_results:
            category_scores[result.score] += result.confidence * weight
        
        # Select best score and confidence
        best_score = max(category_scores.items(), key=lambda x: x[1])
        
        # Calculate overall confidence based on agreement and individual confidences
        confidence = best_score[1]
        
        return ScoringResult(
            score=best_score[0],
            confidence=confidence,
            method="ensemble",
            reasoning="Combined multiple scoring methods",
            details={
                "component_results": [r.to_dict() for r, _ in weighted_results],
                "component_weights": [w for _, w in weighted_results],
                "category_scores": {k.name: v for k, v in category_scores.items()}
            }
        ) 