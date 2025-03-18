"""
Component Specialization Framework

This module provides interfaces and utilities for specialized scoring components
that can adapt to different domains, age groups, and response types.
"""

from typing import Dict, Any, Optional, List, Tuple, Protocol, Set, Union
import logging
from abc import ABC, abstractmethod
from enum import Enum

from .base import Score, ScoringResult, BaseScorer

logger = logging.getLogger(__name__)

class SpecializationDomain(str, Enum):
    """Enumeration of domains for scorer specialization"""
    MOTOR = "MOTOR"
    COMMUNICATION = "COMMUNICATION" 
    SOCIAL = "SOCIAL"
    COGNITIVE = "COGNITIVE"
    GENERAL = "GENERAL"  # For components that work across domains

class SpecializationAgeGroup(str, Enum):
    """Enumeration of age groups for scorer specialization"""
    INFANT = "INFANT"  # 0-12 months
    TODDLER = "TODDLER"  # 13-36 months
    PRESCHOOL = "PRESCHOOL"  # 37-60 months
    ALL_AGES = "ALL_AGES"  # For components that work across all ages

class SpecializationFeature(str, Enum):
    """Enumeration of response features that components may specialize in"""
    SHORT_RESPONSE = "SHORT_RESPONSE"  # Brief, simple responses
    DETAILED_RESPONSE = "DETAILED_RESPONSE"  # Longer, more detailed responses
    AMBIGUOUS_RESPONSE = "AMBIGUOUS_RESPONSE"  # Responses with unclear indicators
    NEGATIVE_INDICATORS = "NEGATIVE_INDICATORS"  # Responses with negative language
    MULTILINGUAL = "MULTILINGUAL"  # Non-English or mixed-language responses
    EMERGING_BEHAVIORS = "EMERGING_BEHAVIORS"  # Behaviors that are just beginning
    ENVIRONMENTAL_CONTEXT = "ENVIRONMENTAL_CONTEXT"  # Responses with environmental factors
    DEVELOPMENTAL_HISTORY = "DEVELOPMENTAL_HISTORY"  # Responses with developmental history
    GENERAL = "GENERAL"  # For components that work across response types

class SpecializedScorer(BaseScorer, ABC):
    """Abstract base class for scorers that have specific domains of expertise"""
    
    @property
    def specialization_domains(self) -> Set[SpecializationDomain]:
        """Get domains this component specializes in
        
        Returns:
            Set of domains where this component performs well
        """
        return {SpecializationDomain.GENERAL}
    
    @property
    def specialization_age_groups(self) -> Set[SpecializationAgeGroup]:
        """Get age groups this component specializes in
        
        Returns:
            Set of age groups where this component performs well
        """
        return {SpecializationAgeGroup.ALL_AGES}
    
    @property
    def specialization_features(self) -> Set[SpecializationFeature]:
        """Get response features this component specializes in
        
        Returns:
            Set of response features where this component performs well
        """
        return {SpecializationFeature.GENERAL}
    
    def get_specialization_score(self, domain: str, age_months: int, 
                                response_length: int) -> float:
        """Calculate how well this component is specialized for the context
        
        Args:
            domain: The developmental domain (e.g., "MOTOR")
            age_months: Child's age in months
            response_length: Length of response text in characters
            
        Returns:
            Specialization score between 0.0 and 2.0 where:
            - 0.0-0.5: Not specialized for this context
            - 0.5-1.0: Somewhat specialized
            - 1.0-1.5: Well specialized
            - 1.5-2.0: Highly specialized
        """
        # Default implementation provides a basic specialization score
        # Subclasses should override with more sophisticated implementations
        
        # Start with a neutral score
        spec_score = 1.0
        
        # Check domain specialization
        domain_spec = SpecializationDomain.GENERAL
        try:
            domain_spec = SpecializationDomain(domain.upper())
        except ValueError:
            pass
            
        if domain_spec in self.specialization_domains:
            spec_score += 0.3
        elif SpecializationDomain.GENERAL not in self.specialization_domains:
            spec_score -= 0.2
            
        # Check age specialization
        age_spec = self._map_age_to_group(age_months)
        if age_spec in self.specialization_age_groups:
            spec_score += 0.3
        elif SpecializationAgeGroup.ALL_AGES not in self.specialization_age_groups:
            spec_score -= 0.2
            
        # Check response type specialization (using length as a simple proxy)
        if response_length < 100 and SpecializationFeature.SHORT_RESPONSE in self.specialization_features:
            spec_score += 0.2
        elif response_length >= 100 and SpecializationFeature.DETAILED_RESPONSE in self.specialization_features:
            spec_score += 0.2
            
        # Constrain to valid range
        return max(0.1, min(2.0, spec_score))
    
    def _map_age_to_group(self, age_months: int) -> SpecializationAgeGroup:
        """Map age in months to an age group
        
        Args:
            age_months: Age in months
            
        Returns:
            Corresponding SpecializationAgeGroup
        """
        if age_months <= 12:
            return SpecializationAgeGroup.INFANT
        elif age_months <= 36:
            return SpecializationAgeGroup.TODDLER
        else:
            return SpecializationAgeGroup.PRESCHOOL

class KeywordSpecializedScorer(SpecializedScorer):
    """Example implementation of specialized keyword scorer"""
    
    @property
    def specialization_domains(self) -> Set[SpecializationDomain]:
        return {SpecializationDomain.GENERAL, SpecializationDomain.COMMUNICATION}
    
    @property
    def specialization_age_groups(self) -> Set[SpecializationAgeGroup]:
        return {SpecializationAgeGroup.ALL_AGES}
    
    @property
    def specialization_features(self) -> Set[SpecializationFeature]:
        return {SpecializationFeature.SHORT_RESPONSE, SpecializationFeature.NEGATIVE_INDICATORS}
        
    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score a response using keyword-based analysis"""
        # Basic implementation - should be overridden by concrete implementations
        response_lower = response.lower()
        
        # Simple keyword matching for demonstration
        keywords = {
            Score.CANNOT_DO: ["cannot", "not able", "doesn't", "does not", "unable", "no"],
            Score.LOST_SKILL: ["used to", "previously", "before", "lost", "regressed", "stopped"],
            Score.EMERGING: ["sometimes", "occasionally", "beginning", "starting", "trying"],
            Score.WITH_SUPPORT: ["with help", "assistance", "supported", "when I", "guidance"],
            Score.INDEPENDENT: ["yes", "always", "consistently", "independently", "by himself", "by herself"]
        }
        
        # Count matches for each category
        matches = {}
        best_score = Score.NOT_RATED
        best_count = 0
        
        for score, words in keywords.items():
            count = sum(1 for word in words if word in response_lower)
            matches[score] = count
            
            if count > best_count:
                best_count = count
                best_score = score
        
        # Calculate confidence based on match count
        confidence = min(0.5 + (best_count * 0.1), 0.9) if best_count > 0 else 0.0
        
        # If no matches or low confidence, return NOT_RATED
        if best_count == 0 or confidence < 0.5:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=confidence,
                method="keyword_specialized",
                reasoning="Insufficient keyword matches for confident scoring"
            )
        
        return ScoringResult(
            score=best_score,
            confidence=confidence,
            method="keyword_specialized",
            reasoning=f"Matched {best_count} keywords for {best_score.name}",
            details={"matches": matches}
        )

class EmbeddingSpecializedScorer(SpecializedScorer):
    """Example implementation of specialized embedding scorer"""
    
    @property
    def specialization_domains(self) -> Set[SpecializationDomain]:
        return {SpecializationDomain.GENERAL, SpecializationDomain.SOCIAL}
    
    @property
    def specialization_age_groups(self) -> Set[SpecializationAgeGroup]:
        return {SpecializationAgeGroup.ALL_AGES}
    
    @property
    def specialization_features(self) -> Set[SpecializationFeature]:
        return {SpecializationFeature.DETAILED_RESPONSE, SpecializationFeature.AMBIGUOUS_RESPONSE}
        
    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score a response using embedding-based semantic analysis"""
        # Basic implementation - should be overridden by concrete implementations
        
        # Example embeddings for each category (simplified for demonstration)
        category_similarities = {
            Score.CANNOT_DO: 0.3,
            Score.LOST_SKILL: 0.2,
            Score.EMERGING: 0.4,
            Score.WITH_SUPPORT: 0.5,
            Score.INDEPENDENT: 0.6
        }
        
        # Find the best matching category
        best_score = max(category_similarities.items(), key=lambda x: x[1])
        score_category, confidence = best_score
        
        # If confidence is too low, return NOT_RATED
        if confidence < 0.5:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=confidence,
                method="embedding_specialized",
                reasoning=f"Best match ({score_category.name}) below similarity threshold",
                details={"similarities": category_similarities}
            )
        
        return ScoringResult(
            score=score_category,
            confidence=confidence,
            method="embedding_specialized",
            reasoning=f"Best semantic match: {score_category.name} with {confidence:.2f} confidence",
            details={"similarities": category_similarities}
        )

class TransformerSpecializedScorer(SpecializedScorer):
    """Example implementation of specialized transformer scorer"""
    
    @property
    def specialization_domains(self) -> Set[SpecializationDomain]:
        return {SpecializationDomain.GENERAL, SpecializationDomain.COGNITIVE}
    
    @property
    def specialization_age_groups(self) -> Set[SpecializationAgeGroup]:
        return {SpecializationAgeGroup.TODDLER, SpecializationAgeGroup.PRESCHOOL}
    
    @property
    def specialization_features(self) -> Set[SpecializationFeature]:
        return {SpecializationFeature.DETAILED_RESPONSE, SpecializationFeature.DEVELOPMENTAL_HISTORY}
        
    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score a response using transformer-based classification"""
        # Basic implementation - should be overridden by concrete implementations
        
        # Simulate transformer classification scores
        classification_scores = {
            "can do this skill independently": 0.7,
            "emerging or developing this skill": 0.2,
            "can do this skill with support or assistance": 0.05,
            "lost this skill after previously having it": 0.03,
            "cannot do this skill": 0.02
        }
        
        # Map classification labels to Score enum
        label_to_score = {
            "can do this skill independently": Score.INDEPENDENT,
            "emerging or developing this skill": Score.EMERGING,
            "can do this skill with support or assistance": Score.WITH_SUPPORT,
            "lost this skill after previously having it": Score.LOST_SKILL,
            "cannot do this skill": Score.CANNOT_DO
        }
        
        # Find the highest scoring label
        best_label = max(classification_scores.items(), key=lambda x: x[1])
        label, confidence = best_label
        score_category = label_to_score[label]
        
        # If confidence is too low, return NOT_RATED
        if confidence < 0.5:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=confidence,
                method="transformer_specialized",
                reasoning=f"Low confidence ({confidence:.2f}) for {score_category.name}",
                details={
                    "labels": list(classification_scores.keys()),
                    "scores": list(classification_scores.values())
                }
            )
        
        return ScoringResult(
            score=score_category,
            confidence=confidence,
            method="transformer_specialized",
            reasoning=f"Classified as {score_category.name} with {confidence:.2f} confidence",
            details={
                "labels": list(classification_scores.keys()),
                "scores": list(classification_scores.values())
            }
        )

class LLMSpecializedScorer(SpecializedScorer):
    """Example implementation of specialized LLM scorer"""
    
    @property
    def specialization_domains(self) -> Set[SpecializationDomain]:
        return {SpecializationDomain.GENERAL}
    
    @property
    def specialization_age_groups(self) -> Set[SpecializationAgeGroup]:
        return {SpecializationAgeGroup.ALL_AGES}
    
    @property
    def specialization_features(self) -> Set[SpecializationFeature]:
        return {
            SpecializationFeature.AMBIGUOUS_RESPONSE, 
            SpecializationFeature.MULTILINGUAL,
            SpecializationFeature.ENVIRONMENTAL_CONTEXT
        }
    
    def get_specialization_score(self, domain: str, age_months: int, 
                            response_length: int) -> float:
        """Calculate how specialized this scorer is for the given context"""
        # LLM scorers are especially good with complex, ambiguous, or multilingual responses
        base_score = 0.7  # Higher base score for LLM
        
        # Adjust based on response length (LLMs handle longer responses better)
        if response_length > 100:
            base_score += 0.2
        
        return min(base_score, 1.0)
        
    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """Score a response using LLM-based analysis"""
        # Basic implementation - should be overridden by concrete implementations
        
        # Simulate LLM scoring
        # In a real implementation, this would call an LLM API
        
        # For demonstration, we'll return a simulated result
        # Normally this would be based on the LLM's output
        
        # Simulate different scores based on response content
        response_lower = response.lower()
        
        if "not" in response_lower or "cannot" in response_lower:
            score_category = Score.CANNOT_DO
            confidence = 0.85
            reasoning = "Response indicates child cannot perform the skill"
        elif "used to" in response_lower or "stopped" in response_lower:
            score_category = Score.LOST_SKILL
            confidence = 0.82
            reasoning = "Response indicates child has lost a previously acquired skill"
        elif "sometimes" in response_lower or "trying" in response_lower:
            score_category = Score.EMERGING
            confidence = 0.78
            reasoning = "Response indicates skill is emerging but inconsistent"
        elif "help" in response_lower or "support" in response_lower:
            score_category = Score.WITH_SUPPORT
            confidence = 0.88
            reasoning = "Response indicates child can perform with assistance"
        elif "yes" in response_lower or "always" in response_lower:
            score_category = Score.INDEPENDENT
            confidence = 0.92
            reasoning = "Response indicates child can perform independently"
        else:
            # If no clear indicators, return a moderate confidence EMERGING score
            score_category = Score.EMERGING
            confidence = 0.65
            reasoning = "No clear indicators found, defaulting to EMERGING based on context"
        
        return ScoringResult(
            score=score_category,
            confidence=confidence,
            method="llm_specialized",
            reasoning=reasoning,
            details={
                "prompt_tokens": len(response.split()),
                "response_tokens": len(reasoning.split()),
                "model": "simulated-llm"
            }
        )

def analyze_response_features(response: str) -> Set[SpecializationFeature]:
    """Analyze a response to detect specialized features
    
    Args:
        response: The response text
        
    Returns:
        Set of detected features
    """
    features = set()
    
    # Basic length-based classification
    if len(response) < 100:
        features.add(SpecializationFeature.SHORT_RESPONSE)
    else:
        features.add(SpecializationFeature.DETAILED_RESPONSE)
    
    # Detect negation patterns
    negation_words = ["not", "don't", "doesn't", "can't", "cannot", "never", "no"]
    if any(word in response.lower() for word in negation_words):
        features.add(SpecializationFeature.NEGATIVE_INDICATORS)
    
    # Detect ambiguity
    ambiguity_phrases = ["sometimes", "occasionally", "not sure", "maybe", "might", "could be"]
    if any(phrase in response.lower() for phrase in ambiguity_phrases):
        features.add(SpecializationFeature.AMBIGUOUS_RESPONSE)
    
    # Detect developmental history
    history_phrases = ["used to", "before", "previously", "started", "began", "history", "developed"]
    if any(phrase in response.lower() for phrase in history_phrases):
        features.add(SpecializationFeature.DEVELOPMENTAL_HISTORY)
    
    # Detect environmental context
    context_phrases = ["at home", "at school", "with others", "when we", "in the", "environment"]
    if any(phrase in response.lower() for phrase in context_phrases):
        features.add(SpecializationFeature.ENVIRONMENTAL_CONTEXT)
    
    # Detect emerging behaviors
    emerging_phrases = ["starting to", "beginning to", "trying to", "attempts", "learning"]
    if any(phrase in response.lower() for phrase in emerging_phrases):
        features.add(SpecializationFeature.EMERGING_BEHAVIORS)
    
    return features

def specialize_ensemble_weights(
    scorers: List[SpecializedScorer],
    domain: str,
    age_months: int,
    response: str
) -> Dict[str, float]:
    """Calculate specialized weights for ensemble scorers
    
    Args:
        scorers: List of specialized scorers
        domain: Developmental domain
        age_months: Age in months
        response: Response text
        
    Returns:
        Dictionary mapping scorer method names to weights
    """
    # Get specialization scores for each scorer
    specialization_scores = {}
    response_length = len(response)
    
    for scorer in scorers:
        method = scorer.__class__.__name__.lower().replace("scorer", "")
        spec_score = scorer.get_specialization_score(domain, age_months, response_length)
        specialization_scores[method] = spec_score
    
    # Normalize weights
    total_score = sum(specialization_scores.values())
    if total_score == 0:
        # Fallback to equal weights
        return {method: 1.0/len(scorers) for method in specialization_scores.keys()}
    
    normalized_weights = {method: score/total_score for method, score in specialization_scores.items()}
    
    return normalized_weights 