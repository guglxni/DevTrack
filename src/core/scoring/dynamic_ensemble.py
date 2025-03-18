"""
Dynamic Ensemble Scoring System

This module provides enhanced ensemble scoring with adaptive weighting
based on confidence scores, domain expertise, and age-specific knowledge.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
import re
import time
import traceback

from .base import EnsembleScorer, BaseScorer, ScoringResult, Score

logger = logging.getLogger(__name__)

class DynamicEnsembleScorer(EnsembleScorer):
    """Enhanced ensemble with adaptive weighting based on confidence, domain, and age"""
    
    def __init__(self, scorers: List[BaseScorer], weights: Optional[List[float]] = None, 
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the dynamic ensemble scorer
        
        Args:
            scorers: List of scoring components to use
            weights: Initial base weights for each scorer
            config: Additional configuration parameters
        """
        super().__init__(scorers, weights, config)
        
        # Convert list weights to dictionary for easier access
        if isinstance(self.weights, list):
            weight_dict = {}
            for i, scorer in enumerate(self.scorers):
                # Use the class name as the scorer identifier
                scorer_id = scorer.__class__.__name__.lower().replace('scorer', '')
                weight_dict[scorer_id] = self.weights[i] if i < len(self.weights) else 1.0
            self.weights = weight_dict
        elif self.weights is None:
            # Initialize with default weights
            self.weights = {
                scorer.__class__.__name__.lower().replace('scorer', ''): 1.0 
                for scorer in self.scorers
            }
            
        self.domain_specialization = self._load_domain_specialization()
        self.age_specialization = self._load_age_specialization()
        self.confidence_power = self.config.get("confidence_power", 2.0)
        self.minimum_weight = self.config.get("minimum_weight", 0.1)
        self.track_performance = self.config.get("track_performance", True)
        self.performance_history = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration values"""
        config = super()._default_config()
        config.update({
            "confidence_power": 2.0,  # Power to raise confidence to (amplifies differences)
            "minimum_weight": 0.1,  # Minimum weight to assign any scorer
            "specialization_enabled": True,  # Whether to use domain/age specialization
            "dynamic_weighting_enabled": True,  # Whether to use confidence-based dynamic weighting
            "track_performance": True,  # Whether to track component performance over time
        })
        return config
    
    def _load_domain_specialization(self) -> Dict[str, Dict[str, float]]:
        """Load domain specialization weights for each scorer
        
        Returns:
            Dictionary mapping domain -> {scorer_name: weight_multiplier}
        """
        # Default specialization values
        default_specialization = {
            "MOTOR": {
                "keyword": 1.2,
                "embedding": 1.0,
                "transformer": 0.9,
                "llm": 1.1
            },
            "COMMUNICATION": {
                "keyword": 0.9,
                "embedding": 1.0,
                "transformer": 1.2,
                "llm": 1.3
            },
            "SOCIAL": {
                "keyword": 0.8,
                "embedding": 1.1,
                "transformer": 1.1,
                "llm": 1.3
            },
            "COGNITIVE": {
                "keyword": 0.9,
                "embedding": 1.2,
                "transformer": 1.1,
                "llm": 1.2
            }
        }
        
        # Override with config values if provided
        return self.config.get("domain_specialization", default_specialization)
    
    def _load_age_specialization(self) -> Dict[str, Dict[str, float]]:
        """Load age specialization weights for each scorer
        
        Returns:
            Dictionary mapping age_bracket -> {scorer_name: weight_multiplier}
        """
        # Default age specialization values
        default_specialization = {
            "infant": {  # 0-12 months
                "keyword": 0.8,
                "embedding": 1.1,
                "transformer": 0.9,
                "llm": 1.3
            },
            "toddler": {  # 13-36 months
                "keyword": 1.0,
                "embedding": 1.0,
                "transformer": 1.1,
                "llm": 1.2
            },
            "preschooler": {  # 37-60 months
                "keyword": 1.1,
                "embedding": 1.0,
                "transformer": 1.2,
                "llm": 1.1
            }
        }
        
        # Override with config values if provided
        return self.config.get("age_specialization", default_specialization)
    
    def _get_age_bracket(self, age_months: int) -> str:
        """Determine age bracket from age in months
        
        Args:
            age_months: Age in months
            
        Returns:
            Age bracket string: infant, toddler, or preschooler
        """
        if age_months <= 12:
            return "infant"
        elif age_months <= 36:
            return "toddler"
        else:
            return "preschooler"
    
    def _apply_specialization(self, weight: float, scorer_name: str, 
                             domain: Optional[str] = None, age_months: Optional[int] = None) -> float:
        """Apply domain and age specialization to weights
        
        Args:
            weight: The original weight for the scorer
            scorer_name: Name of the scorer
            domain: Optional domain for specialization
            age_months: Optional age in months for age-based specialization
            
        Returns:
            Adjusted weight
        """
        if not self.config.get("specialization_enabled", True):
            return weight
            
        # Apply domain specialization if available
        if domain and domain in self.domain_specialization:
            domain_spec = self.domain_specialization[domain]
            if scorer_name in domain_spec:
                weight *= domain_spec[scorer_name]
                
        # Apply age specialization if available
        if age_months is not None:
            age_bracket = self._get_age_bracket(age_months)
            if age_bracket in self.age_specialization:
                age_spec = self.age_specialization[age_bracket]
                if scorer_name in age_spec:
                    weight *= age_spec[scorer_name]
                    
        return weight
    
    def _apply_confidence_weighting(self, weighted_results: List[Tuple[ScoringResult, float]]) -> List[Tuple[ScoringResult, float]]:
        """Apply confidence-based weighting adjustments
        
        Args:
            weighted_results: List of (result, weight) tuples
            
        Returns:
            Updated weighted results with confidence-based adjustments
        """
        if not self.config.get("dynamic_weighting_enabled", True):
            return weighted_results
            
        # Adjust weights based on confidence scores
        confidence_adjusted_weights = []
        for result, base_weight in weighted_results:
            # Increase weight for high-confidence predictions
            confidence_factor = result.confidence ** self.confidence_power
            adjusted_weight = base_weight * confidence_factor
            confidence_adjusted_weights.append((result, adjusted_weight))
            
            if confidence_factor != 1.0:
                logger.debug(f"Adjusted {result.method} weight from {base_weight:.2f} to {adjusted_weight:.2f} "
                            f"based on confidence {result.confidence:.2f}")
        
        return confidence_adjusted_weights
    
    def _enforce_minimum_weights(self, weighted_results: List[Tuple[ScoringResult, float]]) -> List[Tuple[ScoringResult, float]]:
        """Ensure all components maintain minimum influence
        
        Args:
            weighted_results: List of (result, weight) tuples
            
        Returns:
            Updated weighted results with minimum weights enforced
        """
        min_weight = self.minimum_weight
        
        # Enforce minimum weight
        min_adjusted_weights = []
        for result, weight in weighted_results:
            enforced_weight = max(weight, min_weight)
            min_adjusted_weights.append((result, enforced_weight))
        
        return min_adjusted_weights
    
    def _normalize_weights(self, weighted_results: List[Tuple[ScoringResult, float]]) -> List[Tuple[ScoringResult, float]]:
        """Normalize weights to sum to 1.0
        
        Args:
            weighted_results: List of (result, weight) tuples
            
        Returns:
            Updated weighted results with normalized weights
        """
        # Normalize weights
        total_weight = sum(w for _, w in weighted_results)
        if total_weight == 0:
            # Fallback to equal weights if total is zero
            equal_weight = 1.0 / len(weighted_results)
            return [(r, equal_weight) for r, _ in weighted_results]
            
        return [(r, w/total_weight) for r, w in weighted_results]
    
    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a response using weighted ensemble of component scorers
        
        Args:
            response: The response text to score
            milestone_context: Optional context about the milestone being scored
            
        Returns:
            ScoringResult with score, confidence, and reasoning
        """
        # Skip empty responses
        if not response or not response.strip():
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="dynamic_ensemble",
                reasoning="Empty response"
            )
            
        # Extract domain and age if provided in context
        domain = None
        age_months = None
        if milestone_context:
            domain = milestone_context.get('domain')
            age_months = milestone_context.get('age_months')
            
        # Get individual scores from component scorers
        weighted_results = []
        for scorer in self.scorers:
            result = scorer.score(response, milestone_context)
            
            # Get initial weight
            weight = self.weights.get(scorer.__class__.__name__.lower().replace('scorer', ''), 1.0)
            
            # Apply domain and age specialization
            weight = self._apply_specialization(weight, scorer.__class__.__name__.lower().replace('scorer', ''), domain, age_months)
            
            # Apply performance-based adjustment if tracking is enabled
            if self.track_performance and self.performance_history:
                weight = self._analyze_component_performance(
                    weight, scorer.__class__.__name__.lower().replace('scorer', ''), domain, age_months
                )
            
            # Store the result and its weight
            weighted_results.append((result, weight))
            
        # Apply confidence-based weighting
        weighted_results = self._apply_confidence_weighting(weighted_results)
        
        # Determine if we should use enhanced ambiguity and conflict detection
        use_enhanced = self.config.get("use_enhanced_ambiguity_detection", False)
        
        # Enhanced mode: First check for person-dependent patterns which are high-priority indicators
        if use_enhanced and response:
            # Quick check for person-dependent phrases
            person_dependent_patterns = [
                r"\bwith\s+(me|mom|dad|mother|father|parent|caregiver|teacher|therapist)\b",
                r"\b(only|just)\s+with\s+(me|mom|dad|mother|father|parent|caregiver|teacher|therapist)\b",
                r"\bdoes\s+.*?\bwith\s+(me|mom|dad|mother|father|parent|caregiver|teacher|therapist)\b.*?\bbut\s+not\b",
                r"\b(better|worse)\s+with\s+(me|mom|dad|mother|father|parent|caregiver|teacher)\b"
            ]
            
            for pattern in person_dependent_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    # Run full support pattern analysis
                    support_analysis = self._detect_support_patterns(response)
                    if support_analysis.get("has_support_indicators", False) and "person_dependent" in support_analysis.get("support_types", {}):
                        # Person-dependent behavior is a strong indicator of WITH_SUPPORT
                        matched_patterns = support_analysis.get("matched_patterns", [])
                        reasoning = f"Person-dependent behavior detected: {', '.join(matched_patterns[:3])}..."
                        
                        return ScoringResult(
                            score=Score.WITH_SUPPORT,
                            confidence=0.78,  # Increased from 0.75
                            method="dynamic_ensemble/person_dependent",
                            reasoning=reasoning
                        )
        
        # Check for ambiguity in the results using the appropriate method
        if use_enhanced:
            # Use enhanced ambiguity detection with response text analysis
            is_ambiguous = self._check_for_ambiguity(weighted_results, response)
        else:
            # Use standard ambiguity detection without response text analysis
            is_ambiguous = self._check_for_ambiguity(weighted_results)
        
        # Apply consistency bonus for scorers that agree
        weighted_results = self._apply_consistency_bonus(weighted_results)
            
        # Enforce minimum weights and normalize
        weighted_results = self._enforce_minimum_weights(weighted_results)
        weighted_results = self._normalize_weights(weighted_results)
        
        # Handle ambiguous cases with special logic
        if is_ambiguous:
            if use_enhanced:
                # Use enhanced ambiguity handling that considers response text
                return self._handle_ambiguous_case(weighted_results, response)
            else:
                # Use standard ambiguity handling without response text analysis
                return self._handle_ambiguous_case(weighted_results)
            
        # Enhanced mode: Check for special patterns in the response
        if use_enhanced and response:
            # Check for support patterns (person-dependent, needs initiation, etc.)
            support_analysis = self._detect_support_patterns(response)
            if support_analysis.get("has_support_indicators", False) and support_analysis.get("support_confidence", 0.0) > 0.7:
                # Strong support indicators should yield WITH_SUPPORT even if not ambiguous
                support_types = list(support_analysis.get("support_types", {}).keys())
                reasoning = f"Detected strong support indicators: {', '.join(support_types)}. " + \
                           f"Support patterns: {', '.join(support_analysis.get('matched_patterns', []))[:100]}..."
                
                return ScoringResult(
                    score=Score.WITH_SUPPORT,
                    confidence=min(0.92, support_analysis.get("support_confidence", 0.75) + 0.05),  # Increased confidence
                    method="dynamic_ensemble/support_detected",
                    reasoning=reasoning
                )
                
            # Check for complex linguistic patterns that might affect scoring
            complex_analysis = self._analyze_complex_statements(response)
            if complex_analysis.get("has_complex_patterns", False):
                # Get the combined result first
                combined = self._combine_results(weighted_results)
                
                # Apply confidence adjustments based on complex patterns
                total_adjustment = 0
                adjustment_reasons = []
                
                for pattern_type, adjustment_info in complex_analysis.get("interpretation_adjustments", {}).items():
                    if adjustment_info.get("confidence_adjustment", 0) != 0:
                        total_adjustment += adjustment_info.get("confidence_adjustment", 0)
                        adjustment_reasons.append(adjustment_info.get("note", ""))
                
                if total_adjustment != 0:
                    # Increase minimum confidence threshold from 0.5 to 0.55
                    combined.confidence = max(0.55, min(0.97, combined.confidence + total_adjustment))
                    combined.reasoning = f"Complex linguistic patterns detected: {'; '.join(adjustment_reasons)}. " + combined.reasoning
                    combined.method = "dynamic_ensemble/complex_patterns"
                    return combined
            
            # Check for conflicting information directly in the response
            contradiction_analysis = self._detect_conflicting_information(response)
            if contradiction_analysis.get("has_conflicts", False):
                # If we detected conflicting information but our scorers didn't flag it as ambiguous,
                # we'll still use the normal weighted combination but with reduced confidence
                combined = self._combine_results(weighted_results)
                # Reduced confidence penalty from 0.85 to 0.9
                combined.confidence *= 0.9  
                combined.reasoning = f"Contradictory information detected in response ({', '.join(contradiction_analysis.get('matched_patterns', []))}). " + combined.reasoning
                combined.method = "dynamic_ensemble/contradictory"
                return combined
        
        # Combine the weighted results
        return self._combine_results(weighted_results)
    
    def _check_for_ambiguity(self, weighted_results: List[Tuple[ScoringResult, float]], response: str = "") -> bool:
        """Check if the scoring results are ambiguous
        
        Args:
            weighted_results: List of (result, weight) tuples
            response: The original response text (optional)
            
        Returns:
            True if the results are ambiguous
        """
        # First, check for text-based ambiguity if response text is provided
        if response:
            # Patterns that indicate ambiguity
            ambiguity_indicators = [
                r"\bit\s+depends\b",
                r"\bsometimes.*?\bother\s+times\b",
                r"\bnot\s+always\b",
                r"\bvaries\b",
                r"\binconsistent\b",
                r"\bon\s+some\s+days\b",
                r"\bon\s+and\s+off\b",
                r"\bwith\s+certain.*?but\s+not\b",
                r"\bbut\s+at\b",
                r"\bbut\s+when\b",
                r"\bno[\s,]+but\s+also\s+yes\b",
                r"\byes[\s,]+but\s+also\s+no\b",
                r"\bcomplicated\b",
                r"\bat\s+home.*?but",
                r"\bwhen.*?but\s+not\s+when\b"
            ]
            
            # Check for these patterns
            for pattern in ambiguity_indicators:
                if re.search(pattern, response, re.IGNORECASE):
                    logger.debug(f"Text-based ambiguity detected: '{pattern}' in '{response[:50]}...'")
                    return True
        
        # Extract scores (excluding NOT_RATED)
        scores = [r.score for r, _ in weighted_results if r.score != Score.NOT_RATED]
        
        # If we have fewer than 2 valid scores, it's not ambiguous based on scorer disagreement
        if len(scores) < 2:
            return False
        
        # Count occurrences of each score
        score_counts = {}
        for score in scores:
            if score not in score_counts:
                score_counts[score] = 0
            score_counts[score] += 1
        
        # If all scorers agree, it's not ambiguous
        if len(score_counts) == 1:
            return False
        
        # Check if there's a clear majority (more than 60% agreement)
        total_scores = len(scores)
        for count in score_counts.values():
            if count / total_scores > 0.6:
                return False
        
        # Check for extreme disagreement (CANNOT_DO vs INDEPENDENT)
        has_cannot_do = Score.CANNOT_DO in score_counts
        has_independent = Score.INDEPENDENT in score_counts
        
        if has_cannot_do and has_independent:
            logger.debug("Ambiguity detected: extreme disagreement between CANNOT_DO and INDEPENDENT")
            return True
            
        # Check confidence spread if we have multiple distinct scores
        if len(score_counts) > 1:
            # Extract confidences
            confidences = [r.confidence for r, _ in weighted_results if r.score != Score.NOT_RATED]
            confidence_spread = max(confidences) - min(confidences)
            
            # If the spread is low, scorers are conflicted but similarly confident,
            # indicating genuine ambiguity
            if confidence_spread < 0.3:
                logger.debug(f"Ambiguity detected: multiple scores with similar confidence (spread={confidence_spread:.2f})")
                return True
        
        # Not ambiguous
        return False
    
    def _apply_consistency_bonus(self, weighted_results: List[Tuple[ScoringResult, float]]) -> List[Tuple[ScoringResult, float]]:
        """
        Apply a bonus to scorers that agree with each other
        
        Args:
            weighted_results: List of (result, weight) tuples
            
        Returns:
            Updated weighted results with consistency bonus applied
        """
        # Count occurrences of each score
        score_counts = {}
        for result, _ in weighted_results:
            if result.score == Score.NOT_RATED:
                continue
            if result.score not in score_counts:
                score_counts[result.score] = 0
            score_counts[result.score] += 1
        
        # Find the most common score
        most_common_score = None
        max_count = 0
        for score, count in score_counts.items():
            if count > max_count:
                max_count = count
                most_common_score = score
        
        # If we have a most common score, apply bonus to scorers that agree
        if most_common_score is not None and max_count > 1:  # Only apply if at least 2 scorers agree
            bonus = 0.2  # Consistency bonus
            adjusted_results = []
            
            for result, weight in weighted_results:
                if result.score == most_common_score:
                    # Apply bonus
                    adjusted_weight = weight * (1 + bonus)
                    adjusted_results.append((result, adjusted_weight))
                    logger.debug(f"Applied consistency bonus to {result.method}: {weight:.2f} -> {adjusted_weight:.2f}")
                else:
                    adjusted_results.append((result, weight))
            
            return adjusted_results
        
        # If no most common score, return unchanged
        return weighted_results
    
    def _handle_ambiguous_case(self, weighted_results: List[Tuple[ScoringResult, float]], response: str = "") -> ScoringResult:
        """
        Handle ambiguous scoring cases with special logic
        
        Args:
            weighted_results: List of (result, weight) tuples
            response: The original response text (optional)
            
        Returns:
            Resolved scoring result
        """
        # Extract valid results (not NOT_RATED)
        valid_results = [(r, w) for r, w in weighted_results if r.score != Score.NOT_RATED]
        
        # If we have no valid results, return NOT_RATED
        if not valid_results:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="dynamic_ensemble/ambiguous",
                reasoning="No valid scores available"
            )
        
        # Enhanced mode: Check for contradictions in the response text
        if response:
            contradiction_analysis = self._detect_conflicting_information(response)
            if contradiction_analysis.get("has_conflicts", False):
                # For responses with contradictory information, EMERGING is often the most appropriate score
                # as it indicates a skill that is inconsistent or in development
                reasoning = f"Detected contradictory information in response: {', '.join(contradiction_analysis.get('matched_patterns', []))}"
                
                # Check if we have an EMERGING score from any component
                emerging_results = [(r, w) for r, w in valid_results if r.score == Score.EMERGING]
                if emerging_results:
                    # Use the EMERGING score with highest confidence
                    best_emerging = max(emerging_results, key=lambda rw: rw[0].confidence)
                    # Increase confidence - previously was 0.95 multiplier
                    confidence = best_emerging[0].confidence * 0.98  
                    
                    return ScoringResult(
                        score=Score.EMERGING,
                        confidence=confidence,
                        method="dynamic_ensemble/contradictory",
                        reasoning=reasoning + ". Selected EMERGING as most appropriate for contradictory information."
                    )
                
                # If no EMERGING score available, provide a moderate confidence score
                # Increase baseline confidence from 0.65 to 0.72
                return ScoringResult(
                    score=Score.EMERGING,
                    confidence=0.72,
                    method="dynamic_ensemble/contradictory",
                    reasoning=reasoning + ". Defaulting to EMERGING as most appropriate for contradictory information."
                )
                
            # Enhanced mode: Check for support patterns (person-dependent, needs initiation, etc.)
            support_analysis = self._detect_support_patterns(response)
            if support_analysis.get("has_support_indicators", False) and support_analysis.get("support_confidence", 0.0) > 0.65:
                support_types = list(support_analysis.get("support_types", {}).keys())
                reasoning = f"Detected strong support indicators: {', '.join(support_types)}. " + \
                           f"Support patterns: {', '.join(support_analysis.get('matched_patterns', []))[:100]}..."
                
                # Check if we have WITH_SUPPORT score from any component
                with_support_results = [(r, w) for r, w in valid_results if r.score == Score.WITH_SUPPORT]
                if with_support_results:
                    # Increase confidence boost from 1.05 to 1.08
                    best_support = max(with_support_results, key=lambda rw: rw[0].confidence)
                    confidence = min(best_support[0].confidence * 1.08, support_analysis.get("support_confidence", 0.7) + 0.05)
                    
                    return ScoringResult(
                        score=Score.WITH_SUPPORT,
                        confidence=confidence,
                        method="dynamic_ensemble/support_detected",
                        reasoning=reasoning + ". Selected WITH_SUPPORT based on detected support indicators."
                    )
                
                # If no WITH_SUPPORT score available, use the calculated support confidence + 0.05
                return ScoringResult(
                    score=Score.WITH_SUPPORT,
                    confidence=min(0.92, support_analysis.get("support_confidence", 0.7) + 0.05),
                    method="dynamic_ensemble/support_detected",
                    reasoning=reasoning + ". Defaulting to WITH_SUPPORT based on detected support indicators."
                )
                
            # Enhanced mode: Check for complex patterns like double negatives
            complex_analysis = self._analyze_complex_statements(response)
            if complex_analysis.get("has_complex_patterns", False):
                # Handle double negatives which often confuse assessment
                if complex_analysis.get("double_negatives", []):
                    reasoning = f"Detected double negatives: {', '.join(complex_analysis.get('double_negatives', []))[:100]}..."
                    
                    # Check context to interpret double negatives correctly
                    negative_context = re.search(r"\b(struggles?|difficult(y|ies)|problems?|challenges?|hard|tough)\b", response, re.IGNORECASE)
                    positive_context = re.search(r"\b(able|capable|manages?|succeeds?|accomplishe?s?)\b", response, re.IGNORECASE)
                    
                    if negative_context and not positive_context:
                        # Double negative in negative context often indicates low ability
                        # Check if we also have support indicators
                        if support_analysis.get("has_support_indicators", False):
                            # Increase confidence from 0.65 to 0.70
                            return ScoringResult(
                                score=Score.WITH_SUPPORT,
                                confidence=0.70,
                                method="dynamic_ensemble/double_negative",
                                reasoning=reasoning + " In negative context with support indicators, interpreted as WITH_SUPPORT."
                            )
                
                # Handle qualified statements that typically indicate EMERGING
                if complex_analysis.get("qualified_statements", []) and len(complex_analysis.get("qualified_statements", [])) > 0:
                    reasoning = f"Detected qualified statements: {', '.join(complex_analysis.get('qualified_statements', []))[:100]}..."
                    confidence_adj = complex_analysis.get("interpretation_adjustments", {}).get("qualified_statements", {}).get("confidence_adjustment", -0.1)
                    # Reduce the negative confidence adjustment by half
                    confidence_adj = confidence_adj / 2
                    
                    # Increase base confidence from 0.7 to 0.75
                    return ScoringResult(
                        score=Score.EMERGING,
                        confidence=0.75 + confidence_adj,
                        method="dynamic_ensemble/qualified_statements",
                        reasoning=reasoning + " Qualified statements typically indicate EMERGING abilities."
                    )
        
        # Get the scores and their confidences
        score_confidences = {}
        for result, _ in valid_results:
            if result.score not in score_confidences:
                score_confidences[result.score] = []
            score_confidences[result.score].append(result.confidence)
        
        # Calculate average confidence for each score
        avg_confidences = {s: sum(confs)/len(confs) for s, confs in score_confidences.items()}
        
        # Enhanced mode: Check for context-dependent phrases in the response
        is_context_dependent = False
        has_mixed_pattern = False
        
        if response:
            # Check for context-dependent phrases
            context_patterns = [
                r"\bat\s+home\b.*?\bbut\b",
                r"\bin\s+familiar\b.*?\bbut\b",
                r"\bwith\s+(mom|dad|parent|caregiver|teacher|therapist)\b.*?\bbut\b",
                r"\bwhen\s+calm\b.*?\bbut\b",
                r"\bdepends\s+on\s+the\s+(day|context|situation|setting|environment)\b",
            ]
            
            for pattern in context_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    is_context_dependent = True
                    break
            
            # Check for mixed pattern phrases
            mixed_patterns = [
                r"\bsometimes\b.*?\bother\s+times\b",
                r"\boccasionally\b",
                r"\binconsistent\b",
                r"\bmixed\b",
                r"\bvariable\b",
                r"\bon\s+and\s+off\b",
                r"\byes\s+and\s+no\b",
                r"\bno\s+but\s+also\s+yes\b",
                r"\b50(\s+|%)percent\b",
                r"\bhalf\s+the\s+time\b"
            ]
            
            for pattern in mixed_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    has_mixed_pattern = True
                    break
        
        # Enhanced mode: If the response explicitly indicates context dependency or mixed patterns,
        # strongly favor EMERGING or WITH_SUPPORT
        if (is_context_dependent or has_mixed_pattern) and response:
            if Score.EMERGING in score_confidences:
                reasoning = "Response indicates variable performance across contexts. Selected EMERGING as the most appropriate score."
                # Reduce confidence penalty from 0.9 to 0.95
                return ScoringResult(
                    score=Score.EMERGING,
                    confidence=avg_confidences[Score.EMERGING] * 0.95,
                    method="dynamic_ensemble/context_dependent",
                    reasoning=reasoning
                )
            elif Score.WITH_SUPPORT in score_confidences:
                reasoning = "Response indicates need for support in some contexts. Selected WITH_SUPPORT as the most appropriate score."
                # Reduce confidence penalty from 0.9 to 0.95
                return ScoringResult(
                    score=Score.WITH_SUPPORT,
                    confidence=avg_confidences[Score.WITH_SUPPORT] * 0.95,
                    method="dynamic_ensemble/context_dependent",
                    reasoning=reasoning
                )
        
        # Check for EMERGING or WITH_SUPPORT among the scores
        conservative_scores = [Score.EMERGING, Score.WITH_SUPPORT]
        has_conservative = any(s in score_confidences for s in conservative_scores)
        
        # If we have conflicting extreme scores (CANNOT_DO vs INDEPENDENT)
        if Score.CANNOT_DO in score_confidences and Score.INDEPENDENT in score_confidences:
            # If we have a conservative option, use it
            if has_conservative:
                # Find the conservative score with highest confidence
                best_conservative = None
                best_conf = 0
                for score in conservative_scores:
                    if score in avg_confidences and avg_confidences[score] > best_conf:
                        best_conf = avg_confidences[score]
                        best_conservative = score
                
                if best_conservative:
                    reasoning = f"Ambiguous case with conflicting extreme scores. Choosing conservative {best_conservative.name}."
                    # Reduce confidence penalty from 0.9 to 0.93
                    return ScoringResult(
                        score=best_conservative,
                        confidence=best_conf * 0.93,
                        method="dynamic_ensemble/ambiguous",
                        reasoning=reasoning
                    )
            
            # Otherwise, default to EMERGING
            reasoning = "Ambiguous case with conflicting extreme scores. Defaulting to EMERGING."
            # Increase confidence from 0.6 to 0.65
            return ScoringResult(
                score=Score.EMERGING,
                confidence=0.65,
                method="dynamic_ensemble/ambiguous",
                reasoning=reasoning
            )
        
        # Enhanced mode: For mixed/context-dependent responses, prefer more moderate scores
        if (is_context_dependent or has_mixed_pattern) and any(score in [Score.CANNOT_DO, Score.INDEPENDENT] for score in score_confidences.keys()) and response:
            # Check if we have a more moderate score with decent confidence
            for moderate_score in [Score.EMERGING, Score.WITH_SUPPORT]:
                if moderate_score in avg_confidences and avg_confidences[moderate_score] > 0.5:
                    score = moderate_score
                    confidence = avg_confidences[moderate_score]
                    # Determine which extreme score was competing
                    extreme_score = Score.CANNOT_DO if Score.CANNOT_DO in score_confidences else Score.INDEPENDENT
                    reasoning = f"Response indicates variable performance. Selected {score.name} despite higher confidence in {extreme_score.name}."
                    # Reduce confidence penalty from 0.9 to 0.93
                    return ScoringResult(
                        score=score,
                        confidence=confidence * 0.93,
                        method="dynamic_ensemble/ambiguous/moderated",
                        reasoning=reasoning
                    )
        
        # For other ambiguous cases, find the score with highest average confidence
        best_score = max(avg_confidences.items(), key=lambda x: x[1])
        score, confidence = best_score
        
        reasoning = f"Ambiguous case. Selected {score.name} with highest average confidence."
        # Reduce confidence penalty from 0.9 to 0.93
        return ScoringResult(
            score=score,
            confidence=confidence * 0.93,
            method="dynamic_ensemble/ambiguous",
            reasoning=reasoning,
            details={"average_confidences": avg_confidences}
        )
    
    def _combine_results(self, weighted_results: List[Tuple[ScoringResult, float]]) -> ScoringResult:
        """Combine results using weighted voting
        
        Args:
            weighted_results: List of (result, weight) tuples
            
        Returns:
            Combined scoring result
        """
        # Extract only valid results (not NOT_RATED)
        valid_results = [(r, w) for r, w in weighted_results if r.score != Score.NOT_RATED]
        
        # If no valid results, return NOT_RATED
        if not valid_results:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="dynamic_ensemble",
                reasoning="No valid scores available"
            )
        
        # Get the total weight for each score
        score_weights = {}
        total_weight = 0
        
        for result, weight in valid_results:
            if result.score not in score_weights:
                score_weights[result.score] = 0
            score_weights[result.score] += weight
            total_weight += weight
        
        # Find the score with highest total weight
        best_score = max(score_weights.items(), key=lambda item: item[1])
        score = best_score[0]
        weight_ratio = best_score[1] / total_weight if total_weight > 0 else 0
        
        # Calculate average confidence for the winning score
        confidence_sum = 0
        result_count = 0
        reasoning_parts = []
        
        for result, weight in valid_results:
            if result.score == score:
                confidence_sum += result.confidence * weight
                result_count += weight
                if weight > 0.2:  # Only include reasoning from significant contributors
                    reasoning_parts.append(f"{result.method}: {result.reasoning}")
        
        # Calculate weighted average confidence
        average_confidence = confidence_sum / result_count if result_count > 0 else 0
        
        # Adjust confidence based on weight ratio (how dominant the winner is)
        adjusted_confidence = average_confidence * (0.5 + 0.5 * weight_ratio)
        
        # Build our final reasoning
        final_reasoning = f"Selected {score.name} with {weight_ratio:.0%} of total vote weight. "
        
        if reasoning_parts:
            final_reasoning += "Key factors: " + "; ".join(reasoning_parts)
        
        return ScoringResult(
            score=score,
            confidence=adjusted_confidence,
            method="dynamic_ensemble",
            reasoning=final_reasoning,
            details={
                "score_weights": {s.name: w for s, w in score_weights.items()},
                "weight_ratio": weight_ratio,
                "average_confidence": average_confidence
            }
        )

    def _analyze_component_performance(self, weight: float, method: str, 
                               domain: Optional[str] = None, age_months: Optional[int] = None) -> float:
        """Analyze component performance and adjust weight accordingly
        
        Args:
            weight: Original weight
            method: Scorer method identifier
            domain: Optional domain for domain-specific adjustment
            age_months: Optional age in months for age-specific adjustment
            
        Returns:
            Adjusted weight based on performance history
        """
        if not self.performance_history:
            return weight
            
        # Start with base adjustment of 1.0
        performance_adjustment = 1.0
        
        # Factor in domain-specific performance if available
        if domain and domain in self.performance_history.get("domain", {}):
            if method in self.performance_history["domain"][domain]:
                domain_perf = self.performance_history["domain"][domain][method]
                # Apply a modest adjustment based on domain performance
                domain_factor = 1.0 + (domain_perf - 0.5) * 0.4  # Max ±20% adjustment
                performance_adjustment *= domain_factor
                logger.debug(f"Domain performance adjustment for {method} in {domain}: {domain_factor:.2f}")
        
        # Factor in age-specific performance if available
        if age_months is not None:
            age_bracket = self._get_age_bracket(age_months)
            if age_bracket in self.performance_history.get("age_group", {}):
                if method in self.performance_history["age_group"][age_bracket]:
                    age_perf = self.performance_history["age_group"][age_bracket][method]
                    # Apply a modest adjustment based on age group performance
                    age_factor = 1.0 + (age_perf - 0.5) * 0.4  # Max ±20% adjustment
                    performance_adjustment *= age_factor
                    logger.debug(f"Age performance adjustment for {method} in {age_bracket}: {age_factor:.2f}")
        
        # Overall performance adjustment
        if method in self.performance_history.get("overall", {}):
            overall_perf = self.performance_history["overall"][method]
            # Apply a small adjustment based on overall performance
            overall_factor = 1.0 + (overall_perf - 0.5) * 0.2  # Max ±10% adjustment
            performance_adjustment *= overall_factor
            logger.debug(f"Overall performance adjustment for {method}: {overall_factor:.2f}")
            
        # Apply performance adjustment to weight
        adjusted_weight = weight * performance_adjustment
        
        if performance_adjustment != 1.0:
            logger.debug(f"Performance-adjusted {method} weight from {weight:.2f} to {adjusted_weight:.2f}")
            
        return adjusted_weight

    def _update_performance_history(self, method: str, domain: Optional[str], 
                              age_months: Optional[int], score: Score, 
                              correct: bool, confidence: float):
        """
        Update performance history with new feedback
        
        Args:
            method: Scoring method (component name)
            domain: Domain of the milestone
            age_months: Age in months
            score: Predicted score
            correct: Whether the prediction was correct
            confidence: Confidence of the prediction
        """
        if not hasattr(self, "performance_history"):
            self.performance_history = {
                "domain": {},
                "age_group": {},
                "overall": {}
            }
            
        # Get update value - weight correct feedback by confidence
        update_value = 1.0 if correct else 0.0
        
        # Update domain-specific performance
        if domain:
            if domain not in self.performance_history["domain"]:
                self.performance_history["domain"][domain] = {}
            
            if method not in self.performance_history["domain"][domain]:
                self.performance_history["domain"][domain][method] = 0.5  # Initial neutral value
                
            # Exponential moving average for smooth updates
            alpha = 0.2  # Learning rate
            current = self.performance_history["domain"][domain][method]
            self.performance_history["domain"][domain][method] = (1 - alpha) * current + alpha * update_value
        
        # Update age-specific performance
        if age_months is not None:
            age_bracket = self._get_age_bracket(age_months)
            if age_bracket not in self.performance_history["age_group"]:
                self.performance_history["age_group"][age_bracket] = {}
            
            if method not in self.performance_history["age_group"][age_bracket]:
                self.performance_history["age_group"][age_bracket][method] = 0.5  # Initial neutral value
                
            # Exponential moving average
            alpha = 0.2  # Learning rate
            current = self.performance_history["age_group"][age_bracket][method]
            self.performance_history["age_group"][age_bracket][method] = (1 - alpha) * current + alpha * update_value
        
        # Update overall performance
        if method not in self.performance_history["overall"]:
            self.performance_history["overall"][method] = 0.5  # Initial neutral value
            
        # Exponential moving average
        alpha = 0.1  # Lower learning rate for overall performance
        current = self.performance_history["overall"][method]
        self.performance_history["overall"][method] = (1 - alpha) * current + alpha * update_value
        
    def update_performance_feedback(self, result: ScoringResult, 
                              correct_score: Score, 
                              milestone_context: Optional[Dict[str, Any]] = None):
        """
        Update performance history with expert feedback
        
        Args:
            result: The scoring result
            correct_score: The correct score according to expert
            milestone_context: Additional context about the milestone
        """
        if not self.config.get("track_performance", True):
            return
            
        # Extract domain and age if available
        domain = milestone_context.get("domain") if milestone_context else None
        age_months = milestone_context.get("age_months") if milestone_context else None
        
        # Update component performances
        for scorer in self.scorers:
            method = scorer.__class__.__name__.lower()
            if method in result.details.get("component_weights", {}):
                correct = result.score == correct_score
                self._update_performance_history(
                    method=method,
                    domain=domain,
                    age_months=age_months,
                    score=result.score,
                    correct=correct,
                    confidence=result.confidence
                ) 

    def _detect_conflicting_information(self, response: str) -> Dict[str, Any]:
        """
        Detect conflicting information in a response
        
        Args:
            response: Response text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Skip if empty response
        if not response:
            return {"has_conflicts": False}
            
        # Look for common contradiction patterns
        contradiction_patterns = [
            r"\b(yes|no),?\s+but",
            r"(can|does)\s+([^,\.]+),?\s+but\s+(cannot|doesn'?t)",
            r"(sometimes|occasionally).+?(always|never)",
            r"(never|not).+?except",
            r"(only|just)\s+when",
            r"(used\s+to).+?(now|but\s+now)",
            r"(at\s+home).+?(school|daycare).+?but",
            r"(with\s+me).+?(not\s+with|but\s+not\s+with)",
            r"half\s+the\s+time|50\s*%|sometimes\s+yes\s+sometimes\s+no"
        ]
        
        # Check if any patterns match
        has_contradictions = False
        matched_patterns = []
        for pattern in contradiction_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                has_contradictions = True
                matched_patterns.append(match.group(0))
                
        # Return simple analysis
        if has_contradictions:
            return {
                "has_conflicts": True,
                "matched_patterns": matched_patterns,
                "recommendation": "Consider EMERGING score for responses with contradictory information"
            }
        
        return {"has_conflicts": False}

    def _detect_support_patterns(self, response: str) -> Dict[str, Any]:
        """
        Detect patterns indicating that support is needed for a behavior
        
        Args:
            response: Response text to analyze
            
        Returns:
            Dictionary with analysis results including support type and confidence
        """
        # Skip if empty response
        if not response:
            return {"has_support_indicators": False}
            
        # Define patterns for different types of support
        support_patterns = {
            "person_dependent": [
                r"\b(only|just)\s+with\s+(me|mom|dad|mother|father|parent|caregiver|teacher|therapist)\b",
                r"\bwith\s+(me|mom|dad|mother|father|parent|caregiver|teacher|therapist).*?\bbut\s+not\b",
                r"\bdepends\s+on\s+who\b",
                r"\bdifferent(ly)?\s+with\s+different\s+people\b",
                r"\b(better|worse)\s+with\s+(me|mom|dad|mother|father|parent|caregiver|teacher)\b",
                # Additional person-dependent patterns
                r"\bwhen\s+(I|we|mom|dad|teacher|therapist)\s+(help|guide|assist|show|demonstrate)\b",
                r"\b(needs|relies\s+on)\s+(me|us|mom|dad|parent|caregiver|teacher|therapist)\b",
                r"\b(won'?t|doesn'?t|can'?t)\s+do\s+.*?\bwithout\s+(me|us|mom|dad|parent|caregiver|teacher|therapist)\b",
                r"\bonly\s+responds\s+to\s+(me|mom|dad|parent|caregiver|teacher|therapist)\b",
                r"\b(prefers|wants)\s+(me|mom|dad|parent|caregiver|teacher|therapist)\s+to\b"
            ],
            "initiation_dependent": [
                r"\bif\s+(we|I|someone|adult)\s+(start|help|assist|show|guide|prompt|begin|initiate)\b",
                r"\b(need|needs|required?)\s+(to\s+be|someone\s+to)\s+(start|help|remind|show|prompt)\b",
                r"\b(after|once|when)\s+(being|getting)\s+(shown|prompted|reminded|guided|helped|assisted)\b",
                r"\bnever\s+(start|begin|initiate)\s+by\s+(him|her|them)self\b",
                r"\bwon'?t\s+(start|begin|do\s+it)\s+on\s+(his|her|their)\s+own\b",
                # Additional initiation-dependent patterns
                r"\b(with|after|needs)\s+(prompting|prompts|cues|reminders|encouragement)\b",
                r"\b(has\s+to|must\s+be)\s+(prompted|reminded|encouraged|cued)\b",
                r"\b(doesn'?t|won'?t)\s+(do|attempt|try)\s+without\s+(prompting|prompts|cues|reminders)\b",
                r"\bneeds\s+to\s+be\s+(told|asked|directed|instructed)\b",
                r"\b(only|just)\s+when\s+(prompted|reminded|encouraged|directed)\b"
            ],
            "environmental_support": [
                r"\bonly\s+in\s+(familiar|certain|specific|structured)\s+(settings?|environments?|places?|situations?)\b",
                r"\bneeds\s+(visual|physical|verbal|gestural)\s+(cues?|supports?|prompts?)\b",
                r"\bwith\s+(visual|physical|verbal|gestural)\s+support\b",
                r"\brequires\s+(adaptations?|modifications?)\b",
                r"\busing\s+(special|adapted|modified)\s+equipment\b",
                # Additional environmental support patterns
                r"\bwhen\s+the\s+environment\s+is\s+(structured|controlled|quiet|calm|familiar)\b",
                r"\bwith\s+the\s+right\s+(setup|arrangement|conditions|environment)\b",
                r"\b(can'?t|doesn'?t|won'?t)\s+in\s+(noisy|busy|unfamiliar|new|chaotic)\s+environments\b"
            ],
            "partial_assistance": [
                r"\bpartial(ly)?\s+(independent|able|capable)\b",
                r"\bsome\s+assistance\b",
                r"\ba\s+little\s+help\b",
                r"\bhand\s+over\s+hand\b",
                r"\bminimal\s+support\b",
                r"\boccasional\s+help\b",
                # Additional partial assistance patterns
                r"\bwith\s+guidance\b",
                r"\bneeds\s+help\s+with\s+parts\b",
                r"\bcan\s+do\s+some\s+but\s+not\s+all\b",
                r"\bpartially\s+able\b"
            ]
        }
        
        # Check for matches
        support_indicators = {}
        strong_support_evidence = False
        matched_patterns = []
        
        for support_type, patterns in support_patterns.items():
            support_indicators[support_type] = []
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    matched_text = match.group(0)
                    support_indicators[support_type].append(matched_text)
                    matched_patterns.append(matched_text)
                    
                    # Some patterns are strong indicators of WITH_SUPPORT
                    if support_type in ["person_dependent", "initiation_dependent"] and not re.search(r"\b(not|don'?t|no|never)\b", matched_text, re.IGNORECASE):
                        strong_support_evidence = True
        
        # Check for negation of independence
        independence_negation_patterns = [
            r"\b(can'?t|cannot|doesn'?t|won'?t|not)\s+(do|perform)\s+by\s+(him|her|them)sel(f|ves)\b",
            r"\b(not|isn'?t|aren'?t)\s+(fully|completely|totally)\s+independent\b",
            r"\b(always|still)\s+needs\s+(help|support|assistance)\b",
            r"\b(unable|can\'?t)\s+to\s+do\s+without\s+(help|assistance|support|guidance)\b",
            # Additional independence negation patterns
            r"\b(requires|needs)\s+assistance\b",
            r"\bwith\s+(help|assistance|support)\b",
            r"\b(can'?t|cannot|doesn'?t|won'?t|not)\s+do\s+on\s+(his|her|their)\s+own\b",
            r"\b(can'?t|cannot|doesn'?t|won'?t|not)\s+do\s+independently\b"
        ]
        
        independence_negated = False
        for pattern in independence_negation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                independence_negated = True
                matched_patterns.append(re.search(pattern, response, re.IGNORECASE).group(0))
        
        # Check for specific person-dependent phrases that might not be caught by the patterns
        person_dependent_keywords = [
            "with me", "with mom", "with dad", "with parent", "with caregiver", "with teacher", 
            "when I", "when we", "when mom", "when dad", "when teacher"
        ]
        
        for keyword in person_dependent_keywords:
            if keyword.lower() in response.lower() and not support_indicators.get("person_dependent"):
                support_indicators["person_dependent"] = [f"Keyword match: {keyword}"]
                matched_patterns.append(f"Keyword match: {keyword}")
                
        # Determine if we have support indicators
        has_support_indicators = any(indicators for indicators in support_indicators.values()) or independence_negated
        
        # Calculate confidence in WITH_SUPPORT assessment based on evidence
        support_confidence = 0.0
        if has_support_indicators:
            # Base confidence
            support_confidence = 0.6
            
            # Adjust based on number and types of indicators
            total_indicators = sum(len(indicators) for indicators in support_indicators.values())
            if total_indicators > 1:
                # Multiple indicators increase confidence
                support_confidence += min(0.1 * total_indicators, 0.2)
            
            # Strong evidence provides higher confidence
            if strong_support_evidence:
                support_confidence += 0.1
                
            # Multiple types of support provide higher confidence
            types_with_evidence = sum(1 for indicators in support_indicators.values() if indicators)
            if types_with_evidence > 1:
                support_confidence += 0.05 * types_with_evidence
                
            # Independence negation is a strong indicator
            if independence_negated:
                support_confidence += 0.05
                
            # Cap at 0.9
            support_confidence = min(support_confidence, 0.9)
        
        # Return support analysis
        return {
            "has_support_indicators": has_support_indicators,
            "support_types": {k: v for k, v in support_indicators.items() if v},
            "matched_patterns": matched_patterns,
            "independence_negated": independence_negated,
            "strong_support_evidence": strong_support_evidence,
            "support_confidence": support_confidence
        } 

    def _analyze_complex_statements(self, response: str) -> Dict[str, Any]:
        """
        Analyze complex linguistic patterns such as double negatives, qualifiers, 
        and conditional statements that often lead to misinterpretation.
        
        Args:
            response: Response text to analyze
            
        Returns:
            Dictionary with analysis results including identified patterns and interpretations
        """
        if not response:
            return {"has_complex_patterns": False}
            
        # Initialize results dictionary
        results = {
            "has_complex_patterns": False,
            "double_negatives": [],
            "qualified_statements": [],
            "conditional_statements": [],
            "hedge_words": [],
            "interpretation_adjustments": {}
        }
        
        # 1. Detect double negatives
        double_negative_patterns = [
            r"\bnot\s+(\w+\s+){0,5}?never\b",
            r"\bnever\s+(\w+\s+){0,5}?not\b",
            r"\bdon'?t\s+(\w+\s+){0,5}?no\b",
            r"\bno\s+(\w+\s+){0,5}?not\b",
            r"\bcan'?t\s+(\w+\s+){0,5}?no\b",
            r"\bwon'?t\s+(\w+\s+){0,5}?never\b",
            r"\bu(n|m)able\s+(\w+\s+){0,5}?not\b",
            # Additional double negative patterns
            r"\bnot\s+(\w+\s+){0,5}?un(able|willing|likely)\b",
            r"\bnot\s+(\w+\s+){0,5}?impossible\b",
            r"\bnot\s+(\w+\s+){0,5}?without\b",
            r"\bcan'?t\s+(\w+\s+){0,5}?unless\b",
            r"\bwon'?t\s+(\w+\s+){0,5}?unless\b",
            r"\bnever\s+(\w+\s+){0,5}?unless\b",
            r"\brefuses\s+(\w+\s+){0,5}?not\b"
        ]
        
        for pattern in double_negative_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                results["double_negatives"].append(match.group(0))
        
        # 2. Detect qualified statements
        qualified_statement_patterns = [
            r"\b(sometimes|occasionally|rarely|not always|not often|usually not)\b",
            r"\bin (some|certain|specific) (situations|cases|instances)\b",
            r"\bonly (when|if|with)\b",
            r"\bgenerally\s+(doesn't|does not|won't|will not|cannot|can't)\b",
            r"\b(mostly|typically|normally)\s+(doesn't|does not|won't|will not|cannot|can't)\b",
            # Additional qualified statement patterns
            r"\b(on occasion|at times|now and then|every so often)\b",
            r"\b(hit or miss|varies|inconsistent|mixed results)\b",
            r"\b(some days|good days and bad days|depends on the day)\b",
            r"\b(not consistently|not reliably|not predictably)\b",
            r"\b(to some extent|to a degree|partially|somewhat)\b",
            r"\b(making progress|getting better at|improving with|working on)\b",
            r"\b(starting to|beginning to|learning to|trying to)\b",
            r"\b(almost|nearly|close to|on the verge of)\b"
        ]
        
        for pattern in qualified_statement_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                results["qualified_statements"].append(match.group(0))
        
        # 3. Detect conditional statements
        conditional_statement_patterns = [
            r"\bif\s+(?!.*?(start|help|assist|show|guide|prompt|begin|initiate)).*?then\b",
            r"\bwhen\s+(?!.*?(shown|prompted|reminded|guided|helped|assisted)).*?then\b",
            r"\b(unless|except when|except if|apart from when)\b",
            r"\bdepends\s+on\b",
            # Additional conditional statement patterns
            r"\bonly\s+if\b",
            r"\bas\s+long\s+as\b",
            r"\bprovided\s+that\b",
            r"\bin\s+case\s+of\b",
            r"\bunder\s+certain\s+conditions\b",
            r"\bwhen\s+the\s+conditions\s+are\s+right\b"
        ]
        
        for pattern in conditional_statement_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                results["conditional_statements"].append(match.group(0))
        
        # 4. Detect hedge words and phrases
        hedge_patterns = [
            r"\b(sort of|kind of|somewhat|a little|a bit|more or less|approximately)\b",
            r"\b(seem(s|ed)? to|appear(s|ed)? to|look(s|ed) like)\b",
            r"\b(I think|I believe|I guess|I suppose|I assume|might|may|could|possibly)\b",
            r"\b(at times|from time to time|now and then|once in a while)\b",
            # Additional hedge patterns
            r"\b(probably|likely|perhaps|maybe|potentially)\b",
            r"\b(not\s+sure\s+if|not\s+certain\s+if|hard\s+to\s+say\s+if)\b",
            r"\b(as\s+far\s+as\s+I\s+can\s+tell|from\s+what\s+I've\s+seen)\b",
            r"\b(it\s+appears|it\s+seems|it\s+looks\s+like)\b"
        ]
        
        for pattern in hedge_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                results["hedge_words"].append(match.group(0))
        
        # 5. Create interpretations and scoring adjustments
        if results["double_negatives"]:
            # Analyze the context of double negatives to determine the likely meaning
            negative_context = re.search(r"\b(struggles?|difficult(y|ies)|problems?|challenges?|hard|tough)\b", response, re.IGNORECASE)
            positive_context = re.search(r"\b(able|capable|manages?|succeeds?|accomplishe?s?)\b", response, re.IGNORECASE)
            
            if negative_context and not positive_context:
                # Double negative in negative context often indicates low ability
                # Reduce confidence penalty from -0.15 to -0.08
                results["interpretation_adjustments"]["double_negatives"] = {
                    "effect": "likely_negative",
                    "confidence_adjustment": -0.08,
                    "note": "Double negatives in negative context likely indicate inability or difficulty"
                }
            elif positive_context and not negative_context:
                # Double negative in positive context often indicates ability
                # Reduce confidence penalty from -0.15 to -0.08
                results["interpretation_adjustments"]["double_negatives"] = {
                    "effect": "likely_positive",
                    "confidence_adjustment": -0.08,
                    "note": "Double negatives in positive context likely indicate ability"
                }
            else:
                # Mixed or unclear context
                # Reduce confidence penalty from -0.2 to -0.12
                results["interpretation_adjustments"]["double_negatives"] = {
                    "effect": "potential_affirmative",
                    "confidence_adjustment": -0.12,
                    "note": "Double negatives with unclear context create significant ambiguity"
                }
        
        if results["qualified_statements"]:
            # More qualified statements suggest stronger EMERGING pattern
            qualifier_count = len(results["qualified_statements"])
            # Reduce base confidence penalty from -0.1 to -0.06
            confidence_adjustment = -0.06
            if qualifier_count > 1:
                # Reduce scaling factor from 0.02 to 0.015 per qualifier
                confidence_adjustment = -0.06 - (0.015 * min(qualifier_count - 1, 4))  # Up to -0.12 for 5+ qualifiers
                
            results["interpretation_adjustments"]["qualified_statements"] = {
                "effect": "suggests_emerging",
                "confidence_adjustment": confidence_adjustment,
                "note": "Qualifiers often indicate EMERGING rather than CANNOT_DO or INDEPENDENT"
            }
        
        if results["conditional_statements"]:
            # Reduce confidence penalty from -0.1 to -0.06
            results["interpretation_adjustments"]["conditional_statements"] = {
                "effect": "context_dependent",
                "confidence_adjustment": -0.06,
                "note": "Conditional statements suggest performance depends on specific conditions"
            }
        
        if results["hedge_words"]:
            # Reduce scaling factor from -0.05 to -0.03 per hedge word
            results["interpretation_adjustments"]["hedge_words"] = {
                "effect": "uncertainty",
                "confidence_adjustment": -0.03 * min(len(results["hedge_words"]), 3),
                "note": "Hedge words indicate uncertainty in the assessment"
            }
        
        # Set the has_complex_patterns flag
        results["has_complex_patterns"] = (len(results["double_negatives"]) > 0 or 
                                        len(results["qualified_statements"]) > 0 or 
                                        len(results["conditional_statements"]) > 0 or 
                                        len(results["hedge_words"]) > 0)
        
        return results 