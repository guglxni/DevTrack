"""
Modular Scoring System for ASD Assessment

This package contains the components of the improved modular scoring system.
"""

from .base import BaseScorer, ScoringResult
from .keyword_scorer import KeywordBasedScorer
from .embedding_scorer import SemanticEmbeddingScorer
from .transformer_scorer import TransformerBasedScorer
from .confidence_tracker import ConfidenceTracker
from .audit_logger import AuditLogger
from .continuous_learning import ContinuousLearningEngine
from .improved_engine import ImprovedDevelopmentalScoringEngine
from .dynamic_ensemble import DynamicEnsembleScorer
from .component_specialization import (
    SpecializedScorer, 
    SpecializationDomain, 
    SpecializationAgeGroup,
    SpecializationFeature,
    KeywordSpecializedScorer,
    EmbeddingSpecializedScorer,
    TransformerSpecializedScorer,
    LLMSpecializedScorer,
    analyze_response_features,
    specialize_ensemble_weights
)
from .llm_scorer import LLMBasedScorer
from .r2r_enhanced_scorer import R2REnhancedScorer
from .r2r_active_learning import R2RActiveLearningSystems

# New advanced LLM integration modules
from .llm_fine_tuner import LLMFineTuner, fine_tune_llm_for_milestone_scoring
from .reasoning_enhancer import (
    ReasoningEnhancer, 
    ReasoningStep, 
    ReasoningChain, 
    enhance_scoring_with_reasoning
)

__all__ = [
    'BaseScorer',
    'ScoringResult',
    'KeywordBasedScorer',
    'SemanticEmbeddingScorer',
    'TransformerBasedScorer',
    'ConfidenceTracker',
    'AuditLogger',
    'ContinuousLearningEngine',
    'ImprovedDevelopmentalScoringEngine',
    'DynamicEnsembleScorer',
    'SpecializedScorer',
    'SpecializationDomain',
    'SpecializationAgeGroup',
    'SpecializationFeature',
    'KeywordSpecializedScorer',
    'EmbeddingSpecializedScorer',
    'TransformerSpecializedScorer',
    'LLMSpecializedScorer',
    'analyze_response_features',
    'specialize_ensemble_weights',
    'LLMBasedScorer',
    'R2REnhancedScorer',
    'R2RActiveLearningSystems',
    
    # New advanced LLM integration exports
    'LLMFineTuner',
    'fine_tune_llm_for_milestone_scoring',
    'ReasoningEnhancer',
    'ReasoningStep',
    'ReasoningChain',
    'enhance_scoring_with_reasoning'
] 