"""
Semantic Embedding Scorer Module

This module implements a semantic scoring approach using sentence embeddings.
"""

import re
import torch
from typing import Dict, Any, Optional, List
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base import BaseScorer, ScoringResult, Score


class SemanticEmbeddingScorer(BaseScorer):
    """
    Scorer that uses sentence embeddings to compare responses with exemplars
    
    This implementation:
    1. Uses pre-trained sentence embeddings to capture semantic meaning
    2. Compares responses against canonical examples of each category
    3. Provides similarity scores with detailed confidence metrics
    4. Falls back gracefully when embeddings are not available
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the embedding-based scorer"""
        super().__init__(config or self._default_config())
        self._initialize_model()
        
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
            "model_name": "all-MiniLM-L6-v2",  # Default embedding model
            "confidence_threshold": 0.65,       # Minimum confidence threshold
            "similarity_threshold": 0.5,        # Minimum similarity threshold
            "use_milestone_context": True,      # Whether to use milestone context
        }
    
    def _initialize_model(self) -> None:
        """Initialize sentence embedding model"""
        self.embedding_model = None
        self.score_exemplars = {}
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.is_available = False
            return
            
        try:
            # Load the embedding model
            model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            self.is_available = True
            
            # Initialize exemplars for each score
            self._initialize_exemplars()
        except Exception as e:
            self.is_available = False
            print(f"Error initializing embedding model: {e}")
    
    def _initialize_exemplars(self) -> None:
        """Initialize exemplar embeddings for each score category"""
        if not self.is_available or not self.embedding_model:
            return
            
        # Define canonical examples for each score category
        exemplars = {
            Score.CANNOT_DO: [
                "No, child cannot do this at all",
                "Child doesn't know how to do this yet",
                "We haven't seen any signs of this behavior",
                "Not yet, still working on this skill",
                "No, never does this activity" 
            ],
            Score.LOST_SKILL: [
                "Used to do this but stopped",
                "Could do this before but has lost the ability",
                "Previously demonstrated this skill but regressed",
                "No longer shows this behavior that was present before",
                "Has lost this skill in the past few months"
            ],
            Score.EMERGING: [
                "Sometimes shows this behavior but not consistently",
                "Beginning to demonstrate this skill occasionally",
                "Starting to show early signs of this ability",
                "Shows this sometimes, but it's still developing",
                "Inconsistently demonstrates this behavior"
            ],
            Score.WITH_SUPPORT: [
                "Can do this with help or prompting",
                "Needs assistance to complete this task",
                "Does this when supported by an adult",
                "Requires guidance to perform this skill",
                "Can accomplish with verbal reminders"
            ],
            Score.INDEPENDENT: [
                "Yes, does this independently all the time",
                "Completely mastered this skill",
                "Always performs this task without help",
                "Consistently demonstrates this ability",
                "Does this on their own without prompting"
            ]
        }
        
        # Compute embeddings for all exemplars
        with torch.no_grad():
            self.score_exemplars = {}
            for score, examples in exemplars.items():
                # Encode all examples and stack the embeddings
                embeddings = self.embedding_model.encode(examples, convert_to_tensor=True)
                self.score_exemplars[score] = embeddings
    
    def _get_milestone_descriptions(self, score: Score, milestone_context: Dict[str, Any]) -> List[str]:
        """Generate score-specific descriptions based on milestone"""
        if not milestone_context or "behavior" not in milestone_context:
            return []
            
        behavior = milestone_context.get("behavior", "")
        criteria = milestone_context.get("criteria", "")
        
        descriptions = {
            Score.CANNOT_DO: [
                f"Child cannot {behavior.lower()}",
                f"Child does not {behavior.lower()}",
                f"Child is unable to {behavior.lower()}"
            ],
            Score.LOST_SKILL: [
                f"Child used to {behavior.lower()} but no longer does",
                f"Child previously could {behavior.lower()} but has lost this skill",
                f"Child has regressed in ability to {behavior.lower()}"
            ],
            Score.EMERGING: [
                f"Child is beginning to {behavior.lower()} sometimes",
                f"Child occasionally {behavior.lower()}",
                f"Child inconsistently {behavior.lower()}"
            ],
            Score.WITH_SUPPORT: [
                f"Child can {behavior.lower()} with help",
                f"Child {behavior.lower()} when given assistance",
                f"Child needs support to {behavior.lower()}"
            ],
            Score.INDEPENDENT: [
                f"Child can independently {behavior.lower()}",
                f"Child consistently {behavior.lower()} without help",
                f"Child has mastered {behavior.lower()}"
            ]
        }
        
        return descriptions.get(score, [])
    
    def score(self, 
              response: str, 
              milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score response using semantic embeddings
        
        Args:
            response: The response to score
            milestone_context: Optional milestone context (domain, behavior, etc.)
            
        Returns:
            ScoringResult: Score with confidence and reasoning
        """
        # Check if embedding model is available
        if not self.is_available or not self.embedding_model:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="embedding",
                reasoning="Embedding model not available"
            )
        
        response = response.strip()
        if not response:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="embedding",
                reasoning="Empty response"
            )
        
        # Check for ambiguity indicators in the text
        ambiguity_patterns = [
            r"\bsometimes\b",
            r"\boccasionally\b",
            r"\bvaries\b",
            r"\bdepends\b",
            r"\bmixed\b",
            r"\binconsistent\b",
            r"\bvariable\b",
            r"\bnot\s+always\b",
            r"\bnot\s+consistent\b",
            r"\bon\s+and\s+off\b",
            r"\bgood\s+days\s+and\s+bad\s+days\b",
            # Adding more subtle ambiguity patterns
            r"\bstarting\s+to\b",
            r"\bbeginning\s+to\b",
            r"\btrying\s+to\b",
            r"\battempting\s+to\b",
            r"\bin\s+some\s+situations\b",
            r"\bin\s+certain\s+contexts\b",
            r"\bwith\s+certain\s+people\b",
            r"\bstill\s+learning\b",
            r"\bpracticing\b",
            r"\bworking\s+on\b",
            r"\bmaking\s+progress\b",
            r"\bimproving\b",
            r"\bmost\s+of\s+the\s+time\b",
            r"\busually\b",
            r"\bgenerally\b",
            r"\btypically\b",
            r"\boften\b",
            r"\bsome\s+days\b",
            r"\bseveral\s+times\b",
            r"\ba\s+few\s+times\b",
            r"\bmore\s+than\s+before\b",
            r"\bless\s+than\s+before\b",
            r"\bbetter\s+than\s+before\b",
            r"\bworse\s+than\s+before\b",
            r"\bonly\s+when\b"
        ]
        
        # Check for context dependency indicators
        context_dependency_patterns = [
            r"\bat\s+home\b",
            r"\bat\s+school\b",
            r"\bat\s+daycare\b",
            r"\bwith\s+me\b",
            r"\bwith\s+mom\b",
            r"\bwith\s+dad\b",
            r"\bwith\s+parents\b",
            r"\bwith\s+family\b",
            r"\bwith\s+familiar\s+people\b",
            r"\bwith\s+strangers\b",
            r"\bin\s+familiar\s+settings\b",
            r"\bin\s+unfamiliar\s+settings\b",
            r"\bwhen\s+calm\b",
            r"\bwhen\s+excited\b",
            r"\bwhen\s+tired\b",
            r"\bwhen\s+hungry\b",
            r"\bwhen\s+interested\b",
            r"\bwhen\s+motivated\b"
        ]
        
        has_ambiguity = False
        for pattern in ambiguity_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                has_ambiguity = True
                break
        
        has_context_dependency = False
        for pattern in context_dependency_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                has_context_dependency = True
                break
        
        # Check for double negation patterns
        double_negation_patterns = [
            r"not\s+unable",
            r"not\s+impossible",
            r"not\s+never",
            r"not\s+without",
            r"can't\s+not",
            r"doesn't\s+not",
            r"isn't\s+not",
            r"wasn't\s+not"
        ]
        
        has_double_negation = False
        for pattern in double_negation_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                has_double_negation = True
                break
        
        # Encode the response
        with torch.no_grad():
            response_embedding = self.embedding_model.encode(response, convert_to_tensor=True)
            
            # Calculate similarities to each score category
            similarities = {}
            detailed_similarities = {}
            
            for score, exemplar_embeddings in self.score_exemplars.items():
                # Calculate cosine similarity with each exemplar
                similarity_scores = torch.nn.functional.cosine_similarity(
                    response_embedding.unsqueeze(0),
                    exemplar_embeddings
                )
                
                # Store maximum and average similarity
                max_similarity = torch.max(similarity_scores).item()
                avg_similarity = torch.mean(similarity_scores).item()
                similarities[score] = max_similarity  # Use maximum similarity as the score
                
                detailed_similarities[score] = {
                    "max": max_similarity,
                    "avg": avg_similarity,
                    "all": similarity_scores.tolist()
                }
                
                # Add milestone-specific context if available
                if milestone_context and self.config.get("use_milestone_context", True):
                    milestone_descriptions = self._get_milestone_descriptions(score, milestone_context)
                    
                    if milestone_descriptions:
                        # Calculate similarities with milestone-specific descriptions
                        milestone_embeddings = self.embedding_model.encode(milestone_descriptions, convert_to_tensor=True)
                        milestone_similarities = torch.nn.functional.cosine_similarity(
                            response_embedding.unsqueeze(0),
                            milestone_embeddings
                        )
                        
                        # Incorporate milestone similarity (weighted higher)
                        milestone_max = torch.max(milestone_similarities).item()
                        if milestone_max > max_similarity:
                            # Milestone descriptions are a better match
                            similarities[score] = milestone_max
                            detailed_similarities[score]["milestone_max"] = milestone_max
                            detailed_similarities[score]["milestone_all"] = milestone_similarities.tolist()
            
            # Find the best matching score
            if not similarities:
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=0.0,
                    method="embedding",
                    reasoning="No similarities calculated"
                )
                
            # Get sorted similarities
            similarities_sorted = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_score_value, top_similarity = similarities_sorted[0]
            second_score_value, second_similarity = similarities_sorted[1] if len(similarities_sorted) > 1 else (None, 0)
            
            # Check if the similarity meets the threshold
            if top_similarity < self.config.get("similarity_threshold", 0.5):
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=top_similarity,
                    method="embedding",
                    reasoning=f"Best match ({top_score_value.name}) below similarity threshold",
                    details={"similarities": detailed_similarities}
                )
            
            # Calculate confidence based on:
            # 1. Top similarity score
            # 2. Gap between top and second similarities (distinctiveness)
            
            # Calculate distinctiveness
            distinctiveness = top_similarity - second_similarity  # How distinct is the top match?
            
            # Overall confidence combines similarity and distinctiveness
            confidence = 0.7 * top_similarity + 0.3 * distinctiveness
            
            # Adjust for ambiguity and double negations
            ambiguity_adjustment = 0.0
            
            # More nuanced adjustment based on ambiguity type
            if has_ambiguity:
                # Reduce confidence for ambiguous responses
                ambiguity_adjustment -= 0.15
                
                # For ambiguous responses, consider EMERGING as a fallback
                if confidence < 0.6 and top_score_value in [Score.INDEPENDENT, Score.CANNOT_DO]:
                    if second_score_value == Score.EMERGING or second_score_value == Score.WITH_SUPPORT:
                        # If the second best score is EMERGING or WITH_SUPPORT and it's close
                        if distinctiveness < 0.15:
                            top_score_value = second_score_value
                            confidence = max(0.5, confidence + 0.05)  # Slightly boost confidence
            
            # Add more nuanced adjustment if there's context dependency
            if has_context_dependency:
                ambiguity_adjustment -= 0.05
                
                # If the context dependency implies variability and the top score is extreme
                if top_score_value in [Score.INDEPENDENT, Score.CANNOT_DO]:
                    # Check if WITH_SUPPORT or EMERGING are close alternatives
                    for i, (score, sim) in enumerate(similarities_sorted[1:3]):
                        if score in [Score.WITH_SUPPORT, Score.EMERGING] and top_similarity - sim < 0.2:
                            # Prefer the more moderate category for context-dependent behaviors
                            top_score_value = score
                            confidence = max(0.5, confidence - 0.1)  # Reduce confidence due to the adjustment
                            break
            
            # Apply the ambiguity adjustment
            confidence = max(0.4, confidence + ambiguity_adjustment)
            
            if has_double_negation:
                # For double negations, adjust the score if it's a negative category
                if top_score_value in [Score.CANNOT_DO, Score.LOST_SKILL]:
                    # Flip to the opposite category
                    if top_score_value == Score.CANNOT_DO:
                        # Check if INDEPENDENT is a close second
                        if second_score_value == Score.INDEPENDENT and distinctiveness < 0.2:
                            top_score_value = Score.INDEPENDENT
                        else:
                            # Otherwise default to EMERGING
                            top_score_value = Score.EMERGING
                    elif top_score_value == Score.LOST_SKILL:
                        top_score_value = Score.INDEPENDENT
                    
                    # Reduce confidence due to the complexity
                    confidence = max(0.5, confidence - 0.1)
            
            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
            
            # Check confidence threshold
            if confidence < self.config.get("confidence_threshold", 0.65):
                reasoning = f"Confidence too low for {top_score_value.name}"
                if has_ambiguity:
                    reasoning += " (contains ambiguity indicators)"
                if has_context_dependency:
                    reasoning += " (contains context dependency indicators)"
                if has_double_negation:
                    reasoning += " (contains double negation patterns)"
                
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=confidence,
                    method="embedding",
                    reasoning=reasoning,
                    details={"similarities": detailed_similarities}
                )
            
            # Generate reasoning
            reasoning = f"Semantic similarity indicates {top_score_value.name} (similarity: {top_similarity:.2f})"
            
            if has_ambiguity:
                reasoning += " (contains ambiguity indicators)"
            
            if has_context_dependency:
                reasoning += " (contains context dependency indicators)"
            
            if has_double_negation:
                reasoning += " (contains double negation patterns)"
            
            return ScoringResult(
                score=top_score_value,
                confidence=confidence,
                method="embedding",
                reasoning=reasoning,
                details={"similarities": detailed_similarities}
            ) 