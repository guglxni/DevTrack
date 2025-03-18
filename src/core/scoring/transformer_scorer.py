"""
Transformer-Based Scoring Module

This module implements a scoring approach using transformer-based models.
"""

import torch
from typing import Dict, Any, Optional, List, Tuple, Union
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import BaseScorer, ScoringResult, Score


class TransformerBasedScorer(BaseScorer):
    """
    Scorer that uses transformer models for classification
    
    This implementation:
    1. Uses pretrained transformer models for more nuanced understanding
    2. Can utilize zero-shot or fine-tuned classification
    3. Provides detailed confidence and reasoning
    4. Falls back gracefully when transformers are not available
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the transformer-based scorer"""
        super().__init__(config or self._default_config())
        self._initialize_model()
        
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
            "model_name": "facebook/bart-large-mnli",  # Default zero-shot model
            "confidence_threshold": 0.7,               # Minimum confidence threshold
            "use_zero_shot": True,                     # Whether to use zero-shot classification
            "use_milestone_context": True,             # Whether to use milestone context
            "max_length": 512                          # Maximum sequence length
        }
    
    def _initialize_model(self) -> None:
        """Initialize transformer model"""
        self.classifier = None
        
        if not TRANSFORMERS_AVAILABLE:
            self.is_available = False
            return
            
        try:
            if self.config.get("use_zero_shot", True):
                # Initialize zero-shot classification pipeline
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=self.config.get("model_name", "facebook/bart-large-mnli"),
                    tokenizer=self.config.get("model_name", "facebook/bart-large-mnli"),
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # Initialize standard classification model
                model_name = self.config.get("model_name", "distilbert-base-uncased")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
            
            self.is_available = True
        except Exception as e:
            self.is_available = False
            print(f"Error initializing transformer model: {e}")
    
    def _get_class_labels(self) -> List[str]:
        """Get class labels for classification"""
        return [
            "cannot do this skill",
            "lost this skill after previously having it",
            "emerging or developing this skill",
            "can do this skill with support or assistance",
            "can do this skill independently"
        ]
    
    def _format_with_milestone(self, response: str, milestone_context: Dict[str, Any]) -> str:
        """Format the input with milestone context"""
        if not milestone_context or "behavior" not in milestone_context:
            return response
            
        behavior = milestone_context.get("behavior", "")
        prompt = f"For the milestone '{behavior}', the caregiver responded: {response}"
        return prompt
    
    def score_with_zero_shot(self, 
                            response: str, 
                            milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score using zero-shot classification
        
        Args:
            response: Response to score
            milestone_context: Optional milestone context
            
        Returns:
            ScoringResult: Classification result
        """
        # Get class labels for zero-shot
        class_labels = self._get_class_labels()
        
        # Format input with milestone if available
        if milestone_context and self.config.get("use_milestone_context", True):
            input_text = self._format_with_milestone(response, milestone_context)
        else:
            input_text = response
            
        # Run classification
        result = self.classifier(
            input_text, 
            class_labels,
            multi_label=False
        )
        
        # Map result to scores
        label_to_score = {
            "cannot do this skill": Score.CANNOT_DO,
            "lost this skill after previously having it": Score.LOST_SKILL,
            "emerging or developing this skill": Score.EMERGING,
            "can do this skill with support or assistance": Score.WITH_SUPPORT,
            "can do this skill independently": Score.INDEPENDENT
        }
        
        # Get top prediction
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        # Convert to our score enum
        score_value = label_to_score.get(top_label, Score.NOT_RATED)
        
        # Check confidence threshold
        if top_score < self.config.get("confidence_threshold", 0.7):
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=top_score,
                method="transformer_zero_shot",
                reasoning=f"Low confidence ({top_score:.2f}) for {score_value.name}",
                details={
                    "labels": result["labels"],
                    "scores": result["scores"]
                }
            )
            
        return ScoringResult(
            score=score_value,
            confidence=top_score,
            method="transformer_zero_shot",
            reasoning=f"Transformer classified as {score_value.name} with {top_score:.2f} confidence",
            details={
                "labels": result["labels"],
                "scores": result["scores"]
            }
        )
    
    def score_with_classifier(self, 
                             response: str, 
                             milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score using standard classification model
        
        Args:
            response: Response to score
            milestone_context: Optional milestone context
            
        Returns:
            ScoringResult: Classification result
        """
        # Format input with milestone if available
        if milestone_context and self.config.get("use_milestone_context", True):
            input_text = self._format_with_milestone(response, milestone_context)
        else:
            input_text = response
            
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.get("max_length", 512)
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()[0]
        
        # Map to score categories (assuming model was trained with these labels)
        label_to_score = {
            0: Score.CANNOT_DO,
            1: Score.LOST_SKILL,
            2: Score.EMERGING,
            3: Score.WITH_SUPPORT,
            4: Score.INDEPENDENT
        }
        
        # Get top prediction
        top_idx = probs.argmax()
        top_prob = probs[top_idx]
        
        # Convert to our score enum
        score_value = label_to_score.get(top_idx, Score.NOT_RATED)
        
        # Check confidence threshold
        if top_prob < self.config.get("confidence_threshold", 0.7):
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=float(top_prob),
                method="transformer_classifier",
                reasoning=f"Low confidence ({top_prob:.2f}) for {score_value.name}",
                details={
                    "probabilities": probs.tolist(),
                    "predicted_idx": int(top_idx)
                }
            )
            
        return ScoringResult(
            score=score_value,
            confidence=float(top_prob),
            method="transformer_classifier",
            reasoning=f"Transformer classified as {score_value.name} with {top_prob:.2f} confidence",
            details={
                "probabilities": probs.tolist(),
                "predicted_idx": int(top_idx)
            }
        )
    
    def score(self, 
              response: str, 
              milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score response using transformer models
        
        Args:
            response: The response to score
            milestone_context: Optional milestone context
            
        Returns:
            ScoringResult: Score with confidence and reasoning
        """
        # Check if transformer model is available
        if not self.is_available or not self.classifier:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="transformer",
                reasoning="Transformer model not available"
            )
        
        response = response.strip()
        if not response:
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="transformer",
                reasoning="Empty response"
            )
        
        # Choose scoring method based on configuration
        if self.config.get("use_zero_shot", True):
            return self.score_with_zero_shot(response, milestone_context)
        else:
            return self.score_with_classifier(response, milestone_context) 