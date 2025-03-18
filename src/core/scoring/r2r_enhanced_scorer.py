"""
R2R Enhanced Scorer

This module implements a scorer that leverages the R2R (Reason to Retrieve) system
to enhance scoring capabilities through retrieval augmented generation.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import time

from ..retrieval.r2r_client import R2RClient
from .base import BaseScorer, ScoringResult, Score

# Configure logging
logger = logging.getLogger(__name__)

class R2REnhancedScorer(BaseScorer):
    """Enhanced scorer that utilizes R2R for contextual scoring of responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the R2R Enhanced Scorer.
        
        Args:
            config: Configuration dictionary for the scorer and R2R client
        """
        super().__init__(config)
        self.config = config or {}
        
        # Initialize R2R client with focus on local model
        self.r2r_client = R2RClient(
            llm_provider="local",
            llm_config={
                "model_path": self.config.get("model_path", "models/mistral-7b-instruct-v0.2.Q3_K_S.gguf"),
                "temperature": 0.2,
                "max_tokens": 1024
            },
            data_dir=self.config.get("data_dir", "data/documents"),
            r2r_base_url=self.config.get("r2r_base_url", None)
        )
        
        # Track if client is available
        self.client_available = self.r2r_client.model is not None
        if not self.client_available:
            logger.warning("R2R Enhanced Scorer will operate in fallback mode")

    def format_scoring_query(self, response: str, milestone_context: Dict[str, Any]) -> str:
        """
        Format the query for scoring using developmental milestone context.
        
        Args:
            response: User response to score
            milestone_context: Context about the milestone being assessed
            
        Returns:
            str: Formatted query for the scoring model
        """
        milestone_name = milestone_context.get("name", "Unknown milestone")
        age_range = milestone_context.get("age_range", "Unknown age range")
        description = milestone_context.get("description", "")
        domain = milestone_context.get("domain", "General")
        
        query = f"""
        ## Developmental Milestone
        Name: {milestone_name}
        Domain: {domain}
        Age Range: {age_range}
        Description: {description}
        
        ## Caregiver's Response
        {response}
        
        ## Scoring Task
        Analyze this caregiver's response about their child's developmental milestone.
        Does the response indicate the child has achieved this milestone? 
        Consider the age-appropriateness and specificity of the description.
        """
        
        return query.strip()

    def extract_score_from_text(self, text: str) -> Tuple[Score, float, str]:
        """
        Extract a structured score from model-generated text.
        
        Args:
            text: Generated text from LLM
            
        Returns:
            Tuple[Score, float, str]: Score enum, confidence value, and reasoning
        """
        text_lower = text.lower()
        
        # Set default values
        score = Score.NOT_RATED
        confidence = 0.7  # Default confidence
        reasoning = text
        
        # Check for phrases indicating the milestone is not achieved
        negative_indicators = [
            "has not achieved",
            "not yet achieved",
            "has not yet",
            "doesn't indicate",
            "does not indicate",
            "milestone is not present",
            "not consistently demonstrated",
            "not able to",
            "does not demonstrate",
            "no evidence",
            "milestone is absent",
            "has not mastered",
            "still learning",
            "cannot perform",
            "not yet figured out",
            "hasn't figured out",
            "not yet consistently",
            "not yet able",
            "has not"
        ]
        
        # Check for phrases indicating the milestone is emerging
        emerging_indicators = [
            "partially achieved",
            "beginning to develop",
            "starting to show",
            "emerging skill",
            "developing this skill",
            "some evidence",
            "milestone is emerging",
            "shows signs of",
            "in the process of",
            "learning to",
            "getting better at",
            "making progress toward",
            "trying to",
            "sometimes",
            "occasionally",
            "beginning to",
            "started trying",
            "just started"
        ]
        
        # Check for phrases indicating the milestone is achieved
        positive_indicators = [
            "achieved the milestone", 
            "has achieved", 
            "indicates the child has achieved",
            "consistent with having achieved",
            "child has met",
            "milestone is present",
            "milestone is achieved",
            "child demonstrates",
            "yes, based on",
            "appears to have achieved",
            "child is able to",
            "milestone has been achieved",
            "clearly indicates",
            "consistently"
        ]
        
        # First, look for explicit negations that would indicate CANNOT_DO
        if any(indicator in text_lower for indicator in negative_indicators):
            score = Score.CANNOT_DO
        # Then check for emerging indicators
        elif any(indicator in text_lower for indicator in emerging_indicators):
            score = Score.EMERGING
        # Finally check for positive indicators
        elif any(indicator in text_lower for indicator in positive_indicators):
            # Check if this is with support
            if "with support" in text_lower or "with help" in text_lower or "with assistance" in text_lower:
                score = Score.WITH_SUPPORT
            else:
                score = Score.INDEPENDENT
        # Look for phrases that directly indicate the child has not achieved the milestone
        elif "not achieved" in text_lower or "has not" in text_lower:
            score = Score.CANNOT_DO
        # Look for phrases indicating the milestone is achieved but with a qualifier
        elif "has achieved" in text_lower or "is able to" in text_lower:
            score = Score.INDEPENDENT
        # Check for contradictions or ambiguity
        elif "not" in text_lower and "achieved" in text_lower:
            score = Score.CANNOT_DO
        # Simple "yes" or "no" at the beginning
        elif text_lower.strip().startswith("yes"):
            score = Score.INDEPENDENT
        elif text_lower.strip().startswith("no"):
            score = Score.CANNOT_DO
        # If nothing matched, extract from the reasoning
        else:
            # Looking for general expressions about achievement
            if "does not" in text_lower and ("indicate" in text_lower or "suggest" in text_lower):
                score = Score.CANNOT_DO
            elif "suggests that the child has achieved" in text_lower:
                score = Score.INDEPENDENT
            elif "does indicate" in text_lower or "suggests" in text_lower:
                score = Score.INDEPENDENT
        
        # Extract reasoning section if present
        if "--- reasoning ---" in text_lower:
            parts = text.split("--- REASONING ---")
            if len(parts) > 1:
                reasoning = parts[1].strip()
        elif "reasoning:" in text_lower:
            parts = text.split("reasoning:", 1)
            if len(parts) > 1:
                reasoning = parts[1].strip()
        
        return score, confidence, reasoning

    def score(self, response: str, milestone_context: Optional[Dict[str, Any]] = None) -> ScoringResult:
        """
        Score a response against developmental milestones using R2R.
        
        Args:
            response: The caregiver's response to score
            milestone_context: Context about the milestone being assessed
            
        Returns:
            ScoringResult: Structured scoring result with explanation
        """
        start_time = time.time()
        
        # Check if milestone context is provided
        if not milestone_context:
            logger.warning("Milestone context not provided, cannot score response")
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="r2r_enhanced_fallback",
                reasoning="Missing milestone context"
            )
            
        # Check if response is empty
        if not response or not response.strip():
            logger.warning("Empty response, cannot score")
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="r2r_enhanced_fallback",
                reasoning="Empty response"
            )
        
        # Check if R2R client is available
        if not self.client_available or not self.r2r_client.model:
            logger.warning("R2R client not available for scoring")
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="r2r_enhanced_fallback",
                reasoning="R2R client not available"
            )
        
        try:
            # Format the scoring query
            query = self.format_scoring_query(response, milestone_context)
            
            # Using system prompt for expert analysis
            system_prompt = """You are an expert pediatric developmental specialist analyzing caregiver responses about their child's development.
            Your task is to determine if a milestone is present, absent, or emerging based on a response.
            
            Analyze the caregiver's response to identify:
            1. If it directly indicates the milestone is achieved (INDEPENDENT)
            2. If it clearly indicates the milestone is not achieved (CANNOT_DO)
            3. If it suggests the milestone is partially achieved or in development (EMERGING)
            
            Provide your response in this format:
            
            SCORE: [INDEPENDENT/CANNOT_DO/EMERGING]
            CONFIDENCE: [0-10 scale]
            
            --- REASONING ---
            [Your detailed analysis here]
            """
            
            # Generate scoring response using the local model
            result = self.r2r_client.generate(query, system_prompt)
            
            # Check for error
            if "error" in result:
                logger.error(f"Error in generation: {result['error']}")
                return ScoringResult(
                    score=Score.NOT_RATED,
                    confidence=0.0,
                    method="r2r_enhanced_error",
                    reasoning=f"Generation error: {result['error']}"
                )
            
            # Extract the generated text
            generated_text = result.get("text", "")
            
            # Extract score, confidence, and reasoning
            score, confidence, reasoning = self.extract_score_from_text(generated_text)
            
            # Prepare result
            scoring_result = ScoringResult(
                score=score,
                confidence=confidence,
                method="r2r_enhanced",
                reasoning=reasoning if reasoning else generated_text,
                details={
                    "generated_text": generated_text,
                    "sources": result.get("sources", [])
                }
            )
            
            return scoring_result
            
        except Exception as e:
            logger.error(f"Error in R2R scoring: {e}")
            return ScoringResult(
                score=Score.NOT_RATED,
                confidence=0.0,
                method="r2r_enhanced_error",
                reasoning=f"Scoring error: {str(e)}"
            ) 