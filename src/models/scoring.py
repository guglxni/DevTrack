from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

class ParentResponse(BaseModel):
    """Model for a parent response to be scored."""
    id: str = Field(..., description="Unique identifier for the parent response")
    question: str = Field(..., description="The question that was asked")
    milestone_behavior: str = Field(..., description="The milestone behavior to check for")
    response: str = Field(..., description="The parent's response text")
    keywords: Optional[List[str]] = Field(default=None, description="Optional keywords to use for scoring")

class ScoredResponse(BaseModel):
    """Response model for a scored parent response."""
    id: str = Field(..., description="Unique identifier for the scored response")
    parent_response_id: str = Field(..., description="ID of the parent response that was scored")
    score: int = Field(..., description="Numeric score value")
    label: str = Field(..., description="Score label (e.g., INDEPENDENT, EMERGING)")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Reasoning for the score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata about the scoring process")

class ComprehensiveAssessmentRequest(BaseModel):
    """Request model for comprehensive assessment scoring."""
    parent_responses: List[ParentResponse] = Field(..., description="List of parent responses to score") 