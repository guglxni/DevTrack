"""
Improved Scoring API Routes

This module provides FastAPI routes for the improved scoring engine.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import logging
from enum import Enum
import os
from fastapi.responses import HTMLResponse

# Import the improved scoring engine
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
from src.core.scoring.base import Score


# Initialize the router
router = APIRouter(
    prefix="/improved-scoring",
    tags=["improved-scoring"],
    responses={404: {"description": "Not found"}},
)

# Initialize logger
logger = logging.getLogger("improved_scoring_api")


# Create a singleton instance of the engine
_ENGINE_INSTANCE = None


def get_engine() -> ImprovedDevelopmentalScoringEngine:
    """
    Get the singleton instance of the scoring engine
    
    Returns:
        ImprovedDevelopmentalScoringEngine: The scoring engine instance
    """
    global _ENGINE_INSTANCE
    
    if _ENGINE_INSTANCE is None:
        logger.info("Initializing improved scoring engine")
        
        # Initialize with default config
        _ENGINE_INSTANCE = ImprovedDevelopmentalScoringEngine({
            "enable_keyword_scorer": True,
            "enable_embedding_scorer": True,
            "enable_transformer_scorer": False,  # Disabled by default for performance
            "enable_continuous_learning": False,
            "score_weights": {
                "keyword": 0.6,
                "embedding": 0.4
            }
        })
        
    return _ENGINE_INSTANCE


# API Models
class ScoreEnum(str, Enum):
    """Enumeration of possible scores"""
    CANNOT_DO = "CANNOT_DO"
    LOST_SKILL = "LOST_SKILL"
    EMERGING = "EMERGING"
    WITH_SUPPORT = "WITH_SUPPORT"
    INDEPENDENT = "INDEPENDENT"
    NOT_RATED = "NOT_RATED"


class MilestoneContext(BaseModel):
    """Milestone context for scoring"""
    id: str = Field(..., description="Unique identifier for the milestone")
    domain: Optional[str] = Field(None, description="Domain of development")
    behavior: str = Field(..., description="The milestone behavior")
    criteria: Optional[str] = Field(None, description="Criteria for meeting the milestone")
    age_range: Optional[str] = Field(None, description="Age range for the milestone")


class ScoringRequest(BaseModel):
    """Request for scoring a response"""
    response: str = Field(..., description="The response to score")
    milestone_context: MilestoneContext = Field(..., description="Context about the milestone")
    detailed: bool = Field(False, description="Whether to return detailed results")


class ComponentResult(BaseModel):
    """Result from an individual scoring component"""
    score_label: str = Field(..., description="Score label (e.g., INDEPENDENT)")
    score_value: int = Field(..., description="Numeric score value")
    confidence: float = Field(..., description="Confidence score (0-1)")
    method: str = Field(..., description="Scoring method used")
    reasoning: Optional[str] = Field(None, description="Reasoning for the score")


class ScoringResponse(BaseModel):
    """Response containing scoring results"""
    score_name: str = Field(..., description="Score label (e.g., INDEPENDENT)")
    score_value: int = Field(..., description="Numeric score value")
    confidence: float = Field(..., description="Confidence score (0-1)")
    reasoning: Optional[str] = Field(None, description="Reasoning for the score")
    needs_review: bool = Field(False, description="Whether this needs expert review")
    component_results: Optional[List[ComponentResult]] = Field(None, description="Individual component results")


class ExpertFeedbackRequest(BaseModel):
    """Request for providing expert feedback"""
    response: str = Field(..., description="The original response")
    milestone_context: MilestoneContext = Field(..., description="Context about the milestone")
    correct_score: ScoreEnum = Field(..., description="The correct score according to expert")
    notes: Optional[str] = Field(None, description="Optional expert notes")


class PerformanceMetricsResponse(BaseModel):
    """Response containing performance metrics"""
    confidence_metrics: Dict[str, Any] = Field(default_factory=dict, description="Confidence metrics by method")
    audit_statistics: Dict[str, Any] = Field(default_factory=dict, description="Audit log statistics")
    training_statistics: Dict[str, Any] = Field(default_factory=dict, description="Training data statistics")


class ReviewItem(BaseModel):
    """Item that needs expert review"""
    id: str = Field(..., description="Review item ID")
    response: str = Field(..., description="The response text")
    milestone: Dict[str, Any] = Field(..., description="Milestone context")
    predicted_score: str = Field(..., description="Predicted score label")
    confidence: Optional[float] = Field(None, description="Confidence in prediction")
    timestamp: str = Field(..., description="When this was queued for review")


# API Routes
@router.post("/score", response_model=ScoringResponse)
async def score_response(
    request: ScoringRequest,
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_engine)
):
    """
    Score a response using the improved scoring engine
    
    Args:
        request: The scoring request
        
    Returns:
        ScoringResponse: The scoring result
    """
    try:
        result = engine.score_response(
            response=request.response,
            milestone_context=request.milestone_context.dict(),
            detailed=True  # Always get detailed result for API
        )
        
        # Convert to API response model
        return ScoringResponse(
            score_name=result["score_name"],
            score_value=result["score_value"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            needs_review=result.get("needs_review", False),
            component_results=[
                ComponentResult(
                    score_label=comp["score_label"],
                    score_value=comp["score"],
                    confidence=comp["confidence"],
                    method=comp["method"],
                    reasoning=comp.get("reasoning")
                )
                for comp in result.get("component_results", [])
            ] if "component_results" in result else None
        )
    except Exception as e:
        logger.error(f"Error scoring response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scoring response: {str(e)}")


@router.post("/expert-feedback", status_code=204)
async def provide_expert_feedback(
    request: ExpertFeedbackRequest,
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_engine)
):
    """
    Provide expert feedback for a response
    
    Args:
        request: The feedback request
    """
    try:
        # Convert string enum to Score enum
        score_mapping = {
            "CANNOT_DO": Score.CANNOT_DO,
            "LOST_SKILL": Score.LOST_SKILL,
            "EMERGING": Score.EMERGING,
            "WITH_SUPPORT": Score.WITH_SUPPORT,
            "INDEPENDENT": Score.INDEPENDENT,
            "NOT_RATED": Score.NOT_RATED
        }
        
        correct_score = score_mapping.get(request.correct_score.value)
        if correct_score is None:
            raise HTTPException(status_code=400, detail=f"Invalid score: {request.correct_score}")
        
        # Provide feedback
        engine.with_expert_feedback(
            response=request.response,
            milestone_context=request.milestone_context.dict(),
            correct_score=correct_score,
            notes=request.notes
        )
        
        return None
    except Exception as e:
        logger.error(f"Error providing expert feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error providing expert feedback: {str(e)}")


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_engine)
):
    """
    Get performance metrics for the scoring system
    
    Returns:
        PerformanceMetricsResponse: The performance metrics
    """
    try:
        metrics = engine.get_performance_metrics()
        return PerformanceMetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")


@router.get("/reviews", response_model=List[ReviewItem])
async def get_pending_reviews(
    limit: int = Query(10, description="Maximum number of reviews to return"),
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_engine)
):
    """
    Get items needing expert review
    
    Args:
        limit: Maximum number of items to return
        
    Returns:
        List[ReviewItem]: Items needing review
    """
    try:
        pending = engine.get_pending_reviews(limit=limit)
        
        # Convert to API model
        return [
            ReviewItem(
                id=item["id"],
                response=item["response"],
                milestone=item["milestone"],
                predicted_score=item["predicted_score"],
                confidence=item.get("confidence"),
                timestamp=item["timestamp"]
            )
            for item in pending
        ]
    except Exception as e:
        logger.error(f"Error getting pending reviews: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting pending reviews: {str(e)}")


@router.post("/reviews/{review_id}/feedback", status_code=204)
async def provide_review_feedback(
    review_id: str,
    correct_score: ScoreEnum,
    notes: Optional[str] = None,
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_engine)
):
    """
    Provide feedback for a review item
    
    Args:
        review_id: The ID of the review item
        correct_score: The correct score
        notes: Optional expert notes
    """
    try:
        # Convert string enum to Score enum
        score_mapping = {
            "CANNOT_DO": Score.CANNOT_DO,
            "LOST_SKILL": Score.LOST_SKILL,
            "EMERGING": Score.EMERGING,
            "WITH_SUPPORT": Score.WITH_SUPPORT,
            "INDEPENDENT": Score.INDEPENDENT,
            "NOT_RATED": Score.NOT_RATED
        }
        
        score = score_mapping.get(correct_score.value)
        if score is None:
            raise HTTPException(status_code=400, detail=f"Invalid score: {correct_score}")
        
        # Get the learning engine
        if not engine.learning_engine:
            raise HTTPException(status_code=500, detail="Continuous learning is not enabled")
            
        # Add feedback
        success = engine.learning_engine.add_expert_feedback(
            review_id=review_id,
            correct_score=score,
            expert_notes=notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Review item not found: {review_id}")
            
        return None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing review feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error providing review feedback: {str(e)}")


@router.get("/", response_class=HTMLResponse)
async def get_improved_scoring_dashboard():
    """
    Get the Improved Scoring Dashboard HTML page
    """
    dashboard_path = os.path.join("src", "web", "static", "improved-scoring", "index.html")
    with open(dashboard_path, "r") as f:
        return f.read()


@router.get("/expert-feedback.html", response_class=HTMLResponse)
async def get_expert_feedback_page():
    """
    Get the Expert Feedback HTML page
    """
    page_path = os.path.join("src", "web", "static", "improved-scoring", "expert-feedback.html")
    with open(page_path, "r") as f:
        return f.read()


# Function to add these routes to the main API
def add_routes_to_app(app):
    """
    Add the improved scoring routes to a FastAPI app
    
    Args:
        app: The FastAPI app
    """
    app.include_router(router) 