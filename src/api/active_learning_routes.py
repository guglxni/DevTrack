"""
Active Learning API Routes

This module provides API routes for the active learning system,
allowing expert feedback collection and model improvement tracking.
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, Field
import uuid
from datetime import datetime
import logging
from fastapi.responses import HTMLResponse, FileResponse
import os

from src.core.scoring.base import Score
from src.core.scoring.active_learning import ActiveLearningEngine

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/active-learning",
    tags=["active-learning"],
    responses={404: {"description": "Not found"}}
)

# ----- Models -----

class ValueableExample(BaseModel):
    """Example that would be valuable for expert review"""
    response: str = Field(..., description="The response text")
    milestone_context: Dict[str, Any] = Field(..., description="Context about the milestone")
    predicted_score: str = Field(..., description="Predicted score name")
    predicted_score_value: int = Field(..., description="Predicted score value")
    confidence: float = Field(..., description="Confidence in prediction")
    reasoning: Optional[str] = Field(None, description="Reasoning for prediction")
    priority: float = Field(..., description="Priority value for review (higher is more valuable)")
    id: str = Field(..., description="Unique identifier for the example")

class FeedbackRequest(BaseModel):
    """Request to provide expert feedback on a prediction"""
    review_id: str = Field(..., description="ID of the review to provide feedback on")
    correct_score: str = Field(..., description="Correct score according to expert")
    notes: Optional[str] = Field(None, description="Optional notes from expert")

class ModelVersionInfo(BaseModel):
    """Information about a model version"""
    version: str = Field(..., description="Version string (e.g., '1.2.3')")
    timestamp: str = Field(..., description="When this version was created")
    description: str = Field(..., description="Description of this version")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    training_examples_count: int = Field(..., description="Number of training examples used")

class ActiveLearningStats(BaseModel):
    """Statistics about the active learning system"""
    total_examples: int = Field(..., description="Total number of examples in system")
    examples_by_category: Dict[str, int] = Field(..., description="Examples per category")
    pending_reviews: int = Field(..., description="Number of pending expert reviews")
    completed_reviews: int = Field(..., description="Number of completed expert reviews")
    current_model_version: str = Field(..., description="Current model version")
    total_model_versions: int = Field(..., description="Total number of model versions")

# ----- Dependencies -----

def get_active_learning_engine() -> ActiveLearningEngine:
    """Get or create the active learning engine"""
    # In production, you'd want to use a singleton pattern or dependency injection
    try:
        engine = ActiveLearningEngine()
        return engine
    except Exception as e:
        logger.error(f"Error initializing ActiveLearningEngine: {str(e)}")
        # Create a minimal engine with default config
        from src.core.scoring.continuous_learning import ContinuousLearningEngine
        return ContinuousLearningEngine()

# ----- Routes -----

@router.get("/pending-reviews", response_model=List[ValueableExample])
async def get_pending_reviews(
    limit: int = Query(20, description="Maximum number of reviews to return"),
    engine: ActiveLearningEngine = Depends(get_active_learning_engine)
):
    """
    Get pending reviews ordered by priority
    """
    try:
        pending = engine.get_prioritized_reviews(limit)
        
        # Log the number of pending reviews and sample reviews
        logger.info(f"API: Got {len(pending)} pending reviews from engine")
        sample_reviews = [item for item in pending if item.get('id', '').startswith('sample_')]
        logger.info(f"API: Found {len(sample_reviews)} sample reviews in pending reviews")
        
        # Convert to API model
        result = []
        for item in pending:
            try:
                # Log sample review details
                if item.get('id', '').startswith('sample_'):
                    logger.info(f"API: Processing sample review: {item.get('id')}")
                    logger.info(f"API: Sample review milestone_context: {item.get('milestone_context')}")
                
                # Ensure all required fields are present with defaults if missing
                result.append({
                    "response": item.get("response", ""),
                    "milestone_context": item.get("milestone_context", {}),
                    "predicted_score": item.get("predicted_score", "UNKNOWN"),
                    "predicted_score_value": item.get("predicted_score_value", -1),
                    "confidence": item.get("confidence", 0.0),
                    "reasoning": item.get("reasoning", ""),
                    "priority": item.get("priority", 0.5),
                    "id": item.get("id", str(uuid.uuid4()))
                })
            except Exception as e:
                logger.error(f"Error processing review item: {e}")
                # Skip this item and continue
                continue
        
        # Log the final result
        logger.info(f"API: Returning {len(result)} reviews")
        sample_count = sum(1 for item in result if item.get('id', '').startswith('sample_'))
        logger.info(f"API: Returning {sample_count} sample reviews")
        
        return result
    except Exception as e:
        logger.error(f"Error getting pending reviews: {str(e)}")
        # Return empty list instead of error
        return []

@router.post("/feedback", status_code=204)
async def provide_feedback(
    request: FeedbackRequest,
    engine: ActiveLearningEngine = Depends(get_active_learning_engine)
):
    """
    Provide expert feedback on a prediction
    """
    # Convert string score to enum
    try:
        score_enum = getattr(Score, request.correct_score)
    except AttributeError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid score: {request.correct_score}. Valid scores are: {[s.name for s in Score if s != Score.NOT_RATED]}"
        )
    
    # Add feedback
    success = engine.add_expert_feedback(
        review_id=request.review_id,
        correct_score=score_enum,
        expert_notes=request.notes
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Review item with ID {request.review_id} not found"
        )

@router.get("/model-versions", response_model=List[ModelVersionInfo])
async def get_model_versions(
    engine: ActiveLearningEngine = Depends(get_active_learning_engine)
):
    """
    Get the history of model versions
    """
    try:
        return engine.model_versions
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        # Return a default version if there's an error
        return [{
            "version": "0.1.0",
            "timestamp": datetime.now().isoformat(),
            "description": "Initial version",
            "metrics": {},
            "training_examples_count": 0
        }]

@router.get("/statistics", response_model=ActiveLearningStats)
async def get_statistics(
    engine: ActiveLearningEngine = Depends(get_active_learning_engine)
):
    """
    Get statistics about the active learning system
    """
    try:
        # Use the engine's get_system_statistics method directly
        return engine.get_system_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        # Return default statistics
        return {
            "total_examples": 0,
            "examples_by_category": {},
            "pending_reviews": 0,
            "completed_reviews": 0,
            "current_model_version": "0.1.0",
            "total_model_versions": 1
        }

@router.get("/export-interface", response_model=Dict[str, Any])
async def export_interface_data(
    engine: ActiveLearningEngine = Depends(get_active_learning_engine)
):
    """
    Export all data needed for the feedback interface
    """
    return engine.export_feedback_interface_data()

@router.post("/trigger-retraining", status_code=204)
async def trigger_retraining(
    description: str = Body(..., embed=True, description="Description of why retraining was triggered"),
    engine: ActiveLearningEngine = Depends(get_active_learning_engine)
):
    """
    Manually trigger model retraining
    """
    # Retrain models
    engine._retrain_models()
    
    # Update version with description
    engine._increment_model_version(description)

@router.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """
    Get the Active Learning Dashboard HTML page
    """
    dashboard_path = os.path.join("src", "web", "static", "active-learning", "index.html")
    with open(dashboard_path, "r") as f:
        return f.read()

@router.get("/pending-reviews.html", response_class=HTMLResponse)
async def get_pending_reviews_page():
    """
    Get the Pending Reviews HTML page
    """
    page_path = os.path.join("src", "web", "static", "active-learning", "pending-reviews.html")
    with open(page_path, "r") as f:
        return f.read()

@router.get("/statistics.html", response_class=HTMLResponse)
async def get_statistics_page():
    """
    Get the Statistics HTML page
    """
    page_path = os.path.join("src", "web", "static", "active-learning", "statistics.html")
    with open(page_path, "r") as f:
        return f.read()

@router.get("/model-versions.html", response_class=HTMLResponse)
async def get_model_versions_page():
    """
    Get the Model Versions HTML page
    """
    page_path = os.path.join("src", "web", "static", "active-learning", "model-versions.html")
    with open(page_path, "r") as f:
        return f.read()

@router.get("/export-interface.html", response_class=HTMLResponse)
async def get_export_interface_page():
    """
    Get the Export Interface HTML page
    """
    page_path = os.path.join("src", "web", "static", "active-learning", "export-interface.html")
    with open(page_path, "r") as f:
        return f.read()

# Function to register routes with the main app
def add_routes_to_app(app):
    """Add active learning routes to the FastAPI app"""
    app.include_router(router) 