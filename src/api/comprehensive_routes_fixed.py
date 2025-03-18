"""
Comprehensive Assessment Routes

This module provides routes for comprehensive assessment, combining question processing,
keyword management, response analysis, and score recording in a single endpoint.
"""

import logging
import os
import traceback
from typing import Dict, List, Any, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
import httpx

from src.core.scoring.base import Score
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive_routes")

# Create router
router = APIRouter(prefix="/api", tags=["comprehensive"])

# Models
class ComprehensiveAssessment(BaseModel):
    """Comprehensive assessment request model."""
    question: str = Field(..., description="The question text about the child's behavior")
    milestone_behavior: str = Field(..., description="The milestone behavior being assessed")
    parent_response: str = Field(..., description="Parent/caregiver response describing the child's behavior")
    keywords: Optional[Dict[str, List[str]]] = Field(None, description="Optional dictionary of keywords by category")

class ComprehensiveResult(BaseModel):
    """Result model for comprehensive assessment."""
    question_processed: bool = Field(..., description="Whether the question was successfully processed")
    milestone_found: bool = Field(..., description="Whether the milestone was found")
    milestone_details: Optional[Dict] = Field(None, description="Details about the milestone if found")
    keywords_updated: Optional[List[str]] = Field(None, description="Categories that were updated with new keywords")
    score: int = Field(..., description="The determined score (0-4)")
    score_label: str = Field(..., description="The score category (e.g., INDEPENDENT)")
    confidence: float = Field(..., description="Confidence level of the score determination (0-1)")
    domain: Optional[str] = Field(None, description="Developmental domain of the milestone")

# Dependency for scoring engine
def get_scoring_engine():
    """Get the scoring engine from the app module."""
    try:
        from src.app import engine
        logger.info("Successfully retrieved engine from src.app")
        return engine
    except ImportError:
        logger.error("Failed to import engine from src.app")
        try:
            from src.core.scoring.engine import ImprovedDevelopmentalScoringEngine
            logger.info("Creating new engine instance")
            return ImprovedDevelopmentalScoringEngine()
        except ImportError:
            logger.error("Failed to import ImprovedDevelopmentalScoringEngine")
            # Return a dummy engine that will use the smart scoring endpoint
            logger.info("Using smart scoring endpoint as fallback")
            return None

def try_llm_scoring(engine, response: str, milestone_context: dict) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """
    Attempt to score a response using the LLM scorer directly.
    
    Args:
        engine: The scoring engine
        response: The response text
        milestone_context: Context about the milestone
        
    Returns:
        Tuple of (score_value, score_label, confidence) or (None, None, None) if LLM scoring fails
    """
    # Check if LLM scorer is available in the engine
    if not hasattr(engine, '_scorers') or 'llm' not in engine._scorers:
        logger.info("LLM scorer not available in engine")
        return None, None, None
    
    try:
        logger.info(f"Attempting to use LLM-based scoring for response: {response}")
        llm_result = engine._scorers['llm'].score(response, milestone_context)
        
        if llm_result.score != Score.NOT_RATED and llm_result.confidence > 0.5:
            llm_score = llm_result.score.value if hasattr(llm_result.score, 'value') else llm_result.score
            llm_confidence = llm_result.confidence
            llm_score_label = llm_result.score.name if hasattr(llm_result.score, 'name') else None
            
            # Map the score value to a label if not provided
            if llm_score_label is None:
                score_labels = {
                    0: "CANNOT_DO",
                    1: "LOST_SKILL",
                    2: "EMERGING",
                    3: "WITH_SUPPORT",
                    4: "INDEPENDENT",
                    -1: "NOT_RATED"
                }
                llm_score_label = score_labels.get(llm_score, "NOT_RATED")
            
            logger.info(f"LLM scoring successful: {llm_score_label} ({llm_score}) with confidence {llm_confidence:.2f}")
            return llm_score, llm_score_label, llm_confidence
    except Exception as e:
        logger.error(f"Error in LLM scoring: {str(e)}")
    
    return None, None, None

@router.post("/comprehensive-assessment", response_model=ComprehensiveResult)
async def comprehensive_assessment(
    assessment_data: ComprehensiveAssessment,
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_scoring_engine)
):
    """
    Comprehensive endpoint that combines question processing, keyword management,
    response analysis, and score recording in a single call.
    
    This endpoint provides a streamlined way to process a full assessment in one request.
    """
    logger.info(f"Processing question: {assessment_data.question}")
    logger.info(f"Looking for milestone: {assessment_data.milestone_behavior}")
    
    # If engine is None, use the smart scoring endpoint directly
    if engine is None:
        try:
            logger.info("Using smart scoring endpoint as fallback")
            
            # Format the request for the smart scoring endpoint
            smart_request = {
                "parent_responses": [
                    {
                        "id": "direct-test",
                        "question": assessment_data.question,
                        "milestone_behavior": assessment_data.milestone_behavior,
                        "response": assessment_data.parent_response
                    }
                ]
            }
            
            # Call the smart scoring endpoint directly
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8003/smart-scoring/smart-comprehensive-assessment",
                    json=jsonable_encoder(smart_request)
                )
                
                if response.status_code != 200:
                    logger.error(f"Smart scoring endpoint returned error: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=500, detail="Error calling smart scoring endpoint")
                
                # Extract the result
                result = response.json()
                if not result or len(result) == 0:
                    logger.error("Smart scoring endpoint returned empty result")
                    raise HTTPException(status_code=500, detail="Empty result from smart scoring endpoint")
                
                scored_response = result[0]
                
                # Map the result to the ComprehensiveResult model
                return ComprehensiveResult(
                    question_processed=True,
                    milestone_found=True,
                    milestone_details={
                        "behavior": assessment_data.milestone_behavior,
                        "criteria": assessment_data.milestone_behavior,
                        "domain": "Unknown",
                        "age_range": "Unknown"
                    },
                    keywords_updated=[],
                    score=scored_response["score"],
                    score_label=scored_response["label"],
                    confidence=scored_response["confidence"],
                    domain="Unknown"
                )
        except Exception as e:
            logger.error(f"Error in smart scoring fallback: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
    
    # Get all milestones
    try:
        milestones = engine.get_all_milestones()
        logger.info(f"Retrieved {len(milestones)} milestones from API")
    except Exception as e:
        logger.error(f"Error getting milestones: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting milestones: {str(e)}")
    
    # Find the milestone by behavior
    milestone = None
    for m in milestones:
        # Check if m is a dictionary or an object
        try:
            if isinstance(m, dict) and m.get("behavior", "").lower() == assessment_data.milestone_behavior.lower():
                milestone = m
                break
            elif hasattr(m, "behavior") and m.behavior.lower() == assessment_data.milestone_behavior.lower():
                # Convert object to dictionary if it's not a dict
                milestone = {
                    "behavior": m.behavior,
                    "domain": getattr(m, "domain", "Unknown"),
                    "age_range": getattr(m, "age_range", ""),
                    "criteria": getattr(m, "criteria", m.behavior)
                }
                break
        except Exception as e:
            logger.warning(f"Error processing milestone: {str(e)}")
            continue
    
    # Check if milestone was found
    if not milestone:
        logger.warning(f"Milestone not found: {assessment_data.milestone_behavior}")
        return ComprehensiveResult(
            question_processed=True,
            milestone_found=False,
            score=-1,
            score_label="NOT_RATED",
            confidence=0.0,
            keywords_updated=[]
        )
    
    logger.info(f"Found milestone: {milestone['behavior']} (domain: {milestone['domain']})")
    
    # Prepare milestone context
    milestone_context = {
        "behavior": milestone["behavior"],
        "domain": milestone["domain"],
        "age_range": milestone.get("age_range", ""),
        "criteria": milestone.get("criteria", milestone["behavior"])
    }
    
    # Update keywords if provided
    keywords_updated = []
    if assessment_data.keywords:
        # Get the specific milestone key for this milestone only
        milestone_key = engine._get_milestone_key(milestone)
        
        # Update keywords for each category if provided
        for category, keywords in assessment_data.keywords.items():
            if keywords and len(keywords) > 0:
                # Update keyword dictionaries
                engine.keyword_manager.update_keywords(milestone_key, category, keywords)
                keywords_updated.append(category)
    
    # Try LLM scoring first
    llm_score, llm_score_label, llm_confidence = try_llm_scoring(
        engine, 
        assessment_data.parent_response, 
        milestone_context
    )
    
    if llm_score is not None:
        # Use LLM score directly
        score_value = llm_score
        score_label = llm_score_label
        confidence = llm_confidence
        logger.info(f"Using LLM scoring: {score_label} ({score_value})")
    else:
        # Simple keyword-based scoring for common responses
        simple_response = assessment_data.parent_response.lower().strip()
        simple_score = None
        simple_confidence = 0.9  # Increase confidence for simple keyword matching
        
        # Check for simple emerging responses first (to handle cases like "sometimes, but not always")
        if any(keyword in simple_response for keyword in ["sometimes", "occasionally", "starting to", "beginning to", "not always but sometimes", "not always"]):
            simple_score = 2  # EMERGING
            logger.info(f"Simple keyword match found for emerging response: {simple_response}")
        # Check for simple negative responses
        elif any(keyword in simple_response for keyword in ["no", "never", "not at all", "doesn't", "does not", "cannot"]):
            simple_score = 0  # CANNOT_DO
            logger.info(f"Simple keyword match found for negative response: {simple_response}")
        # Check for simple positive responses
        elif any(keyword in simple_response for keyword in ["yes", "always", "usually", "most of the time", "definitely", "absolutely"]):
            simple_score = 4  # INDEPENDENT
            logger.info(f"Simple keyword match found for positive response: {simple_response}")
        
        # Use simple keyword matching directly if we found a match
        use_simple_scoring = simple_score is not None
        
        try:
            # Only use the main scoring engine if we don't have a simple match
            if not use_simple_scoring:
                # Get detailed scoring results to see which components were used
                detailed_result = engine.score_response(
                    assessment_data.parent_response, 
                    milestone_context,
                    detailed=True  # Get detailed results with all scoring components
                )
                
                # Log which scoring components were used
                if isinstance(detailed_result, dict) and "component_results" in detailed_result:
                    for component in detailed_result["component_results"]:
                        logger.info(f"Scoring component: {component['method']} - Score: {component['score']} (Confidence: {component['confidence']:.2f})")
                
                # Extract the final score
                if isinstance(detailed_result, dict) and "component_results" in detailed_result:
                    # If we have component results, extract the score from there
                    score_value = detailed_result.get("score", 0)
                    
                    # Check if score_value is a Score enum
                    if hasattr(score_value, "value"):
                        score_value = score_value.value
                    
                    # Map the score value to a label
                    score_labels = {
                        0: "CANNOT_DO",
                        1: "LOST_SKILL",
                        2: "EMERGING",
                        3: "WITH_SUPPORT",
                        4: "INDEPENDENT",
                        -1: "NOT_RATED"
                    }
                    score_label = score_labels.get(score_value, "NOT_RATED")
                    confidence = detailed_result.get("confidence", 0.7)
                    logger.info(f"Original scoring result: {score_label} ({score_value})")
                elif isinstance(detailed_result, dict):
                    # If we have a dict but no component results, try to get the score directly
                    score_value = detailed_result.get("score", 0)
                    
                    # Check if score_value is a Score enum
                    if hasattr(score_value, "value"):
                        score_value = score_value.value
                        
                    score_label = detailed_result.get("score_label", "NOT_RATED")
                    if "score_label" not in detailed_result:
                        # Map the score value to a label if not provided
                        score_labels = {
                            0: "CANNOT_DO",
                            1: "LOST_SKILL",
                            2: "EMERGING",
                            3: "WITH_SUPPORT",
                            4: "INDEPENDENT",
                            -1: "NOT_RATED"
                        }
                        score_label = score_labels.get(score_value, "NOT_RATED")
                    confidence = detailed_result.get("confidence", 0.7)
                    logger.info(f"Original scoring result: {score_label} ({score_value})")
                else:
                    # Fallback if detailed results not available
                    score_result = detailed_result
                    score_value = score_result.score.value
                    score_label = score_result.score.name
                    confidence = score_result.confidence
                    logger.info(f"Original scoring result: {score_label} ({score_value})")
            else:
                # Use simple keyword matching
                score_value = simple_score
                score_labels = {
                    0: "CANNOT_DO",
                    1: "LOST_SKILL",
                    2: "EMERGING",
                    3: "WITH_SUPPORT",
                    4: "INDEPENDENT",
                    -1: "NOT_RATED"
                }
                score_label = score_labels.get(score_value, "NOT_RATED")
                confidence = simple_confidence
                logger.info(f"Using simple keyword scoring: {score_label} ({score_value})")
        except Exception as e:
            # If the scoring engine fails, use our simple keyword-based scoring
            logger.error(f"Error in scoring engine: {str(e)}")
            
            if use_simple_scoring:
                # Use the simple score we already calculated
                score_value = simple_score
                score_labels = {
                    0: "CANNOT_DO",
                    1: "LOST_SKILL",
                    2: "EMERGING",
                    3: "WITH_SUPPORT",
                    4: "INDEPENDENT",
                    -1: "NOT_RATED"
                }
                score_label = score_labels.get(score_value, "NOT_RATED")
                confidence = simple_confidence
                logger.info(f"Falling back to simple keyword scoring: {score_label} ({score_value})")
            else:
                # No simple score and engine failed, return NOT_RATED
                score_value = -1
                score_label = "NOT_RATED"
                confidence = 0.0
                logger.error(f"No scoring method available for response: {assessment_data.parent_response}")
    
    # Record the score in the engine
    try:
        # Convert score value to Score enum
        score_enum = None
        for score in Score:
            if score.value == score_value:
                score_enum = score
                break
        
        if score_enum is not None:
            engine.set_milestone_score(milestone, score_enum)
            logger.info(f"Score set: {milestone['behavior']} ({milestone['domain']}) = {score_label}")
    except Exception as e:
        logger.error(f"Error setting score: {str(e)}")
    
    logger.info(f"Scored {milestone['behavior']} ({milestone['domain']}): {score_label}")
    
    # Prepare response
    return ComprehensiveResult(
        question_processed=True,
        milestone_found=True,
        milestone_details={
            "behavior": milestone["behavior"],
            "criteria": milestone.get("criteria", milestone["behavior"]),
            "domain": milestone["domain"],
            "age_range": milestone.get("age_range", "")
        },
        keywords_updated=keywords_updated,
        score=score_value,
        score_label=score_label,
        confidence=confidence,
        domain=milestone["domain"]
    )

def add_routes_to_app(app):
    """Add comprehensive assessment routes to the FastAPI app."""
    # Create a new router without the comprehensive-assessment endpoint
    new_router = APIRouter(prefix="/api", tags=["comprehensive"])
    
    # We're not including the comprehensive-assessment endpoint here
    # because we're handling it directly in app.py
    
    # Add the new router to the app
    app.include_router(new_router)
    logger.info("Comprehensive assessment routes have been registered") 
 