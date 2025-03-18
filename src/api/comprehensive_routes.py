"""
Comprehensive Assessment Routes

This module provides API routes for the comprehensive assessment functionality,
combining question processing, keyword management, response analysis, and score recording.
"""

from typing import Dict, List, Optional, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
import logging
import traceback

from src.core.scoring.base import Score
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine

# Configure logging
logger = logging.getLogger(__name__)

# Try to import hybrid scoring if available
try:
    from src.core.hybrid_scoring import hybrid_score_response
    HYBRID_SCORER_AVAILABLE = True
    logger.info("Hybrid scoring is available for comprehensive assessment")
except ImportError:
    HYBRID_SCORER_AVAILABLE = False
    logger.warning("Hybrid scoring not available, falling back to basic scoring")

# Create router
router = APIRouter(tags=["assessment"])

# ---- Models ----

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


# ---- Dependencies ----

def get_scoring_engine():
    """
    Get or create a scoring engine instance
    
    Returns:
        ImprovedDevelopmentalScoringEngine: The scoring engine instance
    """
    # Try to get the engine from the main app
    try:
        from src.app import engine
        logger.info("Successfully retrieved engine from src.app")
        return engine
    except ImportError:
        # Create a new engine if not available from app
        logger.info("Creating new scoring engine instance")
        engine_config = {
            "enable_keyword_scorer": True,
            "enable_embedding_scorer": True,
            "enable_transformer_scorer": True,
            "enable_llm_scorer": True,  # Explicitly enable LLM-based scoring
            "use_tiered_approach": False,  # Use ensemble approach to include all scorers
            "score_weights": {
                "keyword": 0.3,
                "embedding": 0.3,
                "transformer": 0.2,
                "llm": 0.4  # Give higher weight to LLM scorer
            },
            "confidence_threshold": 0.7,
            "llm_threshold": 0.5  # Lower threshold to use LLM more often
        }
        return ImprovedDevelopmentalScoringEngine(config=engine_config)


# ---- Routes ----

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

@router.post("/comprehensive-assessment", status_code=200, response_model=ComprehensiveResult)
async def comprehensive_assessment(
    assessment_data: ComprehensiveAssessment,
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_scoring_engine)
):
    """
    Comprehensive endpoint that combines question processing, keyword management,
    response analysis, and score recording in a single call.
    
    This endpoint provides a streamlined way to process a full assessment in one request.
    """
    try:
        logger.info("Successfully retrieved engine from src.app")
        logger.info(f"Processing question: {assessment_data.question}")
        logger.info(f"Looking for milestone: {assessment_data.milestone_behavior}")
        
        # Step 1: Find the milestone
        # Instead of using engine.find_milestone_by_name, we'll get all milestones from the API
        # and find the matching milestone ourselves
        try:
            # Import the get_milestones function from src.app
            from src.app import get_milestones
            
            # Get all milestones from the API
            all_milestones = await get_milestones()
            logger.info(f"Retrieved {len(all_milestones)} milestones from API")
            
            # Find the milestone by name
            milestone = None
            for m in all_milestones:
                if m["behavior"].lower() == assessment_data.milestone_behavior.lower():
                    milestone = m
                    break
            
            if not milestone:
                # If not found, try the engine's find_milestone_by_name as a fallback
                logger.info(f"Milestone not found in API milestones, trying engine.find_milestone_by_name")
                milestone = engine.find_milestone_by_name(assessment_data.milestone_behavior)
        except Exception as e:
            logger.error(f"Error getting milestones from API: {str(e)}")
            logger.info(f"Falling back to engine.find_milestone_by_name")
            # Fallback to the engine's find_milestone_by_name method
            milestone = engine.find_milestone_by_name(assessment_data.milestone_behavior)
        
        if not milestone:
            raise HTTPException(status_code=404, detail=f"Milestone '{assessment_data.milestone_behavior}' not found")
        
        logger.info(f"Found milestone: {milestone['behavior'] if isinstance(milestone, dict) else milestone.behavior} (domain: {milestone['domain'] if isinstance(milestone, dict) else milestone.domain})")
        
        # Step 2: Update keywords if provided
        keywords_updated = []
        if assessment_data.keywords:
            try:
                milestone_key = engine._get_milestone_key(milestone)
                logger.info(f"Using milestone key: {milestone_key}")
                
                # Update keywords for each category
                for category, keywords in assessment_data.keywords.items():
                    if keywords and len(keywords) > 0:
                        logger.info(f"Updating keywords for category {category}: {keywords}")
                        engine._scorers["keyword"].keyword_manager.update_keywords(milestone_key, category, keywords)
                        keywords_updated.append(category)
            except Exception as e:
                logger.error(f"Error updating keywords: {str(e)}")
        
        # Step 3: Score the response using detailed mode to get all scoring components
        logger.info("Using original scoring logic for response analysis")
        
        # Create milestone context
        milestone_context = {
            "behavior": milestone["behavior"] if isinstance(milestone, dict) else milestone.behavior,
            "domain": milestone["domain"] if isinstance(milestone, dict) else milestone.domain,
            "age_range": milestone["age_range"] if isinstance(milestone, dict) else milestone.age_range,
            "criteria": milestone["criteria"] if isinstance(milestone, dict) else milestone.criteria
        }
        
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
                        logger.info(f"Using ensemble scoring: {score_label} ({score_value})")
            except Exception as e:
                # If the scoring engine fails, use our simple keyword-based scoring
                logger.error(f"Error in scoring engine: {str(e)}")
                logger.info("Falling back to simple keyword-based scoring")
                
                if simple_score is not None:
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
                    logger.info(f"Simple keyword scoring result: {score_label} ({score_value})")
                else:
                    # If no simple match found, default to NOT_RATED
                    score_value = -1
                    score_label = "NOT_RATED"
                    confidence = 0.0
                    logger.info("No scoring method succeeded, defaulting to NOT_RATED")
            
            # If we still have a NOT_RATED score but we found a simple keyword match, use that instead
            if score_value == -1 and simple_score is not None:
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
                logger.info(f"Overriding NOT_RATED with simple keyword match: {score_label} ({score_value})")
            
            # Ensure the score_label is correctly mapped to the score value
            score_labels = {
                0: "CANNOT_DO",
                1: "LOST_SKILL",
                2: "EMERGING",
                3: "WITH_SUPPORT",
                4: "INDEPENDENT",
                -1: "NOT_RATED"
            }
            score_label = score_labels.get(score_value, "NOT_RATED")
            logger.info(f"Final score: {score_label} ({score_value})")
            
            # Step 4: Record the score
            if isinstance(milestone, dict):
                # If milestone is a dict, we need to create a Milestone object
                from src.core.scoring.improved_engine import Milestone
                milestone_obj = Milestone()
                milestone_obj.behavior = milestone["behavior"]
                milestone_obj.domain = milestone["domain"]
                milestone_obj.age_range = milestone["age_range"]
                milestone_obj.criteria = milestone["criteria"]
                engine.set_milestone_score(milestone_obj, Score(score_value))
            else:
                # If milestone is already an object, we can use it directly
                engine.set_milestone_score(milestone, Score(score_value))
            logger.info(f"Scored {milestone['behavior'] if isinstance(milestone, dict) else milestone.behavior} ({milestone['domain'] if isinstance(milestone, dict) else milestone.domain}): {score_label}")
            
            # Step 5: Prepare the response
            logger.info("Preparing response")
            milestone_details = {
                "behavior": milestone["behavior"] if isinstance(milestone, dict) else milestone.behavior,
                "criteria": milestone["criteria"] if isinstance(milestone, dict) else milestone.criteria,
                "domain": milestone["domain"] if isinstance(milestone, dict) else milestone.domain,
                "age_range": milestone["age_range"] if isinstance(milestone, dict) else milestone.age_range
            }
            
            return {
                "question_processed": True,
                "milestone_found": True,
                "milestone_details": milestone_details,
                "keywords_updated": keywords_updated,
                "score": score_value,
                "score_label": score_label,
                "confidence": confidence,
                "domain": milestone["domain"] if isinstance(milestone, dict) else milestone.domain
            }
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error and traceback
        logger.error(f"Error processing comprehensive assessment: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing assessment: {str(e)}")


# Function to register routes with the main app
def add_routes_to_app(app):
    """Add comprehensive assessment routes to the FastAPI app."""
    logger.info("Registering comprehensive assessment routes")
    app.include_router(router) 
        # Step 3: Analyze the response (similar to /score-response endpoint)
        # Try to use the hybrid scorer for more reliable scoring if available
        if HYBRID_SCORER_AVAILABLE:
            logger.info("Using hybrid scoring for response analysis")
            try:
                # Use our enhanced hybrid scoring approach with the provided keywords
                result = hybrid_score_response(
                    assessment_data.milestone_behavior, 
                    assessment_data.parent_response,
                    assessment_data.keywords
                )
                
                score_value = result["score"]
                score_label = result["score_label"]
                confidence = result["confidence"]
                
                logger.info(f"Hybrid scoring result: {score_label} ({score_value}) with confidence {confidence}")
                
                # Map score value to Score enum for the engine
                score_enum = None
                for score in Score:
                    if score.value == score_value:
                        score_enum = score
                        break
                
                # Record the score in the engine if we have a valid score
                if score_enum is not None:
                    engine.set_milestone_score(milestone, score_enum)
                    logger.info(f"Scored {milestone.behavior} ({milestone.domain}): {score_enum.name}")
                else:
                    logger.warning(f"Could not map score value {score_value} to a Score enum")
            except Exception as e:
                logger.error(f"Error in hybrid scoring: {str(e)}\n{traceback.format_exc()}")
                # Fall back to the original scoring logic if hybrid scoring fails
                logger.info("Falling back to original scoring logic due to hybrid scoring error")
                score_enum = await engine.analyze_response(assessment_data.parent_response, milestone)
                score_value = score_enum.value
                score_label = score_enum.name
                confidence = 0.85  # Default confidence
                
                # Record the score
                engine.set_milestone_score(milestone, score_enum)
                logger.info(f"Scored {milestone.behavior} ({milestone.domain}): {score_enum.name}")
        else:
            logger.info("Using original scoring logic for response analysis")
            try:
                # Fall back to the original scoring logic
                score_enum = await engine.analyze_response(assessment_data.parent_response, milestone)
                score_value = score_enum.value
                score_label = score_enum.name
                confidence = 0.85  # Default confidence
                
                logger.info(f"Original scoring result: {score_label} ({score_value})")
                
                # Record the score
                engine.set_milestone_score(milestone, score_enum)
                logger.info(f"Scored {milestone.behavior} ({milestone.domain}): {score_enum.name}")
            except Exception as e:
                logger.error(f"Error in original scoring: {str(e)}\n{traceback.format_exc()}")
                raise
        
        # Step 4: Return the comprehensive result
        logger.info("Preparing response")
        return {
            "question_processed": True,
            "milestone_found": True,
            "milestone_details": milestone_details,
            "keywords_updated": keywords_updated,
            "score": score_value,
            "score_label": score_label,
            "confidence": confidence,
            "domain": milestone.domain
        }
    except HTTPException as http_ex:
        # Re-raise HTTP exceptions with logging
        logger.error(f"HTTP Exception: {http_ex.detail}")
        raise
    except Exception as e:
        # Log the full traceback for better debugging
        error_msg = f"Error processing assessment: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)


# Function to register routes with the main app
def add_routes_to_app(app):
    """Add comprehensive assessment routes to the FastAPI app."""
    logger.info("Registering comprehensive assessment routes")
    app.include_router(router) 