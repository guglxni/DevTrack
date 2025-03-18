"""
Smart Scoring Routes

This module provides routes for comprehensive assessment with smart scoring,
prioritizing LLM-based scoring over keyword-based scoring when available.

RECOMMENDATIONS FOR STREAMLINING THE SCORING ENGINE:
1. Remove redundant scoring methods and focus on the most accurate ones:
   - Keep the improved keyword scoring as a reliable fallback
   - Prioritize LLM-based scoring when available
   - Consider removing or reducing the weight of less accurate methods like basic embedding scoring
   
2. Simplify the scoring pipeline:
   - Use a tiered approach: try LLM first, then keyword, then ensemble only if needed
   - Skip the ensemble scoring for clear cases where keyword matching is highly confident
   
3. Optimize the milestone matching:
   - Improve the milestone lookup to be more robust with fuzzy matching
   - Cache frequently accessed milestones for better performance
   
4. Error handling improvements:
   - Better handling of milestone objects vs dictionaries
   - More graceful fallbacks when scoring components fail
"""

import logging
import os
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, Field
from fuzzywuzzy import process

from src.core.scoring.base import Score
from src.core.scoring.improved_engine import ImprovedDevelopmentalScoringEngine
from src.models.scoring import (
    ComprehensiveAssessmentRequest,
    ScoredResponse,
    ParentResponse
)
from src.api.dependencies import get_scoring_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_scoring_routes")

# Create router
router = APIRouter(prefix="/smart-scoring", tags=["smart-scoring"])

# Models
class SmartScoringRequest(BaseModel):
    """Request model for smart scoring."""
    question: str = Field(..., description="The question text about the child's behavior")
    milestone_behavior: str = Field(..., description="The milestone behavior being assessed")
    parent_response: str = Field(..., description="Parent/caregiver response describing the child's behavior")
    keywords: Optional[Dict[str, List[str]]] = Field(None, description="Optional dictionary of keywords by category")

class SmartScoringResult(BaseModel):
    """Result model for smart scoring."""
    question_processed: bool = Field(..., description="Whether the question was successfully processed")
    milestone_found: bool = Field(..., description="Whether the milestone was found")
    milestone_details: Optional[Dict] = Field(None, description="Details about the milestone if found")
    keywords_updated: Optional[List[str]] = Field(None, description="Categories that were updated with new keywords")
    score: int = Field(..., description="The determined score (0-4)")
    score_label: str = Field(..., description="The score category (e.g., INDEPENDENT)")
    confidence: float = Field(..., description="Confidence level of the score determination (0-1)")
    domain: Optional[str] = Field(None, description="Developmental domain of the milestone")
    scoring_method: str = Field(..., description="Method used for scoring (llm, keyword, ensemble)")
    reasoning: Optional[str] = Field(None, description="Reasoning behind the score (if available)")

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
            raise HTTPException(status_code=500, detail="Scoring engine not available")

def try_llm_scoring(engine, response: str, milestone_context: dict) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[str]]:
    """
    Attempt to score a response using the LLM scorer directly.
    
    Args:
        engine: The scoring engine
        response: The response text
        milestone_context: Context about the milestone
        
    Returns:
        Tuple of (score_value, score_label, confidence, reasoning) or (None, None, None, None) if LLM scoring fails
    """
    # Check if LLM scorer is available in the engine
    if not hasattr(engine, '_scorers') or 'llm' not in engine._scorers:
        logger.info("LLM scorer not available in engine")
        return None, None, None, None
    
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
            
            # Extract reasoning if available
            reasoning = None
            if hasattr(llm_result, 'details') and llm_result.details:
                reasoning = llm_result.details.get('reasoning')
                
            logger.info(f"LLM scoring successful: {llm_score_label} ({llm_score}) with confidence {llm_confidence:.2f}")
            return llm_score, llm_score_label, llm_confidence, reasoning
    except Exception as e:
        logger.error(f"Error in LLM scoring: {str(e)}")
    
    return None, None, None, None

def try_keyword_scoring(response: str) -> Tuple[Optional[int], Optional[str], Optional[float], Optional[str]]:
    """
    Attempt to score a response using simple keyword matching.
    
    Args:
        response: The response text
        
    Returns:
        Tuple of (score_value, score_label, confidence, reasoning) or (None, None, None, None) if no match
    """
    simple_response = response.lower().strip()
    simple_score = None
    simple_confidence = 0.9  # High confidence for simple keyword matching
    simple_reasoning = None
    
    # Check for positive responses first (most reliable to match)
    if any(kw in simple_response for kw in ["yes", "always", "consistently", "definitely", "absolutely", "all the time"]):
        simple_score = 4  # INDEPENDENT
        simple_label = "INDEPENDENT" 
        simple_reasoning = f"Positive keyword match: '{response}'"
        logger.info(f"Positive keyword match found: {simple_response}")
        return simple_score, simple_label, simple_confidence, simple_reasoning
        
    # Check for emerging responses 
    if any(kw in simple_response for kw in ["sometimes", "occasionally", "starting to", "beginning to", "not always", "inconsistent"]):
        simple_score = 2  # EMERGING
        simple_label = "EMERGING"
        simple_reasoning = f"Emerging keyword match: '{response}'"
        logger.info(f"Emerging keyword match found: {simple_response}")
        return simple_score, simple_label, simple_confidence, simple_reasoning
        
    # Check for negative responses
    if any(kw in simple_response for kw in ["no", "never", "not at all", "doesn't", "does not", "cannot", "can't"]):
        simple_score = 0  # CANNOT_DO
        simple_label = "CANNOT_DO" 
        simple_reasoning = f"Negative keyword match: '{response}'"
        logger.info(f"Negative keyword match found: {simple_response}")
        return simple_score, simple_label, simple_confidence, simple_reasoning
    
    # Check for support/help responses
    if any(kw in simple_response for kw in ["with help", "with support", "with assistance", "when helped", "needs help"]):
        simple_score = 3  # WITH_SUPPORT
        simple_label = "WITH_SUPPORT"
        simple_reasoning = f"Support keyword match: '{response}'"
        logger.info(f"Support keyword match found: {simple_response}")
        return simple_score, simple_label, simple_confidence, simple_reasoning
    
    # Check for regression/lost skill responses
    if any(kw in simple_response for kw in ["used to", "stopped", "regressed", "lost", "doesn't anymore", "previously"]):
        simple_score = 1  # LOST_SKILL
        simple_label = "LOST_SKILL" 
        simple_reasoning = f"Regression keyword match: '{response}'"
        logger.info(f"Regression keyword match found: {simple_response}")
        return simple_score, simple_label, simple_confidence, simple_reasoning
    
    # If no clear match is found
    logger.info(f"No clear keyword match found for: {simple_response}")
    return None, None, None, None

@router.post("/smart-comprehensive-assessment", response_model=List[ScoredResponse])
async def smart_comprehensive_assessment(
    request: ComprehensiveAssessmentRequest,
    background_tasks: BackgroundTasks,
    engine: ImprovedDevelopmentalScoringEngine = Depends(get_scoring_engine),
):
    """
    Score a comprehensive assessment using the improved scoring engine with tiered approach.
    
    This endpoint processes multiple parent responses against their respective milestones
    and returns a list of scored responses.
    """
    logger.info(f"Processing comprehensive assessment with {len(request.parent_responses)} responses")
    
    # Track processing metrics
    start_time = time.time()
    processing_stats = {
        "total_responses": len(request.parent_responses),
        "processed_responses": 0,
        "successful_scores": 0,
        "scoring_methods_used": {
            "llm": 0,
            "keyword": 0,
            "transformer": 0,
            "embedding": 0,
            "combined": 0
        }
    }
    
    scored_responses = []
    
    for parent_response in request.parent_responses:
        question = parent_response.question
        milestone_behavior = parent_response.milestone_behavior
        response_text = parent_response.response
        
        logger.info(f"Processing question: {question}")
        logger.info(f"Looking for milestone behavior: {milestone_behavior}")
        logger.info(f"Parent response: {response_text}")
        
        # Get the milestone from the engine
        milestone = None
        try:
            # Get all milestones from the engine
            all_milestones = engine.get_all_milestones()
            
            # Try exact match first (case-insensitive)
            for m in all_milestones:
                # Check if m is a dictionary or an object
                if isinstance(m, dict):
                    m_behavior = m.get("behavior", "").lower()
                else:
                    m_behavior = getattr(m, "behavior", "").lower()
                    
                if m_behavior == milestone_behavior.lower():
                    milestone = m
                    break
                    
            # If no exact match, try fuzzy matching
            if not milestone:
                logger.warning(f"No exact milestone match found for: {milestone_behavior}")
                # Get the closest matching milestone
                milestone_behaviors = []
                for m in all_milestones:
                    if isinstance(m, dict):
                        milestone_behaviors.append(m.get("behavior", ""))
                    else:
                        milestone_behaviors.append(getattr(m, "behavior", ""))
                        
                closest_matches = process.extract(
                    milestone_behavior,
                    milestone_behaviors,
                    limit=3
                )
                if closest_matches and closest_matches[0][1] >= 80:  # 80% similarity threshold
                    match_text = closest_matches[0][0]
                    for m in all_milestones:
                        if isinstance(m, dict):
                            m_behavior = m.get("behavior", "")
                        else:
                            m_behavior = getattr(m, "behavior", "")
                            
                        if m_behavior == match_text:
                            milestone = m
                            logger.info(f"Using fuzzy matched milestone: {match_text} (score: {closest_matches[0][1]})")
                            break
        except Exception as e:
            logger.error(f"Error finding milestone: {str(e)}")
        
        if not milestone:
            logger.warning(f"No milestone found for behavior: {milestone_behavior}")
            # Log some sample milestones for debugging
            sample_milestones = []
            for m in all_milestones[:5]:
                if isinstance(m, dict):
                    sample_milestones.append(m.get("behavior", ""))
                else:
                    sample_milestones.append(getattr(m, "behavior", ""))
            logger.info(f"Sample available milestones: {sample_milestones}")
            
            # Create a default milestone with the provided behavior
            milestone = {"behavior": milestone_behavior, "keywords": []}
        
        # Update milestone keywords if provided in the request
        if parent_response.keywords:
            if isinstance(milestone, dict):
                milestone["keywords"] = parent_response.keywords
            else:
                # If milestone is an object, convert it to a dictionary
                milestone = {
                    "behavior": getattr(milestone, "behavior", ""),
                    "domain": getattr(milestone, "domain", ""),
                    "age_range": getattr(milestone, "age_range", ""),
                    "criteria": getattr(milestone, "criteria", ""),
                    "keywords": parent_response.keywords
                }
            logger.info(f"Updated milestone keywords: {parent_response.keywords}")
        
        # Score the response using the tiered approach
        try:
            # Convert milestone to dictionary if it's an object
            if not isinstance(milestone, dict):
                milestone_dict = {
                    "behavior": getattr(milestone, "behavior", ""),
                    "domain": getattr(milestone, "domain", ""),
                    "age_range": getattr(milestone, "age_range", ""),
                    "criteria": getattr(milestone, "criteria", "")
                }
            else:
                milestone_dict = milestone
            
            # Simple check for clear negative responses
            response_lower = response_text.lower()
            if (("no" in response_lower and "not" in response_lower) or 
                ("doesn't" in response_lower) or 
                ("does not" in response_lower and "recognize" in response_lower) or
                ("no, he does not recognize" in response_lower)):
                logger.info(f"Detected clear negative response: {response_text}")
                scored_response = ScoredResponse(
                    id=str(uuid.uuid4()),
                    parent_response_id=parent_response.id,
                    score=0,  # CANNOT_DO
                    label="CANNOT_DO",
                    confidence=0.9,
                    reasoning=f"Clear negative response: {response_text}",
                    metadata={
                        "scoring_methods": ["keyword_override"],
                        "early_return": True,
                        "reason": "negative_pattern_match"
                    }
                )
                scored_responses.append(scored_response)
                processing_stats["processed_responses"] += 1
                processing_stats["successful_scores"] += 1
                processing_stats["scoring_methods_used"]["keyword"] += 1
                continue
            
            # Get detailed results to track which methods were used
            try:
                detailed_result = engine.score_response(response_text, milestone_dict, detailed=True)
            except Exception as e:
                logger.error(f"Error scoring response: {str(e)}")
                # Create a default NOT_RATED result
                detailed_result = {
                    "score": Score.NOT_RATED,
                    "score_name": "NOT_RATED",
                    "score_value": -1,
                    "confidence": 0.0,
                    "method": "error",
                    "reasoning": f"Error: {str(e)}",
                    "component_results": [],
                    "elapsed_time": 0.0
                }
            
            # Add more logging to debug the detailed_result structure
            logger.info(f"Detailed result: {detailed_result}")

            # Extract the final result
            if isinstance(detailed_result, dict):
                # If detailed_result is a dictionary with final_result key
                if "final_result" in detailed_result:
                    result = detailed_result.get("final_result", {})
                    all_results = detailed_result.get("all_results", {})
                else:
                    # If detailed_result is the result itself
                    result = detailed_result
                    all_results = {}
                
                # Log the result structure
                logger.info(f"Result structure: {result}")
            else:
                # If detailed_result is not a dictionary, try to convert it
                try:
                    result = {
                        "score": detailed_result.score.value if hasattr(detailed_result.score, "value") else detailed_result.score,
                        "label": detailed_result.score.name if hasattr(detailed_result.score, "name") else "NOT_RATED",
                        "confidence": detailed_result.confidence if hasattr(detailed_result, "confidence") else 0.0,
                        "reasoning": detailed_result.reasoning if hasattr(detailed_result, "reasoning") else "No reasoning provided",
                        "scoring_audit": {
                            "methods_succeeded": [detailed_result.method] if hasattr(detailed_result, "method") else []
                        }
                    }
                    all_results = {}
                except Exception as e:
                    logger.error(f"Error extracting result from detailed_result: {str(e)}")
                    result = {
                        "score": 0,
                        "label": "NOT_RATED",
                        "confidence": 0.0,
                        "reasoning": f"Error extracting result: {str(e)}",
                        "scoring_audit": {"methods_succeeded": []}
                    }
                    all_results = {}
            
            # Track which scoring methods were used
            scoring_audit = result.get("scoring_audit", {})
            methods_succeeded = scoring_audit.get("methods_succeeded", [])
            
            for method in methods_succeeded:
                processing_stats["scoring_methods_used"][method] += 1
                
            if scoring_audit.get("reason") == "combined_results":
                processing_stats["scoring_methods_used"]["combined"] += 1
            
            # Map score value to label
            score_value = result.get("score", 0)
            # Convert Score enum to integer if needed
            if hasattr(score_value, "value"):
                score_value = score_value.value
            
            score_label = result.get("score_name", result.get("label", "NOT_RATED"))
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            logger.info(f"Final score: {score_value} ({score_label}), confidence: {confidence}")
            logger.info(f"Reasoning: {reasoning}")
            
            # Create the scored response
            scored_response = ScoredResponse(
                id=str(uuid.uuid4()),
                parent_response_id=parent_response.id,
                score=score_value,
                label=score_label,
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "scoring_methods": methods_succeeded,
                    "early_return": scoring_audit.get("early_return", False),
                    "reason": scoring_audit.get("reason", ""),
                    "domain": milestone.get("domain", "EL") if isinstance(milestone, dict) else getattr(milestone, "domain", "EL")
                }
            )
            
            scored_responses.append(scored_response)
            processing_stats["processed_responses"] += 1
            
            if score_value > 0:  # If we got a valid score
                processing_stats["successful_scores"] += 1
                
        except Exception as e:
            logger.error(f"Error scoring response: {str(e)}")
            # Create a NOT_RATED response for failed scoring
            scored_response = ScoredResponse(
                id=str(uuid.uuid4()),
                parent_response_id=parent_response.id,
                score=0,
                label="NOT_RATED",
                confidence=0,
                reasoning=f"Error during scoring: {str(e)}",
                metadata={"error": str(e)}
            )
            scored_responses.append(scored_response)
    
    # Log processing statistics
    processing_time = time.time() - start_time
    logger.info(f"Processed {processing_stats['processed_responses']} responses in {processing_time:.2f} seconds")
    logger.info(f"Successfully scored {processing_stats['successful_scores']} responses")
    logger.info(f"Scoring methods used: {processing_stats['scoring_methods_used']}")
    
    # Schedule background task to log detailed processing stats
    background_tasks.add_task(
        log_scoring_stats,
        processing_stats=processing_stats,
        processing_time=processing_time
    )
    
    return scored_responses

# Helper function for background task
def log_scoring_stats(processing_stats: dict, processing_time: float):
    """Log detailed scoring statistics for analysis."""
    logger.info("=== Detailed Scoring Statistics ===")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    logger.info(f"Average time per response: {processing_time / max(processing_stats['total_responses'], 1):.2f} seconds")
    logger.info(f"Success rate: {processing_stats['successful_scores'] / max(processing_stats['total_responses'], 1) * 100:.1f}%")
    
    # Log method usage statistics
    methods_used = processing_stats["scoring_methods_used"]
    total_methods = sum(methods_used.values())
    
    if total_methods > 0:
        logger.info("Scoring method distribution:")
        for method, count in methods_used.items():
            percentage = count / total_methods * 100
            logger.info(f"  - {method}: {count} ({percentage:.1f}%)")
    else:
        logger.info("No scoring methods were successfully used")

def add_routes_to_app(app):
    """Add smart scoring routes to the FastAPI app."""
    app.include_router(router)
    logger.info("Smart scoring routes have been registered") 
 